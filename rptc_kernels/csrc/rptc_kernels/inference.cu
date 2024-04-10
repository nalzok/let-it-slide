#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cuda_pipeline.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xFFFFFFFFU
#define HALF_MASK 0x0000FFFFU

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


__global__ static void
decompress_kernel(
    half *__restrict__ out,
    const uint4 *__restrict__ compressed,
    const half *__restrict__ codebook,
    size_t compressed_m,
    size_t compressed_n
) {
    size_t laneId = threadIdx.x % warpSize;
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    for (int elem_idx = threadId; elem_idx < compressed_m * compressed_n; elem_idx += gridDim.x * blockDim.x) {
        uint4 inputs = compressed[elem_idx];
        uint16_t state = __shfl_up_sync(FULL_MASK, inputs.w, 1);
        if (laneId == 0) {
            if (elem_idx % compressed_n == 0) {
                // first element in a row
                state = 0;
            } else {
                const uint16_t *ptr = reinterpret_cast<const uint16_t *>(compressed + elem_idx);
                state = ptr[-1];
            }
        }

        half *__restrict__ output = out + elem_idx * 64;

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = __funnelshift_l(inputs.x, state, 2);
            inputs.x <<= 2;
            output[i] = codebook[state];
        }

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = __funnelshift_l(inputs.y, state, 2);
            inputs.y <<= 2;
            output[16+i] = codebook[state];
        }

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = __funnelshift_l(inputs.z, state, 2);
            inputs.z <<= 2;
            output[32+i] = codebook[state];
        }

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = __funnelshift_l(inputs.w, state, 2);
            inputs.w <<= 2;
            output[48+i] = codebook[state];
        }
    }
}


__host__ extern void decompress(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &out
) {
    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 2);
    TORCH_CHECK(compressed.size(1) % (4 * 32) == 0);    // each warp has 32 threads, each handling an int4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t compressed_m = compressed.size(0);
    size_t compressed_n = compressed.size(1) / 4;
    size_t m = compressed_m;
    size_t n = compressed_n * 64;   // at 2 bit, each int4 has 4x32 bits = 4x16 weights

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.size(0) == 1<<16);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.size(0) == m);
    TORCH_CHECK(out.size(1) == n);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, compressed.get_device());
    size_t grid_size = static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = 1024;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    decompress_kernel<<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint4 *)compressed.data_ptr<int32_t>(),
        (const half *)codebook.data_ptr<c10::Half>(),
        compressed_m,
        compressed_n);
    
    gpuErrchk(cudaPeekAtLastError());
}


template <size_t L>
__global__ static void
decompress_matvec_kernel(
    half *__restrict__ out,
    const uint4 *__restrict__ compressed,
    const half *__restrict__ codebook,
    const half *__restrict__ x,
    size_t compressed_m,
    size_t compressed_n
) {
    const half *__restrict__ codebook_ptr = codebook;

    if constexpr (L < 16) {
        // TODO: what's the lifetime of smem?
        __shared__ half smem[1<<L];
        for (int cb_idx = threadIdx.x; cb_idx < (1<<L)/8; cb_idx += blockDim.x) {
            reinterpret_cast<int4 *>(smem)[cb_idx] = reinterpret_cast<const int4 *>(codebook)[cb_idx];
        }
        codebook_ptr = smem;
    }

    constexpr uint16_t mask = (1<<L) - 1;

    size_t laneId = threadIdx.x % warpSize;
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    for (int elem_idx = threadId; elem_idx < compressed_m * compressed_n; elem_idx += gridDim.x * blockDim.x) {
        uint4 inputs = compressed[elem_idx];
        uint16_t state = __shfl_up_sync(FULL_MASK, inputs.w, 1);
        if (laneId == 0) {
            if (elem_idx % compressed_n == 0) {
                // first element in a row
                state = 0;
            } else {
                const uint16_t *ptr = reinterpret_cast<const uint16_t *>(compressed + elem_idx);
                state = ptr[-1];
            }
        }

        const half *__restrict__ activations = out + (elem_idx % compressed_n) * 64;
        float delta = 0;

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = mask & __funnelshift_l(inputs.x, state, 2);
            inputs.x <<= 2;
            delta = __fmaf_rn(__half2float(codebook_ptr[state]), __half2float(activations[i]), delta);
        }

        activations += 16;

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = mask & __funnelshift_l(inputs.y, state, 2);
            inputs.y <<= 2;
            delta = __fmaf_rn(__half2float(codebook_ptr[state]), __half2float(activations[i]), delta);
        }

        activations += 16;

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = mask & __funnelshift_l(inputs.z, state, 2);
            inputs.z <<= 2;
            delta = __fmaf_rn(__half2float(codebook_ptr[state]), __half2float(activations[i]), delta);
        }

        activations += 16;

        #pragma unroll
        for (int i = 0; i < 16; i += 1) {
            state = mask & __funnelshift_l(inputs.w, state, 2);
            inputs.w <<= 2;
            delta = __fmaf_rn(__half2float(codebook_ptr[state]), __half2float(activations[i]), delta);
        }

        size_t out_idx = elem_idx / compressed_n;
        atomicAdd(out + out_idx, __float2half(delta));
    }
}


template <size_t L>
__host__ static void decompress_matvec(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
) {
    static_assert(L <= 16, "Shift register length should not exceed 16 as the kernel uses uint16_t");

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 2);
    TORCH_CHECK(compressed.size(1) % (4 * 32) == 0);    // each warp has 32 threads, each handling an int4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t compressed_m = compressed.size(0);
    size_t compressed_n = compressed.size(1) / 4;
    size_t m = compressed_m;
    size_t n = compressed_n * 64;   // at 2 bit, each int4 has 4x32 bits = 4x16 weights

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.size(0) == 1<<L);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1);
    TORCH_CHECK(x.size(0) == n);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 1);
    TORCH_CHECK(out.size(0) == m);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, compressed.get_device());
    size_t grid_size = static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = 1024;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    cudaFuncSetAttribute(
            decompress_matvec_kernel<L>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);

    decompress_matvec_kernel<L><<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint4 *)compressed.data_ptr<int32_t>(),
        (const half *)codebook.data_ptr<c10::Half>(),
        (const half *)x.data_ptr<c10::Half>(),
        compressed_m,
        compressed_n);
    
    gpuErrchk(cudaPeekAtLastError());
}

__host__ extern void decompress_matvec_16(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<16>(compressed, codebook, x, out);
}

__host__ extern void decompress_matvec_14(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<14>(compressed, codebook, x, out);
}

__host__ extern void decompress_matvec_12(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<12>(compressed, codebook, x, out);
}

__host__ extern void decompress_matvec_10(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<10>(compressed, codebook, x, out);
}

__host__ extern void decompress_matvec_8(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<8>(compressed, codebook, x, out);
}
