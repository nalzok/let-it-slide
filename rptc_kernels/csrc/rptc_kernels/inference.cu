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

#define MAX_THREADS_PER_BLOCK 256

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

typedef union __align__(4) alias_half_uint16 {
    half2 foo;
    uint16_t bar[2];
} alias_half_uint16;


__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK)
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


__host__ extern float decompress(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &out
) {
    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 2);
    TORCH_CHECK(compressed.size(1) % (4 * 32) == 0);    // each warp has 32 threads, each handling an uint4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t compressed_m = compressed.size(0);
    size_t compressed_n = compressed.size(1) / 4;
    size_t m = compressed_m;
    size_t n = compressed_n * 64;   // at 2 bit, each uint4 has 4x32 bits = 4x16 weights

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
    size_t grid_size = 2 * static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = MAX_THREADS_PER_BLOCK;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    decompress_kernel<<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint4 *)compressed.data_ptr<int32_t>(),
        (const half *)codebook.data_ptr<c10::Half>(),
        compressed_m,
        compressed_n);
    
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return msecTotal;
}


template <size_t L>
__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK)
decompress_matvec_kernel(
    half *__restrict__ out,
    const uint4 *__restrict__ compressed,
    const half *__restrict__ codebook,
    const half2 *__restrict__ x,
    size_t iters_per_thread,
    size_t m,
    size_t n
) {
    const half *__restrict__ codebook_ptr = codebook;
    if constexpr (L < 16) {
        __shared__ half smem_codebook[1<<L];
        size_t offset_quarter = 1<<(L-2);
        // uint32_t *smem_x = reinterpret_cast<uint32_t *>(smem_codebook);
        // uint32_t *smem_y = reinterpret_cast<uint32_t *>(smem_codebook + offset_quarter);
        // uint32_t *smem_w = reinterpret_cast<uint32_t *>(smem_codebook + offset_quarter * 2);
        // uint32_t *smem_z = reinterpret_cast<uint32_t *>(smem_codebook + offset_quarter * 3);

        for (size_t idx = threadIdx.x; idx < (1<<L)/8; idx += blockDim.x) {
            // read in uint4 from global memory, and then write in uint32_t into shared memory
            uint4 quadruple = reinterpret_cast<const uint4 *>(codebook)[idx];
            reinterpret_cast<uint4 *>(smem_codebook)[idx] = quadruple;
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 1));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 2));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 3));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 4));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 5));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 6));
            // asm("prefetchu.L1 [%0];" : : "l" (codebook + idx * 8 + 7));
            // uintptr_t ptr = reinterpret_cast<uintptr_t>(&codebook[idx]);
            // if (ptr % 128 == 0) {
            //     asm("prefetch.local.L1 [%0];" : : "l" (ptr));
            // }
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 1));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 2));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 3));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 4));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 5));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 6));
            // asm("prefetch.global.L1 [%0];" : : "l" (codebook + idx * 8 + 7));
            // smem_x[idx] = quadruple.x;
            // smem_y[idx] = quadruple.y;
            // smem_z[idx] = quadruple.z;
            // smem_w[idx] = quadruple.w;
        }
        codebook_ptr = smem_codebook;
    }

    __syncthreads();

    constexpr uint16_t mask = (1<<L) - 1;
    constexpr uint16_t lane_mask = mask & ~0b111110;

    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t laneId = threadIdx.x % warpSize;
    size_t warpId = threadId / warpSize;
    size_t strideC = blockDim.x * gridDim.x;
    size_t strideX = warpSize * 4;

    uint32_t carry = 0U;
    half2 inners[4] = {
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f),
    };

    for (size_t iter = 0; iter < iters_per_thread; iter += 1) {
        // uint4 elem = compressed[iter * strideC + threadId];

        uint4 elem;
        asm volatile ("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(elem.x), "=r"(elem.y), "=r"(elem.z), "=r"(elem.w)
                : "l" (compressed + iter * strideC + threadId));
        // ld.global.nc.v4.u32     {%r427, %r428, %r429, %r430}, [%rd323];

        // send w in lane X to carry in lane X+1, lane 0 not updated
        carry = __shfl_up_sync(FULL_MASK, elem.w, 1);

        // send w in lane 31 to carry in lane 0, lane 1-31 not updated
        uint32_t next_carry = __shfl_down_sync(FULL_MASK, elem.w, 31);

        uint32_t reg_c[5] = { carry, elem.x, elem.y, elem.z, elem.w };

        alias_half_uint16 reg_w[4][8];
        half2 reg_a[4][8];
        #pragma unroll
        for (size_t k = 0; k < 4; k += 1) {
            #pragma unroll
            for (size_t j = 0; j < 8; j += 1) {
                // TODO: would uint16_t be faster?
                uint32_t state_x = (lane_mask & __funnelshift_l(reg_c[k+1], reg_c[k], 4*j)) | (laneId << 1);
                uint32_t state_y = (lane_mask & __funnelshift_l(reg_c[k+1], reg_c[k], 4*j+2)) | (laneId << 1);

                // state_x = state_x * (2*state_x+1);
                // state_x = state_x * 1664525 + 1013904223;
                // state_y = state_y * (2*state_y+1);
                // state_y = state_y * 1664525 + 1013904223;

                // reg_w[k][j].foo = __halves2half2(codebook_ptr[idx_x], codebook_ptr[idx_y]);
                // TODO: Section 3.2 in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10318209
                // reg_w[k][j].foo = h2cos(__halves2half2(
                //     __float2half(sinf(__uint_as_float(idx_x))),
                //     __float2half(fabsf(__uint_as_float(idx_y)))
                // ));
                asm volatile ("ld.global.nc.b16 %0, [%1];"
                        : "=h"(reg_w[k][j].bar[0])
                        : "l" (codebook_ptr + state_x));
                asm volatile ("ld.global.nc.b16 %0, [%1];"
                        : "=h"(reg_w[k][j].bar[1])
                        : "l" (codebook_ptr + state_y));
                reg_a[k][j] = x[((iter * 4 + k) * 8 + j) * warpSize + laneId];
            }
        }

        #pragma unroll
        for (size_t j = 0; j < 8; j += 1) {
            #pragma unroll
            for (size_t k = 0; k < 4; k += 1) {
                inners[k] = __hfma2(reg_w[k][j].foo, reg_a[k][j], inners[k]);
            }
        }

        carry = next_carry;
    }

    half2 inner01 = __hadd2(inners[0], inners[1]);
    half2 inner23 = __hadd2(inners[2], inners[3]);
    half2 inner0123 = __hadd2(inner01, inner23);

    for (size_t offset = 16; offset > 0; offset /= 2) {
        inner0123 = __hadd2(inner0123, __shfl_down_sync(FULL_MASK, inner0123, offset));
    }

    if (laneId == 0) {
        out[warpId] = __hadd(inner0123.x, inner0123.y);
    }
}


template <size_t L>
__host__ static float decompress_matvec(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
) {
    static_assert(L <= 32, "Shift register length should not exceed 32 as the kernel uses uint32_t");

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 3);
    TORCH_CHECK(compressed.size(2) == 32 * 4);  // each warp reads an uint4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t iters_per_thread = compressed.size(0);
    size_t m = compressed.size(1);
    size_t n = iters_per_thread * 32 * 4 * 16;

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

    size_t block_size = MAX_THREADS_PER_BLOCK;
    TORCH_CHECK(MAX_THREADS_PER_BLOCK % 32 == 0);
    size_t warps_per_block = MAX_THREADS_PER_BLOCK / 32;
    TORCH_CHECK(m % warps_per_block == 0);
    size_t grid_size = m / warps_per_block; // each warp takes care of a row

    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    gpuErrchk(cudaFuncSetAttribute(
                decompress_matvec_kernel<L>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxL1));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    decompress_matvec_kernel<L><<<grid_size, block_size>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint4 *)compressed.data_ptr<int32_t>(),
        (const half *)codebook.data_ptr<c10::Half>(),
        (const half2 *)x.data_ptr<c10::Half>(),
        iters_per_thread,
        m,
        n);
    
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start, stop));

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return msecTotal;
}

__host__ extern float decompress_matvec_16(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<16>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_14(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<14>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_12(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<12>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_10(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<10>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_8(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<8>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_6(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<6>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_4(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<4>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_2(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<2>(compressed, codebook, x, out);
}
