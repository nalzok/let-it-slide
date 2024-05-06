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

#if __CUDA_ARCH__ == 610
#define MAX_THREADS_PER_BLOCK 1024
#else
#define MAX_THREADS_PER_BLOCK 256
#endif

#define FULL_MASK 0xFFFFFFFFU
#define HALF_MASK 0x0000FFFFU

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


#if __CUDA_ARCH__ == 610
constexpr size_t prefetch = 1;
#else
constexpr size_t prefetch = 1;
#endif


__constant__ half2 act[16384];


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


typedef union __align__(16) Ux4 {
    uint4 __align__(16) vec;
    uint32_t __align__(16) elem[4];
} Ux4;


template <size_t L>
__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK)
decompress_matvec_t_kernel(
    half *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half *__restrict__ codebook,
    size_t compressed_m,
    size_t compressed_n
) {
    const half *__restrict__ codebook_ptr = codebook;
    if constexpr (L < 16) {
        __shared__ half smem[1<<L];
        for (int cb_idx = threadIdx.x; cb_idx < (1<<L)/8; cb_idx += blockDim.x) {
            reinterpret_cast<uint4 *>(smem)[cb_idx] = reinterpret_cast<const uint4 *>(codebook)[cb_idx];
        }
        codebook_ptr = smem;
    }

    const uint4 *c = reinterpret_cast<const uint4 *>(compressed);

    constexpr uint16_t mask = (1<<L) - 1;

    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t rowId = threadId; rowId < compressed_m; rowId += prefetch * stride) {
        uint32_t carries[prefetch];
#if __CUDA_ARCH__ == 610
        float2 inners[prefetch];
#else
        half2 inners[prefetch];
#endif

        #pragma unroll
        for (size_t i = 0; i < prefetch; i += 1) {
            carries[i] = 0U;
#if __CUDA_ARCH__ == 610
            inners[i] = make_float2(0.0f, 0.0f);
#else
            inners[i] = __float2half2_rn(0.0f);
#endif
        }

        for (size_t colId = 0; colId < compressed_n / 4; colId += 1) {
            Ux4 reg_c[prefetch];
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                reg_c[i].vec = c[(rowId + i * stride) + colId * compressed_m];
            }

            #pragma unroll
            for (size_t k = 0; k < 4; k += 1) {
                uint32_t reg_sft2[8][prefetch];
                uint32_t reg_sft4[8][prefetch];

                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {

                    // Move k to the inner loop
                    // We don't always need funnelshift; sometimes shift is enough
                    #pragma unroll
                    for (int i = 0; i < prefetch; i += 1) {
                        reg_sft2[j][i] = mask & __funnelshift_l(reg_c[i].elem[k], carries[i], 2);
                        reg_sft4[j][i] = mask & __funnelshift_l(reg_c[i].elem[k], carries[i], 4);
                    }
                }

                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {

                    #pragma unroll
                    for (int i = 0; i < prefetch; i += 1) {
                        carries[i] = reg_sft4[j][i];

                        half2 reg_w = make_half2(
                                codebook_ptr[reg_sft2[j][i]],
                                codebook_ptr[reg_sft4[j][i]]);
                        half2 reg_a = act[colId*32 + k*8 + j];

                        // TODO: Kahan summation?
#if __CUDA_ARCH__ == 610
                        inners[i].x = __fmaf_rn(__half2float(reg_w.x),
                                __half2float(reg_a.x),
                                inners[i].x);
                        inners[i].y = __fmaf_rn(__half2float(reg_w.y),
                                __half2float(reg_a.y),
                                inners[i].y);
#else
                        inners[i] = __hfma2(reg_w, reg_a, inners[i]);
#endif
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < prefetch; i += 1) {
            size_t outId = rowId + i * stride;
#if __CUDA_ARCH__ == 610
            out[outId] = __float2half(__half2float(out[outId]) + inners[i].x + inners[i].y);
#else
            half delta = __hadd(inners[i].x, inners[i].y);
            out[outId] = __hadd(out[outId], delta);
#endif
        }
    }
}


template <size_t L>
__host__ static float decompress_matvec_t(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
) {
    static_assert(L <= 32, "Shift register length should not exceed 32 as the kernel uses uint32_t");

    // FIXME: Why does prefetch = 256 not work?
    CHECK_INPUT(compressed);
    // TORCH_CHECK(compressed.dim() == 2);
    // FIXME: do we need this?
    // TORCH_CHECK(compressed.size(1) % MAX_THREADS_PER_BLOCK == 0);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t m = out.size(0);
    size_t n = x.size(0);
    size_t compressed_m = m;
    size_t compressed_n = n / 16;

    // size_t compressed_m = compressed.size(0);
    // size_t compressed_n = compressed.size(1);
    // size_t m = compressed_m;
    // size_t n = compressed_n * 16;   // at 2 bit, each uint32_t has 32 bits = 16 weights

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.size(0) == 1<<L);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1);
    TORCH_CHECK(x.size(0) == n);
    TORCH_CHECK(x.size(0) % 8 == 0);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    // copy to the constant memory
    gpuErrchk(cudaMemcpyToSymbol(act, x.data_ptr<c10::Half>(), n * 2));

    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 1);
    TORCH_CHECK(out.size(0) == m);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);


    size_t grid_size = m / MAX_THREADS_PER_BLOCK;
    size_t block_size = MAX_THREADS_PER_BLOCK;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    decompress_matvec_t_kernel<L><<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint32_t *)compressed.data_ptr<int32_t>(),
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

__host__ extern float decompress_matvec_t_16(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec_t<16>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_t_14(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec_t<14>(compressed, codebook, x, out);
}
