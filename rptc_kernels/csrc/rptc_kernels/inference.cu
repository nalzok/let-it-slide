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

#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_MP 2

#define FULL_MASK 0xFFFFFFFFU
#define HALF_MASK 0x0000FFFFU

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


#if __CUDA_ARCH__ == 610
constexpr size_t warpsPerRow = 1;
constexpr size_t prefetch = 1;
constexpr size_t warpsPerRowMatvec = 1;
constexpr size_t prefetchMatvec = 8;
constexpr size_t warpsPerRowRowsum = 1;
constexpr size_t prefetchRowsum = 8;
#else
constexpr size_t warpsPerRow = 1;
constexpr size_t prefetch = 2;
constexpr size_t warpsPerRowMatvec = 2;
constexpr size_t prefetchMatvec = 32;
constexpr size_t warpsPerRowRowsum = 2;
constexpr size_t prefetchRowsum = 32;
#endif


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
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
    size_t grid_size = MIN_BLOCKS_PER_MP * static_cast<size_t>(deviceProp.multiProcessorCount);
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
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
decompress_matvec_kernel(
    half *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half *__restrict__ codebook,
    const half *__restrict__ x,
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

    constexpr uint16_t mask = (1<<L) - 1;

    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t laneId = threadIdx.x % warpSize;
    size_t warpId = threadId / warpSize;
    size_t warpCount = gridDim.x * blockDim.x / warpSize;
    size_t stride = warpsPerRow * warpSize;

    // each int4 has 4x32 bits, which corresponds to 4x16 weights at 2 bit
    // both reg_w and reg_a are reused 4 times, hence their sizes
    uint32_t carries[prefetch];
    uint4 reg_c[prefetch];
    size_t reg_w[prefetch][16];
    half2 reg_a[prefetch][8];
#if __CUDA_ARCH__ == 610
    float2 inners[prefetch];
#else
    half2 inners[prefetch];
#endif

    for (size_t rowId = warpId / warpsPerRow;
            rowId < compressed_m;
            rowId += warpCount / warpsPerRow) {
        const uint4 *row = reinterpret_cast<const uint4 *>(compressed + rowId * compressed_n);

        #pragma unroll
        for (size_t i = 0; i < prefetch; i += 1) {
            carries[i] = row[(compressed_n + i * warpSize - 1) % compressed_m].w;
#if __CUDA_ARCH__ == 610
            inners[i] = make_float2(0.0f, 0.0f);
#else
            inners[i] = __float2half2_rn(0.0f);
#endif
        }

        for (size_t colId = threadId % stride;
                colId < compressed_n / 4;
                colId += prefetch * stride) {
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                reg_c[i] = row[colId + i * stride];
            }

            const half2 *act = reinterpret_cast<const half2 *>(x) + threadId % stride;
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {
                    reg_a[i][j] = act[(i*8+j)*stride];
                }
            }
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                carries[i] = __shfl_up_sync(FULL_MASK, reg_c[i].w, 1);  // laneId == 0 is not updated
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] = __funnelshift_l(reg_c[i].x, carries[i], 2*j);
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] &= mask;
                }
            }
            #pragma unroll
            for (size_t j = 0; j < 8; j += 1) {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
                    // TODO: Kahan summation?
#if __CUDA_ARCH__ == 610
                    inners[i].x = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j]]),
                            __half2float(reg_a[i][j].x),
                            inners[i].x);
                    inners[i].y = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j+1]]),
                            __half2float(reg_a[i][j].y),
                            inners[i].y);
#else
                    inners[i] = __hfma2(
                            __halves2half2(codebook_ptr[reg_w[i][2*j]], codebook_ptr[reg_w[i][2*j+1]]),
                            reg_a[i][j],
                            inners[i]);
#endif
                }
            }

            act += prefetch * 8 * stride;
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {
                    reg_a[i][j] = act[(i*8+j)*stride];
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] = __funnelshift_l(reg_c[i].y, reg_c[i].x, 2*j);
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] &= mask;
                }
            }
            #pragma unroll
            for (size_t j = 0; j < 8; j += 1) {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                    inners[i].x = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j]]),
                            __half2float(reg_a[i][j].x),
                            inners[i].x);
                    inners[i].y = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j+1]]),
                            __half2float(reg_a[i][j].y),
                            inners[i].y);
#else
                    inners[i] = __hfma2(
                            __halves2half2(codebook_ptr[reg_w[i][2*j]], codebook_ptr[reg_w[i][2*j+1]]),
                            reg_a[i][j],
                            inners[i]);
#endif
                }
            }

            act += prefetch * 8 * stride;
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {
                    reg_a[i][j] = act[(i*8+j)*stride];
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] = __funnelshift_l(reg_c[i].z, reg_c[i].y, 2*j);
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] &= mask;
                }
            }
            #pragma unroll
            for (size_t j = 0; j < 8; j += 1) {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                    inners[i].x = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j]]),
                            __half2float(reg_a[i][j].x),
                            inners[i].x);
                    inners[i].y = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j+1]]),
                            __half2float(reg_a[i][j].y),
                            inners[i].y);
#else
                    inners[i] = __hfma2(
                            __halves2half2(codebook_ptr[reg_w[i][2*j]], codebook_ptr[reg_w[i][2*j+1]]),
                            reg_a[i][j],
                            inners[i]);
#endif
                }
            }

            act += prefetch * 8 * stride;
            #pragma unroll
            for (size_t i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 8; j += 1) {
                    reg_a[i][j] = act[(i*8+j)*stride];
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] = __funnelshift_l(reg_c[i].w, reg_c[i].z, 2*j);
                }
            }
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                #pragma unroll
                for (size_t j = 0; j < 16; j += 1) {
                    reg_w[i][j] &= mask;
                }
            }
            #pragma unroll
            for (size_t j = 0; j < 8; j += 1) {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                    inners[i].x = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j]]),
                            __half2float(reg_a[i][j].x),
                            inners[i].x);
                    inners[i].y = __fmaf_rn(__half2float(codebook_ptr[reg_w[i][2*j+1]]),
                            __half2float(reg_a[i][j].y),
                            inners[i].y);
#else
                    inners[i] = __hfma2(
                            __halves2half2(codebook_ptr[reg_w[i][2*j]], codebook_ptr[reg_w[i][2*j+1]]),
                            reg_a[i][j],
                            inners[i]);
#endif
                }
            }

            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
                carries[i] = __shfl_down_sync(FULL_MASK, reg_c[i].w, 31);  // only laneId == 0 is updated
            }
        }

        for (size_t offset = 16; offset > 0; offset /= 2) {
            #pragma unroll
            for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                inners[i].x += __shfl_down_sync(FULL_MASK, inners[i].x, offset);
                inners[i].y += __shfl_down_sync(FULL_MASK, inners[i].y, offset);
#else
                inners[i] = __hadd2(inners[i], __shfl_down_sync(FULL_MASK, inners[i], offset));
#endif
            }
        }

        if (laneId == 0) {
            if constexpr (warpsPerRow == 1) {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                    out[rowId] = __float2half(__half2float(out[rowId]) + inners[i].x + inners[i].y);
#else
                    half delta = __hadd(inners[i].x, inners[i].y);
                    out[rowId] = __hadd(out[rowId], delta);
#endif
                }
            } else {
                #pragma unroll
                for (int i = 0; i < prefetch; i += 1) {
#if __CUDA_ARCH__ == 610
                    static_assert(warpsPerRow == 1, "atomicAdd(half *, half) is not supported");
#else
                    atomicAdd(out + rowId, __hadd(inners[i].x, inners[i].y));
#endif
                }
            }
        }
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
    TORCH_CHECK(compressed.dim() == 2);
    TORCH_CHECK(compressed.size(1) % (prefetch * warpsPerRow * 32 * 4) == 0); // 32 as in warpSize, 4 as in uint4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t compressed_m = compressed.size(0);
    size_t compressed_n = compressed.size(1);
    size_t m = compressed_m;
    size_t n = compressed_n * 16;   // at 2 bit, each uint32_t has 32 bits = 16 weights

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
    size_t grid_size = MIN_BLOCKS_PER_MP * static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = MAX_THREADS_PER_BLOCK;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    decompress_matvec_kernel<L><<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const uint32_t *)compressed.data_ptr<int32_t>(),
        (const half *)codebook.data_ptr<c10::Half>(),
        (const half *)x.data_ptr<c10::Half>(),
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


__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
matvec_kernel(
    half *__restrict__ out,
    const half *__restrict__ decompressed,
    const half *__restrict__ x,
    size_t m,
    size_t n
) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t laneId = threadIdx.x % warpSize;
    size_t warpId = threadId / warpSize;
    size_t warpCount = gridDim.x * blockDim.x / warpSize;
    size_t stride = warpsPerRowMatvec * warpSize;

    half2 reg_w[prefetchMatvec], reg_a[prefetchMatvec];
    const half2 *act = reinterpret_cast<const half2 *>(x);

    for (size_t rowId = warpId / warpsPerRowMatvec;
            rowId < m;
            rowId += warpCount / warpsPerRowMatvec) {
        const half2 *row = reinterpret_cast<const half2 *>(decompressed + rowId * n);
#if __CUDA_ARCH__ == 610
        float2 inner = make_float2(0.0f, 0.0f);
#else
        half2 inner = __float2half2_rn(0.0f);
#endif

        for (size_t colId = threadId % stride;
                colId < n / 2;
                colId += prefetchMatvec * stride) {
            #pragma unroll
            for (size_t j = 0; j < prefetchMatvec; j += 1) {
                reg_w[j] = row[colId + j * stride];
                reg_a[j] = act[colId + j * stride];
            }

            #pragma unroll
            for (size_t j = 0; j < prefetchMatvec; j += 1) {
#if __CUDA_ARCH__ == 610
                inner.x = __fmaf_rn(__half2float(reg_w[j].x),
                        __half2float(reg_a[j].x),
                        inner.x);
                inner.y = __fmaf_rn(__half2float(reg_w[j].y),
                        __half2float(reg_a[j].y),
                        inner.y);
#else
                inner = __hfma2(reg_w[j], reg_a[j], inner);
#endif
            }
        }

        for (size_t offset = 16; offset > 0; offset /= 2) {
#if __CUDA_ARCH__ == 610
            inner.x += __shfl_down_sync(FULL_MASK, inner.x, offset);
            inner.y += __shfl_down_sync(FULL_MASK, inner.y, offset);
#else
            inner = __hadd2(inner, __shfl_down_sync(FULL_MASK, inner, offset));
#endif
        }

        if (laneId == 0) {
            if constexpr (warpsPerRowMatvec == 1) {
#if __CUDA_ARCH__ == 610
                out[rowId] = __float2half(__half2float(out[rowId]) + inner.x + inner.y);
#else
                half delta = __hadd(inner.x, inner.y);
                out[rowId] = __hadd(out[rowId], delta);
#endif
            } else {
#if __CUDA_ARCH__ == 610
                static_assert(warpsPerRowMatvec == 1, "atomicAdd(half *, half) is not supported");
#else
                atomicAdd(out + rowId, __hadd(inner.x, inner.y));
#endif
            }
        }
    }
}


__host__ extern float matvec(
    torch::Tensor &decompressed,
    torch::Tensor &x,
    torch::Tensor &out
) {
    CHECK_INPUT(decompressed);
    TORCH_CHECK(decompressed.dim() == 2);
    // 32 as in warpSize, 2 as in half2
    TORCH_CHECK(decompressed.size(1) % (prefetchMatvec * warpsPerRowMatvec * 32 * 2) == 0);
    TORCH_CHECK(decompressed.scalar_type() == torch::kFloat16);

    size_t m = decompressed.size(0);
    size_t n = decompressed.size(1);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1);
    TORCH_CHECK(x.size(0) == n);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 1);
    TORCH_CHECK(out.size(0) == m);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, decompressed.get_device());
    size_t grid_size = MIN_BLOCKS_PER_MP * static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = MAX_THREADS_PER_BLOCK;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    matvec_kernel<<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const half *)decompressed.data_ptr<c10::Half>(),
        (const half *)x.data_ptr<c10::Half>(),
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


__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
rowsum_kernel(
    half *__restrict__ out,
    const half *__restrict__ decompressed,
    size_t m,
    size_t n
) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t laneId = threadIdx.x % warpSize;
    size_t warpId = threadId / warpSize;
    size_t warpCount = gridDim.x * blockDim.x / warpSize;
    size_t stride = warpsPerRowRowsum * warpSize;

    half2 reg_w[prefetchRowsum];

    for (size_t rowId = warpId / warpsPerRowRowsum;
            rowId < m;
            rowId += warpCount / warpsPerRowRowsum) {
        const half2 *row = reinterpret_cast<const half2 *>(decompressed + rowId * n);
#if __CUDA_ARCH__ == 610
        float2 inner = make_float2(0.0f, 0.0f);
#else
        half2 inner = __float2half2_rn(0.0f);
#endif

        for (size_t colId = threadId % stride;
                colId < n / 2;
                colId += prefetchRowsum * stride) {
            #pragma unroll
            for (size_t j = 0; j < prefetchRowsum; j += 1) {
                reg_w[j] = row[colId + j * stride];
            }

            #pragma unroll
            for (size_t j = 0; j < prefetchRowsum; j += 1) {
#if __CUDA_ARCH__ == 610
                inner.x += __half2float(reg_w[j].x);
                inner.y += __half2float(reg_w[j].y);
#else
                inner = __hadd2(inner, reg_w[j]);
#endif
            }
        }

        for (size_t offset = 16; offset > 0; offset /= 2) {
#if __CUDA_ARCH__ == 610
            inner.x += __shfl_down_sync(FULL_MASK, inner.x, offset);
            inner.y += __shfl_down_sync(FULL_MASK, inner.y, offset);
#else
            inner = __hadd2(inner, __shfl_down_sync(FULL_MASK, inner, offset));
#endif
        }

        if (laneId == 0) {
            if constexpr (warpsPerRowRowsum == 1) {
#if __CUDA_ARCH__ == 610
                out[rowId] = __float2half(__half2float(out[rowId]) + inner.x + inner.y);
#else
                half delta = __hadd(inner.x, inner.y);
                out[rowId] = __hadd(out[rowId], delta);
#endif
            } else {
#if __CUDA_ARCH__ == 610
                static_assert(warpsPerRowRowsum == 1, "atomicAdd(half *, half) is not supported");
#else
                atomicAdd(out + rowId, __hadd(inner.x, inner.y));
#endif
            }
        }
    }
}


__host__ extern float rowsum(
    torch::Tensor &decompressed,
    torch::Tensor &out
) {
    CHECK_INPUT(decompressed);
    TORCH_CHECK(decompressed.dim() == 2);
    // 32 as in warpSize, 2 as in half2
    TORCH_CHECK(decompressed.size(1) % (prefetchRowsum * warpsPerRowRowsum * 32 * 2) == 0);
    TORCH_CHECK(decompressed.scalar_type() == torch::kFloat16);

    size_t m = decompressed.size(0);
    size_t n = decompressed.size(1);

    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 1);
    TORCH_CHECK(out.size(0) == m);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, decompressed.get_device());
    size_t grid_size = MIN_BLOCKS_PER_MP * static_cast<size_t>(deviceProp.multiProcessorCount);
    size_t block_size = MAX_THREADS_PER_BLOCK;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    rowsum_kernel<<<grid_size, block_size, 0, stream>>>(
        (half *)out.data_ptr<c10::Half>(),
        (const half *)decompressed.data_ptr<c10::Half>(),
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
