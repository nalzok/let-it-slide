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

#define FULL_MASK 0xFFFFFFFFU
#define HALF_MASK 0x0000FFFFU

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)          do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)
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


typedef union ditto {
    uint32_t u32;
    half2 f16x2;
} ditto;


typedef union ditto2 {
    unsigned long long ull;
    uint64_t u64;
    uint2 u32x2;
    float2 f32x2;
    uint32_t u32[2];
    half2 f16x2[2];
    half2 *ptr2f16x2;
} ditto2;


typedef union ditto4 {
    uint4 u32x4;
    uint32_t u32[4];
    float4 f32x4;
    half2 f16x2[4];
    uint16_t u16[8];
} ditto4;


template <uint32_t L, uint32_t S, uint32_t V>
__global__ static void
__launch_bounds__(MAX_THREADS_PER_BLOCK)
decompress_matvec_kernel(
    half *__restrict__ out,
    const uint4 *__restrict__ compressed,
    const uint4 *__restrict__ codebook,
    const uint4 *__restrict__ x,
    size_t iters_per_thread,
    size_t m,
    size_t n
) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t laneId = threadIdx.x % warpSize;
    size_t warpId = threadId / warpSize;
    size_t strideC = blockDim.x * gridDim.x;

    __shared__ __align__(1<<15) half2 smem_codebook[1<<(S+5)];
    ditto2 smem_codebook_lane = { .ptr2f16x2 = smem_codebook + laneId };
    for (size_t idx = threadIdx.x>>5; idx < 1<<(S-2); idx += blockDim.x>>5) {
        ditto4 quadruple = { .u32x4 = reinterpret_cast<const uint4 *>(codebook)[idx] };
        for (uint32_t i = 0; i < 4; i += 1) {
            smem_codebook_lane.ptr2f16x2[(idx<<7)|(i<<5)] = quadruple.f16x2[i];
        }
    }

    extern __shared__ half2 smem_x[];
    for (size_t idx = threadIdx.x; idx < n/8; idx += blockDim.x) {
        ditto4 quadruple = { .u32x4 = reinterpret_cast<const uint4 *>(x)[idx] };
        for (uint32_t i = 0; i < 4; i += 1) {
            smem_x[idx+i*(n/2/4)] = quadruple.f16x2[i];
        }
    }
    half2 *smem_x_ptr = smem_x + laneId;
    __syncthreads();

    uint32_t carry = 0U;
    half2 inner = __float2half2_rn(0.0f);

    for (size_t iter = 0; iter < iters_per_thread; iter += 1) {
        uint4 elem = compressed[iter * strideC + threadId];

        // send w in lane X to carry in lane X+1, lane 0 not updated
        carry = __shfl_up_sync(FULL_MASK, elem.w, 1);

        // send w in lane 31 to carry in lane 0, lane 1-31 not updated
        uint32_t next_carry = __shfl_down_sync(FULL_MASK, elem.w, 31);

        uint32_t reg_c[8] = {
            elem.x, elem.y, elem.z, elem.w,
            __funnelshift_lc(elem.y, elem.x, 16),
            __funnelshift_lc(elem.z, elem.y, 16),
            __funnelshift_lc(elem.w, elem.z, 16),
            __funnelshift_lc(elem.x, carry, 16),
        };

        #pragma unroll
        for (uint32_t k = 0; k < 8; k += 1) {
            #pragma unroll
            for (uint32_t j = 0; j < 4; j += 1) {
                half2 reg_a = smem_x_ptr[((iter * 8 + k) * 4 + j) * warpSize];
                uint32_t hash = reg_c[k] >> (16-(1<<V)*2-4*j);
                hash = hash * (hash+1);
                uint32_t idx = hash >> 1;
                constexpr uint32_t mask = ((1<<S)-1) << (5+V+1);
                asm volatile ("lop3.b32 %0, %1, %2, %3, 0xCA;"
                        : "=r"(smem_codebook_lane.u32[0])
                        : "r"(mask), "r"(idx), "r"(smem_codebook_lane.u32[0]));
                ditto reg_w = { .f16x2 = *smem_codebook_lane.ptr2f16x2 };
                // sign flip
                constexpr uint32_t toggle = 0b00000000'00000000'10000000'00000000;
                asm volatile ("lop3.b32 %0, %1, %2, %3, 0xCA;"
                        : "=r"(reg_w.u32)
                        : "r"(toggle), "r"(hash), "r"(reg_w.u32));
                inner = __hfma2(reg_w.f16x2, reg_a, inner);
            }
        }

        carry = next_carry;
    }

    for (size_t offset = 16; offset > 0; offset /= 2) {
        inner = __hadd2(inner, __shfl_down_sync(FULL_MASK, inner, offset));
    }

    if (laneId == 0) {
        out[warpId] = __hadd(inner.x, inner.y);
    }
}


template <uint32_t L, uint32_t S, uint32_t V>
__host__ static float decompress_matvec(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
) {
    static_assert(L - (1<<V)*2 <= 16, "Shift register should fit in uint32_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S + 5 + V + 1 <= 15, "We can only use 32 KiB shared memory");
    static_assert(V == 1, "Quantize two weights at a time");

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 3);
    TORCH_CHECK(compressed.size(2) == 32 * 4);  // each warp reads an uint4
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    size_t iters_per_thread = compressed.size(0);
    size_t m = compressed.size(1);
    size_t n = iters_per_thread * 32 * 4 * 16;
    TORCH_CHECK(n % 8 == 0, "The input dimension must be divisible by 8 because we use uint32 to load half16");

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.size(0) == 1<<(S+V));
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

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaEventRecord(start, stream));

    size_t smem_x_size = n * 2;
    decompress_matvec_kernel<L, S, V><<<grid_size, block_size, smem_x_size>>>(
        reinterpret_cast<half *>(out.data_ptr<c10::Half>()),
        reinterpret_cast<const uint4 *>(compressed.data_ptr<int32_t>()),
        reinterpret_cast<const uint4 *>(codebook.data_ptr<c10::Half>()),
        reinterpret_cast<const uint4 *>(x.data_ptr<c10::Half>()),
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

__host__ extern float decompress_matvec_16_6_1(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<16, 6, 1>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_20_6_1(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<20, 6, 1>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_16_8_1(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<16, 8, 1>(compressed, codebook, x, out);
}

__host__ extern float decompress_matvec_20_8_1(
    torch::Tensor &compressed, torch::Tensor &codebook, torch::Tensor &x, torch::Tensor &out
) {
    return decompress_matvec<20, 8, 1>(compressed, codebook, x, out);
}
