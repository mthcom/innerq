#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef N
    #define N 32768
#endif

#define NUM_HEADS 32
#define HEAD_DIM 128
#define G 32

#ifndef HYBRID_P
    #define HYBRID_P 0.01
#endif

#define N_PAD              (((N + G - 1) / G) * G)
#define N_GROUPS           ((N + G - 1) / G)
#define HEAD_DIM_GROUPS    ((HEAD_DIM + G - 1) / G)
#define HIDDEN_QPARAMS_ELEMENTS ((size_t)NUM_HEADS * HEAD_DIM_GROUPS * N_PAD)

#define PACKS_INT2_PER_VEC (N_PAD / 16)
#define PACKS_INT3_PER_VEC (N_GROUPS * 3)

#define LANES_PER_OUT 8
#define OUTS_PER_WARP 4

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} \

__device__ __forceinline__ float subwarpReduceSum(
    float val,
    int lanes_per_out,
    int subwarp_id
) {
    unsigned mask;

    if (lanes_per_out == 4) {
        mask = 0xFu << (subwarp_id * 4);
        val += __shfl_down_sync(mask, val, 2, 4);
        val += __shfl_down_sync(mask, val, 1, 4);
    } else if (lanes_per_out == 8) {
        mask = 0xFFu << (subwarp_id * 8);
        val += __shfl_down_sync(mask, val, 4, 8);
        val += __shfl_down_sync(mask, val, 2, 8);
        val += __shfl_down_sync(mask, val, 1, 8);
    } else {
        mask = 0xFFFFu << (subwarp_id * 16);
        val += __shfl_down_sync(mask, val, 8, 16);
        val += __shfl_down_sync(mask, val, 4, 16);
        val += __shfl_down_sync(mask, val, 2, 16);
        val += __shfl_down_sync(mask, val, 1, 16);
    }

    return val;
}

__device__ __forceinline__ float simple_rand(unsigned int seed) {
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    unsigned int result = (word >> 22u) ^ word;
    return (float)result / 4294967295.0f;
}

__global__ void init_random_half(half* data, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = simple_rand((unsigned int)idx) * 2.0f - 1.0f;
        data[idx] = __float2half(val);
    }
}

__global__ void init_random_half_lastdim_padded(
    half* data,
    size_t size,
    int logical_n,
    int padded_n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int token = (int)(idx % padded_n);
        if (token < logical_n) {
            float val = simple_rand((unsigned int)idx) * 2.0f - 1.0f;
            data[idx] = __float2half(val);
        } else {
            data[idx] = __float2half(0.0f);
        }
    }
}

__global__ void init_random_uint32(uint32_t* data, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int state = (unsigned int)idx * 747796405u + 2891336453u;
        unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        unsigned int result = (word >> 22u) ^ word;
        data[idx] = result;
    }
}

__global__ void init_hybrid_scales(half* scales, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = simple_rand((unsigned int)idx) * 8.0f;
        float prob = simple_rand(((unsigned int)idx) ^ 0x55555555u);

        if (prob < HYBRID_P) {
            val = -val;
        }

        scales[idx] = __float2half(val);
    }
}

#ifndef ONLYHYBRID
__global__ void av_asym_group_H_int2(
    const half* Attn,
    const uint32_t* V,
    const half* scales,
    const half* zeros,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;
    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;   // 0..7
    int sublane = lane % LANES_PER_OUT;  // 0..3
    int dim = dim_pack * OUTS_PER_WARP + subout;

    int dim_group = dim / G;

    float sum = 0.0f;

    #pragma unroll 1
    for (int p = sublane; p < PACKS_INT2_PER_VEC; p += LANES_PER_OUT) {
        uint32_t packed_v = V[(head * HEAD_DIM + dim) * PACKS_INT2_PER_VEC + p];

        int token_base = p * 16;

        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[head * N_PAD + token_base]);

        float4 a0 = a_vec[0];
        float4 a1 = a_vec[1];

        half* a_h0 = reinterpret_cast<half*>(&a0);
        half* a_h1 = reinterpret_cast<half*>(&a1);

        const half* s_ptr = &scales[(head * HEAD_DIM_GROUPS + dim_group) * N_PAD + token_base];

        const half* z_ptr = &zeros[(head * HEAD_DIM_GROUPS + dim_group) * N_PAD + token_base];

        const float4* s_vec = reinterpret_cast<const float4*>(s_ptr);
        const float4* z_vec = reinterpret_cast<const float4*>(z_ptr);

        float4 s0 = s_vec[0];
        float4 s1 = s_vec[1];
        float4 z0 = z_vec[0];
        float4 z1 = z_vec[1];

        half* s_h0 = reinterpret_cast<half*>(&s0);
        half* s_h1 = reinterpret_cast<half*>(&s1);
        half* z_h0 = reinterpret_cast<half*>(&z0);
        half* z_h1 = reinterpret_cast<half*>(&z1);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed_v >> (i * 2)) & 0x3;

            float a = __half2float(a_h0[i]);
            float s = __half2float(s_h0[i]);
            float z = __half2float(z_h0[i]);

            sum += a * (val * s + z);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed_v >> ((i + 8) * 2)) & 0x3;

            float a = __half2float(a_h1[i]);
            float s = __half2float(s_h1[i]);
            float z = __half2float(z_h1[i]);

            sum += a * (val * s + z);
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}

__global__ void av_sym_group_N_int2(
    const half* Attn,
    const uint32_t* V,
    const half* scales,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;
    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;
    int sublane = lane % LANES_PER_OUT;
    int dim = dim_pack * OUTS_PER_WARP + subout;

    int v_base = (head * HEAD_DIM + dim) * PACKS_INT2_PER_VEC;
    int qparam_base = head * HEAD_DIM + dim;

    float sum = 0.0f;

    #pragma unroll 1
    for (int group_idx = sublane; group_idx < N_GROUPS; group_idx += LANES_PER_OUT) {
        int p0 = group_idx * 2;
        int p1 = p0 + 1;

        uint32_t packed0 = V[v_base + p0];
        uint32_t packed1 = V[v_base + p1];

        float s = __half2float(scales[group_idx * NUM_HEADS * HEAD_DIM + qparam_base]);

        int a_start = head * N_PAD + group_idx * G;
        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[a_start]);

        float4 a0 = a_vec[0];
        float4 a1 = a_vec[1];
        float4 a2 = a_vec[2];
        float4 a3 = a_vec[3];

        half* a_h0 = reinterpret_cast<half*>(&a0);
        half* a_h1 = reinterpret_cast<half*>(&a1);
        half* a_h2 = reinterpret_cast<half*>(&a2);
        half* a_h3 = reinterpret_cast<half*>(&a3);

        float local = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed0 >> (i * 2)) & 0x3;
            local += __half2float(a_h0[i]) * static_cast<float>(val);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed0 >> ((i + 8) * 2)) & 0x3;
            local += __half2float(a_h1[i]) * static_cast<float>(val);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed1 >> (i * 2)) & 0x3;
            local += __half2float(a_h2[i]) * static_cast<float>(val);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed1 >> ((i + 8) * 2)) & 0x3;
            local += __half2float(a_h3[i]) * static_cast<float>(val);
        }

        sum += local * s;
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}

__global__ void av_fp16(
    const half* Attn,
    const half* V,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;
    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;
    int sublane = lane % LANES_PER_OUT;
    int dim = dim_pack * OUTS_PER_WARP + subout;

    float sum = 0.0f;

    int chunks = N_PAD / 8;
    #pragma unroll 1
    for (int c = sublane; c < chunks; c += LANES_PER_OUT) {
        int offset = c * 8;
        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[head * N_PAD + offset]);
        const float4* v_vec = reinterpret_cast<const float4*>(&V[(head * HEAD_DIM + dim) * N_PAD + offset]);

        float4 a_f = a_vec[0];
        float4 v_f = v_vec[0];
        half* a_h = reinterpret_cast<half*>(&a_f);
        half* v_h = reinterpret_cast<half*>(&v_f);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += __half2float(a_h[i]) * __half2float(v_h[i]);
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}

__device__ __forceinline__ int extract_3bit(
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    int idx
) {
    int bit_pos = idx * 3;
    int word_idx = bit_pos / 32;
    int bit_offset = bit_pos % 32;

    if (bit_offset <= 29) {
        uint32_t w = word_idx == 0 ? w0 : (word_idx == 1 ? w1 : w2);
        return (w >> bit_offset) & 0x7;
    } else {
        uint32_t w_low = word_idx == 0 ? w0 : w1;
        uint32_t w_high = word_idx == 0 ? w1 : w2;
        int bits_in_low = 32 - bit_offset;
        int val_low = w_low >> bit_offset;
        int val_high = (w_high & ((1 << (3 - bits_in_low)) - 1)) << bits_in_low;
        return val_low | val_high;
    }
}

__device__ __forceinline__ int extract_3bit_signed(
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    int idx
) {
    int val = extract_3bit(w0, w1, w2, idx);
    return (val ^ 4) - 4;
}

__global__ void av_sym_group_N_int3(
    const half* Attn,
    const uint32_t* V,
    const half* scales,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;
    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;
    int sublane = lane % LANES_PER_OUT;
    int dim = dim_pack * OUTS_PER_WARP + subout;

    float sum = 0.0f;

    #pragma unroll 1
    for (int group_idx = sublane; group_idx < N_GROUPS; group_idx += LANES_PER_OUT) {
        int v_start = (head * HEAD_DIM + dim) * PACKS_INT3_PER_VEC + group_idx * 3;
        uint32_t w0 = V[v_start];
        uint32_t w1 = V[v_start + 1];
        uint32_t w2 = V[v_start + 2];

        float s = __half2float(scales[(group_idx * NUM_HEADS + head) * HEAD_DIM + dim]);

        int a_start = head * N_PAD + group_idx * G;
        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[a_start]);
        float4 a0 = a_vec[0];
        float4 a1 = a_vec[1];
        float4 a2 = a_vec[2];
        float4 a3 = a_vec[3];
        half* a_h0 = reinterpret_cast<half*>(&a0);
        half* a_h1 = reinterpret_cast<half*>(&a1);
        half* a_h2 = reinterpret_cast<half*>(&a2);
        half* a_h3 = reinterpret_cast<half*>(&a3);

        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = extract_3bit_signed(w0, w1, w2, i);
            local += __half2float(a_h0[i]) * static_cast<float>(val);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = extract_3bit_signed(w0, w1, w2, i + 8);
            local += __half2float(a_h1[i]) * static_cast<float>(val);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = extract_3bit_signed(w0, w1, w2, i + 16);
            local += __half2float(a_h2[i]) * static_cast<float>(val);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = extract_3bit_signed(w0, w1, w2, i + 24);
            local += __half2float(a_h3[i]) * static_cast<float>(val);
        }
        sum += local * s;
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}

__device__ __forceinline__ uint32_t extract_3bit_unsigned_tq(
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    int idx
) {
    int bit = idx * 3;
    uint64_t x;

    if (bit < 32) {
        x = static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
    } else {
        bit -= 32;
        x = static_cast<uint64_t>(w1) | (static_cast<uint64_t>(w2) << 32);
    }
    return static_cast<uint32_t>((x >> bit) & 0x7u);
}

__global__ void av_turboquant_int3(
    const half* __restrict__ Attn,
    const uint32_t* __restrict__ V,
    const half* __restrict__ codebook,
    half* __restrict__ Out
) {
    __shared__ float cb_s[8];

    if (threadIdx.x < 8) {
        cb_s[threadIdx.x] = __half2float(codebook[threadIdx.x]);
    }

    __syncthreads();

    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;

    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;
    int sublane = lane % LANES_PER_OUT;

    int dim = dim_pack * OUTS_PER_WARP + subout;

    float sum = 0.0f;

    #pragma unroll 1
    for (int group_idx = sublane; group_idx < N_GROUPS; group_idx += LANES_PER_OUT) {
        int v_start = (head * HEAD_DIM + dim) * PACKS_INT3_PER_VEC + group_idx * 3;

        uint32_t w0 = V[v_start];
        uint32_t w1 = V[v_start + 1];
        uint32_t w2 = V[v_start + 2];

        int a_start = head * N_PAD + group_idx * G;

        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[a_start]);

        float4 a0 = a_vec[0];
        float4 a1 = a_vec[1];
        float4 a2 = a_vec[2];
        float4 a3 = a_vec[3];

        half* a_h0 = reinterpret_cast<half*>(&a0);
        half* a_h1 = reinterpret_cast<half*>(&a1);
        half* a_h2 = reinterpret_cast<half*>(&a2);
        half* a_h3 = reinterpret_cast<half*>(&a3);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t idx = extract_3bit_unsigned_tq(w0, w1, w2, i);
            sum += __half2float(a_h0[i]) * cb_s[idx];
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t idx = extract_3bit_unsigned_tq(w0, w1, w2, i + 8);
            sum += __half2float(a_h1[i]) * cb_s[idx];
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t idx = extract_3bit_unsigned_tq(w0, w1, w2, i + 16);
            sum += __half2float(a_h2[i]) * cb_s[idx];
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t idx = extract_3bit_unsigned_tq(w0, w1, w2, i + 24);
            sum += __half2float(a_h3[i]) * cb_s[idx];
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}
#endif

__global__ void av_hybrid_group_N_int2(
    const half* __restrict__ Attn,
    const uint32_t* __restrict__ V,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int dim_packs_per_head = HEAD_DIM / OUTS_PER_WARP;
    int head = warp_global / dim_packs_per_head;
    int dim_pack = warp_global % dim_packs_per_head;
    if (head >= NUM_HEADS) return;

    int subout = lane / LANES_PER_OUT;
    int sublane = lane % LANES_PER_OUT;
    int dim = dim_pack * OUTS_PER_WARP + subout;

    int v_base = (head * HEAD_DIM + dim) * PACKS_INT2_PER_VEC;
    int qparam_base = head * HEAD_DIM + dim;

    float sum = 0.0f;

    #pragma unroll 1
    for (int group_idx = sublane; group_idx < N_GROUPS; group_idx += LANES_PER_OUT) {
        int p0 = group_idx * 2;
        int p1 = p0 + 1;

        uint32_t packed0 = V[v_base + p0];
        uint32_t packed1 = V[v_base + p1];

        int qidx = group_idx * NUM_HEADS * HEAD_DIM + qparam_base;

        float s_raw = __half2float(scales[qidx]);
        float s = fabsf(s_raw);

        float z = s_raw < 0.0f ? __half2float(zeros[qidx]) : 0.0f;

        int a_start = head * N_PAD + group_idx * G;
        const float4* a_vec = reinterpret_cast<const float4*>(&Attn[a_start]);

        float4 a0 = a_vec[0];
        float4 a1 = a_vec[1];
        float4 a2 = a_vec[2];
        float4 a3 = a_vec[3];

        half* a_h0 = reinterpret_cast<half*>(&a0);
        half* a_h1 = reinterpret_cast<half*>(&a1);
        half* a_h2 = reinterpret_cast<half*>(&a2);
        half* a_h3 = reinterpret_cast<half*>(&a3);

        float unscaled_sum = 0.0f;
        float sum_a = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed0 >> (i * 2)) & 0x3;
            float av = __half2float(a_h0[i]);
            unscaled_sum += av * val;
            sum_a += av;
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed0 >> ((i + 8) * 2)) & 0x3;
            float av = __half2float(a_h1[i]);
            unscaled_sum += av * val;
            sum_a += av;
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed1 >> (i * 2)) & 0x3;
            float av = __half2float(a_h2[i]);
            unscaled_sum += av * val;
            sum_a += av;
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed1 >> ((i + 8) * 2)) & 0x3;
            float av = __half2float(a_h3[i]);
            unscaled_sum += av * val;
            sum_a += av;
        }

        sum += unscaled_sum * s + sum_a * z;
    }

    sum = subwarpReduceSum(sum, LANES_PER_OUT, subout);

    if (sublane == 0) {
        Out[head * HEAD_DIM + dim] = __float2half(sum);
    }
}

template <typename Func>
void run_benchmark(const char* name, Func kernel_launch) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 10; ++i) { // Warmup phase
        kernel_launch();
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    int iters = 100;

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        kernel_launch();
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "AV,N:" << N << "," << name << ",Latency:"
              << (ms / iters) * 1000.0f
              << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    size_t attn_elements = (size_t)NUM_HEADS * N_PAD;
    size_t v_fp16_elements = (size_t)NUM_HEADS * HEAD_DIM * N_PAD;
    size_t v_int2_elements = (size_t)NUM_HEADS * HEAD_DIM * PACKS_INT2_PER_VEC;
    size_t v_int3_elements = (size_t)NUM_HEADS * HEAD_DIM * PACKS_INT3_PER_VEC;
    size_t out_elements = (size_t)NUM_HEADS * HEAD_DIM;
    size_t scales_elements = HIDDEN_QPARAMS_ELEMENTS;

    half* d_Attn = nullptr;
    half* d_V_fp16 = nullptr;
    half* d_Out = nullptr;

    uint32_t* d_V_int2 = nullptr;
    uint32_t* d_V_int3 = nullptr;
    uint32_t* d_V_uint3 = nullptr;

    half* d_scales = nullptr;
    half* d_zeros = nullptr;

    half* d_turboquant_codebook = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Attn, attn_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_V_fp16, v_fp16_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_V_int2, v_int2_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_V_int3, v_int3_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_V_uint3, v_int3_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_Out, out_elements * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&d_scales, scales_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_zeros, scales_elements * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&d_turboquant_codebook, 8 * sizeof(half)));

    half h_turboquant_codebook[8];

    h_turboquant_codebook[0] = __float2half(-1.0000000f);
    h_turboquant_codebook[1] = __float2half(-0.5000000f);
    h_turboquant_codebook[2] = __float2half(-0.2500000f);
    h_turboquant_codebook[3] = __float2half(-0.1250000f);
    h_turboquant_codebook[4] = __float2half( 0.1250000f);
    h_turboquant_codebook[5] = __float2half( 0.2500000f);
    h_turboquant_codebook[6] = __float2half( 0.5000000f);
    h_turboquant_codebook[7] = __float2half( 1.0000000f);

    CHECK_CUDA(cudaMemcpy(
        d_turboquant_codebook,
        h_turboquant_codebook,
        8 * sizeof(half),
        cudaMemcpyHostToDevice
    ));

    int init_threads = 256;

    auto launch_half = [&](half* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_half<<<blocks, init_threads>>>(ptr, elements);
        CHECK_CUDA(cudaGetLastError());
    };

    auto launch_half_padded = [&](half* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_half_lastdim_padded<<<blocks, init_threads>>>(ptr, elements, N, N_PAD);
        CHECK_CUDA(cudaGetLastError());
    };

    auto launch_uint32 = [&](uint32_t* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_uint32<<<blocks, init_threads>>>(ptr, elements);
        CHECK_CUDA(cudaGetLastError());
    };

    launch_half_padded(d_Attn, attn_elements);
    launch_half_padded(d_V_fp16, v_fp16_elements);

    launch_uint32(d_V_int2, v_int2_elements);
    launch_uint32(d_V_int3, v_int3_elements);
    launch_uint32(d_V_uint3, v_int3_elements);

    launch_half(d_zeros, scales_elements);

    {
        size_t blocks = (scales_elements + init_threads - 1) / init_threads;
        init_hybrid_scales<<<blocks, init_threads>>>(d_scales, scales_elements);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 128;
    int warps_per_block = threads / 32;

    int output_warps = NUM_HEADS * (HEAD_DIM / OUTS_PER_WARP);
    int blocks = (output_warps + warps_per_block - 1) / warps_per_block;

#ifndef ONLYHYBRID
    run_benchmark("int2_sym_Ndim", [&]() {
        av_sym_group_N_int2<<<blocks, threads>>>(
            d_Attn,
            d_V_int2,
            d_scales,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("fp16", [&]() {
        av_fp16<<<blocks, threads>>>(
            d_Attn,
            d_V_fp16,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("int3_sym_Ndim", [&]() {
        av_sym_group_N_int3<<<blocks, threads>>>(
            d_Attn,
            d_V_int3,
            d_scales,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("int3_turboquant", [&]() {
        av_turboquant_int3<<<blocks, threads>>>(
            d_Attn,
            d_V_uint3,
            d_turboquant_codebook,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("int2_asym_headdim", [&]() {
        av_asym_group_H_int2<<<blocks, threads>>>(
            d_Attn,
            d_V_int2,
            d_scales,
            d_zeros,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());
#endif

    run_benchmark("int2_hybrid_Ndim", [&]() {
        av_hybrid_group_N_int2<<<blocks, threads>>>(
            d_Attn,
            d_V_int2,
            d_scales,
            d_zeros,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_Attn);
    cudaFree(d_V_fp16);
    cudaFree(d_V_int2);
    cudaFree(d_V_uint3);
    cudaFree(d_V_int3);
    cudaFree(d_Out);
    cudaFree(d_scales);
    cudaFree(d_zeros);
    cudaFree(d_turboquant_codebook);

    return 0;
}
