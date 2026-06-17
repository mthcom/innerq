#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// #define N 32768
#define NUM_HEADS 32
#define HEAD_DIM 128
#define D (NUM_HEADS * HEAD_DIM)
#define G 32

#ifndef HYBRID_P
    #define HYBRID_P 0.01
#endif

#define GROUPS_PER_HEAD     (HEAD_DIM / G)        // 4
#define PACKS_INT2_PER_HEAD (HEAD_DIM / 16)       // 8
#define PACKS_INT3_PER_HEAD (GROUPS_PER_HEAD * 3) // 12
#define PACKS_INT4_PER_HEAD (HEAD_DIM / 8) // 128 / 8 = 16 uint32 packs
#define N_GROUPS            ((N + G - 1) / G)

#define LANES_PER_HEAD 4
#define HEADS_PER_WARP 8

#define TURBOQUANT_CODEBOOK_SIZE 16

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
    int lanes_per_head,
    int subwarp_id
) {
    unsigned mask;

    if (lanes_per_head == 4) {
        mask = 0xFu << (subwarp_id * 4);
        val += __shfl_down_sync(mask, val, 2, 4);
        val += __shfl_down_sync(mask, val, 1, 4);
    } else if (lanes_per_head == 8) {
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

__global__ void init_random_norms_float(float* norms, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float r = simple_rand((unsigned int)idx); // [0, 1]
        norms[idx] = 5.5f + 2.0f * r; // [5.5, 7.5]
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

__global__ void init_turboquant_codebook(half* codebook) {
    int i = threadIdx.x;

    if (i < TURBOQUANT_CODEBOOK_SIZE) {
        float x = ((float)i - 7.5f) / 7.5f; // dummy numbres, roughly [-1, 1]
        codebook[i] = __float2half(x);
    }
}

__global__ void gemv_asym_group_N_int2(
    const half* Q,
    const uint32_t* K,
    const half* scales,
    const half* zeros,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int token = warp_global / (NUM_HEADS / HEADS_PER_WARP);
    int head_pack = warp_global % (NUM_HEADS / HEADS_PER_WARP);
    if (token >= N) return;

    int subhead  = lane / LANES_PER_HEAD;
    int sublane  = lane % LANES_PER_HEAD;
    int head = head_pack * HEADS_PER_WARP + subhead;

    int group_token = token / G;

    float sum = 0.0f;

    #pragma unroll
    for (int p = 0; p < (PACKS_INT2_PER_HEAD / LANES_PER_HEAD); ++p) {
        int pack_idx = sublane + p * LANES_PER_HEAD;

        uint32_t packed_k = K[(token * NUM_HEADS + head) * PACKS_INT2_PER_HEAD + pack_idx];

        int q_start = head * HEAD_DIM + pack_idx * 16;
        int scale_start = (group_token * NUM_HEADS + head) * HEAD_DIM + pack_idx * 16;

        const float4* q_vec = reinterpret_cast<const float4*>(&Q[q_start]);
        const float4* s_vec = reinterpret_cast<const float4*>(&scales[scale_start]);
        const float4* z_vec = reinterpret_cast<const float4*>(&zeros[scale_start]);

        float4 q0 = q_vec[0];
        float4 q1 = q_vec[1];
        float4 s0 = s_vec[0];
        float4 s1 = s_vec[1];
        float4 z0 = z_vec[0];
        float4 z1 = z_vec[1];

        half* q_h0 = reinterpret_cast<half*>(&q0);
        half* q_h1 = reinterpret_cast<half*>(&q1);
        half* s_h0 = reinterpret_cast<half*>(&s0);
        half* s_h1 = reinterpret_cast<half*>(&s1);
        half* z_h0 = reinterpret_cast<half*>(&z0);
        half* z_h1 = reinterpret_cast<half*>(&z1);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed_k >> (i * 2)) & 0x3;
            float deq = val * __half2float(s_h0[i]) + __half2float(z_h0[i]);
            sum += __half2float(q_h0[i]) * deq;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int val = (packed_k >> ((i + 8) * 2)) & 0x3;
            float deq = val * __half2float(s_h1[i]) + __half2float(z_h1[i]);
            sum += __half2float(q_h1[i]) * deq;
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_HEAD, subhead);

    if (sublane == 0) {
        Out[token * NUM_HEADS + head] = __float2half(sum);
    }
}

__global__ void gemv_fp16(
    const half* Q,
    const half* K,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int token = warp_global / (NUM_HEADS / HEADS_PER_WARP);
    int head_pack = warp_global % (NUM_HEADS / HEADS_PER_WARP);
    if (token >= N) return;

    int subhead  = lane / LANES_PER_HEAD;  // 0..7
    int sublane  = lane % LANES_PER_HEAD;  // 0..3
    int head = head_pack * HEADS_PER_WARP + subhead;

    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < (HEAD_DIM / 8) / LANES_PER_HEAD; ++j) { // (16 / 4) = 4
        int chunk  = sublane + j * LANES_PER_HEAD;  // 0..15
        int offset = chunk * 8;

        int q_start = head * HEAD_DIM + offset;
        int k_start = (token * NUM_HEADS + head) * HEAD_DIM + offset;

        const float4* q_vec = reinterpret_cast<const float4*>(&Q[q_start]);
        const float4* k_vec = reinterpret_cast<const float4*>(&K[k_start]);

        float4 q_f = q_vec[0];
        float4 k_f = k_vec[0];

        half* q_h = reinterpret_cast<half*>(&q_f);
        half* k_h = reinterpret_cast<half*>(&k_f);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += __half2float(q_h[i]) * __half2float(k_h[i]);
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_HEAD, subhead);

    if (sublane == 0) {
        Out[token * NUM_HEADS + head] = __float2half(sum);
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

__global__ void gemv_sym_group_D_int3(
    const half* Q,
    const uint32_t* K,
    const half* scales,
    half* Out
) {
    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int token = warp_global / (NUM_HEADS / HEADS_PER_WARP);
    int head_pack = warp_global % (NUM_HEADS / HEADS_PER_WARP);

    if (token >= N) return;

    int subhead = lane / LANES_PER_HEAD;
    int sublane = lane % LANES_PER_HEAD;

    int head = head_pack * HEADS_PER_WARP + subhead;

    float sum = 0.0f;

    int group_idx = sublane;

    int k_start = (token * NUM_HEADS + head) * PACKS_INT3_PER_HEAD + group_idx * 3;

    uint32_t w0 = K[k_start];
    uint32_t w1 = K[k_start + 1];
    uint32_t w2 = K[k_start + 2];

    float s = __half2float(
        scales[(token * NUM_HEADS + head) * GROUPS_PER_HEAD + group_idx]
    );

    int q_start = head * HEAD_DIM + group_idx * G;

    const float4* q_vec = reinterpret_cast<const float4*>(&Q[q_start]);

    float4 q0 = q_vec[0];
    float4 q1 = q_vec[1];
    float4 q2 = q_vec[2];
    float4 q3 = q_vec[3];

    half* q_h0 = reinterpret_cast<half*>(&q0);
    half* q_h1 = reinterpret_cast<half*>(&q1);
    half* q_h2 = reinterpret_cast<half*>(&q2);
    half* q_h3 = reinterpret_cast<half*>(&q3);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int val = extract_3bit_signed(w0, w1, w2, i);
        sum += __half2float(q_h0[i]) * (val * s);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int val = extract_3bit_signed(w0, w1, w2, i + 8);
        sum += __half2float(q_h1[i]) * (val * s);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int val = extract_3bit_signed(w0, w1, w2, i + 16);
        sum += __half2float(q_h2[i]) * (val * s);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int val = extract_3bit_signed(w0, w1, w2, i + 24);
        sum += __half2float(q_h3[i]) * (val * s);
    }

    sum = subwarpReduceSum(sum, LANES_PER_HEAD, subhead);

    if (sublane == 0) {
        Out[token * NUM_HEADS + head] = __float2half(sum);
    }
}

__global__ void gemv_turboquant_int4(
    const half* __restrict__ Q,
    const uint32_t* __restrict__ K,
    const half* __restrict__ codebook,
    const float* __restrict__ K_norms,   // [N, NUM_HEADS]
    half* __restrict__ Out
) {
    __shared__ float cb[TURBOQUANT_CODEBOOK_SIZE];

    int tid = threadIdx.x;

    if (tid < TURBOQUANT_CODEBOOK_SIZE) {
        cb[tid] = __half2float(codebook[tid]);
    }

    __syncthreads();

    int warp_global = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    constexpr int HEAD_PACKS_PER_TOKEN = NUM_HEADS / HEADS_PER_WARP;

    int token = warp_global / HEAD_PACKS_PER_TOKEN;
    int head_pack = warp_global % HEAD_PACKS_PER_TOKEN;

    if (token >= N) return;

    int subhead = lane / LANES_PER_HEAD;          // 0..7
    int sublane = lane & (LANES_PER_HEAD - 1);    // 0..3

    int head = head_pack * HEADS_PER_WARP + subhead;

    float sum = 0.0f;

    #pragma unroll
    for (int p = 0; p < (PACKS_INT4_PER_HEAD / LANES_PER_HEAD); ++p) {
        int pack_idx = sublane + p * LANES_PER_HEAD;

        uint32_t packed_k = K[(token * NUM_HEADS + head) * PACKS_INT4_PER_HEAD + pack_idx];

        int q_start = head * HEAD_DIM + pack_idx * 8;

        const float4* q_vec = reinterpret_cast<const float4*>(&Q[q_start]);

        float4 q_raw = q_vec[0];

        const half* q_h = reinterpret_cast<const half*>(&q_raw);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int code = (packed_k >> (i * 4)) & 0xF;
            float k_val = cb[code];

            sum += __half2float(q_h[i]) * k_val;
        }
    }

    sum = subwarpReduceSum(sum, LANES_PER_HEAD, subhead);

    if (sublane == 0) {
        float k_norm = K_norms[token * NUM_HEADS + head];
        sum *= k_norm;

        Out[token * NUM_HEADS + head] = __float2half(sum);
    }
}

template <typename Func>
void run_benchmark(const char* name, Func kernel_launch) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 10; ++i) { // Warmup
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

    std::cout << "QK^T,N:" << N << "," << name << ",Latency:"
              << (ms / iters) * 1000.0f
              << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    size_t q_elements = NUM_HEADS * HEAD_DIM;
    size_t k_fp16_elements = (size_t)N * NUM_HEADS * HEAD_DIM;
    size_t k_int2_elements = (size_t)N * NUM_HEADS * PACKS_INT2_PER_HEAD;
    size_t k_int3_elements = (size_t)N * NUM_HEADS * PACKS_INT3_PER_HEAD;
    size_t out_elements = (size_t)N * NUM_HEADS;
    size_t k_int4_elements = (size_t)N * NUM_HEADS * PACKS_INT4_PER_HEAD;

    size_t scales_D_elements = (size_t)N * NUM_HEADS * GROUPS_PER_HEAD;
    size_t scales_N_elements = (size_t)N_GROUPS * NUM_HEADS * HEAD_DIM;
    size_t norm_elements = (size_t)N * NUM_HEADS;

    half* d_Q = nullptr;
    half* d_K_fp16 = nullptr;
    half* d_Out = nullptr;

    uint32_t* d_K_int2 = nullptr;
    uint32_t* d_K_int3 = nullptr;
    uint32_t* d_K_turbo_int4 = nullptr;

    half* d_scales_D = nullptr;
    half* d_scales_N = nullptr;
    half* d_zeros_N = nullptr;
    half* d_turbo_codebook = nullptr;
    float* d_turbo_norms = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, q_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_K_fp16, k_fp16_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_K_int2, k_int2_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_K_int3, k_int3_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_Out, out_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_K_turbo_int4, k_int4_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_turbo_codebook, TURBOQUANT_CODEBOOK_SIZE * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_turbo_norms, norm_elements * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_scales_D, scales_D_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_scales_N, scales_N_elements * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_zeros_N, scales_N_elements * sizeof(half)));

    int init_threads = 256;

    auto launch_half = [&](half* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_half<<<blocks, init_threads>>>(ptr, elements);
        CHECK_CUDA(cudaGetLastError());
    };

    auto launch_float = [&](float* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_norms_float<<<blocks, init_threads>>>(ptr, elements);
        CHECK_CUDA(cudaGetLastError());
    };

    auto launch_uint32 = [&](uint32_t* ptr, size_t elements) {
        size_t blocks = (elements + init_threads - 1) / init_threads;
        init_random_uint32<<<blocks, init_threads>>>(ptr, elements);
        CHECK_CUDA(cudaGetLastError());
    };

    launch_half(d_Q, q_elements);
    launch_half(d_K_fp16, k_fp16_elements);

    launch_uint32(d_K_int2, k_int2_elements);
    launch_uint32(d_K_int3, k_int3_elements);
    launch_uint32(d_K_turbo_int4, k_int4_elements);

    launch_half(d_zeros_N, scales_N_elements);

    {
        size_t blocks = (scales_D_elements + init_threads - 1) / init_threads;
        init_hybrid_scales<<<blocks, init_threads>>>(d_scales_D, scales_D_elements);
        CHECK_CUDA(cudaGetLastError());
    }

    {
        size_t blocks = (scales_N_elements + init_threads - 1) / init_threads;
        init_hybrid_scales<<<blocks, init_threads>>>(d_scales_N, scales_N_elements);
        CHECK_CUDA(cudaGetLastError());
    }
    init_turboquant_codebook<<<1, 32>>>(d_turbo_codebook);
    launch_float(d_turbo_norms, norm_elements);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    int threads = 128;
    int warps_per_block = threads / 32;

    int warps = N * (NUM_HEADS / HEADS_PER_WARP);
    int blocks = (warps + warps_per_block - 1) / warps_per_block;

    run_benchmark("int2_asym_Ndim", [&]() {
        gemv_asym_group_N_int2<<<blocks, threads>>>(
            d_Q,
            d_K_int2,
            d_scales_N,
            d_zeros_N,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("fp16", [&]() {
        gemv_fp16<<<blocks, threads>>>(
            d_Q,
            d_K_fp16,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("int3_sym_headdim", [&]() {
        gemv_sym_group_D_int3<<<blocks, threads>>>(
            d_Q,
            d_K_int3,
            d_scales_D,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    run_benchmark("int4_turboquant", [&]() {
        gemv_turboquant_int4<<<blocks, threads>>>(
            d_Q,
            d_K_turbo_int4,
            d_turbo_codebook,
            d_turbo_norms,
            d_Out
        );
    });
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_Q);
    cudaFree(d_K_fp16);
    cudaFree(d_K_int2);
    cudaFree(d_K_int3);
    cudaFree(d_K_turbo_int4);
    cudaFree(d_Out);
    cudaFree(d_scales_D);
    cudaFree(d_scales_N);
    cudaFree(d_zeros_N);
    cudaFree(d_turbo_codebook);
    cudaFree(d_turbo_norms);

    return 0;
}
