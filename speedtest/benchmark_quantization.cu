#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

#define NUM_HEADS 32
#define HEAD_DIM 128
// #define N 32
#define G 32
#define D_MODEL (NUM_HEADS * HEAD_DIM)
#define GROUPS_PER_HEAD (HEAD_DIM / G)

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} \

__inline__ __device__ uint32_t pack_2bit_warp(uint32_t q_val, int lane_id) {
    uint32_t my_bits = (q_val & 0x3) << ((lane_id % 16) * 2);

    for (int offset = 8; offset > 0; offset /= 2) {
        my_bits |= __shfl_down_sync(0xffffffff, my_bits, offset, 16);
    }

    return my_bits;
}

__inline__ __device__ void pack_3bit_warp(
    uint32_t q,
    int lane,
    uint32_t& w0,
    uint32_t& w1,
    uint32_t& w2
) {
    w0 = 0;
    w1 = 0;
    w2 = 0;

    int bit = lane * 3;

    if (bit < 32) {
        w0 = q << bit;

        if (bit > 29) {
            w1 = q >> (32 - bit);
        }
    }
    else if (bit < 64) {
        int local = bit - 32;

        w1 = q << local;

        if (bit > 61) {
            w2 = q >> (64 - bit);
        }
    }
    else {
        w2 = q << (bit - 64);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        w0 |= __shfl_down_sync(0xffffffff, w0, offset);
        w1 |= __shfl_down_sync(0xffffffff, w1, offset);
        w2 |= __shfl_down_sync(0xffffffff, w2, offset);
    }
}

__inline__ __device__ uint32_t pack_4bit_subwarp(uint32_t q_val, int lane_id) {
    uint32_t my_bits = (q_val & 0xF) << ((lane_id & 7) * 4);

    my_bits |= __shfl_down_sync(0xffffffff, my_bits, 4, 8);
    my_bits |= __shfl_down_sync(0xffffffff, my_bits, 2, 8);
    my_bits |= __shfl_down_sync(0xffffffff, my_bits, 1, 8);

    return my_bits;
}

__inline__ __device__ float warpAllReduceMax(float val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float warpAllReduceMin(float val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


__global__ void symmetricQuantization(
    const float* __restrict__ input,
    uint32_t* __restrict__ packed_q,
    half* __restrict__ scales,
    int num_groups
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int group_id = tid / 32;
    if (group_id >= num_groups) return;
    float val = input[tid];

    float max_abs = warpAllReduceMax(fabsf(val));
    max_abs = __shfl_sync(0xffffffff, max_abs, 0);

    float scale_sym = fmaxf(max_abs / 2.0f, 1e-7f);

    float q_sym_f = roundf(val / scale_sym);
    q_sym_f = fmaxf(-2.0f, fminf(1.0f, q_sym_f));

    uint32_t final_q = static_cast<uint32_t>(q_sym_f + 2.0f);
    uint32_t packed = pack_2bit_warp(final_q, lane_id);

    if (lane_id == 0) {
        packed_q[group_id * 2] = packed;
        scales[group_id] = __float2half(scale_sym);
    }

    if (lane_id == 16) {
        packed_q[group_id * 2 + 1] = packed;
    }
}

__global__ void symmetricQuantization3Bit(
    const float* __restrict__ input,
    uint32_t* __restrict__ packed_q_3bit,
    half* __restrict__ scales,
    int num_groups
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int group_id = tid / 32;
    if (group_id >= num_groups) return;
    float val = input[tid];

    float max_abs = warpAllReduceMax(fabsf(val));
    max_abs = __shfl_sync(0xffffffff, max_abs, 0);

    float scale_sym = fmaxf(max_abs / 4.0f, 1e-7f);

    float q_sym_f = roundf(val / scale_sym);
    q_sym_f = fmaxf(-4.0f, fminf(3.0f, q_sym_f));

    uint32_t q_val = static_cast<uint32_t>(q_sym_f + 4.0f) & 0x7;

    uint32_t my_w0, my_w1, my_w2;
    pack_3bit_warp(q_val, lane_id, my_w0, my_w1, my_w2);

    if (lane_id == 0) {
        *reinterpret_cast<uint3*>(&packed_q_3bit[group_id * 3]) = make_uint3(my_w0, my_w1, my_w2);
        scales[group_id] = __float2half(scale_sym);
    }
}

__global__ void hybridQuantization(
    const float* __restrict__ input,
    uint32_t* __restrict__ packed_q,
    half* __restrict__ scales,
    half* __restrict__ zps,
    int num_groups
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int group_id = tid / 32;
    if (group_id >= num_groups) return;
    float val = input[tid];
    float min_val = warpAllReduceMin(val);
    float max_val = warpAllReduceMax(val);
    float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    float scale_sym = fmaxf(max_abs / 2.0f, 1e-7f);

    float q_sym_f = roundf(val / scale_sym);
    q_sym_f = fmaxf(-2.0f, fminf(1.0f, q_sym_f));

    float dq_sym = q_sym_f * scale_sym;
    float local_err_sym = fabsf(val - dq_sym);
    float scale_asym = fmaxf((max_val - min_val) / 3.0f, 1e-7f);
    float zp_asym = min_val;

    float q_asym_f = roundf((val - zp_asym) / scale_asym);
    q_asym_f = fmaxf(0.0f, fminf(3.0f, q_asym_f));

    float dq_asym = q_asym_f * scale_asym + zp_asym;
    float local_err_asym = fabsf(val - dq_asym);

    float err_sym = warpAllReduceSum(local_err_sym);
    float err_asym = warpAllReduceSum(local_err_asym);
    bool use_asym = err_asym < err_sym;
    uint32_t final_q = use_asym ? static_cast<uint32_t>(q_asym_f) : static_cast<uint32_t>(q_sym_f + 2.0f);
    uint32_t packed = pack_2bit_warp(final_q, lane_id);

    if (lane_id == 0) {
        packed_q[group_id * 2] = packed;
        if (use_asym) {
            unsigned short h_bits = __half_as_ushort(__float2half(scale_asym));
            h_bits |= 0x8000;
            scales[group_id] = __ushort_as_half(h_bits);
            zps[group_id] = __float2half(zp_asym);
        } else {
            unsigned short h_bits = __half_as_ushort(__float2half(scale_sym));
            h_bits &= 0x7FFF;
            scales[group_id] = __ushort_as_half(h_bits);
            zps[group_id] = __float2half(0.0f);
        }
    }

    if (lane_id == 16) {
        packed_q[group_id * 2 + 1] = packed;
    }
}

__global__ void asymmetricQuantization(
    const float* __restrict__ input,
    uint32_t* __restrict__ packed_q,
    half* __restrict__ scales,
    half* __restrict__ zps,
    int num_groups
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int group_id = tid / 32;
    if (group_id >= num_groups) return;
    float val = input[tid];
    float min_val = warpAllReduceMin(val);
    float max_val = warpAllReduceMax(val);
    float scale = fmaxf((max_val - min_val) / 3.0f, 1e-7f);
    float zp = min_val;
    float q_f = roundf((val - zp) / scale);
    uint32_t final_q = static_cast<uint32_t>(fmaxf(0.0f, fminf(3.0f, q_f)));
    uint32_t packed = pack_2bit_warp(final_q, lane_id);

    if (lane_id == 0) {
        packed_q[group_id * 2] = packed;
        scales[group_id] = __float2half(scale);
        zps[group_id] = __float2half(zp);
    }

    if (lane_id == 16) {
        packed_q[group_id * 2 + 1] = packed;
    }
}

__global__ void turboQuantMSE4Bit(
    const float* __restrict__ input,
    const float* __restrict__ codebook,
    uint32_t* __restrict__ packed_indices_4bit,
    float* __restrict__ norms,
    int num_vectors
) {
    int lane_id = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int vector_id = blockIdx.x * warps_per_block + warp_in_block;
    if (vector_id >= num_vectors) return;
    const int base = vector_id * HEAD_DIM;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int d = lane_id; d < HEAD_DIM; d += 32) {
        float v = input[base + d];
        sum_sq += v * v;
    }

    float norm_sq = warpAllReduceSum(sum_sq);
    float norm = fmaxf(sqrtf(norm_sq), 1e-12f);

    if (lane_id == 0) {
        norms[vector_id] = norm;
    }

    #pragma unroll
    for (int d = lane_id; d < HEAD_DIM; d += 32) {
        float y = input[base + d] / norm;
        int best_idx = 0;
        float best_dist = fabsf(y - codebook[0]);

        #pragma unroll
        for (int c = 1; c < 16; ++c) {
            float dist = fabsf(y - codebook[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }

        uint32_t q = static_cast<uint32_t>(best_idx) & 0xF;
        uint32_t packed = pack_4bit_subwarp(q, lane_id);

        if ((lane_id & 7) == 0) {
            int packed_word = d >> 3;
            packed_indices_4bit[vector_id * (HEAD_DIM / 8) + packed_word] = packed;
        }
    }
}

__global__ void turboQuantMSE3Bit(
    const float* __restrict__ input,
    const float* __restrict__ codebook_3bit,   // length 8
    uint32_t* __restrict__ packed_indices_3bit,
    float* __restrict__ norms,
    int num_vectors
) {
    constexpr int WORDS_PER_GROUP = 3;

    int lane_id = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int vector_id = blockIdx.x * warps_per_block + warp_in_block;
    if (vector_id >= num_vectors) return;
    const int base = vector_id * HEAD_DIM;
    float sum_sq = 0.0f;

    #pragma unroll
    for (int d = lane_id; d < HEAD_DIM; d += 32) {
        float v = input[base + d];
        sum_sq += v * v;
    }

    float norm_sq = warpAllReduceSum(sum_sq);
    float norm = fmaxf(sqrtf(norm_sq), 1e-12f);

    if (lane_id == 0) {
        norms[vector_id] = norm;
    }

    constexpr int GROUPS_PER_VECTOR = HEAD_DIM / 32;

    #pragma unroll
    for (int g = 0; g < GROUPS_PER_VECTOR; ++g) {
        int d = g * 32 + lane_id;
        float y = input[base + d] / norm;
        int best_idx = 0;
        float best_dist = fabsf(y - codebook_3bit[0]);

        #pragma unroll
        for (int c = 1; c < 8; ++c) {
            float dist = fabsf(y - codebook_3bit[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }

        uint32_t q = static_cast<uint32_t>(best_idx) & 0x7;
        uint32_t w0, w1, w2;

        pack_3bit_warp(q,lane_id,w0,w1,w2);

        if (lane_id == 0) {
            int out_idx = vector_id * (GROUPS_PER_VECTOR * WORDS_PER_GROUP) + g * WORDS_PER_GROUP;

            packed_indices_3bit[out_idx + 0] = w0;
            packed_indices_3bit[out_idx + 1] = w1;
            packed_indices_3bit[out_idx + 2] = w2;
        }
    }
}

template <typename LaunchFunc>
float run_benchmark(
    const char* name,
    LaunchFunc launch,
    int warmup,
    int iters
) {
    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) {
        launch();
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; i++) {
        launch();
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float us_per_iter = 1000.0f * ms / static_cast<float>(iters);

    std::cout << name << us_per_iter << " us/iter\n";

    return us_per_iter;
}

int main() {
    const int totalN = N * NUM_HEADS * HEAD_DIM;
    const int num_groups = N * NUM_HEADS * GROUPS_PER_HEAD;
    const int tq_num_vectors = N * NUM_HEADS;
    const int tq_packed_words_per_vector = HEAD_DIM / 8;

    std::vector<float> host_input(totalN);

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int hg = 0; hg < GROUPS_PER_HEAD; hg++) {
                int group_id = n * NUM_HEADS * GROUPS_PER_HEAD + h * GROUPS_PER_HEAD + hg;
                bool make_asym = group_id % 100 == 0;

                for (int i = 0; i < G; i++) {
                    int elem_idx = n * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + hg * G + i;

                    if (make_asym) {
                        host_input[elem_idx] = static_cast<float>(i % 4);
                    } else {
                        host_input[elem_idx] = static_cast<float>(i % 4) - 2.0f;
                    }
                }
            }
        }
    }

    float* d_input = nullptr;
    uint32_t* d_packed_q = nullptr;
    uint32_t* d_packed_q_3bit = nullptr;
    half* d_scales = nullptr;
    half* d_zps = nullptr;
    uint32_t* d_tq_packed_indices_4bit = nullptr;
    float* d_tq_norms = nullptr;
    float* d_tq_codebook_4bit = nullptr;
    uint32_t* d_tq_packed_indices_3bit = nullptr;
    float* d_tq_codebook_3bit = nullptr;

    constexpr int tq_3bit_words_per_vector = (HEAD_DIM / 32) * 3;

    CHECK_CUDA(cudaMalloc(&d_tq_packed_indices_3bit, tq_num_vectors * tq_3bit_words_per_vector * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_tq_codebook_3bit, 8 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input, totalN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_packed_q, num_groups * 2 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_packed_q_3bit, num_groups * 3 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scales, num_groups * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_zps, num_groups * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        host_input.data(),
        totalN * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMalloc(
        &d_tq_packed_indices_4bit,
        tq_num_vectors * tq_packed_words_per_vector * sizeof(uint32_t)
    ));

    CHECK_CUDA(cudaMalloc(
        &d_tq_norms,
        tq_num_vectors * sizeof(float)
    ));

    CHECK_CUDA(cudaMalloc(
        &d_tq_codebook_4bit,
        16 * sizeof(float)
    ));

    std::vector<float> host_tq_codebook_4bit = {
        -0.35f,-0.28f,-0.22f,-0.17f,
        -0.12f,-0.08f,-0.045f,-0.015f,
         0.015f,0.045f,0.08f,0.12f,
         0.17f,0.22f,0.28f,0.35f
    };

    CHECK_CUDA(cudaMemcpy(
        d_tq_codebook_4bit,
        host_tq_codebook_4bit.data(),
        16 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    std::vector<float> host_tq_codebook_3bit = {
        -0.30f,-0.20f,-0.12f,-0.04f,
         0.04f,0.12f,0.20f,0.30f
    };

    CHECK_CUDA(cudaMemcpy(
        d_tq_codebook_3bit,
        host_tq_codebook_3bit.data(),
        8 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const int blockSize = 256;
    const int numBlocks = (totalN + blockSize - 1) / blockSize;
    const int warpsPerBlock = blockSize / 32;
    const int tqNumBlocks = (tq_num_vectors + warpsPerBlock - 1) / warpsPerBlock;
    const int warmup = 10;
    const int iters = 100;

    std::cout << "N: " << N
              << " NUM_HEADS: " << NUM_HEADS
              << " HEAD_DIM: " << HEAD_DIM
              << " D_MODEL: " << D_MODEL
              << " G: " << G
              << "\n";

    std::cout << "-------------------------------------------\n";

    run_benchmark(
        "2-Bit Symmetric Time:          ",
        [&]() {
            symmetricQuantization<<<numBlocks, blockSize>>>(
                d_input,
                d_packed_q,
                d_scales,
                num_groups
            );
        },
        warmup,
        iters
    );

    run_benchmark(
        "3-Bit Symmetric Time:          ",
        [&]() {
            symmetricQuantization3Bit<<<numBlocks, blockSize>>>(
                d_input,
                d_packed_q_3bit,
                d_scales,
                num_groups
            );
        },
        warmup,
        iters
    );

    run_benchmark(
        "2-Bit Asymmetric Time:         ",
        [&]() {
            asymmetricQuantization<<<numBlocks, blockSize>>>(
                d_input,
                d_packed_q,
                d_scales,
                d_zps,
                num_groups
            );
        },
        warmup,
        iters
    );

    run_benchmark(
        "2-Bit Hybrid Time:             ",
        [&]() {
            hybridQuantization<<<numBlocks, blockSize>>>(
                d_input,
                d_packed_q,
                d_scales,
                d_zps,
                num_groups
            );
        },
        warmup,
        iters
    );

    run_benchmark(
        "TurboQuantMSE 4-Bit Time:      ",
        [&]() {
            turboQuantMSE4Bit<<<tqNumBlocks, blockSize>>>(
                d_input,
                d_tq_codebook_4bit,
                d_tq_packed_indices_4bit,
                d_tq_norms,
                tq_num_vectors
            );
        },
        warmup,
        iters
    );

    run_benchmark(
        "TurboQuantMSE 3-Bit Time:      ",
        [&]() {
            turboQuantMSE3Bit<<<tqNumBlocks, blockSize>>>(
                d_input,
                d_tq_codebook_3bit,
                d_tq_packed_indices_3bit,
                d_tq_norms,
                tq_num_vectors
            );
        },
        warmup,
        iters
    );

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_packed_q));
    CHECK_CUDA(cudaFree(d_packed_q_3bit));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_zps));
    CHECK_CUDA(cudaFree(d_tq_packed_indices_4bit));
    CHECK_CUDA(cudaFree(d_tq_norms));
    CHECK_CUDA(cudaFree(d_tq_codebook_4bit));
    CHECK_CUDA(cudaFree(d_tq_packed_indices_3bit));
    CHECK_CUDA(cudaFree(d_tq_codebook_3bit));

    return 0;
}
