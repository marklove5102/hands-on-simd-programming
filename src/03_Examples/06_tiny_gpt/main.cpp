#include "../../include/simd_utils.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int SEQ_LEN = 8;
constexpr int EMBED_DIM = 64;
constexpr int NUM_HEADS = 4;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 16
constexpr int FF_DIM = 128;
constexpr int VOCAB_SIZE = 64;
constexpr int NUM_BLOCKS = 61;
constexpr float EPS = 1e-5f;

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

float horizontal_sum(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void transpose_matrix(const float* src, float* dst, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void split_heads(const float* src, float* dst) {
    for (int s = 0; s < SEQ_LEN; ++s) {
        for (int h = 0; h < NUM_HEADS; ++h) {
            const float* from = src + s * EMBED_DIM + h * HEAD_DIM;
            float* to = dst + (h * SEQ_LEN + s) * HEAD_DIM;
            std::copy(from, from + HEAD_DIM, to);
        }
    }
}

void combine_heads(const float* src, float* dst) {
    for (int s = 0; s < SEQ_LEN; ++s) {
        for (int h = 0; h < NUM_HEADS; ++h) {
            const float* from = src + (h * SEQ_LEN + s) * HEAD_DIM;
            float* to = dst + s * EMBED_DIM + h * HEAD_DIM;
            std::copy(from, from + HEAD_DIM, to);
        }
    }
}

void softmax_inplace(float* row, int len) {
    float max_val = row[0];
    for (int i = 1; i < len; ++i) {
        max_val = std::max(max_val, row[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < len; ++i) {
        row[i] *= inv;
    }
}

void rmsnorm_scalar(const float* input, const float* gamma, float* output, int length, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < length; ++i) {
        sum_sq += input[i] * input[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / length + eps);
    for (int i = 0; i < length; ++i) {
        output[i] = input[i] * gamma[i % EMBED_DIM] * scale;
    }
}

void rmsnorm_simd(const float* input, const float* gamma, float* output, int length, float eps) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float sum_sq = horizontal_sum(acc);
    for (; i < length; ++i) {
        sum_sq += input[i] * input[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / length + eps);
    __m256 scale_vec = _mm256_set1_ps(scale);
    i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 g = _mm256_loadu_ps(gamma + (i % EMBED_DIM));
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(v, g), scale_vec);
        _mm256_storeu_ps(output + i, result);
    }
    for (; i < length; ++i) {
        output[i] = input[i] * gamma[i % EMBED_DIM] * scale;
    }
}

void relu_scalar(float* data, int length) {
    for (int i = 0; i < length; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void relu_simd(float* data, int length) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_max_ps(zero, v));
    }
    for (; i < length; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void residual_add_scalar(const float* a, const float* b, float* out, int length) {
    for (int i = 0; i < length; ++i) {
        out[i] = a[i] + b[i];
    }
}

void residual_add_simd(const float* a, const float* b, float* out, int length) {
    int i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < length; ++i) {
        out[i] = a[i] + b[i];
    }
}

void matmul_scalar(const float* A, const float* B_T, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float* a_ptr = A + i * K;
            const float* b_ptr = B_T + j * K;
            for (int k = 0; k < K; ++k) {
                sum += a_ptr[k] * b_ptr[k];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_simd(const float* A, const float* B_T, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a_ptr = A + i * K;
            const float* b_ptr = B_T + j * K;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(a_ptr + k);
                __m256 b = _mm256_loadu_ps(b_ptr + k);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            float sum = horizontal_sum(acc);
            for (; k < K; ++k) {
                sum += a_ptr[k] * b_ptr[k];
            }
            C[i * N + j] = sum;
        }
    }
}

struct FloatLinear {
    int out_dim = 0;
    int in_dim = 0;
    std::vector<float> weights_T;
};

struct QuantizedLinear {
    int out_dim = 0;
    int in_dim = 0;
    std::vector<int8_t> weights_T;
    std::vector<float> scales;
};

struct LinearPair {
    FloatLinear fp32;
    QuantizedLinear q8;
};

struct BlockWeights {
    LinearPair wq;
    LinearPair wk;
    LinearPair wv;
    LinearPair wo;
    LinearPair wff1;
    LinearPair wff2;
    std::vector<float> gamma1;
    std::vector<float> gamma2;
};

struct ModelWeights {
    std::vector<float> embedding;
    std::vector<BlockWeights> blocks;
    LinearPair logits;
};

float attention_scale() {
    return 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
}

void quantize_into(const FloatLinear& src, QuantizedLinear& dst) {
    dst.out_dim = src.out_dim;
    dst.in_dim = src.in_dim;
    dst.weights_T.resize(static_cast<size_t>(dst.out_dim) * dst.in_dim);
    dst.scales.resize(dst.out_dim);
    for (int row = 0; row < dst.out_dim; ++row) {
        const float* src_row = src.weights_T.data() + row * dst.in_dim;
        float max_abs = 0.0f;
        for (int col = 0; col < dst.in_dim; ++col) {
            max_abs = std::max(max_abs, std::abs(src_row[col]));
        }
        float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        dst.scales[row] = scale;
        float inv_scale = scale > 0.0f ? (1.0f / scale) : 0.0f;
        int8_t* dst_row = dst.weights_T.data() + row * dst.in_dim;
        for (int col = 0; col < dst.in_dim; ++col) {
            float scaled = src_row[col] * inv_scale;
            int value = static_cast<int>(std::round(scaled));
            value = std::max(-127, std::min(127, value));
            dst_row[col] = static_cast<int8_t>(value);
        }
    }
}

float dot_q8_simd(const int8_t* w_row, const float* x, int length, float scale) {
    __m256 acc = _mm256_setzero_ps();
    __m256 scale_vec = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= length - 16; i += 16) {
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + i));
        __m128i lo_bytes = packed;
        __m128i hi_bytes = _mm_srli_si128(packed, 8);
        __m128i lo_i16 = _mm_cvtepi8_epi16(lo_bytes);
        __m128i hi_i16 = _mm_cvtepi8_epi16(hi_bytes);
        __m256i lo_i32 = _mm256_cvtepi16_epi32(lo_i16);
        __m256i hi_i32 = _mm256_cvtepi16_epi32(hi_i16);
        __m256 w_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(lo_i32), scale_vec);
        __m256 w_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(hi_i32), scale_vec);
        __m256 x_lo = _mm256_loadu_ps(x + i);
        __m256 x_hi = _mm256_loadu_ps(x + i + 8);
        acc = _mm256_fmadd_ps(w_lo, x_lo, acc);
        acc = _mm256_fmadd_ps(w_hi, x_hi, acc);
    }
    for (; i <= length - 8; i += 8) {
        __m128i packed8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_row + i));
        __m128i i16 = _mm_cvtepi8_epi16(packed8);
        __m256i i32 = _mm256_cvtepi16_epi32(i16);
        __m256 w_vec = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale_vec);
        __m256 x_vec = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
    }
    float sum = horizontal_sum(acc);
    for (; i < length; ++i) {
        sum += static_cast<float>(w_row[i]) * scale * x[i];
    }
    return sum;
}

void matmul_q8_simd(const float* A, const QuantizedLinear& W_T, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        const float* a_ptr = A + i * K;
        for (int j = 0; j < N; ++j) {
            const int8_t* w_row = W_T.weights_T.data() + j * K;
            float scale = W_T.scales[j];
            C[i * N + j] = dot_q8_simd(w_row, a_ptr, K, scale);
        }
    }
}

struct StageTimes {
    double embed = 0.0;
    double rms1 = 0.0;
    double qkv = 0.0;
    double attn_scores = 0.0;
    double attn_softmax = 0.0;
    double attn_context = 0.0;
    double attn_proj = 0.0;
    double residual1 = 0.0;
    double rms2 = 0.0;
    double ffn_expand = 0.0;
    double activation = 0.0;
    double ffn_contract = 0.0;
    double residual2 = 0.0;
    double logits = 0.0;
    double sampling = 0.0;
    double total = 0.0;

    StageTimes& accumulate(const StageTimes& other) {
        embed += other.embed;
        rms1 += other.rms1;
        qkv += other.qkv;
        attn_scores += other.attn_scores;
        attn_softmax += other.attn_softmax;
        attn_context += other.attn_context;
        attn_proj += other.attn_proj;
        residual1 += other.residual1;
        rms2 += other.rms2;
        ffn_expand += other.ffn_expand;
        activation += other.activation;
        ffn_contract += other.ffn_contract;
        residual2 += other.residual2;
        logits += other.logits;
        sampling += other.sampling;
        total += other.total;
        return *this;
    }

    StageTimes& scale(double factor) {
        embed *= factor;
        rms1 *= factor;
        qkv *= factor;
        attn_scores *= factor;
        attn_softmax *= factor;
        attn_context *= factor;
        attn_proj *= factor;
        residual1 *= factor;
        rms2 *= factor;
        ffn_expand *= factor;
        activation *= factor;
        ffn_contract *= factor;
        residual2 *= factor;
        logits *= factor;
        sampling *= factor;
        total *= factor;
        return *this;
    }
};

struct ScalarKernels {
    static void rmsnorm(const float* input, const std::vector<float>& gamma, float* output, int length, float eps) {
        rmsnorm_scalar(input, gamma.data(), output, length, eps);
    }

    static void apply_linear(const LinearPair& weight, const float* input, float* output, int M, int K, int N) {
        matmul_scalar(input, weight.fp32.weights_T.data(), output, M, K, N);
    }

    static void add_residual(const float* a, const float* b, float* out, int length) {
        residual_add_scalar(a, b, out, length);
    }

    static void activation(float* data, int length) {
        relu_scalar(data, length);
    }

    static void matmul_float(const float* A, const float* B_T, float* C, int M, int K, int N) {
        matmul_scalar(A, B_T, C, M, K, N);
    }
};

struct SimdKernels {
    static void rmsnorm(const float* input, const std::vector<float>& gamma, float* output, int length, float eps) {
        rmsnorm_simd(input, gamma.data(), output, length, eps);
    }

    static void apply_linear(const LinearPair& weight, const float* input, float* output, int M, int K, int N) {
        matmul_q8_simd(input, weight.q8, output, M, K, N);
    }

    static void add_residual(const float* a, const float* b, float* out, int length) {
        residual_add_simd(a, b, out, length);
    }

    static void activation(float* data, int length) {
        relu_simd(data, length);
    }

    static void matmul_float(const float* A, const float* B_T, float* C, int M, int K, int N) {
        matmul_simd(A, B_T, C, M, K, N);
    }
};

std::vector<float> random_matrix_T(int rows, int cols, std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    std::vector<float> original(static_cast<size_t>(rows) * cols);
    for (float& v : original) {
        v = dist(rng);
    }
    std::vector<float> transposed(static_cast<size_t>(rows) * cols);
    transpose_matrix(original.data(), transposed.data(), rows, cols);
    return transposed;
}

LinearPair make_linear_pair(int out_dim, int in_dim, std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    LinearPair pair;
    pair.fp32.out_dim = out_dim;
    pair.fp32.in_dim = in_dim;
    pair.fp32.weights_T = random_matrix_T(out_dim, in_dim, rng, dist);
    pair.q8.out_dim = out_dim;
    pair.q8.in_dim = in_dim;
    quantize_into(pair.fp32, pair.q8);
    return pair;
}

void initialize(ModelWeights& weights) {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

    weights.embedding.resize(static_cast<size_t>(VOCAB_SIZE) * EMBED_DIM);
    for (float& v : weights.embedding) {
        v = dist(rng);
    }

    weights.blocks.clear();
    weights.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        BlockWeights block;
        block.wq = make_linear_pair(EMBED_DIM, EMBED_DIM, rng, dist);
        block.wk = make_linear_pair(EMBED_DIM, EMBED_DIM, rng, dist);
        block.wv = make_linear_pair(EMBED_DIM, EMBED_DIM, rng, dist);
        block.wo = make_linear_pair(EMBED_DIM, EMBED_DIM, rng, dist);
        block.wff1 = make_linear_pair(FF_DIM, EMBED_DIM, rng, dist);
        block.wff2 = make_linear_pair(EMBED_DIM, FF_DIM, rng, dist);
        block.gamma1.assign(EMBED_DIM, 1.0f);
        block.gamma2.assign(EMBED_DIM, 1.0f);
        weights.blocks.push_back(std::move(block));
    }

    weights.logits = make_linear_pair(VOCAB_SIZE, EMBED_DIM, rng, dist);
}

inline double elapsed_us(const Clock::time_point& start, const Clock::time_point& end) {
    return static_cast<double>(std::chrono::duration_cast<Microseconds>(end - start).count());
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return diff;
}

template <typename Kernels>
void decode_impl(const ModelWeights& weights,
                 const std::vector<int>& tokens,
                 std::vector<float>& block_output,
                 std::vector<float>& logits_out,
                 int& next_token,
                 StageTimes* times) {
    StageTimes local;

    Clock::time_point total_start;
    if (times) {
        total_start = Clock::now();
    }

    constexpr int token_dim = SEQ_LEN * EMBED_DIM;
    constexpr int ff_dim = SEQ_LEN * FF_DIM;

    std::array<float, token_dim> hidden{};
    std::array<float, token_dim> norm1{};
    std::array<float, token_dim> Q{};
    std::array<float, token_dim> K{};
    std::array<float, token_dim> V{};
    std::array<float, NUM_HEADS * SEQ_LEN * HEAD_DIM> Q_heads{};
    std::array<float, NUM_HEADS * SEQ_LEN * HEAD_DIM> K_heads{};
    std::array<float, NUM_HEADS * SEQ_LEN * HEAD_DIM> V_heads{};
    std::array<float, NUM_HEADS * HEAD_DIM * SEQ_LEN> K_heads_T{};
    std::array<float, NUM_HEADS * HEAD_DIM * SEQ_LEN> V_heads_T{};
    std::array<float, NUM_HEADS * SEQ_LEN * HEAD_DIM> context_heads{};
    std::array<float, token_dim> context{};
    std::array<float, token_dim> attn_proj{};
    std::array<float, token_dim> residual1{};
    std::array<float, token_dim> norm2{};
    std::array<float, ff_dim> ff1{};
    std::array<float, token_dim> ff2{};
    std::array<float, token_dim> residual2{};
    std::array<float, SEQ_LEN * SEQ_LEN> scores{};
    std::array<float, VOCAB_SIZE> logits{};

    auto record = [&](double& field, const Clock::time_point& start_tp, const Clock::time_point& end_tp) {
        if (times) {
            field += elapsed_us(start_tp, end_tp);
        }
    };

    auto t0 = Clock::now();
    for (int t = 0; t < SEQ_LEN; ++t) {
        const float* src = weights.embedding.data() + tokens[t] * EMBED_DIM;
        std::copy(src, src + EMBED_DIM, hidden.data() + t * EMBED_DIM);
    }
    record(local.embed, t0, Clock::now());

    const float scale = static_cast<float>(attention_scale());
    for (std::size_t block_idx = 0; block_idx < weights.blocks.size(); ++block_idx) {
        const BlockWeights& block = weights.blocks[block_idx];

        t0 = Clock::now();
        Kernels::rmsnorm(hidden.data(), block.gamma1, norm1.data(), token_dim, EPS);
        record(local.rms1, t0, Clock::now());

        t0 = Clock::now();
        Kernels::apply_linear(block.wq, norm1.data(), Q.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
        Kernels::apply_linear(block.wk, norm1.data(), K.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
        Kernels::apply_linear(block.wv, norm1.data(), V.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
        record(local.qkv, t0, Clock::now());

        split_heads(Q.data(), Q_heads.data());
        split_heads(K.data(), K_heads.data());
        split_heads(V.data(), V_heads.data());

        for (int h = 0; h < NUM_HEADS; ++h) {
            const float* q_head = Q_heads.data() + h * SEQ_LEN * HEAD_DIM;
            const float* k_head = K_heads.data() + h * SEQ_LEN * HEAD_DIM;
            const float* v_head = V_heads.data() + h * SEQ_LEN * HEAD_DIM;
            float* k_t = K_heads_T.data() + h * HEAD_DIM * SEQ_LEN;
            float* v_t = V_heads_T.data() + h * HEAD_DIM * SEQ_LEN;
            transpose_matrix(k_head, k_t, SEQ_LEN, HEAD_DIM);
            transpose_matrix(v_head, v_t, SEQ_LEN, HEAD_DIM);

            t0 = Clock::now();
            Kernels::matmul_float(q_head, k_t, scores.data(), SEQ_LEN, HEAD_DIM, SEQ_LEN);
            record(local.attn_scores, t0, Clock::now());

            auto softmax_start = Clock::now();
            for (float& s : scores) {
                s *= scale;
            }
            for (int row = 0; row < SEQ_LEN; ++row) {
                softmax_inplace(scores.data() + row * SEQ_LEN, SEQ_LEN);
            }
            record(local.attn_softmax, softmax_start, Clock::now());

            float* ctx = context_heads.data() + h * SEQ_LEN * HEAD_DIM;
            t0 = Clock::now();
            Kernels::matmul_float(scores.data(), v_t, ctx, SEQ_LEN, SEQ_LEN, HEAD_DIM);
            record(local.attn_context, t0, Clock::now());
        }

        combine_heads(context_heads.data(), context.data());

        t0 = Clock::now();
        Kernels::apply_linear(block.wo, context.data(), attn_proj.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
        record(local.attn_proj, t0, Clock::now());

        t0 = Clock::now();
        Kernels::add_residual(hidden.data(), attn_proj.data(), residual1.data(), token_dim);
        record(local.residual1, t0, Clock::now());

        t0 = Clock::now();
        Kernels::rmsnorm(residual1.data(), block.gamma2, norm2.data(), token_dim, EPS);
        record(local.rms2, t0, Clock::now());

        t0 = Clock::now();
        Kernels::apply_linear(block.wff1, norm2.data(), ff1.data(), SEQ_LEN, EMBED_DIM, FF_DIM);
        record(local.ffn_expand, t0, Clock::now());

        t0 = Clock::now();
        Kernels::activation(ff1.data(), ff_dim);
        record(local.activation, t0, Clock::now());

        t0 = Clock::now();
        Kernels::apply_linear(block.wff2, ff1.data(), ff2.data(), SEQ_LEN, FF_DIM, EMBED_DIM);
        record(local.ffn_contract, t0, Clock::now());

        t0 = Clock::now();
        Kernels::add_residual(residual1.data(), ff2.data(), residual2.data(), token_dim);
        record(local.residual2, t0, Clock::now());

        std::copy(residual2.begin(), residual2.end(), hidden.begin());
    }

    block_output.assign(hidden.begin(), hidden.end());

    const float* last_token = hidden.data() + (SEQ_LEN - 1) * EMBED_DIM;
    t0 = Clock::now();
    Kernels::apply_linear(weights.logits, last_token, logits.data(), 1, EMBED_DIM, VOCAB_SIZE);
    record(local.logits, t0, Clock::now());

    logits_out.assign(logits.begin(), logits.end());

    std::vector<float> probs(logits.begin(), logits.end());
    auto samp_start = Clock::now();
    softmax_inplace(probs.data(), VOCAB_SIZE);
    next_token = static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    record(local.sampling, samp_start, Clock::now());

    if (times) {
        local.total += elapsed_us(total_start, Clock::now());
        times->accumulate(local);
    }
}


} // namespace

int main() {
    ModelWeights weights;
    initialize(weights);

    std::vector<int> prompt = {3, 17, 12, 8, 5, 9, 2, 0};

    std::vector<float> scalar_hidden;
    std::vector<float> scalar_logits;
    std::vector<float> simd_hidden;
    std::vector<float> simd_logits;
    int scalar_next = -1;
    int simd_next = -1;

    constexpr int warmup = 5;
    for (int i = 0; i < warmup; ++i) {
        decode_impl<ScalarKernels>(weights, prompt, scalar_hidden, scalar_logits, scalar_next, nullptr);
        decode_impl<SimdKernels>(weights, prompt, simd_hidden, simd_logits, simd_next, nullptr);
    }

    StageTimes scalar_accum;
    StageTimes simd_accum;
    constexpr int iterations = 20;
    for (int i = 0; i < iterations; ++i) {
        decode_impl<ScalarKernels>(weights, prompt, scalar_hidden, scalar_logits, scalar_next, &scalar_accum);
    }
    for (int i = 0; i < iterations; ++i) {
        decode_impl<SimdKernels>(weights, prompt, simd_hidden, simd_logits, simd_next, &simd_accum);
    }

    scalar_accum.scale(1.0 / iterations);
    simd_accum.scale(1.0 / iterations);

    float hidden_diff = max_abs_diff(scalar_hidden, simd_hidden);
    float logits_diff = max_abs_diff(scalar_logits, simd_logits);

    std::cout << "Tiny GPT block (scalar vs SIMD quantized)\n";
    std::cout << "Decoder blocks: " << weights.blocks.size() << "\n";
    std::cout << "Prompt tokens: ";
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << prompt[i];
    }
    std::cout << "\n";
    std::cout << "Scalar next token: " << scalar_next << "\n";
    std::cout << "SIMD next token:   " << simd_next << "\n";
    std::cout << "Max hidden diff:   " << hidden_diff << "\n";
    std::cout << "Max logits diff:   " << logits_diff << "\n";

    struct ComponentRow {
        std::string name;
        int count;
        double StageTimes::*field;
    };

    const int block_count = static_cast<int>(weights.blocks.size());
    const int per_head = block_count * NUM_HEADS;

    std::vector<ComponentRow> components = {
        {"embedding", 1, &StageTimes::embed},
        {"rmsnorm_1", block_count, &StageTimes::rms1},
        {"qkv_linear", block_count, &StageTimes::qkv},
        {"attention_scores", per_head, &StageTimes::attn_scores},
        {"attention_softmax", per_head, &StageTimes::attn_softmax},
        {"attention_context", per_head, &StageTimes::attn_context},
        {"attention_projection", block_count, &StageTimes::attn_proj},
        {"residual_1", block_count, &StageTimes::residual1},
        {"rmsnorm_2", block_count, &StageTimes::rms2},
        {"ffn_expand", block_count, &StageTimes::ffn_expand},
        {"activation", block_count, &StageTimes::activation},
        {"ffn_contract", block_count, &StageTimes::ffn_contract},
        {"residual_2", block_count, &StageTimes::residual2},
        {"logits_projection", 1, &StageTimes::logits},
        {"sampling", 1, &StageTimes::sampling}
    };

    double total_scalar = scalar_accum.total;
    double total_simd = simd_accum.total;
    double total_saved = total_scalar - total_simd;

    namespace fs = std::filesystem;
    fs::path out_dir = fs::current_path().parent_path().parent_path().parent_path() / "artifacts";
    fs::create_directories(out_dir);

    fs::path csv_path = out_dir / "tiny_gpt_components.csv";
    std::ofstream csv(csv_path);
    if (csv) {
        csv << "stage,count,scalar_total_us,simd_total_us,speedup,time_saved_us,contribution_pct\n";
        for (const auto& comp : components) {
            double scalar_val = scalar_accum.*(comp.field);
            double simd_val = simd_accum.*(comp.field);
            double saved = scalar_val - simd_val;
            double speedup = simd_val > 0.0 ? scalar_val / simd_val : 0.0;
            double pct = (total_saved != 0.0) ? (saved / total_saved * 100.0) : 0.0;
            csv << comp.name << ',' << comp.count << ',' << scalar_val << ',' << simd_val << ','
                << speedup << ',' << saved << ',' << pct << '\n';
        }
        double overall_speedup = total_simd > 0.0 ? total_scalar / total_simd : 0.0;
        csv << "overall," << block_count << ',' << total_scalar << ',' << total_simd << ','
            << overall_speedup << ',' << total_saved << ',' << 100.0 << '\n';
    } else {
        std::cerr << "Failed to write " << csv_path << "\n";
    }

    set_benchmark_suite("03_Examples/06_tiny_gpt");
    auto scalar_decode = [&]() {
        std::vector<float> hidden;
        std::vector<float> logits;
        int next = -1;
        decode_impl<ScalarKernels>(weights, prompt, hidden, logits, next, nullptr);
    };
    auto simd_decode = [&]() {
        std::vector<float> hidden;
        std::vector<float> logits;
        int next = -1;
        decode_impl<SimdKernels>(weights, prompt, hidden, logits, next, nullptr);
    };

    benchmark_comparison("tiny_gpt_decode", scalar_decode, simd_decode, 50);

    std::cout << "Scalar average total: " << total_scalar << " us\n";
    std::cout << "SIMD average total:   " << total_simd << " us\n";
    std::cout << "Overall speedup:      " << (total_simd > 0.0 ? total_scalar / total_simd : 0.0) << "x\n";

    return 0;
}
