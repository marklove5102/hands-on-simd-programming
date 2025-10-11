#include "../../include/simd_utils.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int SEQ_LEN = 8;
constexpr int EMBED_DIM = 64;
constexpr int NUM_HEADS = 4;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 16
constexpr int FF_DIM = 128;
constexpr float EPS = 1e-5f;

using MatMulFn = void(*)(const float*, const float*, float*, int, int, int);
using RMSNormFn = void(*)(const float*, const float*, float*, int, float);
using ActivationFn = void(*)(float*, int);

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
    for (int i = 1; i < len; ++i) max_val = std::max(max_val, row[i]);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < len; ++i) row[i] *= inv;
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
    for (; i < length; ++i) sum_sq += input[i] * input[i];
    float scale = 1.0f / std::sqrt(sum_sq / length + eps);
    __m256 scale_vec = _mm256_set1_ps(scale);
    for (i = 0; i <= length - 8; i += 8) {
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
    for (int i = 0; i < length; ++i) data[i] = std::max(0.0f, data[i]);
}

void relu_simd(float* data, int length) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_max_ps(zero, v));
    }
    for (; i < length; ++i) data[i] = std::max(0.0f, data[i]);
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
            for (; k < K; ++k) sum += a_ptr[k] * b_ptr[k];
            C[i * N + j] = sum;
        }
    }
}

struct ModelWeights {
    std::vector<float> Wq_T, Wk_T, Wv_T, Wo_T, Wff1_T, Wff2_T;
    std::vector<float> gamma1, gamma2;
};

struct ModelInputs {
    std::vector<float> tokens;
};

double attention_scale() {
    return 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
}

void initialize(ModelWeights& weights, ModelInputs& inputs) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto fill_and_transpose = [&](int rows, int cols, std::vector<float>& storage) {
        std::vector<float> tmp(rows * cols);
        for (float& v : tmp) v = dist(rng);
        storage.resize(cols * rows);
        transpose_matrix(tmp.data(), storage.data(), rows, cols);
    };

    weights.Wq_T.reserve(EMBED_DIM * EMBED_DIM);
    weights.Wk_T.reserve(EMBED_DIM * EMBED_DIM);
    weights.Wv_T.reserve(EMBED_DIM * EMBED_DIM);
    weights.Wo_T.reserve(EMBED_DIM * EMBED_DIM);
    weights.Wff1_T.reserve(FF_DIM * EMBED_DIM);
    weights.Wff2_T.reserve(EMBED_DIM * FF_DIM);

    fill_and_transpose(EMBED_DIM, EMBED_DIM, weights.Wq_T);
    fill_and_transpose(EMBED_DIM, EMBED_DIM, weights.Wk_T);
    fill_and_transpose(EMBED_DIM, EMBED_DIM, weights.Wv_T);
    fill_and_transpose(EMBED_DIM, EMBED_DIM, weights.Wo_T);
    fill_and_transpose(EMBED_DIM, FF_DIM, weights.Wff1_T);
    fill_and_transpose(FF_DIM, EMBED_DIM, weights.Wff2_T);

    weights.gamma1.assign(EMBED_DIM, 1.0f);
    weights.gamma2.assign(EMBED_DIM, 1.0f);

    inputs.tokens.resize(SEQ_LEN * EMBED_DIM);
    for (float& v : inputs.tokens) v = dist(rng);
}

struct StageTimes {
    double rms1 = 0.0;
    double qkv = 0.0;
    double attn_scores = 0.0;
    double attn_context = 0.0;
    double attn_proj = 0.0;
    double rms2 = 0.0;
    double ff1 = 0.0;
    double activation = 0.0;
    double ff2 = 0.0;
    double total = 0.0;
};

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;

void run_block(const ModelWeights& weights,
               const ModelInputs& inputs,
               std::vector<float>& output,
               MatMulFn matmul_fn,
               RMSNormFn rms_fn,
               ActivationFn activation_fn,
               StageTimes* times = nullptr) {
    const int token_dim = SEQ_LEN * EMBED_DIM;
    output.assign(token_dim, 0.0f);

    auto add_duration = [](StageTimes* st, double& field,
                           const Clock::time_point& start,
                           const Clock::time_point& end) {
        if (st) {
            field += std::chrono::duration_cast<Microseconds>(end - start).count();
        }
    };

    Clock::time_point block_start;
    if (times) {
        block_start = Clock::now();
    }

    std::vector<float> norm1(token_dim);
    Clock::time_point t0 = Clock::now();
    rms_fn(inputs.tokens.data(), weights.gamma1.data(), norm1.data(), token_dim, EPS);
    add_duration(times, times->rms1, t0, Clock::now());

    std::vector<float> Q(token_dim), K(token_dim), V(token_dim);
    t0 = Clock::now();
    matmul_fn(norm1.data(), weights.Wq_T.data(), Q.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
    matmul_fn(norm1.data(), weights.Wk_T.data(), K.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
    matmul_fn(norm1.data(), weights.Wv_T.data(), V.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
    add_duration(times, times->qkv, t0, Clock::now());

    std::vector<float> Q_heads(NUM_HEADS * SEQ_LEN * HEAD_DIM);
    std::vector<float> K_heads(NUM_HEADS * SEQ_LEN * HEAD_DIM);
    std::vector<float> V_heads(NUM_HEADS * SEQ_LEN * HEAD_DIM);
    split_heads(Q.data(), Q_heads.data());
    split_heads(K.data(), K_heads.data());
    split_heads(V.data(), V_heads.data());

    std::vector<float> context_heads(NUM_HEADS * SEQ_LEN * HEAD_DIM, 0.0f);
    std::vector<float> K_heads_T(NUM_HEADS * HEAD_DIM * SEQ_LEN);
    std::vector<float> V_heads_T(NUM_HEADS * HEAD_DIM * SEQ_LEN);
    std::vector<float> scores(SEQ_LEN * SEQ_LEN);
    const float scale = static_cast<float>(attention_scale());

    for (int h = 0; h < NUM_HEADS; ++h) {
        const float* q_head = Q_heads.data() + h * SEQ_LEN * HEAD_DIM;
        const float* k_head = K_heads.data() + h * SEQ_LEN * HEAD_DIM;
        const float* v_head = V_heads.data() + h * SEQ_LEN * HEAD_DIM;
        float* k_t = K_heads_T.data() + h * HEAD_DIM * SEQ_LEN;
        float* v_t = V_heads_T.data() + h * HEAD_DIM * SEQ_LEN;
        transpose_matrix(k_head, k_t, SEQ_LEN, HEAD_DIM);
        transpose_matrix(v_head, v_t, SEQ_LEN, HEAD_DIM);

        t0 = Clock::now();
        matmul_fn(q_head, k_t, scores.data(), SEQ_LEN, HEAD_DIM, SEQ_LEN);
        add_duration(times, times->attn_scores, t0, Clock::now());
        for (float& s : scores) s *= scale;
        for (int row = 0; row < SEQ_LEN; ++row) {
            softmax_inplace(scores.data() + row * SEQ_LEN, SEQ_LEN);
        }
        float* ctx = context_heads.data() + h * SEQ_LEN * HEAD_DIM;
        t0 = Clock::now();
        matmul_fn(scores.data(), v_t, ctx, SEQ_LEN, SEQ_LEN, HEAD_DIM);
        add_duration(times, times->attn_context, t0, Clock::now());
    }

    std::vector<float> context(token_dim);
    combine_heads(context_heads.data(), context.data());

    std::vector<float> attn_proj(token_dim);
    t0 = Clock::now();
    matmul_fn(context.data(), weights.Wo_T.data(), attn_proj.data(), SEQ_LEN, EMBED_DIM, EMBED_DIM);
    add_duration(times, times->attn_proj, t0, Clock::now());

    std::vector<float> residual1(token_dim);
    for (int i = 0; i < token_dim; ++i) {
        residual1[i] = inputs.tokens[i] + attn_proj[i];
    }

    std::vector<float> norm2(token_dim);
    t0 = Clock::now();
    rms_fn(residual1.data(), weights.gamma2.data(), norm2.data(), token_dim, EPS);
    add_duration(times, times->rms2, t0, Clock::now());

    std::vector<float> ff1(SEQ_LEN * FF_DIM);
    t0 = Clock::now();
    matmul_fn(norm2.data(), weights.Wff1_T.data(), ff1.data(), SEQ_LEN, EMBED_DIM, FF_DIM);
    add_duration(times, times->ff1, t0, Clock::now());

    t0 = Clock::now();
    activation_fn(ff1.data(), static_cast<int>(ff1.size()));
    add_duration(times, times->activation, t0, Clock::now());

    std::vector<float> ff2(token_dim);
    t0 = Clock::now();
    matmul_fn(ff1.data(), weights.Wff2_T.data(), ff2.data(), SEQ_LEN, FF_DIM, EMBED_DIM);
    add_duration(times, times->ff2, t0, Clock::now());

    for (int i = 0; i < token_dim; ++i) {
        output[i] = residual1[i] + ff2[i];
    }

    if (times) {
        times->total += std::chrono::duration_cast<Microseconds>(Clock::now() - block_start).count();
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return diff;
}

} // namespace

int main() {
    ModelWeights weights;
    ModelInputs inputs;
    initialize(weights, inputs);

    std::vector<float> scalar_output, simd_output;

    StageTimes scalar_stage{}, simd_stage{};
    auto scalar_block_times = [&]() {
        run_block(weights, inputs, scalar_output, matmul_scalar, rmsnorm_scalar, relu_scalar, &scalar_stage);
    };
    auto simd_block_times = [&]() {
        run_block(weights, inputs, simd_output, matmul_simd, rmsnorm_simd, relu_simd, &simd_stage);
    };

    constexpr int stage_iterations = 10;
    for (int i = 0; i < stage_iterations; ++i) {
        scalar_block_times();
    }
    for (int i = 0; i < stage_iterations; ++i) {
        simd_block_times();
    }

    auto normalize = [&](StageTimes& st) {
        st.rms1 /= stage_iterations;
        st.qkv /= stage_iterations;
        st.attn_scores /= stage_iterations;
        st.attn_context /= stage_iterations;
        st.attn_proj /= stage_iterations;
        st.rms2 /= stage_iterations;
        st.ff1 /= stage_iterations;
        st.activation /= stage_iterations;
        st.ff2 /= stage_iterations;
        st.total /= stage_iterations;
    };
    normalize(scalar_stage);
    normalize(simd_stage);

    auto scalar_block = [&]() {
        run_block(weights, inputs, scalar_output, matmul_scalar, rmsnorm_scalar, relu_scalar);
    };
    auto simd_block = [&]() {
        run_block(weights, inputs, simd_output, matmul_simd, rmsnorm_simd, relu_simd);
    };

    scalar_block();
    simd_block();

    float diff = max_abs_diff(scalar_output, simd_output);
    std::cout << "Max |scalar - simd| difference: " << diff << "\n";

    struct ComponentRow {
        std::string name;
        int count;
        double scalar_us;
        double simd_us;
    };

    std::vector<ComponentRow> components = {
        {"rmsnorm", 2, scalar_stage.rms1 + scalar_stage.rms2, simd_stage.rms1 + simd_stage.rms2},
        {"qkv_projections", 1, scalar_stage.qkv, simd_stage.qkv},
        {"attention_scores", 1, scalar_stage.attn_scores, simd_stage.attn_scores},
        {"context_projection", 1, scalar_stage.attn_context, simd_stage.attn_context},
        {"output_projection", 1, scalar_stage.attn_proj, simd_stage.attn_proj},
        {"ffn_expand", 1, scalar_stage.ff1, simd_stage.ff1},
        {"activation", 1, scalar_stage.activation, simd_stage.activation},
        {"ffn_contract", 1, scalar_stage.ff2, simd_stage.ff2}
    };

    double sum_scalar = 0.0;
    double sum_simd = 0.0;
    for (const auto& c : components) {
        sum_scalar += c.scalar_us;
        sum_simd += c.simd_us;
    }
    double others_scalar = std::max(0.0, scalar_stage.total - sum_scalar);
    double others_simd = std::max(0.0, simd_stage.total - sum_simd);
    components.push_back({"others", 1, others_scalar, others_simd});

    double total_scalar = scalar_stage.total;
    double total_simd = simd_stage.total;
    double total_saved = total_scalar - total_simd;

    namespace fs = std::filesystem;
    fs::path out_path = fs::current_path().parent_path().parent_path().parent_path() / "artifacts" / "attention_components.csv";
    fs::create_directories(out_path.parent_path());
    std::ofstream file(out_path);
    if (file) {
        file << "component,count,scalar_total_us,simd_total_us,speedup,time_saved_us,contribution_pct\n";
        for (const auto& c : components) {
            double saved = c.scalar_us - c.simd_us;
            double speedup = c.simd_us > 0.0 ? c.scalar_us / c.simd_us : 0.0;
            double pct = (total_saved > 0.0) ? (saved / total_saved * 100.0) : 0.0;
            file << c.name << ',' << c.count << ',' << c.scalar_us << ',' << c.simd_us << ','
                 << speedup << ',' << saved << ',' << pct << '\n';
        }
        double overall_speedup = total_simd > 0.0 ? total_scalar / total_simd : 0.0;
        file << "overall,1," << total_scalar << ',' << total_simd << ','
             << overall_speedup << ',' << total_saved << ',' << 100.0 << '\n';
    } else {
        std::cerr << "Failed to write attention_components.csv" << std::endl;
    }

    set_benchmark_suite("03_Examples/05_mha_block");
    benchmark_comparison("attention_block", scalar_block, simd_block, 50);

    std::cout << "First token (scalar vs SIMD):\n";
    for (int d = 0; d < EMBED_DIM; ++d) {
        if (d && d % 8 == 0) std::cout << "\n";
        std::cout << scalar_output[d] << " / " << simd_output[d] << "  ";
    }
    std::cout << "\n";

    return 0;
}
