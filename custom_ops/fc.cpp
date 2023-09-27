// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc.hpp"

#define OV_THREAD OV_THREAD_TBB
#include "openvino/core/parallel.hpp"

#include "profiler.hpp"

// global thread_local profiler
static thread_local ProfilerManager myprofiler("fc.json");

#ifdef _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#include <x86intrin.h>
#endif

namespace llm {
namespace experimental {

FC::FC(const ov::OutputVector &args, Config cfg) : Op({args}), m_config(cfg) { constructor_validate_and_infer_types(); }

std::shared_ptr<ov::Node> FC::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<FC>(new_args, m_config);
}

bool FC::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("quant_type", m_config.quant_type);
    visitor.on_attribute("llama_quant_type", m_config.llama_quant_type);
    visitor.on_attribute("llama_group_k", m_config.llama_group_k);
    visitor.on_attribute("llama_group_n", m_config.llama_group_n);
    visitor.on_attribute("K", m_config.K);
    visitor.on_attribute("N", m_config.N);
    visitor.on_attribute("bits", m_config.bits);
    visitor.on_attribute("evaluate_qweight", m_config.evaluate_qweight);
    return true;
}

void FC::validate_and_infer_types() {
    if (m_config.evaluate_qweight) {
        // in this mode, output quantized weight tensors instead of matmul results
        // int8 array + scale array
        auto group_k = m_config.llama_group_k;
        auto group_n = m_config.llama_group_n;
        assert(group_k % 4 == 0);
        assert(group_n % 8 == 0);

        auto K = m_config.K;
        auto N = m_config.N;
        auto Kgroups = (K + group_k - 1) / group_k;
        auto Ngroups = (N + group_n - 1) / group_n;

        // every 4 rows in (group_k x group_n) sub-block is interleaved to become (group_k/4 by group_n*4)
        set_output_type(0, ov::element::i8, ov::PartialShape{Ngroups, Kgroups, group_k / 4, group_n * 4});
        // scale is shared along group_k
        set_output_type(1, ov::element::f32, ov::PartialShape{Kgroups, Ngroups * group_n});
        return;
    }

    // [B, M, K]
    auto data_pshape = get_input_partial_shape(0);
    auto ndim = data_pshape.size();
    assert(ndim == 3);

    ov::PartialShape result_pshape{data_pshape[0], data_pshape[1], m_config.N};

    // std::cout << result_pshape << std::endl;
    set_output_type(0, get_input_element_type(0), result_pshape);
}

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float hmax_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_max_ps(res, _mm256_castps256_ps128(x));
    res = _mm_max_ps(res, _mm_movehl_ps(res, res));
    res = _mm_max_ps(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float get_amax(float *x, int len) {
    int i = 0;
    float amax = 0;
#if __AVX2__
    // https://stackoverflow.com/questions/63599391/find-absolute-in-avx
    auto sign_bit = _mm256_set1_ps(-0.0f);
    auto v_max_abs = _mm256_setzero_ps();
    for (; i + 8 <= len; i += 8) {
        auto v = _mm256_loadu_ps(x + i);
        v = _mm256_andnot_ps(sign_bit, v);
        v_max_abs = _mm256_max_ps(v_max_abs, v);
    }
    amax = hmax_float_8(v_max_abs);
#endif
    for (; i < len; i++) {
        auto a = std::abs(x[i]);
        if (amax < a)
            amax = a;
    }
    return amax;
}

// round(x * id)
static inline void quant_row_q8_0(float *x, int8_t *qx, int len, float id) {
    int i = 0;
#if __AVX2__
    auto v_id = _mm256_set1_ps(id);
    for (; i + 8 <= len; i += 8) {
        auto v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, v_id);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);

        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packs_epi16(packed, packed);
        _mm_storeu_si64(qx + i, packed);
    }
#endif
    for (; i < len; i++) {
        qx[i] = std::round(x[i] * id);
    }
}

struct VNNI_INT8_Sequence {
    __m256i operator()(__m256i acc, const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
        return _mm256_dpbssd_epi32(acc, x, y);
#elif __AVXVNNI__
        // Get absolute values of x vectors (x becomes u8 : 0~128)
        const __m256i ax = _mm256_sign_epi8(x, x);
        // Sign the values of the y vectors (negative sign of x is combined with y)
        const __m256i sy = _mm256_sign_epi8(y, x);
        return _mm256_dpbusd_epi32(acc, ax, sy);
#elif __AVX2__
        // Get absolute values of x vectors (x becomes u8 : 0~128)
        const __m256i ax = _mm256_sign_epi8(x, x);
        // Sign the values of the y vectors (negative sign of x is combined with y)
        const __m256i sy = _mm256_sign_epi8(y, x);
        const __m256i ones = _mm256_set1_epi16(1);

        // u8 x s8
        const __m256i dot = _mm256_maddubs_epi16(ax, sy);
        return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, ones));
#else
#error "at least AVX2 is required!"
#endif
    }
};

bool FC::quant_q8_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized,
                    d_tensor::PlainTensor<float> wei_scales) const {
    // raw weight input is NxK (transpose_b is true)
    // strides is decreasing, so inner-most dimension is at higher ranks
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    size_t group_k = m_config.llama_group_k;
    size_t group_n = m_config.llama_group_n;
    assert(group_k % 4 == 0);
    assert(group_n % 8 == 0);
    size_t Kgroups = (K + group_k - 1) / group_k;
    size_t Ngroups = (N + group_n - 1) / group_n;

    wei_quantized.assert_dims({Ngroups, Kgroups, group_k / 4, group_n * 4});
    wei_scales.assert_dims({Kgroups, Ngroups * group_n});

    // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
    // and each column of 32x32 sub-block share a quantization scales
    ov::parallel_for(Ngroups, [&](int nb) {
        auto n0 = nb * group_n;
        for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k) {
            for (int ni = 0; ni < group_n; ni++) {
                // derive quantization scales from group_k :  round(x * qs)
                //  amax = np.abs(weight_np_pad).max(axis=3, keepdims=True)
                //  d = amax / ((1 << 7) - 1)
                //  id = np.where(np.abs(d) == 0, 0, 1 / d)
                //  weight_np_quantized = np.round(weight_np_pad * id).astype(np.int8)

                // symetric quantization
                auto src_n = n0 + ni;
                float amax = get_amax(&wei({src_n, k0}), group_k);
                // x = (d * q)
                // q = x / d = x * id
                float d = amax / 127;
                float id = (d != 0) ? (1.0f / d) : 0;

                // save dequantize scale for runtime to use
                wei_scales({kb, src_n}) = d;

                for (int ki = 0; ki < group_k; ki += 4) {
                    for (int i = 0; i < 4; i++) {
                        auto src_k = k0 + ki + i;
                        if (src_n < N && src_k < K) {
                            wei_quantized({nb, kb, ki / 4, ni * 4 + i}) = std::roundf(wei({src_n, src_k}) * id);
                        } else {
                            wei_quantized({nb, kb, ki / 4, ni * 4 + i}) = 0;
                        }
                    }
                }
            }
        }
    });
    // std::cout << "K,N=" << K << "," << N << std::endl;
    // std::cout << wei_quantized << std::endl;
    // std::cout << wei_scales << std::endl;
    return true;
}

bool FC::evaluate_q8_0(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> wei_scales, d_tensor::PlainTensor<float> output) const {
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int group_k = m_config.llama_group_k;
    int group_n = m_config.llama_group_n;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto BM = B * M;
    auto y_stride = output.stride(1);

    // std::cout << "evaluate_q8_0  B,M,K=" << B << "," << M << ", " << K << " ; Ngroups,Kgroups= " << Ngroups << "," <<
    // Kgroups << " nthr=" << tbb::this_task_arena::max_concurrency() << std::endl;

    VNNI_INT8_Sequence vnni_i8;
    assert(K == m_config.K);

    assert(group_n == 32);

    // dynamically quantize whole inputs
    const_cast<FC *>(this)->x_quantized.resize({B, M, Kgroups * group_k});
    const_cast<FC *>(this)->x_scales.resize({B, M, Kgroups});
    {
        auto prof = myprofiler.Profile("quantize");
        // kernel is light-weight to parallel, unless we have multiple rows
        ov::parallel_for2d(B, M, [&](int b, int m) {
            // a single row quantized in K groups
            float *q8_xd = &x_scales({b, m, 0});
            int8_t *q8_xq = &x_quantized({b, m, 0});
            float *raw_x = &input({b, m, 0});
            for (int kb = 0; kb < Kgroups; kb++, raw_x += group_k, q8_xq += group_k) {
                auto amax = get_amax(raw_x, group_k);
                // x = (d * quantized)
                // quantized = round(x / d) = round(x * id)
                const float d = amax / 127;
                const float id = (d != 0) ? (1.0f / d) : 0;

                q8_xd[kb] = d;
                quant_row_q8_0(raw_x, q8_xq, group_k, id);
            }
        });
    }

    {
        auto prof = myprofiler.Profile("vnni");
        ov::parallel_for(Ngroups, [&](int nb) {
            int n0 = nb * group_n;
            float *py = &output({0, 0, n0});
            // B & M dimensions are collapsed as 1 dimension
            for (int b = 0; b < B; b++) {
                for (int m = 0; m < M; m++, py += y_stride) {
                    const float *q8_xd = &x_scales({b, m, 0});
                    const int8_t *q8_xq = &x_quantized({b, m, 0});
                    const int8_t *q8_weight = &wei_quantized({nb, 0, 0, 0});
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++) {

                        // K group is smallest quantization unit which shares single scale
                        auto acci0 = _mm256_setzero_si256();
                        auto acci1 = _mm256_setzero_si256();
                        auto acci2 = _mm256_setzero_si256();
                        auto acci3 = _mm256_setzero_si256();
                        const __m256i ones = _mm256_set1_epi16(1);
                        for (int ki = 0; ki < group_k; ki += 4, q8_weight += 32 * 4, q8_xq += 4) {
                            // 4x32 vnni kernel
                            __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq));
                            __m256i y0 = _mm256_loadu_si256((const __m256i *)(q8_weight));
                            __m256i y1 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32));
                            __m256i y2 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32 * 2));
                            __m256i y3 = _mm256_loadu_si256((const __m256i *)(q8_weight + 32 * 3));

                            acci0 = vnni_i8(acci0, x0, y0);
                            acci1 = vnni_i8(acci1, x0, y1);
                            acci2 = vnni_i8(acci2, x0, y2);
                            acci3 = vnni_i8(acci3, x0, y3);
                        }
                        // load de-quantize coeff and combine with input's dequantize coeff
                        // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t *>(&wei_scales({kb, n0}));
                        const float *f32_scale = &wei_scales({kb, n0});
                        auto dx = _mm256_broadcast_ss(q8_xd);

                        auto d0 = _mm256_loadu_ps(f32_scale);
                        auto d1 = _mm256_loadu_ps(f32_scale + 8 * 1);
                        auto d2 = _mm256_loadu_ps(f32_scale + 8 * 2);
                        auto d3 = _mm256_loadu_ps(f32_scale + 8 * 3);

                        // auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)f16_scale));
                        // auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(f16_scale + 8 * 1)));
                        // auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(f16_scale + 8 * 2)));
                        // auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(f16_scale + 8 * 3)));

                        d0 = _mm256_mul_ps(d0, dx);
                        d1 = _mm256_mul_ps(d1, dx);
                        d2 = _mm256_mul_ps(d2, dx);
                        d3 = _mm256_mul_ps(d3, dx);

                        // dequantize
                        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(acci0), acc0);
                        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(acci1), acc1);
                        acc2 = _mm256_fmadd_ps(d2, _mm256_cvtepi32_ps(acci2), acc2);
                        acc3 = _mm256_fmadd_ps(d3, _mm256_cvtepi32_ps(acci3), acc3);
                    }

                    // output 32 results
                    _mm256_storeu_ps(py + 8 * 0, acc0);
                    _mm256_storeu_ps(py + 8 * 1, acc1);
                    _mm256_storeu_ps(py + 8 * 2, acc2);
                    _mm256_storeu_ps(py + 8 * 3, acc3);
                }
            }
        });
    }

    return true;
}

bool FC::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const {
    if (m_config.evaluate_qweight) {
        return quant_q8_0(&inputs[0], &outputs[0], &outputs[1]);
    }

    evaluate_q8_0(&inputs[0], &inputs[1], &inputs[2], &outputs[0]);
    return true;
}

} // namespace experimental
} // namespace llm
