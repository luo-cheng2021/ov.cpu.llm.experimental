// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc.hpp"

#define OV_THREAD OV_THREAD_TBB
#include "openvino/core/parallel.hpp"

// global thread_local profiler
#if 0
#include "profiler.hpp"
    static thread_local ProfilerManager myprofiler("fc.json");
#define PROFILE(var, name) auto var = myprofiler.Profile(name)
#else
#define PROFILE(var, name)
#endif

#include "intrinsic_helpers.hpp"

namespace llm {
namespace experimental {

FC::FC(const ov::OutputVector &args, Config cfg) : Op({args}), m_config(cfg) { constructor_validate_and_infer_types(); }

std::shared_ptr<ov::Node> FC::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<FC>(new_args, m_config);
}

bool FC::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("quant_type", m_config.quant_type);
    visitor.on_attribute("K", m_config.K);
    visitor.on_attribute("N", m_config.N);
    visitor.on_attribute("evaluate_qweight", m_config.evaluate_qweight);

    m_qtype = quantType::Unkown;
    if (m_config.quant_type == "F16") {
        m_qtype = quantType::F16;
    }
    if (m_config.quant_type == "Q8_0") {
        m_qtype = quantType::Q8_0;
    }
    if (m_config.quant_type == "Q8_C") {
        m_qtype = quantType::Q8_C;
    }
    if (m_config.quant_type == "Q4_0") {
        m_qtype = quantType::Q4_0;
    }
    if (m_config.quant_type == "Q4_C") {
        m_qtype = quantType::Q4_C;
    }
    return true;
}

struct f16_block {
    ov::float16 w[32];
};

struct q8_0_block {
    int8_t w[32 / 4][32 * 4];
    ov::float16 wd[32];
    int8_t &at(int k, int n) { return w[k >> 2][(n * 4) + (k & 3)]; }
};

struct q8_c_block {
    int8_t w[32 / 4][32 * 4];
    int8_t &at(int k, int n) { return w[k >> 2][(n * 4) + (k & 3)]; }
};

struct q4_0_block {
    // whole 4-bit 32x32 block distributed as following
    //    8x(4x32) each 2 adjacent (4x32) is combined into low/high part of a 8bit 4x32
    int8_t w[4][32 * 4];
    ov::float16 wd[32];

    void set(int k, int n, int8_t v) {
        // assert(v >= -8 && v < 7)
        auto &value = w[k >> 3][(n * 4) + (k & 3)];
        bool is_high_4bit = ((k / 4) & 1);
        if (is_high_4bit) {
            value = (value & 0x0F) | (v << 4);
        } else {
            value = (value & 0xF0) | (v & 0x0F);
        }
    }
};

struct q4_c_block {
    int8_t w[4][32 * 4];
    void set(int k, int n, int8_t v) {
        // assert(v >= -8 && v < 7)
        auto &value = w[k >> 3][(n * 4) + (k & 3)];
        bool is_high_4bit = ((k / 4) & 1);
        if (is_high_4bit) {
            value = (value & 0x0F) | (v << 4);
        } else {
            value = (value & 0xF0) | (v & 0x0F);
        }
    }
};

void FC::validate_and_infer_types() {
    auto K = m_config.K;
    auto N = m_config.N;

    if (m_config.evaluate_qweight) {
        // in this mode, output quantized weight tensors instead of matmul results

        // by-default no output
        set_output_size(0);

        // every 4 rows in (group_k x group_n) sub-block is interleaved to become (group_k/4 by group_n*4)
        // totally group_n scales for sub-block is "embedded" in quantized weights in f16 format, so
        // it's 2*group_n more i8 elements
        switch (m_qtype) {
        case quantType::Q8_C: {
            constexpr int group_k = 32;
            constexpr int group_n = 32;
            auto Ngroups = (N + group_n - 1) / group_n;
            auto Kgroups = (K + group_k - 1) / group_k;
            // per-OC quantized
            set_output_type(0, ov::element::i8, ov::PartialShape{Ngroups, Kgroups, sizeof(q8_c_block)});
            // per-OC scale
            set_output_type(1, ov::element::f32, ov::PartialShape{Ngroups * group_n});
        } break;
        case quantType::Q4_C: {
            constexpr int group_k = 32;
            constexpr int group_n = 32;
            auto Ngroups = (N + group_n - 1) / group_n;
            auto Kgroups = (K + group_k - 1) / group_k;
            // per-OC quantized
            set_output_type(0, ov::element::i8, ov::PartialShape{Ngroups, Kgroups, sizeof(q4_c_block)});
            // per-OC scale : in float16 format, use i32 to WA cpu plugin's internal reordering of f16
            set_output_type(1, ov::element::i32, ov::PartialShape{Ngroups * group_n / 2});
        } break;
        case quantType::Q8_0: {
            constexpr int group_k = 32;
            constexpr int group_n = 32;
            auto Kgroups = (K + group_k - 1) / group_k;
            auto Ngroups = (N + group_n - 1) / group_n;
            set_output_type(0, ov::element::i8, ov::PartialShape{Ngroups, Kgroups, sizeof(q8_0_block)});
        } break;
        case quantType::Q4_0: {
            constexpr int group_k = 32;
            constexpr int group_n = 32;
            auto Kgroups = (K + group_k - 1) / group_k;
            auto Ngroups = (N + group_n - 1) / group_n;
            set_output_type(0, ov::element::i8, ov::PartialShape{Ngroups, Kgroups, sizeof(q4_0_block)});
        } break;
        case quantType::F16: {
            constexpr int group_n = 32;
            auto Ngroups = (N + group_n - 1) / group_n;
            // WA, use i32 containes 2 float16
            set_output_type(0, ov::element::i32, ov::PartialShape{Ngroups, K, sizeof(f16_block) / sizeof(int32_t)});
        } break;
        default:
            break;
        }
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

void FC::dynamic_quantize_x(d_tensor::PlainTensor<float> &input, size_t Kgroups, size_t group_k) {
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    PROFILE(prof, "quantize");

    // dynamically quantize whole inputs
    x_quantized.resize({B, M, Kgroups * group_k});
    x_scales.resize({B, M, Kgroups});

    // kernel is light-weight to parallel, unless we have multiple rows
    ov::parallel_for2d(B, M, [&](size_t b, size_t m) {
        // a single row quantized in K groups
        float *q8_xd = &x_scales({b, m, 0});
        int8_t *q8_xq = &x_quantized({b, m, 0});
        float *raw_x = &input({b, m, 0});
        for (size_t kb = 0; kb < Kgroups; kb++, raw_x += group_k, q8_xq += group_k) {
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

bool FC::quant_Q8_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized) const {
    // raw weight input is NxK (transpose_b is true)
    // strides is decreasing, so inner-most dimension is at higher ranks
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    size_t group_k = 32;
    size_t group_n = 32;
    size_t Kgroups = (K + group_k - 1) / group_k;
    size_t Ngroups = (N + group_n - 1) / group_n;

    wei_quantized.assert_dims({Ngroups, Kgroups, sizeof(q8_0_block)});

    // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
    // and each column of 32x32 sub-block share a quantization scales
    ov::parallel_for(Ngroups, [&](size_t nb) {
        auto n0 = nb * group_n;
        q8_0_block *wq8 = reinterpret_cast<q8_0_block *>(&wei_quantized({nb, 0, 0}));
        for (size_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq8++) {
            // w_q composed of
            for (size_t ni = 0; ni < group_n; ni++) {
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
                wq8->wd[ni] = d;

                for (int ki = 0; ki < group_k; ki += 4) {
                    for (int i = 0; i < 4; i++) {
                        auto src_k = k0 + ki + i;
                        int8_t w_quantized = 0;
                        if (src_n < N && src_k < K) {
                            w_quantized = std::roundf(wei({src_n, src_k}) * id);
                        }
                        wq8->w[ki / 4][ni * 4 + i] = w_quantized;
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

bool FC::evaluate_Q8_0(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> output) const {
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    size_t group_k = 32;
    size_t group_n = 32;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto y_stride = output.stride(1);

    VNNI_INT8_Sequence vnni_i8;
    assert(K == m_config.K);

    const_cast<FC *>(this)->dynamic_quantize_x(input, Kgroups, group_k);

    PROFILE(prof, "vnni");
    ov::parallel_for(Ngroups, [&](size_t nb) {
        auto n0 = nb * group_n;
        float *py = &output({0, 0, n0});
        // B & M dimensions are collapsed as 1 dimension
        for (size_t b = 0; b < B; b++) {
            for (size_t m = 0; m < M; m++, py += y_stride) {
                const float *q8_xd = &x_scales({b, m, 0});
                const int8_t *q8_xq = &x_quantized({b, m, 0});

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                const q8_0_block *wq8 = reinterpret_cast<q8_0_block *>(&wei_quantized({nb, 0, 0}));
                for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq8++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    auto *q8_weight = wq8->w[0];
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
                    auto dx = _mm256_broadcast_ss(q8_xd);

                    auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wq8->wd));
                    auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq8->wd[8 * 1])));
                    auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq8->wd[8 * 2])));
                    auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq8->wd[8 * 3])));

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

    return true;
}

//====================================================================================================================
bool FC::quant_Q4_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized) const {
    // raw weight input is NxK (transpose_b is true)
    // strides is decreasing, so inner-most dimension is at higher ranks
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    size_t group_k = 32;
    size_t group_n = 32;
    size_t Kgroups = (K + group_k - 1) / group_k;
    size_t Ngroups = (N + group_n - 1) / group_n;

    wei_quantized.assert_dims({Ngroups, Kgroups, sizeof(q4_0_block)});

    // each 32x32 sub-block is further interleaved every 4-rows into (32/4)x(32*4)
    // and each column of 32x32 sub-block share a quantization scales
    ov::parallel_for(Ngroups, [&](size_t nb) {
        auto n0 = nb * group_n;
        q4_0_block *wq4 = reinterpret_cast<q4_0_block *>(&wei_quantized({nb, 0, 0}));
        for (size_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq4++) {
            // w_q composed of
            for (size_t ni = 0; ni < group_n; ni++) {
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
                float d = amax / 7;
                float id = (d != 0) ? (1.0f / d) : 0;

                // save dequantize scale for runtime to use
                // *16 because de-compress 4bits into 8bits is easier to put
                // 4bits at high-end of 8bits data, which imply multiplication of 16
                // ,thus dequantize need to cancel this
                wq4->wd[ni] = d / 16;

                for (int ki = 0; ki < group_k; ki++) {
                    auto src_k = k0 + ki;
                    int8_t w_quantized = 0;
                    if (src_n < N && src_k < K) {
                        w_quantized = std::roundf(wei({src_n, src_k}) * id);
                    }
                    wq4->set(ki, ni, w_quantized);
                }
            }
        }
    });
    return true;
}

bool FC::evaluate_Q4_0(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> output) const {
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto y_stride = output.stride(1);

    VNNI_INT8_Sequence vnni_i8;
    assert(K == m_config.K);

    const_cast<FC *>(this)->dynamic_quantize_x(input, Kgroups, group_k);

    PROFILE(prof, "vnni");
    ov::parallel_for(Ngroups, [&](size_t nb) {
        auto n0 = nb * group_n;
        float *py = &output({0, 0, n0});
        // B & M dimensions are collapsed as 1 dimension
        for (size_t b = 0; b < B; b++) {
            for (size_t m = 0; m < M; m++, py += y_stride) {
                const float *q8_xd = &x_scales({b, m, 0});
                const int8_t *q8_xq = &x_quantized({b, m, 0});

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                const q4_0_block *wq4 = reinterpret_cast<q4_0_block *>(&wei_quantized({nb, 0, 0}));
                for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq4++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    const __m256i high4_mask = _mm256_set1_epi32(0xF0F0F0F0);
                    auto *q4_weight = wq4->w[0];
                    for (int ki = 0; ki < group_k; ki += 8, q4_weight += 32 * 4, q8_xq += 8) {
                        // low 4bit 4x32 blocks
                        __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq));
                        __m256i y0 = _mm256_loadu_si256((const __m256i *)(q4_weight));
                        __m256i y1 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32));
                        __m256i y2 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 2));
                        __m256i y3 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 3));

                        acci0 = vnni_i8(acci0, x0, _mm256_and_si256(_mm256_slli_epi16(y0, 4), high4_mask));
                        acci1 = vnni_i8(acci1, x0, _mm256_and_si256(_mm256_slli_epi16(y1, 4), high4_mask));
                        acci2 = vnni_i8(acci2, x0, _mm256_and_si256(_mm256_slli_epi16(y2, 4), high4_mask));
                        acci3 = vnni_i8(acci3, x0, _mm256_and_si256(_mm256_slli_epi16(y3, 4), high4_mask));

                        // high 4bit
                        __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq + 4));
                        acci0 = vnni_i8(acci0, x1, _mm256_and_si256(y0, high4_mask));
                        acci1 = vnni_i8(acci1, x1, _mm256_and_si256(y1, high4_mask));
                        acci2 = vnni_i8(acci2, x1, _mm256_and_si256(y2, high4_mask));
                        acci3 = vnni_i8(acci3, x1, _mm256_and_si256(y3, high4_mask));
                    }
                    // load de-quantize coeff and combine with input's dequantize coeff
                    // const u_int16_t *f16_scale = reinterpret_cast<const u_int16_t *>(&wei_scales({kb, n0}));
                    auto dx = _mm256_broadcast_ss(q8_xd);

                    auto d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wq4->wd));
                    auto d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 1])));
                    auto d2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 2])));
                    auto d3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(&wq4->wd[8 * 3])));

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

    return true;
}

//==========================================================================================
bool FC::quant_Q8_C(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized,
                    d_tensor::PlainTensor<float> wei_scales) const {
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    constexpr size_t group_k = 32;
    constexpr size_t group_n = 32;
    auto Ngroups = (N + group_n - 1) / group_n;
    auto Kgroups = (K + group_k - 1) / group_k;

    ov::parallel_for(Ngroups * group_n, [&](size_t n) {
        if (n >= N) {
            wei_scales({n}) = 0;
            return;
        }

        float amax = 0.0f;
        for (size_t k = 0; k < K; k++) {
            auto a = std::abs(wei({n, k}));
            if (amax < a)
                amax = a;
        }
        // x = (d * q)
        // q = x / d = x * id
        float d = amax / 127;
        float id = (d != 0) ? (1.0f / d) : 0;

        wei_scales({n}) = d;

        // quantize column n
        auto nb = n / group_n;
        auto noff = (n - nb * group_n);
        q8_c_block *wq8 = reinterpret_cast<q8_c_block *>(&wei_quantized({nb, 0, 0}));
        for (size_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq8++) {
            for (size_t ki = 0; ki < group_k; ki++) {
                auto src_k = k0 + ki;
                if (src_k < K) {
                    wq8->at(ki, noff) = std::roundf(wei({n, src_k}) * id);
                } else {
                    wq8->at(ki, noff) = 0;
                }
            }
        }
    });
    return true;
}

bool FC::evaluate_Q8_C(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> wei_scales, d_tensor::PlainTensor<float> output) const {
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto y_stride = output.stride(1);

    VNNI_INT8_Sequence vnni_i8;
    assert(K == m_config.K);

    const_cast<FC *>(this)->dynamic_quantize_x(input, Kgroups, group_k);

    PROFILE(prof, "vnni");
    ov::parallel_for(Ngroups, [&](size_t nb) {
        size_t n0 = nb * group_n;
        float *py = &output({0, 0, n0});
        float *pwei_scales = &wei_scales({n0});
        // B & M dimensions are collapsed as 1 dimension
        for (size_t b = 0; b < B; b++) {
            for (size_t m = 0; m < M; m++, py += y_stride) {
                const float *q8_xd = &x_scales({b, m, 0});
                const int8_t *q8_xq = &x_quantized({b, m, 0});

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                const q8_c_block *wq8 = reinterpret_cast<q8_c_block *>(&wei_quantized({nb, 0, 0}));
                for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq8++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    auto *q8_weight = wq8->w[0];
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
                    auto dx = _mm256_broadcast_ss(q8_xd);
                    // dequantize per-group k
                    acc0 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci0), acc0);
                    acc1 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci1), acc1);
                    acc2 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci2), acc2);
                    acc3 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci3), acc3);
                }

                // dequant per-n (OC)
                auto d0 = _mm256_loadu_ps(pwei_scales);
                auto d1 = _mm256_loadu_ps(pwei_scales + 8);
                auto d2 = _mm256_loadu_ps(pwei_scales + 8 * 2);
                auto d3 = _mm256_loadu_ps(pwei_scales + 8 * 3);

                acc0 = _mm256_mul_ps(d0, acc0);
                acc1 = _mm256_mul_ps(d1, acc1);
                acc2 = _mm256_mul_ps(d2, acc2);
                acc3 = _mm256_mul_ps(d3, acc3);

                // output 32 results
                _mm256_storeu_ps(py + 8 * 0, acc0);
                _mm256_storeu_ps(py + 8 * 1, acc1);
                _mm256_storeu_ps(py + 8 * 2, acc2);
                _mm256_storeu_ps(py + 8 * 3, acc3);
            }
        }
    });

    return true;
}

bool FC::quant_Q4_C(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized,
                    d_tensor::PlainTensor<int32_t> wei_scales_i32) const {
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    constexpr size_t group_k = 32;
    constexpr size_t group_n = 32;
    auto Ngroups = (N + group_n - 1) / group_n;
    auto Kgroups = (K + group_k - 1) / group_k;

    d_tensor::PlainTensor<ov::float16> wei_scales;
    wei_scales.resize({Ngroups * group_n}, reinterpret_cast<ov::float16 *>(wei_scales_i32.data()));

    ov::parallel_for(Ngroups * group_n, [&](size_t n) {
        if (n >= N) {
            wei_scales({n}) = 0;
            return;
        }

        float amax = 0.0f;
        for (size_t k = 0; k < K; k++) {
            auto a = std::abs(wei({n, k}));
            if (amax < a)
                amax = a;
        }
        // x = (d * q)
        // q = x / d = x * id
        float d = amax / 7;
        float id = (d != 0) ? (1.0f / d) : 0;

        wei_scales({n}) = d / 16;

        // quantize column n
        auto nb = n / group_n;
        auto noff = (n - nb * group_n);
        q4_c_block *wq4 = reinterpret_cast<q4_c_block *>(&wei_quantized({nb, 0, 0}));
        for (size_t kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, wq4++) {
            for (size_t ki = 0; ki < group_k; ki++) {
                auto src_k = k0 + ki;
                int8_t w_quantized = 0;
                if (src_k < K) {
                    w_quantized = std::roundf(wei({n, src_k}) * id);
                }
                wq4->set(ki, noff, w_quantized);
            }
        }
    });
    return true;
}

bool FC::evaluate_Q4_C(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<int32_t> wei_scales_i32, d_tensor::PlainTensor<float> output) const {
    auto Ngroups = wei_quantized.size(0);
    auto Kgroups = wei_quantized.size(1);
    int group_k = 32;
    int group_n = 32;
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    auto y_stride = output.stride(1);

    d_tensor::PlainTensor<ov::float16> wei_scales;
    wei_scales.resize({Ngroups * group_n}, reinterpret_cast<ov::float16 *>(wei_scales_i32.data()));

    VNNI_INT8_Sequence vnni_i8;
    assert(K == m_config.K);

    const_cast<FC *>(this)->dynamic_quantize_x(input, Kgroups, group_k);

    PROFILE(prof, "vnni");
    ov::parallel_for(Ngroups, [&](size_t nb) {
        size_t n0 = nb * group_n;
        float *py = &output({0, 0, n0});
        ov::float16 *pwei_scales = &wei_scales({n0});
        // B & M dimensions are collapsed as 1 dimension
        for (size_t b = 0; b < B; b++) {
            for (size_t m = 0; m < M; m++, py += y_stride) {
                const float *q8_xd = &x_scales({b, m, 0});
                const int8_t *q8_xq = &x_quantized({b, m, 0});

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                const q4_c_block *wq4 = reinterpret_cast<q4_c_block *>(&wei_quantized({nb, 0, 0}));
                for (int kb = 0, k0 = 0; kb < Kgroups; kb++, k0 += group_k, q8_xd++, wq4++) {
                    // K group is smallest quantization unit which shares single scale
                    auto acci0 = _mm256_setzero_si256();
                    auto acci1 = _mm256_setzero_si256();
                    auto acci2 = _mm256_setzero_si256();
                    auto acci3 = _mm256_setzero_si256();
                    const __m256i high4_mask = _mm256_set1_epi32(0xF0F0F0F0);
                    auto *q4_weight = wq4->w[0];
                    for (int ki = 0; ki < group_k; ki += 8, q4_weight += 32 * 4, q8_xq += 8) {
                        // 4x32 vnni kernel
                        __m256i x0 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq));
                        __m256i y0 = _mm256_loadu_si256((const __m256i *)(q4_weight));
                        __m256i y1 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32));
                        __m256i y2 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 2));
                        __m256i y3 = _mm256_loadu_si256((const __m256i *)(q4_weight + 32 * 3));

                        acci0 = vnni_i8(acci0, x0, _mm256_and_si256(_mm256_slli_epi16(y0, 4), high4_mask));
                        acci1 = vnni_i8(acci1, x0, _mm256_and_si256(_mm256_slli_epi16(y1, 4), high4_mask));
                        acci2 = vnni_i8(acci2, x0, _mm256_and_si256(_mm256_slli_epi16(y2, 4), high4_mask));
                        acci3 = vnni_i8(acci3, x0, _mm256_and_si256(_mm256_slli_epi16(y3, 4), high4_mask));
                        // high 4bit
                        __m256i x1 = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(q8_xq + 4));
                        acci0 = vnni_i8(acci0, x1, _mm256_and_si256(y0, high4_mask));
                        acci1 = vnni_i8(acci1, x1, _mm256_and_si256(y1, high4_mask));
                        acci2 = vnni_i8(acci2, x1, _mm256_and_si256(y2, high4_mask));
                        acci3 = vnni_i8(acci3, x1, _mm256_and_si256(y3, high4_mask));
                    }
                    auto dx = _mm256_broadcast_ss(q8_xd);
                    // dequantize per-group k
                    acc0 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci0), acc0);
                    acc1 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci1), acc1);
                    acc2 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci2), acc2);
                    acc3 = _mm256_fmadd_ps(dx, _mm256_cvtepi32_ps(acci3), acc3);
                }

                // dequant per-n (OC)
                auto d0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(pwei_scales)));
                auto d1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(pwei_scales + 8)));
                auto d2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(pwei_scales + 8 * 2)));
                auto d3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(pwei_scales + 8 * 3)));

                acc0 = _mm256_mul_ps(d0, acc0);
                acc1 = _mm256_mul_ps(d1, acc1);
                acc2 = _mm256_mul_ps(d2, acc2);
                acc3 = _mm256_mul_ps(d3, acc3);

                // output 32 results
                _mm256_storeu_ps(py + 8 * 0, acc0);
                _mm256_storeu_ps(py + 8 * 1, acc1);
                _mm256_storeu_ps(py + 8 * 2, acc2);
                _mm256_storeu_ps(py + 8 * 3, acc3);
            }
        }
    });

    return true;
}

// F16 weights are converted to F32 at runtime before FMA is applied
bool FC::quant_F16_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int32_t> wei_f16_i32) const {
    size_t N = wei.size(0);
    size_t K = wei.size(1);
    constexpr int group_n = 32;
    auto Ngroups = (N + group_n - 1) / group_n;

    d_tensor::PlainTensor<f16_block> wei_f16;
    wei_f16.resize({Ngroups, K}, reinterpret_cast<f16_block *>(wei_f16_i32.data()));

    ov::parallel_for(Ngroups, [&](size_t nb) {
        size_t n0 = nb * group_n;
        for (size_t k = 0; k < K; k++) {
            f16_block &blk = wei_f16({nb, k});
            for (size_t ni = 0; ni < 32; ni++) {
                blk.w[ni] = wei({n0 + ni, k});
            }
        }
    });
    return true;
}

bool FC::evaluate_F16_0(d_tensor::PlainTensor<float> input, d_tensor::PlainTensor<int32_t> wei_f16_i32,
                        d_tensor::PlainTensor<float> output) const {
    auto B = input.size(0);
    auto M = input.size(1);
    auto K = input.size(2);
    constexpr int group_n = 32;
    auto Ngroups = wei_f16_i32.size(0);
    auto y_stride = output.stride(1);

    d_tensor::PlainTensor<f16_block> wei_f16;
    wei_f16.resize({Ngroups, K}, reinterpret_cast<f16_block *>(wei_f16_i32.data()));

    ov::parallel_for(Ngroups, [&](size_t nb) {
        size_t n0 = nb * group_n;
        float *py = &output({0, 0, n0});
        // B & M dimensions are collapsed as 1 dimension
        for (size_t b = 0; b < B; b++) {
            for (size_t m = 0; m < M; m++, py += y_stride) {
                float *px = &input({b, m, 0});
                f16_block *wf16 = &wei_f16({nb, 0});
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                for (size_t k = 0; k < K; k++, wf16++) {
                    auto rx = _mm256_set1_ps(px[k]);
                    auto d0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(wf16->w)));
                    auto d1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(wf16->w + 8)));
                    auto d2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(wf16->w + 8 * 2)));
                    auto d3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i *>(wf16->w + 8 * 3)));
                    acc0 = _mm256_fmadd_ps(rx, d0, acc0);
                    acc1 = _mm256_fmadd_ps(rx, d1, acc1);
                    acc2 = _mm256_fmadd_ps(rx, d2, acc2);
                    acc3 = _mm256_fmadd_ps(rx, d3, acc3);
                }
                _mm256_storeu_ps(py + 8 * 0, acc0);
                _mm256_storeu_ps(py + 8 * 1, acc1);
                _mm256_storeu_ps(py + 8 * 2, acc2);
                _mm256_storeu_ps(py + 8 * 3, acc3);
            }
        }
    });
    return true;
}

bool FC::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const {
    switch (m_qtype) {
    case quantType::F16:
        if (m_config.evaluate_qweight) {
            return quant_F16_0(&inputs[0], &outputs[0]);
        }
        return evaluate_F16_0(&inputs[0], &inputs[1], &outputs[0]);
    case quantType::Q8_0:
        if (m_config.evaluate_qweight) {
            return quant_Q8_0(&inputs[0], &outputs[0]);
        }
        return evaluate_Q8_0(&inputs[0], &inputs[1], &outputs[0]);
    case quantType::Q4_0:
        if (m_config.evaluate_qweight) {
            return quant_Q4_0(&inputs[0], &outputs[0]);
        }
        return evaluate_Q4_0(&inputs[0], &inputs[1], &outputs[0]);
    case quantType::Q8_C:
        if (m_config.evaluate_qweight) {
            return quant_Q8_C(&inputs[0], &outputs[0], &outputs[1]);
        }
        return evaluate_Q8_C(&inputs[0], &inputs[1], &inputs[2], &outputs[0]);
    case quantType::Q4_C:
        if (m_config.evaluate_qweight) {
            return quant_Q4_C(&inputs[0], &outputs[0], &outputs[1]);
        }
        return evaluate_Q4_C(&inputs[0], &inputs[1], &inputs[2], &outputs[0]);
    default:
        std::cout << "Unknown quant Type !" << std::endl;
        assert(false);
        break;
    }
    return false;
}

} // namespace experimental
} // namespace llm
