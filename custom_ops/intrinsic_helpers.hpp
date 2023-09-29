#ifdef _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#include <x86intrin.h>
#endif

#include <cmath>

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

static inline void get_min_max(float *x, int len, float &vmin, float &vmax) {
    int i = 1;
    vmin = vmax = x[0];
    for (; i < len; i++) {
        auto a = x[i];
        if (vmax < a)
            vmax = a;
        if (vmin > a)
            vmin = a;
    }
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
struct VNNI_Sequence {
    __m256i operator()(__m256i acc, const __m256i x_s8, const __m256i y_u8) {
#if __AVXVNNI__
        return _mm256_dpbusd_epi32(acc, y_u8, x_s8);
#elif __AVX2__
        const __m256i ones = _mm256_set1_epi16(1);
        // u8 x s8
        const __m256i dot = _mm256_maddubs_epi16(y_u8, x_s8);
        return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, ones));
#else
#error "at least AVX2 is required!"
#endif
    }
};

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
