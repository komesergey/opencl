#include <iostream>
#include <random>
#include <algorithm>
#include <xmmintrin.h>
#include <immintrin.h>
#include <chrono>

float dot_normal(const float *vec1, const float *vec2, unsigned n) {
    float sum = 0;
    for(unsigned i = 0; i < n; ++i)
        sum += vec1[i] * vec2[i];
    return sum;
}

float dot_sse(const float *vec1, const float *vec2, unsigned n) {
    __m128 u = {0};
    for (unsigned i = 0; i < n; i += 4) {
#ifdef __GNUC__
        __m128 w = _mm_load_ps(&vec1[i]);
        __m128 x = _mm_load_ps(&vec2[i]);
#else
        __m128 w = _mm_loadu_ps(&vec1[i]);
        __m128 x = _mm_loadu_ps(&vec2[i]);
#endif
        x = _mm_mul_ps(w, x);
        u = _mm_add_ps(u, x);
    }

#ifdef __GNUC__
    __attribute__((aligned(16))) float t[4] = {0};
#else
    float t[4] = {0};
#endif
#ifdef __GNUC__
    _mm_store_ps(t, u);
#else
    _mm_storeu_ps(t, u);
#endif
    return t[0] + t[1] + t[2] + t[3];
}

float dot_avx(const float *vec1, const float *vec2, unsigned n) {
    __m256 u = {0};
    for(unsigned i = 0; i < n; i += 8) {
#ifdef __GNUC__
        __m256 w = _mm256_load_ps(&vec1[i]);
        __m256 x = _mm256_load_ps(&vec2[i]);
#else
        __m256 w = _mm256_loadu_ps(&vec1[i]);
        __m256 x = _mm256_loadu_ps(&vec2[i]);
#endif
        x = _mm256_mul_ps(w, x);
        u = _mm256_add_ps(u, x);
    }
#ifdef __GNUC__
    __attribute__((aligned(32))) float t[8] = {0};
#else
    float t[8] = {0};
#endif

#ifdef __GNUC__
    _mm256_store_ps(t, u);
#else
    _mm256_storeu_ps(t, u);
#endif
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

float dot_avx_2(const float *vec1, const float *vec2, unsigned n) {
    __m256 u1 = {0};
    __m256 u2 = {0};
    for(unsigned i = 0; i < n; i += 16) {
#ifdef __GNUC__
        __m256 w1 = _mm256_load_ps(&vec1[i]);
        __m256 w2 = _mm256_load_ps(&vec1[i + 8]);
        __m256 x1 = _mm256_load_ps(&vec2[i]);
        __m256 x2 = _mm256_load_ps(&vec2[i + 8]);
#else
        __m256 w1 = _mm256_loadu_ps(&vec1[i]);
        __m256 w2 = _mm256_loadu_ps(&vec1[i + 8]);
        __m256 x1 = _mm256_loadu_ps(&vec2[i]);
        __m256 x2 = _mm256_loadu_ps(&vec2[i + 8]);
#endif
        x1 = _mm256_mul_ps(w1, x1);
        x2 = _mm256_mul_ps(w2, x2);
        u1 = _mm256_add_ps(u1, x1);
        u2 = _mm256_add_ps(u2, x2);
    }
    u1 = _mm256_add_ps(u1, u2);
#ifdef __GNUC__
    __attribute__((aligned(32))) float t[8] = {0};
#else
    float t[8] = {0};
#endif

#ifdef __GNUC__
    _mm256_store_ps(t, u1);
#else
    _mm256_storeu_ps(t, u1);
#endif
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

template<class T> long long calc_for_a_moment(T t) {
    auto start = std::chrono::high_resolution_clock::now();

    t();

    auto done = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(done - start).count();
}

/*
    Len:          8 dot_sse:        300 dot_avx:        400 dot_avx_2:        400 dot_normal:        400100
    Len:         16 dot_sse:        200 dot_avx:        200 dot_avx_2:        300 dot_normal:        200100
    Len:         32 dot_sse:        200 dot_avx:        200 dot_avx_2:        300 dot_normal:        200100
    Len:         64 dot_sse:        300 dot_avx:        300 dot_avx_2:        300 dot_normal:        400100
    Len:        128 dot_sse:        300 dot_avx:        300 dot_avx_2:        300 dot_normal:        500100
    Len:        256 dot_sse:        400 dot_avx:        400 dot_avx_2:        400 dot_normal:        800100
    Len:        512 dot_sse:        800 dot_avx:        500 dot_avx_2:      14500 dot_normal:       1700100
    Len:       1024 dot_sse:       1500 dot_avx:        900 dot_avx_2:        900 dot_normal:       3100100
    Len:       2048 dot_sse:       2900 dot_avx:       1700 dot_avx_2:       1600 dot_normal:       6300100
    Len:       4096 dot_sse:       6100 dot_avx:       3300 dot_avx_2:       2900 dot_normal:      12800100
    Len:       8192 dot_sse:      10900 dot_avx:       6000 dot_avx_2:       5400 dot_normal:      24100100
    Len:      16384 dot_sse:      33100 dot_avx:      18700 dot_avx_2:      15400 dot_normal:      63000100
    Len:      32768 dot_sse:      70300 dot_avx:      38200 dot_avx_2:      77200 dot_normal:     130900100
    Len:      65536 dot_sse:      88600 dot_avx:      46400 dot_avx_2:      43600 dot_normal:     186900100
    Len:     131072 dot_sse:     382500 dot_avx:     256700 dot_avx_2:     131600 dot_normal:     742800100
    Len:     262144 dot_sse:     354900 dot_avx:     175000 dot_avx_2:     178100 dot_normal:     806500100
    Len:     524288 dot_sse:     693900 dot_avx:     355600 dot_avx_2:     310000 dot_normal:    1432400100
    Len:    1048576 dot_sse:    2409300 dot_avx:     854000 dot_avx_2:     686600 dot_normal:    3255400100
 */

int main() {
    const unsigned len_begin = 8;
    const unsigned len_end   = 1024 * 1024;
    const unsigned len_fact  = 2;
    const unsigned run_ms    = 250;

    std::mt19937 rng;
    std::uniform_real_distribution<> dst(-1, 1);

    for(unsigned len = len_begin; len <= len_end; len *= len_fact) {
#ifdef __GNUC__
        auto *p1 = new __attribute__((aligned(32))) float[len + 8];
        auto *p2 = new __attribute__((aligned(32))) float[len + 8];
#else
        auto *p1 = new float[len + 8];
        auto *p2 = new float[len + 8];
#endif
        float *vec1 = p1;
        float *vec2 = p2;
        while(reinterpret_cast<long>(vec1) % 32) ++vec1;
        while(reinterpret_cast<long>(vec2) % 32) ++vec2;
        std::generate(vec1, vec1 + len, [&rng, &dst](){ return dst(rng); });
        std::generate(vec2, vec2 + len, [&rng, &dst](){ return dst(rng); });

        std::cout << printf("Len: %10d dot_sse: %10lld dot_avx: %10lld dot_avx_2: %10lld dot_normal: %10lld",
                            len,
                            calc_for_a_moment([vec1, vec2, len](){ return dot_sse   (vec1, vec2, len); }),
                            calc_for_a_moment([vec1, vec2, len](){ return dot_avx   (vec1, vec2, len); }),
                            calc_for_a_moment([vec1, vec2, len](){ return dot_avx_2 (vec1, vec2, len); }),
                            calc_for_a_moment([vec1, vec2, len](){ return dot_normal(vec1, vec2, len); })
        ) << std::endl;

        delete[] p1;
        delete[] p2;
    }
}
