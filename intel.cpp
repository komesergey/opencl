#include <iostream>
#include <random>
#include <algorithm>
#include <xmmintrin.h>
#include <immintrin.h>
#include <chrono>

float dot_normal(const float *vec1, const float *vec2, unsigned n)
{
    float sum = 0;
    for(unsigned i = 0; i < n; ++i)
        sum += vec1[i] * vec2[i];
    return sum;
}

float dot_sse(const float *vec1, const float *vec2, unsigned n)
{
    __m128 u = {0};
    for (unsigned i = 0; i < n; i += 4)
    {
        __m128 w = _mm_load_ps(&vec1[i]);
        __m128 x = _mm_load_ps(&vec2[i]);

        x = _mm_mul_ps(w, x);
        u = _mm_add_ps(u, x);
    }
    __attribute__((aligned(16))) float t[4] = {0};
    _mm_store_ps(t, u);
    return t[0] + t[1] + t[2] + t[3];
}

float dot_avx(const float *vec1, const float *vec2, unsigned n)
{
    __m256 u = {0};
    for(unsigned i = 0; i < n; i += 8)
    {
        __m256 w = _mm256_load_ps(&vec1[i]);
        __m256 x = _mm256_load_ps(&vec2[i]);

        x = _mm256_mul_ps(w, x);
        u = _mm256_add_ps(u, x);
    }
    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, u);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

float dot_avx_2(const float *vec1, const float *vec2, unsigned n)
{
    __m256 u1 = {0};
    __m256 u2 = {0};
    for(unsigned i = 0; i < n; i += 16)
    {
        __m256 w1 = _mm256_load_ps(&vec1[i]);
        __m256 w2 = _mm256_load_ps(&vec1[i + 8]);
        __m256 x1 = _mm256_load_ps(&vec2[i]);
        __m256 x2 = _mm256_load_ps(&vec2[i + 8]);

        x1 = _mm256_mul_ps(w1, x1);
        x2 = _mm256_mul_ps(w2, x2);
        u1 = _mm256_add_ps(u1, x1);
        u2 = _mm256_add_ps(u2, x2);
    }
    u1 = _mm256_add_ps(u1, u2);

    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, u1);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

template<class T> long long calc_for_a_moment(T t)
{
    auto start = std::chrono::high_resolution_clock::now();

    t();

    auto done = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(done - start).count();
}


int main()
{
    const unsigned len_begin = 8;
    const unsigned len_end   = 1024 * 1024;
    const unsigned len_fact  = 2;
    const unsigned run_ms    = 250;

    std::mt19937 rng;
    std::uniform_real_distribution<> dst(-1, 1);

    for(unsigned len = len_begin; len <= len_end; len *= len_fact)
    {
        auto *p1 = new __attribute__((aligned(32))) float[len + 8];
        auto *p2 = new __attribute__((aligned(32))) float[len + 8];
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
