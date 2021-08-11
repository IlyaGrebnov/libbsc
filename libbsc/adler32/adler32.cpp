/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Adler-32 checksum functions                               */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2021 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#include <stdlib.h>
#include <memory.h>
#include <string.h>

#include "adler32.h"

#include "../platform/platform.h"
#include "../libbsc.h"

#define BASE 65521UL
#define NMAX 5536

#define DO1(buf, i) { sum1 += (buf)[i]; sum2 += sum1; }
#define DO2(buf, i) DO1(buf, i); DO1(buf, i + 1);
#define DO4(buf, i) DO2(buf, i); DO2(buf, i + 2);
#define DO8(buf, i) DO4(buf, i); DO4(buf, i + 4);
#define DO16(buf)   DO8(buf, 0); DO8(buf, 8);
#define MOD(a)      a %= BASE

#if defined(LIBBSC_DYNAMIC_CPU_DISPATCH)
    unsigned int bsc_adler32_avx(const unsigned char * T, int n, int features);
    unsigned int bsc_adler32_ssse3(const unsigned char * T, int n, int features);
    unsigned int bsc_adler32_sse2(const unsigned char * T, int n, int features);

    #if LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE2
        unsigned int bsc_adler32(const unsigned char * T, int n, int features)
        {
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_AVX)   { return bsc_adler32_avx  (T, n, features); }
            if (bsc_get_cpu_features() >= LIBBSC_CPU_FEATURE_SSSE3) { return bsc_adler32_ssse3(T, n, features); }

            return bsc_adler32_sse2(T, n, features);
        }
    #endif

    #if LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_AVX
        #define ADLER32_FUNCTION_NAME       bsc_adler32_avx
    #elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSSE3
        #define ADLER32_FUNCTION_NAME       bsc_adler32_ssse3
    #elif LIBBSC_CPU_FEATURE == LIBBSC_CPU_FEATURE_SSE2
        #define ADLER32_FUNCTION_NAME       bsc_adler32_sse2
    #endif
#else
    #define ADLER32_FUNCTION_NAME           bsc_adler32
#endif

#if defined(ADLER32_FUNCTION_NAME)

unsigned int ADLER32_FUNCTION_NAME (const unsigned char * T, int n, int features)
{
    unsigned int sum1 = 1;
    unsigned int sum2 = 0;

    while (n >= NMAX)
    {
#if LIBBSC_CPU_FEATURE >= LIBBSC_CPU_FEATURE_SSSE3
        const __m128i tap1 = _mm_setr_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17);
        const __m128i tap2 = _mm_setr_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        const __m128i zero = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m128i ones = _mm_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1);

        __m128i v_ps = _mm_set_epi32(0, 0, 0, sum1 * (NMAX / 32));
        __m128i v_s2 = _mm_set_epi32(0, 0, 0, sum2);
        __m128i v_s1 = _mm_set_epi32(0, 0, 0, 0);

        for (int i = 0; i < NMAX / 32; ++i)
        {
            const __m128i bytes1 = _mm_loadu_si128((__m128i *)(T));
            const __m128i bytes2 = _mm_loadu_si128((__m128i *)(T + 16));

            v_ps = _mm_add_epi32(v_ps, v_s1);
            
            v_s1 = _mm_add_epi32(v_s1, _mm_sad_epu8(bytes1, zero));
            v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(_mm_maddubs_epi16(bytes1, tap1), ones));

            v_s1 = _mm_add_epi32(v_s1, _mm_sad_epu8(bytes2, zero));
            v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(_mm_maddubs_epi16(bytes2, tap2), ones));

            T += 32;
        }

        v_s2 = _mm_add_epi32(v_s2, _mm_slli_epi32(v_ps, 5));

        v_s1 = _mm_add_epi32(v_s1, _mm_shuffle_epi32(v_s1, _MM_SHUFFLE(2, 3, 0, 1)));
        v_s1 = _mm_add_epi32(v_s1, _mm_shuffle_epi32(v_s1, _MM_SHUFFLE(1, 0, 3, 2)));
        sum1 += _mm_cvtsi128_si32(v_s1);

        v_s2 = _mm_add_epi32(v_s2, _mm_shuffle_epi32(v_s2, _MM_SHUFFLE(2, 3, 0, 1)));
        v_s2 = _mm_add_epi32(v_s2, _mm_shuffle_epi32(v_s2, _MM_SHUFFLE(1, 0, 3, 2)));
        sum2 = _mm_cvtsi128_si32(v_s2);

#else
        for (int i = 0; i < NMAX / 16; ++i)
        {
            DO16(T); T += 16;
        }
#endif

        MOD(sum1); MOD(sum2); n -= NMAX;
    }

    while (n >= 16)
    {
        DO16(T); T += 16; n -= 16;
    }

    while (n > 0)
    {
        DO1(T, 0); T += 1; n -= 1;
    }

    MOD(sum1); MOD(sum2);

    return sum1 | (sum2 << 16);
}

#endif

/*-----------------------------------------------------------*/
/* End                                           adler32.cpp */
/*-----------------------------------------------------------*/
