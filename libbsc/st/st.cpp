/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Sort Transform of order 3, 4, 5 and 6                     */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@libbsc.com>

The bsc and libbsc is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The bsc and libbsc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the bsc and libbsc. If not, see http://www.gnu.org/licenses/.

Please see the files COPYING and COPYING.LIB for full copyright information.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

/*--

Sort Transform is patented by Michael Schindler under US patent 6,199,064.
However for research purposes this algorithm is included in this software.
So if you are of the type who should worry about this (making money) worry away.
The author shall have no liability with respect to the infringement of
copyrights, trade secrets or any patents by this software. In no event will
the author be liable for any lost revenue or profits or other special,
indirect and consequential damages.

Sort Transform is disabled by default and can be enabled by defining the
preprocessor macro LIBBSC_SORT_TRANSFORM_SUPPORT at compile time.

--*/

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

#include <stdlib.h>
#include <memory.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "st.h"

#include "../libbsc.h"
#include "../common/common.h"

#define ALPHABET_SQRT_SIZE  (16)

static int bsc_st3_transform_sequential(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    memset(bucket, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 1] & 0xf;
    for (int i = 0; i < n; ++i)
    {
        unsigned char C1 = T[i];
        bucket[(C0 << 8) | C1]++;
        C0 = C1 & 0xf;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum - bucket[i];
    }

    int pos = bucket[((T[1] & 0xf) << 8) | T[2]];

    unsigned int W = (T[n - 1] << 16) | (T[0] << 8) | T[1];
    for (int i = 0; i < n; ++i)
    {
        W = (W << 8) | T[i + 2];
        P[bucket[W & 0x00000fff]++] = W >> 12;
    }

    memset(bucket, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

    unsigned char P0 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char P1 = T[i];
        bucket[(P0 << 4) | (P1 >> 4)]++;
        P0 = P1;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum;
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] & 0x00000fff]] = (unsigned char)(P[i] >> 12);
    }
    int index = bucket[P[pos] & 0x00000fff];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] & 0x00000fff]] = (unsigned char)(P[i] >> 12);
    }

    return index;
}

static int bsc_st4_transform_sequential(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    memset(bucket, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C1 = T[i];
        bucket[(C0 << 8) | C1]++;
        C0 = C1;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum - bucket[i];
    }

    int pos = bucket[(T[2] << 8) | T[3]];

    unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C = (unsigned char)(W >> 24);
        W = (W << 8) | T[i + 3];
        P[bucket[W & 0x0000ffff]++] = (W & 0xffff0000) | C;
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 16]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 16];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 16]] = P[i] & 0xff;
    }

    return index;
}

static int bsc_st5_transform_sequential(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    memset(bucket, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 2] & 0xf;
    unsigned char C1 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C2 = T[i];
        bucket[(C0 << 16) | (C1 << 8) | C2]++;
        C0 = C1 & 0xf; C1 = C2;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum - bucket[i];
    }

    int pos = bucket[((T[2] & 0xf) << 16) | (T[3] << 8) | T[4]];

    unsigned char L = T[n - 1];
    unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
    for (int i = 0; i < n; ++i)
    {
        unsigned int V = (W & 0xfffff000) | L;
        L = (unsigned char)(W >> 24); W = (W << 8) | T[i + 4];
        P[bucket[W & 0x000fffff]++] = V;
    }

    memset(bucket, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

    unsigned char P0 = T[n - 2];
    unsigned char P1 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char P2 = T[i];
        bucket[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
        P0 = P1; P1 = P2;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum;
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 12]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 12];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 12]] = P[i] & 0xff;
    }

    return index;
}

static int bsc_st6_transform_sequential(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    memset(bucket, 0, ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

    for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

    unsigned char C0 = T[n - 2], C1 = T[n - 1];
    for (int i = 0; i < n; ++i)
    {
        unsigned char C2 = T[i];
        bucket[(C0 << 16) | (C1 << 8) | C2]++;
        C0 = C1; C1 = C2;
    }

    for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
    {
        sum += bucket[i];
        bucket[i] = sum - bucket[i];
    }

    int pos = bucket[(T[3] << 16) | (T[4] << 8) | T[5]];

    unsigned int W0 = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
    unsigned int W1 = (T[    2] << 24) | (T[    3] << 16) | (T[4] << 8) | T[5];
    for (int i = 0; i < n; ++i)
    {
        W0 = (W0 << 8) | T[i + 2]; W1 = (W1 << 8) | T[i + 6];
        P[bucket[W1 >> 8]++] = (W0 << 8) | (W0 >> 24);
    }

    for (int i = n - 1; i >= pos; --i)
    {
        T[--bucket[P[i] >> 8]] = P[i] & 0xff;
    }
    int index = bucket[P[pos] >> 8];
    for (int i = pos - 1; i >= 0; --i)
    {
        T[--bucket[P[i] >> 8]] = P[i] & 0xff;
    }

    return index;
}

#ifdef _OPENMP

static int bsc_st3_transform_parallel(unsigned char * T, unsigned int * P, int * bucket0, int n)
{
    if (int * bucket1 = (int *)bsc_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        int pos, index;

        for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

        #pragma omp parallel default(shared) num_threads(2)
        {
            int nThreads = omp_get_num_threads();
            int threadId = omp_get_thread_num();

            if (nThreads == 1)
            {
                index = bsc_st3_transform_sequential(T, P, bucket0, n);
            }
            else
            {
                int median = n / 2;

                {
                    if (threadId == 0)
                    {
                        memset(bucket0, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char C0 = T[n - 1] & 0xf;
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char C1 = T[i];
                            bucket0[(C0 << 8) | C1]++;
                            C0 = C1 & 0xf;
                        }
                    }
                    else
                    {
                        memset(bucket1, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char C0 = T[median - 1] & 0xf;
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char C1 = T[i];
                            bucket1[(C0 << 8) | C1]++;
                            C0 = C1 & 0xf;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        for (int sum1 = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                            bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                        }

                        pos = bucket0[((T[1] & 0xf) << 8) | T[2]];
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        unsigned int W = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                        for (int i = 0; i < median; ++i)
                        {
                            W = (W << 8) | T[i + 2];
                            P[bucket0[W & 0x00000fff]++] = W >> 12;
                        }
                    }
                    else
                    {
                        unsigned int W = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                        for (int i = n - 1; i >= median; --i)
                        {
                            P[bucket1[W & 0x00000fff]--] = W >> 12;
                            W = (W >> 8) | (T[i - 2] << 24);
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        memset(bucket0, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char P1 = T[i];
                            bucket0[(P0 << 4) | (P1 >> 4)]++;
                            P0 = P1;
                        }
                    }
                    else
                    {
                        memset(bucket1, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char P1 = T[i];
                            bucket1[(P0 << 4) | (P1 >> 4)]++;
                            P0 = P1;
                        }

                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        for (int sum1 = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                            bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        if (pos < median)
                        {
                            for (int i = 0; i < pos; ++i)
                            {
                                T[bucket0[P[i] & 0x00000fff]++] = (unsigned char)(P[i] >> 12);
                            }
                            index = bucket0[P[pos] & 0x00000fff];
                            for (int i = pos; i < median; ++i)
                            {
                                T[bucket0[P[i] & 0x00000fff]++] = (unsigned char)(P[i] >> 12);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < median; ++i)
                            {
                                T[bucket0[P[i] & 0x00000fff]++] = (unsigned char)(P[i] >> 12);
                            }
                        }
                    }
                    else
                    {
                        if (pos >= median)
                        {
                            for (int i = n - 1; i > pos; --i)
                            {
                                T[bucket1[P[i] & 0x00000fff]--] = (unsigned char)(P[i] >> 12);
                            }
                            index = bucket1[P[pos] & 0x00000fff];
                            for (int i = pos; i >= median; --i)
                            {
                                T[bucket1[P[i] & 0x00000fff]--] = (unsigned char)(P[i] >> 12);
                            }
                        }
                        else
                        {
                            for (int i = n - 1; i >= median; --i)
                            {
                                T[bucket1[P[i] & 0x00000fff]--] = (unsigned char)(P[i] >> 12);
                            }
                        }
                    }
                }
            }
        }

        bsc_free(bucket1);
        return index;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st4_transform_parallel(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    if (int * bucket0 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        if (int * bucket1 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int pos, index;

            for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

            #pragma omp parallel default(shared) num_threads(2)
            {
                int nThreads = omp_get_num_threads();
                int threadId = omp_get_thread_num();

                if (nThreads == 1)
                {
                    index = bsc_st4_transform_sequential(T, P, bucket, n);
                }
                else
                {
                    int median = n / 2;

                    {
                        if (threadId == 0)
                        {
                            memset(bucket0, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                            unsigned char C0 = T[n - 1];
                            for (int i = 0; i < median; ++i)
                            {
                                unsigned char C1 = T[i];
                                bucket0[(C0 << 8) | C1]++;
                                C0 = C1;
                            }
                        }
                        else
                        {
                            memset(bucket1, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                            unsigned char C0 = T[median - 1];
                            for (int i = median; i < n; ++i)
                            {
                                unsigned char C1 = T[i];
                                bucket1[(C0 << 8) | C1]++;
                                C0 = C1;
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int sum1 = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                            {
                                int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                                bucket[i] = sum0; bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                            }

                            pos = bucket[(T[2] << 8) | T[3]];
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            for (int i = 0; i < median; ++i)
                            {
                                unsigned char C = (unsigned char)(W >> 24);
                                W = (W << 8) | T[i + 3];
                                P[bucket0[W & 0x0000ffff]++] = (W & 0xffff0000) | C;
                            }
                        }
                        else
                        {
                            unsigned int W = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            for (int i = n - 1; i >= median; --i)
                            {
                                unsigned char C = T[i - 1];
                                P[bucket1[W & 0x0000ffff]--] = (W & 0xffff0000) | C;
                                W = (W >> 8) | (C << 24);
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i) bucket0[i] = bucket[i];

                            if (pos < median)
                            {
                                for (int i = 0; i < pos; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                                index = bucket0[P[pos] >> 16];
                                for (int i = pos; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = 0; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 16]++] = P[i] & 0xff;
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE - 1; ++i) bucket1[i] = bucket[i + 1] - 1;
                            bucket1[ALPHABET_SIZE * ALPHABET_SIZE - 1] = n - 1;

                            if (pos >= median)
                            {
                                for (int i = n - 1; i > pos; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                                index = bucket1[P[pos] >> 16];
                                for (int i = pos; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = n - 1; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 16]--] = P[i] & 0xff;
                                }
                            }
                        }
                    }
                }
            }

            bsc_free(bucket1); bsc_free(bucket0);
            return index;
        };
        bsc_free(bucket0);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st5_transform_parallel(unsigned char * T, unsigned int * P, int * bucket0, int n)
{
    if (int * bucket1 = (int *)bsc_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        int pos, index;

        for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

        #pragma omp parallel default(shared) num_threads(2)
        {
            int nThreads = omp_get_num_threads();
            int threadId = omp_get_thread_num();

            if (nThreads == 1)
            {
                index = bsc_st5_transform_sequential(T, P, bucket0, n);
            }
            else
            {
                int median = n / 2;

                {
                    if (threadId == 0)
                    {
                        memset(bucket0, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char C0 = T[n - 2] & 0xf;
                        unsigned char C1 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char C2 = T[i];
                            bucket0[(C0 << 16) | (C1 << 8) | C2]++;
                            C0 = C1 & 0xf; C1 = C2;
                        }
                    }
                    else
                    {
                        memset(bucket1, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char C0 = T[median - 2] & 0xf;
                        unsigned char C1 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char C2 = T[i];
                            bucket1[(C0 << 16) | (C1 << 8) | C2]++;
                            C0 = C1 & 0xf; C1 = C2;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        for (int sum1 = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                            bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                        }

                        pos = bucket0[((T[2] & 0xf) << 16) | (T[3] << 8) | T[4]];
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        unsigned char L = T[n - 1];
                        unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned int V = (W & 0xfffff000) | L;

                            L = (unsigned char)(W >> 24); W = (W << 8) | T[i + 4];
                            P[bucket0[W & 0x000fffff]++] = V;
                        }
                    }
                    else
                    {
                        unsigned char L = T[n - 1];
                        unsigned int  W = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];
                        for (int i = n - 1; i >= median; --i)
                        {
                            unsigned int S = W & 0x000fffff;

                            W = (W >> 8) | (L << 24); L = T[i - 1];
                            P[bucket1[S]--] = (W & 0xfffff000) | L;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        memset(bucket0, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[n - 2];
                        unsigned char P1 = T[n - 1];
                        for (int i = 0; i < median; ++i)
                        {
                            unsigned char P2 = T[i];
                            bucket0[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
                            P0 = P1; P1 = P2;
                        }
                    }
                    else
                    {
                        memset(bucket1, 0, ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                        unsigned char P0 = T[median - 2];
                        unsigned char P1 = T[median - 1];
                        for (int i = median; i < n; ++i)
                        {
                            unsigned char P2 = T[i];
                            bucket1[(P0 << 12) | (P1 << 4) | (P2 >> 4)]++;
                            P0 = P1; P1 = P2;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        for (int sum1 = 0, i = 0; i < ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                        {
                            int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                            bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                        }
                    }

                    #pragma omp barrier
                }

                {
                    if (threadId == 0)
                    {
                        if (pos < median)
                        {
                            for (int i = 0; i < pos; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                            index = bucket0[P[pos] >> 12];
                            for (int i = pos; i < median; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                        }
                        else
                        {
                            for (int i = 0; i < median; ++i)
                            {
                                T[bucket0[P[i] >> 12]++] = P[i] & 0xff;
                            }
                        }
                    }
                    else
                    {
                        if (pos >= median)
                        {
                            for (int i = n - 1; i > pos; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                            index = bucket1[P[pos] >> 12];
                            for (int i = pos; i >= median; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                        }
                        else
                        {
                            for (int i = n - 1; i >= median; --i)
                            {
                                T[bucket1[P[i] >> 12]--] = P[i] & 0xff;
                            }
                        }
                    }
                }
            }
        }

        bsc_free(bucket1);
        return index;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static int bsc_st6_transform_parallel(unsigned char * T, unsigned int * P, int * bucket, int n)
{
    if (int * bucket0 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
    {
        if (int * bucket1 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int pos, index;

            for (int i = 0; i < LIBBSC_HEADER_SIZE; ++i) T[n + i] = T[i];

            #pragma omp parallel default(shared) num_threads(2)
            {
                int nThreads = omp_get_num_threads();
                int threadId = omp_get_thread_num();

                if (nThreads == 1)
                {
                    index = bsc_st6_transform_sequential(T, P, bucket, n);
                }
                else
                {
                    int median = n / 2;

                    {
                        if (threadId == 0)
                        {
                            memset(bucket0, 0, ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                            unsigned char C0 = T[n - 2], C1 = T[n - 1];
                            for (int i = 0; i < median; ++i)
                            {
                                unsigned char C2 = T[i];
                                bucket0[(C0 << 16) | (C1 << 8) | C2]++;
                                C0 = C1; C1 = C2;
                            }
                        }
                        else
                        {
                            memset(bucket1, 0, ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                            unsigned char C0 = T[median - 2], C1 = T[median - 1];
                            for (int i = median; i < n; ++i)
                            {
                                unsigned char C2 = T[i];
                                bucket1[(C0 << 16) | (C1 << 8) | C2]++;
                                C0 = C1; C1 = C2;
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int sum1 = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                            {
                                int sum0 = sum1; sum1 += bucket0[i] + bucket1[i];

                                bucket[i] = sum0; bucket0[i] = sum0; bucket1[i]= sum1 - 1;
                            }

                            pos = bucket[(T[3] << 16) | (T[4] << 8) | T[5]];
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            unsigned int W0 = (T[n - 2] << 24) | (T[n - 1] << 16) | (T[0] << 8) | T[1];
                            unsigned int W1 = (T[    2] << 24) | (T[    3] << 16) | (T[4] << 8) | T[5];
                            for (int i = 0; i < median; ++i)
                            {
                                W0 = (W0 << 8) | T[i + 2]; W1 = (W1 << 8) | T[i + 6];
                                P[bucket0[W1 >> 8]++] = (W0 << 8) | (W0 >> 24);
                            }
                        }
                        else
                        {
                            unsigned int W0 = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];
                            unsigned int W1 = (T[    3] << 24) | (T[4] << 16) | (T[5] << 8) | T[6];
                            for (int i = n - 1; i >= median; --i)
                            {
                                W0 = (W0 >> 8) | (T[i - 1] << 24); W1 = (W1 >> 8) | (T[i + 3] << 24);
                                P[bucket1[W1 >> 8]--] = (W0 << 8) | (W0 >> 24);
                            }
                        }

                        #pragma omp barrier
                    }

                    {
                        if (threadId == 0)
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE; ++i) bucket0[i] = bucket[i];

                            if (pos < median)
                            {
                                for (int i = 0; i < pos; ++i)
                                {
                                    T[bucket0[P[i] >> 8]++] = P[i] & 0xff;
                                }
                                index = bucket0[P[pos] >> 8];
                                for (int i = pos; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 8]++] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = 0; i < median; ++i)
                                {
                                    T[bucket0[P[i] >> 8]++] = P[i] & 0xff;
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE - 1; ++i) bucket1[i] = bucket[i + 1] - 1;
                            bucket1[ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE - 1] = n - 1;

                            if (pos >= median)
                            {
                                for (int i = n - 1; i > pos; --i)
                                {
                                    T[bucket1[P[i] >> 8]--] = P[i] & 0xff;
                                }
                                index = bucket1[P[pos] >> 8];
                                for (int i = pos; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 8]--] = P[i] & 0xff;
                                }
                            }
                            else
                            {
                                for (int i = n - 1; i >= median; --i)
                                {
                                    T[bucket1[P[i] >> 8]--] = P[i] & 0xff;
                                }
                            }
                        }
                    }
                }
            }

            bsc_free(bucket1); bsc_free(bucket0);
            return index;
        };
        bsc_free(bucket0);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_st3_encode(unsigned char * T, int n, int features)
{
    if ((T == NULL) || (n < 0))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return 0;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef _OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st3_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st3_transform_sequential(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st4_encode(unsigned char * T, int n, int features)
{
    if ((T == NULL) || (n < 0))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return 0;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef _OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st4_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st4_transform_sequential(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st5_encode(unsigned char * T, int n, int features)
{
    if ((T == NULL) || (n < 0))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return 0;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_malloc(ALPHABET_SQRT_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef _OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024))
            {
                index = bsc_st5_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st5_transform_sequential(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st6_encode(unsigned char * T, int n, int features)
{
    if ((T == NULL) || (n < 0))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return 0;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (int * bucket = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            int index = LIBBSC_NO_ERROR;

#ifdef _OPENMP

            if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 4 * 1024 * 1024))
            {
                index = bsc_st6_transform_parallel(T, P, bucket, n);
            }
            else

#endif

            {
                index = bsc_st6_transform_sequential(T, P, bucket, n);
            }

            bsc_free(bucket); bsc_free(P);
            return index;
        };
        bsc_free(P);
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static bool bsc_unst_sort(unsigned char * T, unsigned int * P, unsigned int * count, unsigned int * bucket, int n, int k)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    memset(P     , 0, n * sizeof(unsigned int));
    memset(count , 0, ALPHABET_SIZE * sizeof(unsigned int));
    memset(bucket, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int));

    bool failBack = false;
    {
        for (int i = 0; i < n; ++i) count[T[i]]++;
        for (int sum = 0, c = 0; c < ALPHABET_SIZE; ++c)
        {
            if (count[c] >= 0x800000) failBack = true;

            sum += count[c]; count[c] = sum - count[c];
            if ((int)count[c] != sum)
            {
                unsigned int * bucket_p = &bucket[c << 8];
                for (int i = count[c]; i < sum; ++i) bucket_p[T[i]]++;
            }
        }
    }

    for (int c = 0; c < ALPHABET_SIZE; ++c)
    {
        for (int d = 0; d < c; ++d)
        {
            int t = bucket[(d << 8) | c]; bucket[(d << 8) | c] = bucket[(c << 8) | d]; bucket[(c << 8) | d] = t;
        }
    }

    if (k == 3)
    {
        for (int sum = 0, w = 0; w < ALPHABET_SIZE * ALPHABET_SIZE; ++w)
        {
            if (bucket[w] > 0)
            {
                P[sum] = 1; sum += bucket[w];
            }
        }

        return failBack;
    }

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int sum = 0, w = 0; w < ALPHABET_SIZE * ALPHABET_SIZE; ++w)
    {
        sum += bucket[w]; bucket[w] = sum - bucket[w];
        for (int i = bucket[w]; i < sum; ++i)
        {
            unsigned char c = T[i];
            if (group[c] != w)
            {
                group[c] = w; P[index[c]] = 0x80000000;
            }
            index[c]++;
        }
    }

    unsigned int mask0 = 0x80000000, mask1 = 0x40000000;
    for (int round = 5; round <= k; ++round, mask0 >>= 1, mask1 >>= 1)
    {
        memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
        memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

        for (int g = 0, i = 0; i < n; ++i)
        {
            if (P[i] & mask0) g = i;

            unsigned char c = T[i];
            if (group[c] != g)
            {
                group[c] = g; P[index[c]] += mask1;
            }
            index[c]++;
        }
    }

    return failBack;
}

static void bsc_unst_reconstruct_case1(unsigned char * T, unsigned int * P, unsigned int * count, int n, int start)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = (c << 24) | index[c];
        }
        else
        {
            P[i] = (c << 24) | 0x800000 | group[c]; P[group[c]]++;
        }
        index[c]++;
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = u & 0x7fffff;
            u = P[p];
        }

        T[i] = u >> 24; P[p]--; p = u & 0x7fffff;
    }
}

static void bsc_unst_reconstruct_case2(unsigned char * T, unsigned int * P, unsigned int * count, int n, int start)
{
    unsigned int index[ALPHABET_SIZE];
             int group[ALPHABET_SIZE];

    memset(index, 0, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = (c << 24) | index[c];
        }
        else
        {
            P[i] = (c << 24) | 0x800000 | (i - group[c]); P[group[c]]++;
        }
        index[c]++;
    }

    for (int p = start, i = n - 1; i >= 0; --i)
    {
        unsigned int u = P[p];
        if (u & 0x800000)
        {
            p = p - (u & 0x7fffff);
            u = P[p];
        }

        unsigned char c = u >> 24;
        T[i] = c; P[p]--; p = (u & 0x7fffff) + count[c];
    }
}

static INLINE int bsc_unst_search(int index, unsigned int * p, unsigned int v)
{
    while (p[index + 1] <= v) index++;
    return index;
}

#define ST_NUM_FASTBITS (10)

static void bsc_unst_reconstruct_case3(unsigned char * T, unsigned int * P, unsigned int * count, int n, int start)
{
    unsigned char   fastbits[1 << ST_NUM_FASTBITS];
    unsigned int    index[ALPHABET_SIZE + 1];
             int    group[ALPHABET_SIZE];

    memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));
    memset(group, 0xff, ALPHABET_SIZE * sizeof(int));

    for (int g = 0, i = 0; i < n; ++i)
    {
        if (P[i] > 0) g = i;

        unsigned char c = T[i];
        if (group[c] < g)
        {
            group[c] = i; P[i] = index[c];
        }
        else
        {
            P[i] = 0x80000000 | group[c]; P[group[c]]++;
        }
        index[c]++;
    }

    {
        int shift = 0; while (((n - 1) >> shift) >= (1 << ST_NUM_FASTBITS)) shift++;

        {
            memcpy(index, count, ALPHABET_SIZE * sizeof(unsigned int));

            index[ALPHABET_SIZE] = n;
            for (int c = 0, v = 0; c < ALPHABET_SIZE; ++c)
            {
                if (index[c] != index[c + 1])
                {
                    for (; v <= (int)((index[c + 1] - 1) >> shift); ++v) fastbits[v] = c;
                }
            }
        }

        if (P[start] & 0x80000000)
        {
            start = P[start] & 0x7fffffff;
        }

        T[0] = bsc_unst_search(fastbits[start >> shift], index, start);
        P[start]--; start = P[start] + 1;

        for (int p = start, i = n - 1; i >= 1; --i)
        {
            unsigned int u = P[p];
            if (u & 0x80000000)
            {
                p = u & 0x7fffffff;
                u = P[p];
            }

            T[i] = bsc_unst_search(fastbits[p >> shift], index, p);
            P[p]--; p = u;
        }
    }
}

static void bsc_unst_reconstruct(unsigned char * T, unsigned int * P, unsigned int * count, int n, int index, bool failBack)
{
    if (n < 0x800000)   return bsc_unst_reconstruct_case1(T, P, count, n, index);
    if (!failBack)      return bsc_unst_reconstruct_case2(T, P, count, n, index);
    if (failBack)       return bsc_unst_reconstruct_case3(T, P, count, n, index);
}

int bsc_st3_decode(unsigned char * T, int n, int index, int features)
{
    if ((T == NULL) || (n < 0) || (index < 0) || (index >= n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (unsigned int * bucket = (unsigned int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int)))
        {
            unsigned int count[ALPHABET_SIZE];

            bool failBack = bsc_unst_sort(T, P, count, bucket, n, 3);
            bsc_unst_reconstruct(T, P, count, n, index, failBack);

            bsc_free(bucket); bsc_free(P);
            return LIBBSC_NO_ERROR;
        };
        bsc_free(P);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st4_decode(unsigned char * T, int n, int index, int features)
{
    if ((T == NULL) || (n < 0) || (index < 0) || (index >= n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (unsigned int * bucket = (unsigned int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int)))
        {
            unsigned int count[ALPHABET_SIZE];

            bool failBack = bsc_unst_sort(T, P, count, bucket, n, 4);
            bsc_unst_reconstruct(T, P, count, n, index, failBack);

            bsc_free(bucket); bsc_free(P);
            return LIBBSC_NO_ERROR;
        };
        bsc_free(P);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st5_decode(unsigned char * T, int n, int index, int features)
{
    if ((T == NULL) || (n < 0) || (index < 0) || (index >= n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (unsigned int * bucket = (unsigned int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int)))
        {
            unsigned int count[ALPHABET_SIZE];

            bool failBack = bsc_unst_sort(T, P, count, bucket, n, 5);
            bsc_unst_reconstruct(T, P, count, n, index, failBack);

            bsc_free(bucket); bsc_free(P);
            return LIBBSC_NO_ERROR;
        };
        bsc_free(P);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_st6_decode(unsigned char * T, int n, int index, int features)
{
    if ((T == NULL) || (n < 0) || (index < 0) || (index >= n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }

    if (unsigned int * P = (unsigned int *)bsc_malloc(n * sizeof(unsigned int)))
    {
        if (unsigned int * bucket = (unsigned int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(unsigned int)))
        {
            unsigned int count[ALPHABET_SIZE];

            bool failBack = bsc_unst_sort(T, P, count, bucket, n, 6);
            bsc_unst_reconstruct(T, P, count, n, index, failBack);

            bsc_free(bucket); bsc_free(P);
            return LIBBSC_NO_ERROR;
        };
        bsc_free(P);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

/*-----------------------------------------------------------*/
/* End                                                st.cpp */
/*-----------------------------------------------------------*/
