/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Burrows Wheeler Transform                                 */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2010 Ilya Grebnov <ilya.grebnov@libbsc.com>

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

#include <stdlib.h>
#include <memory.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "bwt.h"
#include "divsufsort/divsufsort.h"

#include "../common/common.h"
#include "../libbsc.h"

int bsc_bwt_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features)
{
    int index = divbwt(T, T, NULL, n, num_indexes, indexes, features & LIBBSC_FEATURE_MULTITHREADING);
    switch (index)
    {
        case -1 : return LIBBSC_BAD_PARAMETER;
        case -2 : return LIBBSC_NOT_ENOUGH_MEMORY;
    }
    return index;
}

static bool bsc_unbwt_sort(unsigned char * T, unsigned int * P, unsigned int * bucket, int n, int index)
{
    memset(bucket, 0, ALPHABET_SIZE * sizeof(unsigned int));

    for (int i = 0; i < index; ++i)
    {
        unsigned char c = T[i];
        P[i] = ((bucket[c]++) << 8) | c;
    }
    for (int i = index; i < n; ++i)
    {
        unsigned char c = T[i];
        P[i + 1] = ((bucket[c]++) << 8) | c;
    }

    bool failBack = false;
    for (int sum = 1, i = 0; i < ALPHABET_SIZE; ++i)
    {
        if (bucket[i] >= 0x1000000) failBack = true;
        sum += bucket[i]; bucket[i] = sum - bucket[i];
    }

    return failBack;
}

static INLINE int bsc_unbwt_binarysearch(unsigned int * p, unsigned int v)
{
    int index = 0;

    if (p[index + 127] <= v) index += 128;
    if (p[index +  63] <= v) index +=  64;
    if (p[index +  31] <= v) index +=  32;
    if (p[index +  15] <= v) index +=  16;
    if (p[index +   7] <= v) index +=   8;
    if (p[index +   3] <= v) index +=   4;
    if (p[index +   1] <= v) index +=   2;
    if (p[index +   0] <= v) index +=   1;

    return index;
}

static void bsc_unbwt_reconstruct_sequential(unsigned char * T, unsigned int * P, unsigned int * bucket, int n, int index, bool failBack)
{
    if (!failBack)
    {
        for (int p = 0, i = n - 1; i >= 0; --i)
        {
            unsigned int  u = P[p];
            unsigned char c = u & 0xff;
            T[i] = c; p = (u >> 8) + bucket[c];
        }
    }
    else
    {
        unsigned char L[ALPHABET_SIZE];
        for (int sum = n + 1, i = ALPHABET_SIZE - 1; i >= 0; --i)
        {
            bucket[i] = sum - bucket[i]; sum -= bucket[i];
        }
        int d = 0;
        for (int c = 0, sum = 1; c < ALPHABET_SIZE; ++c)
        {
            int p = bucket[c];
            if (p > 0)
            {
                L[d++] = (unsigned char)c;
                bucket[c] = sum; sum += p;
            }
        }
        for (int i = 0; i < index; ++i)
        {
            unsigned char c = T[i];
            P[bucket[c]++] = i;
        }
        for (int i = index; i < n; ++i)
        {
            unsigned char c = T[i];
            P[bucket[c]++] = i + 1;
        }
        for (int c = 0; c < d; ++c) bucket[c] = bucket[L[c]];
        for (int c = d; c < ALPHABET_SIZE; ++c) bucket[c] = n + 2;

        if (d != ALPHABET_SIZE)
        {
            for (int i = 0, p = index; i < n; ++i)
            {
                T[i] = L[bsc_unbwt_binarysearch(bucket, p)];
                p = P[p];
            }
        }
        else
        {
            for (int i = 0, p = index; i < n; ++i)
            {
                T[i] = bsc_unbwt_binarysearch(bucket, p);
                p = P[p];
            }
        }
    }
}

#ifdef _OPENMP

static void bsc_unbwt_reconstruct_parallel(unsigned char * T, unsigned int * P, unsigned int * bucket, int n, int index, int * indexes, bool failBack)
{
    int mod = n / 8;
    {
        mod |= mod >> 1;  mod |= mod >> 2;
        mod |= mod >> 4;  mod |= mod >> 8;
        mod |= mod >> 16; mod >>= 1; mod++;
    }

    int nBlocks = 1 + (n - 1) / mod;

    if (!failBack)
    {
        #pragma omp parallel for default(shared) schedule(dynamic, 1)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int p           = (blockId < nBlocks - 1) ? indexes[blockId] + 1    : 0;
            int blockStart  = (blockId < nBlocks - 1) ? mod * blockId + mod - 1 : n - 1;
            int blockEnd    = mod * blockId;

            for (int i = blockStart; i >= blockEnd; --i)
            {
                unsigned int  u = P[p];
                unsigned char c = u & 0xff;
                T[i] = c; p = (u >> 8) + bucket[c];
            }
        }
    }
    else
    {
        unsigned char L[ALPHABET_SIZE];
        for (int sum = n + 1, i = ALPHABET_SIZE - 1; i >= 0; --i)
        {
            bucket[i] = sum - bucket[i]; sum -= bucket[i];
        }
        int d = 0;
        for (int c = 0, sum = 1; c < ALPHABET_SIZE; ++c)
        {
            int p = bucket[c];
            if (p > 0)
            {
                L[d++] = (unsigned char)c;
                bucket[c] = sum; sum += p;
            }
        }
        for (int i = 0; i < index; ++i)
        {
            unsigned char c = T[i];
            P[bucket[c]++] = i;
        }
        for (int i = index; i < n; ++i)
        {
            unsigned char c = T[i];
            P[bucket[c]++] = i + 1;
        }
        for (int c = 0; c < d; ++c) bucket[c] = bucket[L[c]];
        for (int c = d; c < ALPHABET_SIZE; ++c) bucket[c] = n + 2;

        #pragma omp parallel for default(shared) schedule(dynamic, 1)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int p           = (blockId > 0          ) ? indexes[blockId - 1] + 1    : index;
            int blockEnd    = (blockId < nBlocks - 1) ? mod * blockId + mod         : n;
            int blockStart  = mod * blockId;

            if (d != ALPHABET_SIZE)
            {
                for (int i = blockStart; i < blockEnd; ++i)
                {
                    T[i] = L[bsc_unbwt_binarysearch(bucket, p)];
                    p = P[p];
                }
            }
            else
            {
                for (int i = blockStart; i < blockEnd; ++i)
                {
                    T[i] = bsc_unbwt_binarysearch(bucket, p);
                    p = P[p];
                }
            }
        }
    }
}

#endif

int bsc_bwt_decode(unsigned char * T, int n, int index, unsigned char num_indexes, int * indexes, int features)
{
    if ((T == NULL) || (n < 0) || (index <= 0) || (index > n))
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (n <= 1)
    {
        return LIBBSC_NO_ERROR;
    }
    if (unsigned int * P = (unsigned int *)bsc_malloc((n + 1) * sizeof(unsigned int)))
    {
        unsigned int bucket[ALPHABET_SIZE];

        bool failBack = bsc_unbwt_sort(T, P, bucket, n, index);

#ifdef _OPENMP

        int mod = n / 8;
        {
            mod |= mod >> 1;  mod |= mod >> 2;
            mod |= mod >> 4;  mod |= mod >> 8;
            mod |= mod >> 16; mod >>= 1;
        }

        if ((features & LIBBSC_FEATURE_MULTITHREADING) && (n >= 64 * 1024) && (num_indexes == (unsigned char)((n - 1) / (mod + 1))) && (indexes != NULL))
        {
            bsc_unbwt_reconstruct_parallel(T, P, bucket, n, index, indexes, failBack);
        }
        else

#endif

        {
            bsc_unbwt_reconstruct_sequential(T, P, bucket, n, index, failBack);
        }

        bsc_free(P);
        return LIBBSC_NO_ERROR;
    };
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-----------------------------------------------------------*/
/* End                                               bwt.cpp */
/*-----------------------------------------------------------*/
