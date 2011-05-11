/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Detectors of blocksize, recordsize and contexts reorder.  */
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

#include <stdlib.h>
#include <memory.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "../common/common.h"
#include "../libbsc.h"
#include "../filters.h"

#include "core/tables.h"

#define DETECTORS_MAX_RECORD_SIZE   4
#define DETECTORS_NUM_BLOCKS        48
#define DETECTORS_BLOCK_SIZE        24576

struct BscBlockModel
{
    struct
    {
        int leftCount, rightCount;
        int leftFrequencies[ALPHABET_SIZE];
        int rightFrequencies[ALPHABET_SIZE];
    } contexts[ALPHABET_SIZE];
};

struct BscReorderingModel
{
    struct
    {
        int count;
        int frequencies[ALPHABET_SIZE];
    } contexts[DETECTORS_MAX_RECORD_SIZE][ALPHABET_SIZE];
};

int bsc_detect_segments_sequential(BscBlockModel * model, const unsigned char * input, int n)
{
    memset(model, 0, sizeof(BscBlockModel));

    for (int context = 0, i = 0; i < n; ++i)
    {
        unsigned char symbol = input[i];
        model->contexts[context].rightFrequencies[symbol]++;
        context = (unsigned char)((context << 5) ^ symbol);
    }

    long long entropy = 0;
    for (int context = 0; context < ALPHABET_SIZE; ++context)
    {
        for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
        {
            model->contexts[context].rightCount += model->contexts[context].rightFrequencies[symbol];
            entropy -= bsc_entropy(model->contexts[context].rightFrequencies[symbol]);
        }
        entropy += bsc_entropy(model->contexts[context].rightCount);
    }

    int blockSize = n;

    long long leftEntropy = 0, rightEntropy = entropy, bestEntropy = entropy - (entropy >> 5) - (65536LL * 12 * 1024);
    for (int context = 0, i = 0; i < n; ++i)
    {
        if (rightEntropy + leftEntropy < bestEntropy)
        {
            bestEntropy = rightEntropy + leftEntropy;
            blockSize   = i;
        }

        unsigned char symbol = input[i];

        rightEntropy    += bsc_delta(--model->contexts[context].rightFrequencies[symbol]);
        leftEntropy     -= bsc_delta(model->contexts[context].leftFrequencies[symbol]++);
        rightEntropy    -= bsc_delta(--model->contexts[context].rightCount);
        leftEntropy     += bsc_delta(model->contexts[context].leftCount++);

        context = (unsigned char)((context << 5) ^ symbol);
    }

    return blockSize;
}

#ifdef _OPENMP

int bsc_detect_segments_parallel(BscBlockModel * model0, BscBlockModel * model1, const unsigned char * input, int n)
{
    int globalBlockSize = n; long long globalBestEntropy;

    #pragma omp parallel default(shared) num_threads(2)
    {
        int nThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();

        if (nThreads == 1)
        {
            globalBlockSize = bsc_detect_segments_sequential(model0, input, n);
        }
        else
        {
            int median = n / 2;

            {
                if (threadId == 0)
                {
                    memset(model0, 0, sizeof(BscBlockModel));

                    int context = 0;
                    for (int i = 0; i < median; ++i)
                    {
                        unsigned char symbol = input[i];
                        model0->contexts[context].rightFrequencies[symbol]++;
                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }
                else
                {
                    memset(model1, 0, sizeof(BscBlockModel));

                    int context = (unsigned char)((input[median - 2] << 5) ^ input[median - 1]);
                    for (int i = median; i < n; ++i)
                    {
                        unsigned char symbol = input[i];
                        model1->contexts[context].leftFrequencies[symbol]++;
                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }

                #pragma omp barrier
            }

            {
                if (threadId == 0)
                {
                    long long entropy = 0;
                    for (int context = 0; context < ALPHABET_SIZE; ++context)
                    {
                        int count = 0;
                        for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
                        {
                            int frequency = model0->contexts[context].rightFrequencies[symbol] + model1->contexts[context].leftFrequencies[symbol];
                            model0->contexts[context].rightFrequencies[symbol] = model1->contexts[context].leftFrequencies[symbol] = frequency;

                            entropy -= bsc_entropy(frequency); count += frequency;
                        }
                        entropy += bsc_entropy(count); model0->contexts[context].rightCount = model1->contexts[context].leftCount = count;
                    }

                    globalBestEntropy = entropy;
                }

                #pragma omp barrier
            }

            {
                int localBlockSize = n; long long localBestEntropy = globalBestEntropy - (globalBestEntropy >> 5) - (65536LL * 12 * 1024);

                #pragma omp barrier

                if (threadId == 0)
                {
                    long long leftEntropy = 0, rightEntropy = globalBestEntropy;
                    for (int context = 0, i = 0; i < median; ++i)
                    {
                        if (rightEntropy + leftEntropy < localBestEntropy)
                        {
                            localBestEntropy = rightEntropy + leftEntropy;
                            localBlockSize   = i;
                        }

                        unsigned char symbol = input[i];

                        rightEntropy    += bsc_delta(--model0->contexts[context].rightFrequencies[symbol]);
                        leftEntropy     -= bsc_delta(model0->contexts[context].leftFrequencies[symbol]++);
                        rightEntropy    -= bsc_delta(--model0->contexts[context].rightCount);
                        leftEntropy     += bsc_delta(model0->contexts[context].leftCount++);

                        context = (unsigned char)((context << 5) ^ symbol);
                    }
                }
                else
                {
                    long long leftEntropy = globalBestEntropy, rightEntropy = 0;
                    for (int i = n - 1; i >= median; --i)
                    {
                        unsigned char   symbol  = input[i];
                        int             context = (unsigned char)((input[i - 2] << 5) ^ input[i - 1]);

                        rightEntropy    -= bsc_delta(model1->contexts[context].rightFrequencies[symbol]++);
                        leftEntropy     += bsc_delta(--model1->contexts[context].leftFrequencies[symbol]);
                        rightEntropy    += bsc_delta(model1->contexts[context].rightCount++);
                        leftEntropy     -= bsc_delta(--model1->contexts[context].leftCount);

                        if (rightEntropy + leftEntropy < localBestEntropy)
                        {
                            localBestEntropy = rightEntropy + leftEntropy;
                            localBlockSize   = i;
                        }
                    }
                }

                #pragma omp critical
                {
                    if (globalBestEntropy > localBestEntropy)
                    {
                        globalBlockSize = localBlockSize; globalBestEntropy = localBestEntropy;
                    }
                }
            }
        }
    }

    return globalBlockSize;
}


#endif

int bsc_detect_segments_recursive(BscBlockModel * model0, BscBlockModel * model1, const unsigned char * input, int n, int * segments, int k, int features)
{
    if (n < DETECTORS_BLOCK_SIZE || k == 1)
    {
        segments[0] = n;
        return 1;
    }

    int blockSize = n;

#ifdef _OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        blockSize = bsc_detect_segments_parallel(model0, model1, input, n);
    }
    else

#endif

    {
        blockSize = bsc_detect_segments_sequential(model0, input, n);
    }

    if (blockSize == n)
    {
        segments[0] = n;
        return 1;
    }

    int leftResult = bsc_detect_segments_recursive(model0, model1, input, blockSize, segments, k - 1, features);
    if (leftResult < LIBBSC_NO_ERROR) return leftResult;

    int rightResult = bsc_detect_segments_recursive(model0, model1, input + blockSize, n - blockSize, segments + leftResult, k - leftResult, features);
    if (rightResult < LIBBSC_NO_ERROR) return rightResult;

    return leftResult + rightResult;
}

int bsc_detect_segments(const unsigned char * input, int n, int * segments, int k, int features)
{
    if (n < DETECTORS_BLOCK_SIZE || k == 1)
    {
        segments[0] = n;
        return 1;
    }

    if (BscBlockModel * model0 = (BscBlockModel *)bsc_malloc(sizeof(BscBlockModel)))
    {
        if (BscBlockModel * model1 = (BscBlockModel *)bsc_malloc(sizeof(BscBlockModel)))
        {
            int result = bsc_detect_segments_recursive(model0, model1, input, n, segments, k, features);

            bsc_free(model1); bsc_free(model0);

            return result;
        }
        bsc_free(model0);
    };

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

static long long bsc_estimate_contextsorder(const unsigned char * input, int n)
{
    int frequencies[ALPHABET_SIZE][3];

    memset(frequencies, 0, sizeof(frequencies));

    unsigned char MTF0 = 0;
    unsigned char MTF1 = 1;
    unsigned char MTFC = 0;

    for (int i = 0; i < n; ++i)
    {
        unsigned char C = input[i];
        if (C == MTF0)
        {
            frequencies[MTFC][0]++; MTFC = MTFC << 2;
        }
        else
        {
            if (C == MTF1)
            {
                frequencies[MTFC][1]++; MTFC = (MTFC << 2) | 1;
            }
            else
            {
                frequencies[MTFC][2]++; MTFC = (MTFC << 2) | 2;
            }
            MTF1 = MTF0; MTF0 = C;
        }
    }

    long long entropy = 0;
    for (int context = 0; context < ALPHABET_SIZE; ++context)
    {
        int count = 0;
        for (int rank = 0; rank < 3; ++rank)
        {
            count += frequencies[context][rank];
            entropy -= bsc_entropy(frequencies[context][rank]);
        }
        entropy += bsc_entropy(count);
    }

    return entropy;
}

int bsc_detect_contextsorder(const unsigned char * input, int n, int features)
{
    int sortingContexts = LIBBSC_NOT_ENOUGH_MEMORY;

    if ((n > DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) && (features & LIBBSC_FEATURE_FASTMODE))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE * sizeof(unsigned char)))
        {
            int blockStride = (((n - DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) / DETECTORS_NUM_BLOCKS) / 48) * 48;

            for (int block = 0; block < DETECTORS_NUM_BLOCKS; ++block)
            {
                memcpy(buffer + block * DETECTORS_BLOCK_SIZE, input + block * (DETECTORS_BLOCK_SIZE + blockStride), DETECTORS_BLOCK_SIZE);
            }

            sortingContexts = bsc_detect_contextsorder(buffer, DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE, features);

            bsc_free(buffer);
        }

        return sortingContexts;
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n * sizeof(unsigned char)))
    {
        if (int * bucket0 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
        {
            memset(bucket0, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

            if (int * bucket1 = (int *)bsc_malloc(ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int)))
            {
                memset(bucket1, 0, ALPHABET_SIZE * ALPHABET_SIZE * sizeof(int));

                unsigned char C0 = input[n - 1];
                for (int i = 0; i < n; ++i)
                {
                    unsigned char C1 = input[i];
                    bucket0[(C0 << 8) | C1]++;
                    bucket1[(C1 << 8) | C0]++;
                    C0 = C1;
                }

                for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                {
                    sum += bucket0[i];
                    bucket0[i] = sum - bucket0[i];
                }

                unsigned char F0 = input[n - 2];
                unsigned char F1 = input[n - 1];
                for (int i = 0; i < n; ++i)
                {
                    unsigned char F2 = input[i];
                    buffer[bucket0[(F1 << 8) | F2]++] = F0;
                    F0 = F1; F1 = F2;
                }

                long long following = bsc_estimate_contextsorder(buffer, n);

                for (int sum = 0, i = 0; i < ALPHABET_SIZE * ALPHABET_SIZE; ++i)
                {
                    sum += bucket1[i];
                    bucket1[i] = sum - bucket1[i];
                }

                unsigned char P0 = input[1];
                unsigned char P1 = input[0];
                for (int i = n - 1; i >= 0; --i)
                {
                    unsigned char P2 = input[i];
                    buffer[bucket1[(P1 << 8) | P2]++] = P0;
                    P0 = P1; P1 = P2;
                }

                long long preceding = bsc_estimate_contextsorder(buffer, n);

                sortingContexts = (preceding < following) ? LIBBSC_CONTEXTS_PRECEDING : LIBBSC_CONTEXTS_FOLLOWING;

                bsc_free(bucket1);
            }
            bsc_free(bucket0);
        };
        bsc_free(buffer);
    }

    return sortingContexts;
}

long long bsc_estimate_reordering(BscReorderingModel * model, int recordSize)
{
    long long entropy = 0;
    for (int record = 0; record < recordSize; ++record)
    {
        for (int context = 0; context < ALPHABET_SIZE; ++context)
        {
            int count = 0;
            for (int symbol = 0; symbol < ALPHABET_SIZE; ++symbol)
            {
                int frequency = model->contexts[record][context].frequencies[symbol];
                count += frequency; entropy -= bsc_entropy(frequency);
            }
            entropy += (65536LL * 8 * (count < 256 ? count : 256)) + bsc_entropy(count);
        }
    }
    return entropy;
}

int bsc_detect_recordsize(const unsigned char * input, int n, int features)
{
    int result = LIBBSC_NOT_ENOUGH_MEMORY;

    if ((n > DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) && (features & LIBBSC_FEATURE_FASTMODE))
    {
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE * sizeof(unsigned char)))
        {
            int blockStride = (((n - DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE) / DETECTORS_NUM_BLOCKS) / 48) * 48;

            for (int block = 0; block < DETECTORS_NUM_BLOCKS; ++block)
            {
                memcpy(buffer + block * DETECTORS_BLOCK_SIZE, input + block * (DETECTORS_BLOCK_SIZE + blockStride), DETECTORS_BLOCK_SIZE);
            }

            result = bsc_detect_recordsize(buffer, DETECTORS_NUM_BLOCKS * DETECTORS_BLOCK_SIZE, features);

            bsc_free(buffer);
        }

        return result;
    }

    if (BscReorderingModel * model = (BscReorderingModel *)bsc_malloc(sizeof(BscReorderingModel)))
    {
        long long Entropy[DETECTORS_MAX_RECORD_SIZE];

        if ((n % 48) != 0) n = n - (n % 48);

        for (int recordSize = 1; recordSize <= DETECTORS_MAX_RECORD_SIZE; ++recordSize)
        {
            memset(model, 0, sizeof(BscReorderingModel));

            if (recordSize == 1)
            {
                int ctx0 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[0][ctx0].frequencies[c1]++; ctx0 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[0][ctx0].frequencies[c2]++; ctx0 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[0][ctx0].frequencies[c3]++; ctx0 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[0][ctx0].frequencies[c5]++; ctx0 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[0][ctx0].frequencies[c6]++; ctx0 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[0][ctx0].frequencies[c7]++; ctx0 = c7;
                }
            }

            if (recordSize == 2)
            {
                int ctx0 = 0, ctx1 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[0][ctx0].frequencies[c2]++; ctx0 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[1][ctx1].frequencies[c3]++; ctx1 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[1][ctx1].frequencies[c5]++; ctx1 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[0][ctx0].frequencies[c6]++; ctx0 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[1][ctx1].frequencies[c7]++; ctx1 = c7;
                }
            }

            if (recordSize == 3)
            {
                int ctx0 = 0, ctx1 = 0, ctx2 = 0;
                for (int i = 0; i < n; i += 6)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[2][ctx2].frequencies[c2]++; ctx2 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[0][ctx0].frequencies[c3]++; ctx0 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[1][ctx1].frequencies[c4]++; ctx1 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[2][ctx2].frequencies[c5]++; ctx2 = c5;
                }
            }

            if (recordSize == 4)
            {
                int ctx0 = 0, ctx1 = 0, ctx2 = 0, ctx3 = 0;
                for (int i = 0; i < n; i += 8)
                {
                    unsigned char c0 = input[i + 0]; model->contexts[0][ctx0].frequencies[c0]++; ctx0 = c0;
                    unsigned char c1 = input[i + 1]; model->contexts[1][ctx1].frequencies[c1]++; ctx1 = c1;
                    unsigned char c2 = input[i + 2]; model->contexts[2][ctx2].frequencies[c2]++; ctx2 = c2;
                    unsigned char c3 = input[i + 3]; model->contexts[3][ctx3].frequencies[c3]++; ctx3 = c3;
                    unsigned char c4 = input[i + 4]; model->contexts[0][ctx0].frequencies[c4]++; ctx0 = c4;
                    unsigned char c5 = input[i + 5]; model->contexts[1][ctx1].frequencies[c5]++; ctx1 = c5;
                    unsigned char c6 = input[i + 6]; model->contexts[2][ctx2].frequencies[c6]++; ctx2 = c6;
                    unsigned char c7 = input[i + 7]; model->contexts[3][ctx3].frequencies[c7]++; ctx3 = c7;
                }
            }

            if (recordSize > 4)
            {
                int Context[DETECTORS_MAX_RECORD_SIZE] = { 0 };
                for (int record = 0, i = 0; i < n; ++i)
                {
                    model->contexts[record][Context[record]].frequencies[input[i]]++;
                    Context[record] = input[i]; record++; if (record == recordSize) record = 0;
                }
            }

            Entropy[recordSize - 1] = bsc_estimate_reordering(model, recordSize);
        }

        long long bestSize = Entropy[0] - (Entropy[0] >> 4) - (65536LL * 8 * 1024);

        result = 1;
        for (int recordSize = 1; recordSize <= DETECTORS_MAX_RECORD_SIZE; ++recordSize)
        {
            if (bestSize > Entropy[recordSize - 1]) { bestSize = Entropy[recordSize - 1]; result = recordSize; }
        }

        bsc_free(model);
    };

    return result;
}

/*-------------------------------------------------*/
/* End                               detectors.cpp */
/*-------------------------------------------------*/
