/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Second stage encoding functions                           */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2026 Ilya Grebnov <ilya.grebnov@gmail.com>

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
#include <stdint.h>
#include <memory.h>

#include "coder.h"

#include "../libbsc.h"
#include "../platform/platform.h"

#include "qlfc/qlfc.h"

int bsc_coder_init(int features)
{
    int result = LIBBSC_NO_ERROR;

    if (result == LIBBSC_NO_ERROR) result = bsc_qlfc_init(features);

    return result;
}

static INLINE int bsc_coder_num_blocks(int n)
{
    struct entry { int blocks; int threshold; };

    static const struct entry break_points[] =
    {
        { 128, 128 * 128 * 65536 },
        {  96,  96 * 96 * 65536 },
        {  64,  64 * 64 * 65536 },
        {  48,  48 * 48 * 65536 },
        {  32,  32 * 32 * 65536 },
        {  24,  24 * 24 * 65536 },
        {  16,  16 * 16 * 65536 },
        {  12,  12 * 12 * 65536 },
        {   8,   8 * 8 * 65536 },
        {   6,   6 * 6 * 65536 },
        {   4,   4 * 4 * 65536 },
        {   2,   2 * 2 * 65536 },
    };

    for (int i = 0; i < sizeof(break_points) / sizeof(break_points[0]); i += 1)
    {
        if (n >= break_points[i].threshold) { return break_points[i].blocks; }
    }

    return 1;
}

int bsc_coder_encode_block(const unsigned char * input, unsigned char * output, int inputSize, int outputSize, int coder)
{
    if (coder == LIBBSC_CODER_QLFC_STATIC)   return bsc_qlfc_static_encode_block  (input, output, inputSize, outputSize);
    if (coder == LIBBSC_CODER_QLFC_ADAPTIVE) return bsc_qlfc_adaptive_encode_block(input, output, inputSize, outputSize);
    if (coder == LIBBSC_CODER_QLFC_FAST)     return bsc_qlfc_fast_encode_block    (input, output, inputSize, outputSize);

    return LIBBSC_BAD_PARAMETER;
}

void bsc_coder_split_blocks(const unsigned char * input, int n, int nBlocks, int * blockStart, int * blockSize)
{
    int rankSize = 0;
    for (int i = 1; i < n; i += 32)
    {
        rankSize += input[i] != input[i - 1];
    }

    if (rankSize > nBlocks)
    {
        int blockRankSize = rankSize / nBlocks;

        blockStart[0] = 0; rankSize = 0;
        for (int id = 0, i = 1; i < n; i += 32)
        {
            rankSize += input[i] != input[i - 1];
            if (rankSize == blockRankSize)
            {
                rankSize = 0;

                blockSize[id] = i - blockStart[id];
                id++; blockStart[id] = i;

                if (id == nBlocks - 1) break;
            }
        }
        blockSize[nBlocks - 1] = n - blockStart[nBlocks - 1];
    }
    else
    {
        for (int p = 0; p < nBlocks; ++p)
        {
            blockStart[p] = (n / nBlocks) * p;
            blockSize[p]  = (p != nBlocks - 1) ? n / nBlocks : n - (n / nBlocks) * (nBlocks - 1);
        }
    }
}

/* In place (libbsc's own call: output = input + LIBBSC_HEADER_SIZE) the coder returns LIBBSC_NOT_ENOUGH_MEMORY only when */
/* its scratch, which stands where libbsc's buffer stood, cannot be allocated; any other failure is LIBBSC_NOT_COMPRESSIBLE. */
static bool bsc_coder_output_in_input(const unsigned char * input, const unsigned char * output, int n)
{
    return ((uintptr_t)output >= (uintptr_t)input) && ((uintptr_t)output < (uintptr_t)input + (uintptr_t)n);
}

/* In place the loop below writes sub-block b once sub-block b + 1 is encoded. No result is larger than its sub-block, */
/* so with d = output - input and the header of 1 + 8 * nBlocks bytes those bytes end at most d + 1 + 8 * nBlocks + */
/* blockSize[0] + ... + blockSize[b] bytes from the input, and the first sub-block not yet read starts at blockSize[0] + */
/* ... + blockSize[b + 1]: every write stays behind the reads when every sub-block from 1 to nBlocks - 2 is at least d + 1 + */
/* 8 * nBlocks bytes. One sub-block, or any other split, is encoded into a scratch and copied. The two slots are taken only */
/* when they ask no more than that scratch of n + 4096 bytes: 2 * (maxBlockSize + 4096) <= n + 4096. */
static bool bsc_coder_can_encode_inplace(const unsigned char * input, const unsigned char * output, int n, int nBlocks, const int * blockSize, int maxBlockSize)
{
    if (nBlocks < 2) return false;
    if (2 * ((size_t)maxBlockSize + 4096) > (size_t)n + 4096) return false;

    uintptr_t bound = ((uintptr_t)output - (uintptr_t)input) + 1 + 8 * (uintptr_t)nBlocks;
    for (int blockId = 1; blockId < nBlocks - 1; ++blockId)
    {
        if ((uintptr_t)blockSize[blockId] < bound) return false;
    }

    return true;
}

int bsc_coder_compress_serial(const unsigned char * input, unsigned char * output, int n, int coder)
{
    int compressedStart[ALPHABET_SIZE];
    int compressedSize[ALPHABET_SIZE];
    int compressionResult[ALPHABET_SIZE];

    int nBlocks   = bsc_coder_num_blocks(n);
    int outputPtr = 1 + 8 * nBlocks;

    int maxBlockSize = 0;
    if (nBlocks > 1)
    {
        bsc_coder_split_blocks(input, n, nBlocks, compressedStart, compressedSize);
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            if (maxBlockSize < compressedSize[blockId]) maxBlockSize = compressedSize[blockId];
        }
    }

    const bool inplace = bsc_coder_output_in_input(input, output, n);
    if (inplace && !bsc_coder_can_encode_inplace(input, output, n, nBlocks, compressedSize, maxBlockSize))
    {
        unsigned char * scratch = (unsigned char *)bsc_malloc((n + 4096) * sizeof(unsigned char));
        if (scratch == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;

        int result = bsc_coder_compress_serial(input, scratch, n, coder);
        if (result >= LIBBSC_NO_ERROR) { memcpy(output, scratch, result); } else { result = LIBBSC_NOT_COMPRESSIBLE; }

        bsc_free(scratch);

        return result;
    }

    if (nBlocks == 1)
    {
        int result = bsc_coder_encode_block(input, output + 1, n, n - 1, coder);
        if (result >= LIBBSC_NO_ERROR) result = (output[0] = 1, result + 1);

        return result;
    }

    size_t          slotSize = (size_t)maxBlockSize + 4096;
    unsigned char * scratch  = NULL;
    if (inplace)
    {
        scratch = (unsigned char *)bsc_malloc(2 * slotSize * sizeof(unsigned char));
        if (scratch == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    void * model = bsc_malloc(bsc_qlfc_model_size(coder));
    unsigned char * buffer = (unsigned char *)bsc_malloc(maxBlockSize * sizeof(unsigned char));
    if (model == NULL || buffer == NULL)
    {
        if (scratch != NULL) bsc_free(scratch);
        if (buffer != NULL) bsc_free(buffer);
        if (model != NULL) bsc_free(model);
        return inplace ? LIBBSC_NOT_COMPRESSIBLE : LIBBSC_NOT_ENOUGH_MEMORY;
    }

    if (!inplace) output[0] = nBlocks;

    int pendingId = -1, pendingPtr = 0;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        int inputStart  = compressedStart[blockId];
        int inputSize   = compressedSize[blockId];
        int outputSize  = inputSize; if (outputSize > n - outputPtr) outputSize = n - outputPtr;

        unsigned char * target = inplace ? scratch + slotSize * (blockId & 1) : output + outputPtr;

        int result = bsc_qlfc_encode_block_model(input + inputStart, target, buffer, inputSize, outputSize, coder, model);
        if (result < LIBBSC_NO_ERROR)
        {
            if (outputPtr + inputSize >= n) { outputPtr = LIBBSC_NOT_COMPRESSIBLE; break; }
            result = inputSize; memcpy(target, input + inputStart, inputSize);
        }

        compressionResult[blockId] = result;

        if (inplace)
        {
            if (pendingId >= 0) memcpy(output + pendingPtr, scratch + slotSize * (pendingId & 1), compressionResult[pendingId]);
            pendingId = blockId; pendingPtr = outputPtr;
        }
        else
        {
            bsc_unaligned_store<int>(output + 1 + 8 * blockId + 0, inputSize);
            bsc_unaligned_store<int>(output + 1 + 8 * blockId + 4, result);
        }

        outputPtr += result;
    }

    if (inplace && outputPtr >= LIBBSC_NO_ERROR)
    {
        memcpy(output + pendingPtr, scratch + slotSize * (pendingId & 1), compressionResult[pendingId]);

        output[0] = nBlocks;
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            bsc_unaligned_store<int>(output + 1 + 8 * blockId + 0, compressedSize[blockId]);
            bsc_unaligned_store<int>(output + 1 + 8 * blockId + 4, compressionResult[blockId]);
        }
    }

    if (scratch != NULL) bsc_free(scratch);
    bsc_free(buffer); bsc_free(model);

    return outputPtr;
}

#ifdef LIBBSC_OPENMP

void bsc_coder_split_blocks_parallel(const unsigned char * input, int n, int nBlocks, int * blockStart, int * blockSize, int * threadRankSize)
{
    int numSamples  = (n + 30) / 32;
    int threadId    = omp_get_thread_num();
    int numThreads  = omp_get_num_threads();
    int sampleStart = (int)(((long long)numSamples * (threadId + 0)) / numThreads);
    int sampleEnd   = (int)(((long long)numSamples * (threadId + 1)) / numThreads);

    int rankSize = 0;
    for (int i = 1 + 32 * sampleStart; i < 1 + 32 * sampleEnd; i += 32)
    {
        rankSize += input[i] != input[i - 1];
    }

    threadRankSize[threadId] = rankSize;

    #pragma omp barrier

    int totalRankSize = 0, threadRankStart = 0;
    for (int t = 0; t < numThreads; ++t)
    {
        if (t == threadId) threadRankStart = totalRankSize;
        totalRankSize += threadRankSize[t];
    }

    if (totalRankSize > nBlocks)
    {
        int blockRankSize = totalRankSize / nBlocks;

        rankSize = threadRankStart;
        for (int id = rankSize / blockRankSize + 1, i = 1 + 32 * sampleStart; (i < 1 + 32 * sampleEnd) && (id < nBlocks); i += 32)
        {
            rankSize += input[i] != input[i - 1];
            if (rankSize == id * blockRankSize) { blockStart[id] = i; id++; }
        }

        #pragma omp barrier

        #pragma omp single
        {
            blockStart[0] = 0;
            for (int id = 0; id < nBlocks - 1; ++id) { blockSize[id] = blockStart[id + 1] - blockStart[id]; }
            blockSize[nBlocks - 1] = n - blockStart[nBlocks - 1];
        }
    }
    else
    {
        #pragma omp single
        {
            for (int p = 0; p < nBlocks; ++p)
            {
                blockStart[p] = (n / nBlocks) * p;
                blockSize[p]  = (p != nBlocks - 1) ? n / nBlocks : n - (n / nBlocks) * (nBlocks - 1);
            }
        }
    }
}

int bsc_coder_compress_parallel(const unsigned char * input, unsigned char * output, int n, int coder)
{
    int nBlocks = bsc_coder_num_blocks(n);

    if (unsigned char * buffer = (unsigned char *)bsc_malloc((n + 4096 * nBlocks) * sizeof(unsigned char)))
    {
        int compressionResult[ALPHABET_SIZE];
        int compressedStart[ALPHABET_SIZE];
        int compressedSize[ALPHABET_SIZE];
        int threadRankSize[ALPHABET_SIZE];

        int result  = LIBBSC_NO_ERROR;

        int numThreads = omp_get_max_threads() / omp_get_num_threads();
        if (numThreads > nBlocks) numThreads = nBlocks;
        if (numThreads < 1) numThreads = 1;

        const bool inplace = bsc_coder_output_in_input(input, output, n);

        if (!inplace) output[0] = nBlocks;
        #pragma omp parallel num_threads(numThreads) if(numThreads > 1)
        {
            if (omp_get_num_threads() == 1)
            {
                result = bsc_coder_compress_serial(input, output, n, coder);
            }
            else
            {
                bsc_coder_split_blocks_parallel(input, n, nBlocks, compressedStart, compressedSize, threadRankSize);

                #pragma omp for ordered schedule(dynamic, 1)
                for (int blockId = 0; blockId < nBlocks; ++blockId)
                {
                    int blockStart   = compressedStart[blockId];
                    int blockSize    = compressedSize[blockId];
                    int outputPtr    = 1 + 8 * nBlocks;

                    compressionResult[blockId] = bsc_coder_encode_block(input + blockStart, buffer + blockStart + 4096 * blockId, blockSize, blockSize, coder);
                    if (compressionResult[blockId] < LIBBSC_NO_ERROR) compressionResult[blockId] = blockSize;

                    if (inplace)
                    {
                        if (compressionResult[blockId] == blockSize) memcpy(buffer + blockStart + 4096 * blockId, input + blockStart, blockSize);
                    }
                    else
                    {
                        bsc_unaligned_store<int>(output + 1 + 8 * blockId + 0, blockSize);
                        bsc_unaligned_store<int>(output + 1 + 8 * blockId + 4, compressionResult[blockId]);
                    }

                    #pragma omp ordered
                    {
                        for (int p = 0; p < blockId; ++p) { outputPtr += compressionResult[p]; }
                    }

                    if (!inplace && outputPtr + compressionResult[blockId] < n)
                    {
                        if (compressionResult[blockId] != blockSize)
                        {
                            memcpy(output + outputPtr, buffer + blockStart + 4096 * blockId, compressionResult[blockId]);
                        }
                        else
                        {
                            memcpy(output + outputPtr, input + blockStart, compressionResult[blockId]);
                        }
                    }

                    if (!inplace && blockId == nBlocks - 1)
                    {
                        result = outputPtr + compressionResult[blockId];
                        if (result >= n) result = LIBBSC_NOT_COMPRESSIBLE;
                    }
                }

                if (inplace)
                {
                    int total = 1 + 8 * nBlocks;
                    for (int p = 0; p < nBlocks; ++p) { total += compressionResult[p]; }

                    if (total < n)
                    {
                        #pragma omp for schedule(dynamic, 1)
                        for (int blockId = 0; blockId < nBlocks; ++blockId)
                        {
                            int outputPtr = 1 + 8 * nBlocks;
                            for (int p = 0; p < blockId; ++p) { outputPtr += compressionResult[p]; }

                            memcpy(output + outputPtr, buffer + compressedStart[blockId] + 4096 * blockId, compressionResult[blockId]);
                        }

                        #pragma omp single
                        {
                            output[0] = nBlocks;
                            for (int blockId = 0; blockId < nBlocks; ++blockId)
                            {
                                bsc_unaligned_store<int>(output + 1 + 8 * blockId + 0, compressedSize[blockId]);
                                bsc_unaligned_store<int>(output + 1 + 8 * blockId + 4, compressionResult[blockId]);
                            }

                            result = total;
                        }
                    }
                    else
                    {
                        #pragma omp single
                        {
                            result = LIBBSC_NOT_COMPRESSIBLE;
                        }
                    }
                }
            }
        }

        bsc_free(buffer);

        return result;
    }
    return bsc_coder_output_in_input(input, output, n) ? LIBBSC_NOT_COMPRESSIBLE : LIBBSC_NOT_ENOUGH_MEMORY;
}

#endif

int bsc_coder_compress(const unsigned char * input, unsigned char * output, int n, int coder, int features)
{
    if ((coder != LIBBSC_CODER_QLFC_STATIC) && (coder != LIBBSC_CODER_QLFC_ADAPTIVE) && (coder != LIBBSC_CODER_QLFC_FAST))
    {
        return LIBBSC_BAD_PARAMETER;
    }

#ifdef LIBBSC_OPENMP

    if ((bsc_coder_num_blocks(n) != 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        return bsc_coder_compress_parallel(input, output, n, coder);
    }

#endif

    return bsc_coder_compress_serial(input, output, n, coder);
}


int bsc_coder_decode_block(const unsigned char * input, unsigned char * output, int coder)
{
    if (coder == LIBBSC_CODER_QLFC_STATIC)   return bsc_qlfc_static_decode_block  (input, output);
    if (coder == LIBBSC_CODER_QLFC_ADAPTIVE) return bsc_qlfc_adaptive_decode_block(input, output);
    if (coder == LIBBSC_CODER_QLFC_FAST)     return bsc_qlfc_fast_decode_block    (input, output);

    return LIBBSC_BAD_PARAMETER;
}

int bsc_coder_decompress(const unsigned char * input, unsigned char * output, int coder, int features)
{
    if ((coder != LIBBSC_CODER_QLFC_STATIC) && (coder != LIBBSC_CODER_QLFC_ADAPTIVE) && (coder != LIBBSC_CODER_QLFC_FAST))
    {
        return LIBBSC_BAD_PARAMETER;
    }

    int nBlocks = input[0];
    if (nBlocks == 1)
    {
        return bsc_coder_decode_block(input + 1, output, coder);
    }

    int decompressionResult[ALPHABET_SIZE];

    int pendingBlocks = nBlocks;

#ifdef LIBBSC_OPENMP

    int numThreads = omp_get_max_threads() / omp_get_num_threads();
    if (numThreads > nBlocks) numThreads = nBlocks;

    if ((numThreads > 1) && (features & LIBBSC_FEATURE_MULTITHREADING))
    {
        #pragma omp parallel for schedule(dynamic, 1) num_threads(numThreads)
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            int inputPtr  = 0; int inputSize;
            int outputPtr = 0; int outputSize;

            inputPtr += 1 + 8 * nBlocks;

            for (int p = 0; p < blockId; ++p)
            {
                inputSize  = bsc_unaligned_load<int>(input + 1 + 8 * p + 4);
                outputSize = bsc_unaligned_load<int>(input + 1 + 8 * p + 0);

                inputPtr += inputSize; outputPtr += outputSize;
            }

            inputSize  = bsc_unaligned_load<int>(input + 1 + 8 * blockId + 4);
            outputSize = bsc_unaligned_load<int>(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                decompressionResult[blockId] = bsc_coder_decode_block(input + inputPtr, output + outputPtr, coder);
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }

        pendingBlocks = 0;
        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            if (decompressionResult[blockId] == LIBBSC_NOT_ENOUGH_MEMORY) pendingBlocks++;
        }
    }

#endif

    if (pendingBlocks > 0)
    {
        void * model = NULL;

        for (int blockId = 0; blockId < nBlocks; ++blockId)
        {
            if (pendingBlocks < nBlocks && decompressionResult[blockId] != LIBBSC_NOT_ENOUGH_MEMORY) continue;

            int inputPtr  = 0; int inputSize;
            int outputPtr = 0; int outputSize;

            inputPtr += 1 + 8 * nBlocks;

            for (int p = 0; p < blockId; ++p)
            {
                inputSize  = bsc_unaligned_load<int>(input + 1 + 8 * p + 4);
                outputSize = bsc_unaligned_load<int>(input + 1 + 8 * p + 0);

                inputPtr += inputSize; outputPtr += outputSize;
            }

            inputSize  = bsc_unaligned_load<int>(input + 1 + 8 * blockId + 4);
            outputSize = bsc_unaligned_load<int>(input + 1 + 8 * blockId + 0);

            if (inputSize != outputSize)
            {
                if (model == NULL) model = bsc_malloc(bsc_qlfc_model_size(coder));
                decompressionResult[blockId] = model != NULL ? bsc_qlfc_decode_block_model(input + inputPtr, output + outputPtr, coder, model) : LIBBSC_NOT_ENOUGH_MEMORY;
            }
            else
            {
                decompressionResult[blockId] = inputSize; memcpy(output + outputPtr, input + inputPtr, inputSize);
            }
        }

        if (model != NULL) bsc_free(model);
    }

    int dataSize = 0, result = LIBBSC_NO_ERROR;
    for (int blockId = 0; blockId < nBlocks; ++blockId)
    {
        if (decompressionResult[blockId] < LIBBSC_NO_ERROR) result = decompressionResult[blockId];
        dataSize += decompressionResult[blockId];
    }

    return (result == LIBBSC_NO_ERROR) ? dataSize : result;
}

/*-----------------------------------------------------------*/
/* End                                             coder.cpp */
/*-----------------------------------------------------------*/
