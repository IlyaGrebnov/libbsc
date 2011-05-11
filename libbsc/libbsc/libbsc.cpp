/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Compression/decompression functions                       */
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

#include <stdlib.h>
#include <string.h>
#include <memory.h>

#include "../st/st.h"
#include "../lzp/lzp.h"
#include "../bwt/bwt.h"
#include "../qlfc/qlfc.h"
#include "../crc32/crc32.h"

#include "../common/common.h"
#include "../libbsc.h"

int bsc_init(int features)
{
    return bsc_qlfc_init(features);
}

int bsc_store(const unsigned char * input, unsigned char * output, int n)
{
    memmove(output + LIBBSC_HEADER_SIZE, input, n);
    *(int *)(output +  0) = n + LIBBSC_HEADER_SIZE;
    *(int *)(output +  4) = n;
    *(int *)(output +  8) = 0;
    *(int *)(output + 12) = 0;
    *(int *)(output + 16) = bsc_crc32(output + LIBBSC_HEADER_SIZE, n);
    *(int *)(output + 20) = bsc_crc32(output, 20);
    return n + LIBBSC_HEADER_SIZE;
}

int bsc_compress_inplace(unsigned char * data, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    int mode = 0;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;
        default : return LIBBSC_BAD_PARAMETER;
    }
    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n <= LIBBSC_HEADER_SIZE)
    {
        memmove(data + LIBBSC_HEADER_SIZE, data, n);
        *(int *)(data +  0) = n + LIBBSC_HEADER_SIZE;
        *(int *)(data +  4) = n;
        *(int *)(data +  8) = 0;
        *(int *)(data + 12) = 0;
        *(int *)(data + 16) = bsc_crc32(data + LIBBSC_HEADER_SIZE, n);
        *(int *)(data + 20) = bsc_crc32(data, 20);
        return n + LIBBSC_HEADER_SIZE;
    }

    int lzSize = n;
    if (mode != (mode & 0xff))
    {
        unsigned char * buffer = (unsigned char *)bsc_malloc(n);
        if (buffer == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;

        lzSize = bsc_lzp_encode(data, buffer, n, lzpHashSize, lzpMinLen);
        if (lzSize < LIBBSC_NO_ERROR)
        {
            lzSize = n; mode &= 0xff;
        }
        else
        {
            memcpy(data, buffer, lzSize);
        }

        bsc_free(buffer);
    }

    if (lzSize <= LIBBSC_HEADER_SIZE)
    {
        blockSorter = LIBBSC_BLOCKSORTER_BWT;
        mode = (mode & 0xffffff00) | LIBBSC_BLOCKSORTER_BWT;
    }

    int index = LIBBSC_BAD_PARAMETER; num_indexes = 0;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st3_encode(data, lzSize, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st4_encode(data, lzSize, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st5_encode(data, lzSize, features); break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(data, lzSize, &num_indexes, indexes, features); break;
        default : return LIBBSC_BAD_PARAMETER;
    }

    if (n < 64 * 1024) num_indexes = 0;

    if (index < LIBBSC_NO_ERROR)
    {
        return index;
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize + 4096))
    {
        int result = bsc_qlfc_compress(data, buffer, lzSize, features);
        if (result >= LIBBSC_NO_ERROR) memcpy(data + LIBBSC_HEADER_SIZE, buffer, result);
        bsc_free(buffer);
        if ((result < LIBBSC_NO_ERROR) || (result + 1 + 4 * num_indexes >= n))
        {
            return LIBBSC_NOT_COMPRESSIBLE;
        }
        {
            if (num_indexes > 0)
            {
                memcpy(data + LIBBSC_HEADER_SIZE + result, indexes, 4 * num_indexes);
            }
            data[LIBBSC_HEADER_SIZE + result + 4 * num_indexes] = num_indexes;
            result += 1 + 4 * num_indexes;
        }
        *(int *)(data +  0) = result + LIBBSC_HEADER_SIZE;
        *(int *)(data +  4) = n;
        *(int *)(data +  8) = mode;
        *(int *)(data + 12) = index;
        *(int *)(data + 16) = bsc_crc32(data + LIBBSC_HEADER_SIZE, result);
        *(int *)(data + 20) = bsc_crc32(data, 20);
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_compress(const unsigned char * input, unsigned char * output, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    if (input == output)
    {
        return bsc_compress_inplace(output, n, lzpHashSize, lzpMinLen, blockSorter, features);
    }

    int mode = 0;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;
        default : return LIBBSC_BAD_PARAMETER;
    }
    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n <= LIBBSC_HEADER_SIZE)
    {
        memcpy(output + LIBBSC_HEADER_SIZE, input, n);
        *(int *)(output +  0) = n + LIBBSC_HEADER_SIZE;
        *(int *)(output +  4) = n;
        *(int *)(output +  8) = 0;
        *(int *)(output + 12) = 0;
        *(int *)(output + 16) = bsc_crc32(output + LIBBSC_HEADER_SIZE, n);
        *(int *)(output + 20) = bsc_crc32(output, 20);
        return n + LIBBSC_HEADER_SIZE;
    }
    int lzSize = 0;
    if (mode != (mode & 0xff))
    {
        lzSize = bsc_lzp_encode(input, output, n, lzpHashSize, lzpMinLen);
        if (lzSize < LIBBSC_NO_ERROR)
        {
            mode &= 0xff;
        }
    }
    if (mode == (mode & 0xff))
    {
        lzSize = n; memcpy(output, input, n);
    }

    if (lzSize <= LIBBSC_HEADER_SIZE)
    {
        blockSorter = LIBBSC_BLOCKSORTER_BWT;
        mode = (mode & 0xffffff00) | LIBBSC_BLOCKSORTER_BWT;
    }

    int index = LIBBSC_BAD_PARAMETER; num_indexes = 0;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st3_encode(output, lzSize, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st4_encode(output, lzSize, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st5_encode(output, lzSize, features); break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(output, lzSize, &num_indexes, indexes, features); break;
        default : return LIBBSC_BAD_PARAMETER;
    }

    if (n < 64 * 1024) num_indexes = 0;

    if (index < LIBBSC_NO_ERROR)
    {
        return index;
    }

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize + 4096))
    {
        int result = bsc_qlfc_compress(output, buffer, lzSize, features);
        if (result >= LIBBSC_NO_ERROR) memcpy(output + LIBBSC_HEADER_SIZE, buffer, result);
        bsc_free(buffer);
        if ((result < LIBBSC_NO_ERROR) || (result + 1 + 4 * num_indexes >= n))
        {
            memcpy(output + LIBBSC_HEADER_SIZE, input, n);
            *(int *)(output +  0) = n + LIBBSC_HEADER_SIZE;
            *(int *)(output +  4) = n;
            *(int *)(output +  8) = 0;
            *(int *)(output + 12) = 0;
            *(int *)(output + 16) = bsc_crc32(output + LIBBSC_HEADER_SIZE, n);
            *(int *)(output + 20) = bsc_crc32(output, 20);
            return n + LIBBSC_HEADER_SIZE;
        }
        {
            if (num_indexes > 0)
            {
                memcpy(output + LIBBSC_HEADER_SIZE + result, indexes, 4 * num_indexes);
            }
            output[LIBBSC_HEADER_SIZE + result + 4 * num_indexes] = num_indexes;
            result += 1 + 4 * num_indexes;
        }
        *(int *)(output +  0) = result + LIBBSC_HEADER_SIZE;
        *(int *)(output +  4) = n;
        *(int *)(output +  8) = mode;
        *(int *)(output + 12) = index;
        *(int *)(output + 16) = bsc_crc32(output + LIBBSC_HEADER_SIZE, result);
        *(int *)(output + 20) = bsc_crc32(output, 20);
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_block_info(const unsigned char * blockHeader, int headerSize, int * pBlockSize, int * pDataSize)
{
    if (headerSize < LIBBSC_HEADER_SIZE)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(blockHeader + 20) != bsc_crc32(blockHeader, 20))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int blockSize    = *(int *)(blockHeader +  0);
    int dataSize     = *(int *)(blockHeader +  4);
    int mode         = *(int *)(blockHeader +  8);
    int index        = *(int *)(blockHeader + 12);

    int lzpMinLen    = (mode >>  8) & 0xff;
    int lzpHashSize  = (mode >> 16) & 0xff;
    int blockSorter  = (mode >>  0) & 0xff;

    int test_mode = 0;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST5 : test_mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST4 : test_mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST3 : test_mode = LIBBSC_BLOCKSORTER_ST3; break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : test_mode = LIBBSC_BLOCKSORTER_BWT; break;
        default : if (blockSorter > 0) return LIBBSC_DATA_CORRUPT;
    }
    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_DATA_CORRUPT;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_DATA_CORRUPT;
        test_mode += (lzpMinLen << 8);
        test_mode += (lzpHashSize << 16);
    }

    if (test_mode != mode)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (blockSize < LIBBSC_HEADER_SIZE || blockSize > LIBBSC_HEADER_SIZE + dataSize)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (index < 0 || index > dataSize)
    {
        return LIBBSC_DATA_CORRUPT;
    }

    if (pBlockSize != NULL) *pBlockSize = blockSize;
    if (pDataSize != NULL) *pDataSize = dataSize;

    return LIBBSC_NO_ERROR;
}

int bsc_decompress_inplace(unsigned char * data, int inputSize, int outputSize, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    int blockSize = 0, dataSize = 0;

    int info = bsc_block_info(data, inputSize, &blockSize, &dataSize);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(data + 16) != bsc_crc32(data + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int mode = *(int *)(data + 8);
    if (mode == 0)
    {
        memmove(data, data + LIBBSC_HEADER_SIZE, dataSize);
        return LIBBSC_NO_ERROR;
    }

    int index = *(int *)(data + 12);
    int blockSorter = (mode >> 0) & 0xff;

    num_indexes = data[blockSize - 1];
    if (num_indexes > 0)
    {
        memcpy(indexes, data + blockSize - 1 - 4 * num_indexes, 4 * num_indexes);
    }

    int lzSize = LIBBSC_NO_ERROR;
    {
        unsigned char * buffer = (unsigned char *)bsc_malloc(blockSize);
        if (buffer == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;

        memcpy(buffer, data, blockSize);

        lzSize = bsc_qlfc_decompress(buffer + LIBBSC_HEADER_SIZE, data, features);

        bsc_free(buffer);
    }
    if (lzSize < LIBBSC_NO_ERROR)
    {
        return lzSize;
    }

    int result;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st3_decode(data, lzSize, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st4_decode(data, lzSize, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st5_decode(data, lzSize, index, features); break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(data, lzSize, index, num_indexes, indexes, features); break;
        default : return LIBBSC_DATA_CORRUPT;
    }
    if (result < LIBBSC_NO_ERROR)
    {
        return result;
    }

    if (mode != (mode & 0xff))
    {
        int lzpMinLen   = (mode >>  8) & 0xff;
        int lzpHashSize = (mode >> 16) & 0xff;
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize))
        {
            memcpy(buffer, data, lzSize);
            result = bsc_lzp_decode(buffer, data, lzSize, lzpHashSize, lzpMinLen);
            bsc_free(buffer);
            if (result < LIBBSC_NO_ERROR)
            {
                return result;
            }
            return result == dataSize ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT;
        }
        return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    return lzSize == dataSize ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT;
}

int bsc_decompress(const unsigned char * input, int inputSize, unsigned char * output, int outputSize, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    if (input == output)
    {
        return bsc_decompress_inplace(output, inputSize, outputSize, features);
    }

    int blockSize = 0, dataSize = 0;

    int info = bsc_block_info(input, inputSize, &blockSize, &dataSize);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(input + 16) != bsc_crc32(input + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE))
    {
        return LIBBSC_DATA_CORRUPT;
    }

    int mode = *(int *)(input + 8);
    if (mode == 0)
    {
        memcpy(output, input + LIBBSC_HEADER_SIZE, dataSize);
        return LIBBSC_NO_ERROR;
    }

    int lzSize = bsc_qlfc_decompress(input + LIBBSC_HEADER_SIZE, output, features);
    if (lzSize < LIBBSC_NO_ERROR)
    {
        return lzSize;
    }

    int index = *(int *)(input + 12);
    int blockSorter = (mode >> 0) & 0xff;

    num_indexes = input[blockSize - 1];
    if (num_indexes > 0)
    {
        memcpy(indexes, input + blockSize - 1 - 4 * num_indexes, 4 * num_indexes);
    }

    int result;
    switch (blockSorter)
    {

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5)

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st3_decode(output, lzSize, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st4_decode(output, lzSize, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st5_decode(output, lzSize, index, features); break;

#endif

        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(output, lzSize, index, num_indexes, indexes, features); break;
        default : return LIBBSC_DATA_CORRUPT;
    }
    if (result < LIBBSC_NO_ERROR)
    {
        return result;
    }

    if (mode != (mode & 0xff))
    {
        int lzpMinLen   = (mode >>  8) & 0xff;
        int lzpHashSize = (mode >> 16) & 0xff;
        if (unsigned char * buffer = (unsigned char *)bsc_malloc(lzSize))
        {
            memcpy(buffer, output, lzSize);
            result = bsc_lzp_decode(buffer, output, lzSize, lzpHashSize, lzpMinLen);
            bsc_free(buffer);
            if (result < LIBBSC_NO_ERROR)
            {
                return result;
            }
            return result == dataSize ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT;
        }
        return LIBBSC_NOT_ENOUGH_MEMORY;
    }

    return lzSize == dataSize ? LIBBSC_NO_ERROR : LIBBSC_DATA_CORRUPT;
}

/*-------------------------------------------------*/
/* End                                  libbsc.cpp */
/*-------------------------------------------------*/
