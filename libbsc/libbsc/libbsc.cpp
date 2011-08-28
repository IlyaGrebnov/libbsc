/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Compression/decompression functions                       */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@gmail.com>

See file AUTHORS for a full list of contributors.

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

#include "../platform/platform.h"
#include "../libbsc.h"

#include "../adler32/adler32.h"
#include "../bwt/bwt.h"
#include "../lzp/lzp.h"
#include "../qlfc/qlfc.h"
#include "../st/st.h"

int bsc_init(int features)
{
    int result = LIBBSC_NO_ERROR;

    if (result == LIBBSC_NO_ERROR) result = bsc_platform_init(features);
    if (result == LIBBSC_NO_ERROR) result = bsc_qlfc_init(features);

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

    if (result == LIBBSC_NO_ERROR) result = bsc_st_init(features);

#endif

    return result;
}

int bsc_store(const unsigned char * input, unsigned char * output, int n, int features)
{
    memmove(output + LIBBSC_HEADER_SIZE, input, n);
    *(int *)(output +  0) = n + LIBBSC_HEADER_SIZE;
    *(int *)(output +  4) = n;
    *(int *)(output +  8) = 0;
    *(int *)(output + 12) = 0;
    *(int *)(output + 16) = bsc_adler32(output + LIBBSC_HEADER_SIZE, n, features);
    *(int *)(output + 20) = bsc_adler32(output, 20, features);
    return n + LIBBSC_HEADER_SIZE;
}

int bsc_compress_inplace(unsigned char * data, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int features)
{
    int             indexes[256];
    unsigned char   num_indexes;

    int mode = 0;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }
    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n < 0 || n > 1073741824) return LIBBSC_BAD_PARAMETER;
    if (n <= LIBBSC_HEADER_SIZE)
    {
        memmove(data + LIBBSC_HEADER_SIZE, data, n);
        *(int *)(data +  0) = n + LIBBSC_HEADER_SIZE;
        *(int *)(data +  4) = n;
        *(int *)(data +  8) = 0;
        *(int *)(data + 12) = 0;
        *(int *)(data + 16) = bsc_adler32(data + LIBBSC_HEADER_SIZE, n, features);
        *(int *)(data + 20) = bsc_adler32(data, 20, features);
        return n + LIBBSC_HEADER_SIZE;
    }

    int lzSize = n;
    if (mode != (mode & 0xff))
    {
        unsigned char * buffer = (unsigned char *)bsc_malloc(n);
        if (buffer == NULL) return LIBBSC_NOT_ENOUGH_MEMORY;

        lzSize = bsc_lzp_compress(data, buffer, n, lzpHashSize, lzpMinLen, features);
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
        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(data, lzSize, &num_indexes, indexes, features); break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st_encode(data, lzSize, 3, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st_encode(data, lzSize, 4, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st_encode(data, lzSize, 5, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : index = bsc_st_encode(data, lzSize, 6, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : index = bsc_st_encode(data, lzSize, 7, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : index = bsc_st_encode(data, lzSize, 8, features); break;

#endif

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
        *(int *)(data + 16) = bsc_adler32(data + LIBBSC_HEADER_SIZE, result, features);
        *(int *)(data + 20) = bsc_adler32(data, 20, features);
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_compress(const unsigned char * input, unsigned char * output, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int features)
{
    if (input == output)
    {
        return bsc_compress_inplace(output, n, lzpHashSize, lzpMinLen, blockSorter, features);
    }

    int             indexes[256];
    unsigned char   num_indexes;

    int mode = 0;
    switch (blockSorter)
    {
        case LIBBSC_BLOCKSORTER_BWT : mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

        default : return LIBBSC_BAD_PARAMETER;
    }
    if (lzpMinLen != 0 || lzpHashSize != 0)
    {
        if (lzpMinLen < 4 || lzpMinLen > 255) return LIBBSC_BAD_PARAMETER;
        if (lzpHashSize < 10 || lzpHashSize > 28) return LIBBSC_BAD_PARAMETER;
        mode += (lzpMinLen << 8);
        mode += (lzpHashSize << 16);
    }
    if (n < 0 || n > 1073741824) return LIBBSC_BAD_PARAMETER;
    if (n <= LIBBSC_HEADER_SIZE)
    {
        memcpy(output + LIBBSC_HEADER_SIZE, input, n);
        *(int *)(output +  0) = n + LIBBSC_HEADER_SIZE;
        *(int *)(output +  4) = n;
        *(int *)(output +  8) = 0;
        *(int *)(output + 12) = 0;
        *(int *)(output + 16) = bsc_adler32(output + LIBBSC_HEADER_SIZE, n, features);
        *(int *)(output + 20) = bsc_adler32(output, 20, features);
        return n + LIBBSC_HEADER_SIZE;
    }
    int lzSize = 0;
    if (mode != (mode & 0xff))
    {
        lzSize = bsc_lzp_compress(input, output, n, lzpHashSize, lzpMinLen, features);
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
        case LIBBSC_BLOCKSORTER_BWT : index = bsc_bwt_encode(output, lzSize, &num_indexes, indexes, features); break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : index = bsc_st_encode(output, lzSize, 3, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : index = bsc_st_encode(output, lzSize, 4, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : index = bsc_st_encode(output, lzSize, 5, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : index = bsc_st_encode(output, lzSize, 6, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : index = bsc_st_encode(output, lzSize, 7, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : index = bsc_st_encode(output, lzSize, 8, features); break;

#endif

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
            *(int *)(output + 16) = bsc_adler32(output + LIBBSC_HEADER_SIZE, n, features);
            *(int *)(output + 20) = bsc_adler32(output, 20, features);
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
        *(int *)(output + 16) = bsc_adler32(output + LIBBSC_HEADER_SIZE, result, features);
        *(int *)(output + 20) = bsc_adler32(output, 20, features);
        return result + LIBBSC_HEADER_SIZE;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_block_info(const unsigned char * blockHeader, int headerSize, int * pBlockSize, int * pDataSize, int features)
{
    if (headerSize < LIBBSC_HEADER_SIZE)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(blockHeader + 20) != bsc_adler32(blockHeader, 20, features))
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
        case LIBBSC_BLOCKSORTER_BWT : test_mode = LIBBSC_BLOCKSORTER_BWT; break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : test_mode = LIBBSC_BLOCKSORTER_ST3; break;
        case LIBBSC_BLOCKSORTER_ST4 : test_mode = LIBBSC_BLOCKSORTER_ST4; break;
        case LIBBSC_BLOCKSORTER_ST5 : test_mode = LIBBSC_BLOCKSORTER_ST5; break;
        case LIBBSC_BLOCKSORTER_ST6 : test_mode = LIBBSC_BLOCKSORTER_ST6; break;
        case LIBBSC_BLOCKSORTER_ST7 : test_mode = LIBBSC_BLOCKSORTER_ST7; break;
        case LIBBSC_BLOCKSORTER_ST8 : test_mode = LIBBSC_BLOCKSORTER_ST8; break;

#endif

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

    int info = bsc_block_info(data, inputSize, &blockSize, &dataSize, features);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(data + 16) != bsc_adler32(data + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, features))
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
        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(data, lzSize, index, num_indexes, indexes, features); break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st_decode(data, lzSize, 3, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st_decode(data, lzSize, 4, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st_decode(data, lzSize, 5, index, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : result = bsc_st_decode(data, lzSize, 6, index, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : result = bsc_st_decode(data, lzSize, 7, index, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : result = bsc_st_decode(data, lzSize, 8, index, features); break;

#endif

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
            result = bsc_lzp_decompress(buffer, data, lzSize, lzpHashSize, lzpMinLen, features);
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

    int info = bsc_block_info(input, inputSize, &blockSize, &dataSize, features);
    if (info != LIBBSC_NO_ERROR)
    {
        return info;
    }

    if (inputSize < blockSize || outputSize < dataSize)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (*(unsigned int *)(input + 16) != bsc_adler32(input + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, features))
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
        case LIBBSC_BLOCKSORTER_BWT : result = bsc_bwt_decode(output, lzSize, index, num_indexes, indexes, features); break;

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

        case LIBBSC_BLOCKSORTER_ST3 : result = bsc_st_decode(output, lzSize, 3, index, features); break;
        case LIBBSC_BLOCKSORTER_ST4 : result = bsc_st_decode(output, lzSize, 4, index, features); break;
        case LIBBSC_BLOCKSORTER_ST5 : result = bsc_st_decode(output, lzSize, 5, index, features); break;
        case LIBBSC_BLOCKSORTER_ST6 : result = bsc_st_decode(output, lzSize, 6, index, features); break;
        case LIBBSC_BLOCKSORTER_ST7 : result = bsc_st_decode(output, lzSize, 7, index, features); break;
        case LIBBSC_BLOCKSORTER_ST8 : result = bsc_st_decode(output, lzSize, 8, index, features); break;

#endif

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
            result = bsc_lzp_decompress(buffer, output, lzSize, lzpHashSize, lzpMinLen, features);
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
