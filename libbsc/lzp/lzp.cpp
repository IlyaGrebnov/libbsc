/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Lempel Ziv Prediction                                     */
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
#include <string.h>

#include "lzp.h"

#include "../common/common.h"
#include "../libbsc.h"

#define LZP_MATCH_FLAG 0xf2

int bsc_lzp_encode(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (n < 16)
    {
        return LIBBSC_BAD_PARAMETER;
    }
    if (int * Contexts = (int *)bsc_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        memset(Contexts, 0, (int)(1 << hashSize) * sizeof(int));

        unsigned int hashMask = (int)(1 << hashSize) - 1;

        const unsigned char * inputStart = input;
        const unsigned char * inputEnd = input + n;
        const unsigned char * outputStart = output;
        const unsigned char * outputEOB = output + n - 16;

        unsigned int ctx = 0;
        for (int i = 0; i < 4; ++i)
        {
            ctx = (ctx << 8) | (*output++ = *input++);
        }

        const unsigned char * hptr = input;

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int hashIndex = ((ctx >> 15) ^ ctx ^ (ctx >> 3)) & hashMask;
            int hashValue = Contexts[hashIndex]; Contexts[hashIndex] = (int)(input - inputStart);
            if (hashValue > 0)
            {
                const unsigned char * ptr = inputStart + hashValue;
                if ((input + minLen <= inputEnd) && (*(unsigned int *)(input + minLen - 4) == *(unsigned int *)(ptr + minLen - 4)))
                {
                    if (hptr > input)
                    {
                        if (*(unsigned int *)hptr != *(unsigned int *)(ptr + (hptr - input)))
                        {
                            goto MATCH_NOT_FOUND;
                        }
                    }

                    int len = 0;
                    for (; input + len + 4 <= inputEnd; len += 4)
                    {
                        if (*(unsigned int *)(input + len) != *(unsigned int *)(ptr + len)) break;
                    }
                    if (len + 3 < minLen)
                    {
                        if (hptr < input + len) hptr = input + len;
                        goto MATCH_NOT_FOUND;
                    }
                    for (; input + len < inputEnd; ++len)
                    {
                        if (input[len] != ptr[len]) break;
                    }
                    if (len < minLen) goto MATCH_NOT_FOUND;
                    input += len; ctx = (input[-1] + (input[-2] << 8) + (input[-3] << 16) + (input[-4] << 24));
                    *output++ = LZP_MATCH_FLAG;
                    for (len -= minLen; len >= 254; len -= 254)
                    {
                        *output++ = 254;
                        if (output >= outputEOB) break;
                    }
                    *output++ = (unsigned char)(len);
                }
                else
                {
MATCH_NOT_FOUND:
                    unsigned char next = *output++ = *input++; ctx = (ctx << 8) | next;
                    if (next == LZP_MATCH_FLAG)
                    {
                        *output++ = 255;
                    }
                }

            }
            else
            {
                ctx = (ctx << 8) | (*output++ = *input++);
            }
        }
        bsc_free(Contexts);
        if (output >= outputEOB)
        {
            return LIBBSC_UNEXPECTED_EOB;
        }
        return (int)(output - outputStart);
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_lzp_decode(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (n < 4)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }
    if (int * Contexts = (int *)bsc_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        memset(Contexts, 0, (int)(1 << hashSize) * sizeof(int));

        unsigned int hashMask = (int)(1 << hashSize) - 1;

        const unsigned char * inputEnd = input + n;
        const unsigned char * outputStart = output;

        unsigned int ctx = 0;
        for (int i = 0; i < 4; ++i)
        {
            ctx = (ctx << 8) | (*output++ = *input++);
        }

        while (input < inputEnd)
        {
            unsigned int hashIndex = ((ctx >> 15) ^ ctx ^ (ctx >> 3)) & hashMask;
            int hashValue = Contexts[hashIndex]; Contexts[hashIndex] = (int)(output - outputStart);
            if (*input == LZP_MATCH_FLAG && hashValue > 0)
            {
                input++;
                if (*input != 255)
                {
                    for (int len = minLen; true; )
                    {
                        len += *input;
                        if (*input++ != 254)
                        {
                            const unsigned char * ptr = outputStart + hashValue;
                            while (len--) *output++ = *ptr++;
                            ctx = (output[-1] + (output[-2] << 8) + (output[-3] << 16) + (output[-4] << 24));
                            break;
                        }
                    }
                }
                else
                {
                    input++; ctx = (ctx << 8) | (*output++ = LZP_MATCH_FLAG);
                }
            }
            else
            {
                ctx = (ctx << 8) | (*output++ = *input++);
            }
        }
        bsc_free(Contexts);
        return (int)(output - outputStart);
    }
    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-----------------------------------------------------------*/
/* End                                               lzp.cpp */
/*-----------------------------------------------------------*/
