/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Lempel Ziv Prediction                                     */
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

#include <stdlib.h>
#include <memory.h>
#include <string.h>

#include "lzp.h"

#include "../common/common.h"
#include "../libbsc.h"

#define LIBBSC_LZP_MATCH_FLAG 0xF2

int bsc_lzp_encode(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (n < 16)
    {
        return LIBBSC_BAD_PARAMETER;
    }

    if (int * lookup = (int *)bsc_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        memset(lookup, 0, (int)(1 << hashSize) * sizeof(int));

        unsigned int            mask        = (int)(1 << hashSize) - 1;
        const unsigned char *   inputStart  = input;
        const unsigned char *   inputEnd    = input + n;
        const unsigned char *   outputStart = output;
        const unsigned char *   outputEOB   = output + n - 32;

        unsigned int context = 0;
        for (int i = 0; i < 4; ++i)
        {
            context = (context << 8) | (*output++ = *input++);
        }

        const unsigned char * heuristic      = input;
        const unsigned char * inputMinLenEnd = input + n - minLen - 8;
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);
            if (value > 0)
            {
                const unsigned char * reference = inputStart + value;
                if ((*(unsigned int *)(input + minLen - 4) == *(unsigned int *)(reference + minLen - 4)) && (*(unsigned int *)(input) == *(unsigned int *)(reference)))
                {
                    if ((heuristic > input) && (*(unsigned int *)heuristic != *(unsigned int *)(reference + (heuristic - input))))
                    {
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    int len = 4;
                    for (; input + len < inputMinLenEnd; len += 4)
                    {
                        if (*(unsigned int *)(input + len) != *(unsigned int *)(reference + len)) break;
                    }
                    if (len < minLen)
                    {
                        if (heuristic < input + len) heuristic = input + len;
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    if (input[len] == reference[len]) len++;
                    if (input[len] == reference[len]) len++;
                    if (input[len] == reference[len]) len++;

                    input += len; context = (input[-1] + (input[-2] << 8) + (input[-3] << 16) + (input[-4] << 24));
                    
                    *output++ = LIBBSC_LZP_MATCH_FLAG; 
                    
                    len -= minLen; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; }

                    *output++ = (unsigned char)(len);
                }
                else
                {

LIBBSC_LZP_MATCH_NOT_FOUND:
                    
                    unsigned char next = *output++ = *input++; context = (context << 8) | next;
                    if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
                }
            }
            else
            {
                context = (context << 8) | (*output++ = *input++);
            }
        }

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);
            if (value > 0)
            {
                unsigned char next = *output++ = *input++; context = (context << 8) | next;
                if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
            }
            else
            {
                context = (context << 8) | (*output++ = *input++);
            }
        }

        bsc_free(lookup);

        return (output >= outputEOB) ? LIBBSC_UNEXPECTED_EOB : (int)(output - outputStart);
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_lzp_decode(const unsigned char * input, unsigned char * output, int n, int hashSize, int minLen)
{
    if (n < 4)
    {
        return LIBBSC_UNEXPECTED_EOB;
    }

    if (int * lookup = (int *)bsc_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        memset(lookup, 0, (int)(1 << hashSize) * sizeof(int));

        unsigned int            mask        = (int)(1 << hashSize) - 1;
        const unsigned char *   inputEnd    = input + n;
        const unsigned char *   outputStart = output;

        unsigned int context = 0;
        for (int i = 0; i < 4; ++i)
        {
            context = (context << 8) | (*output++ = *input++);
        }

        while (input < inputEnd)
        {
            unsigned int index = ((context >> 15) ^ context ^ (context >> 3)) & mask;
            int value = lookup[index]; lookup[index] = (int)(output - outputStart);
            if (*input == LIBBSC_LZP_MATCH_FLAG && value > 0)
            {
                input++;
                if (*input != 255)
                {
                    int len = minLen; while (true) { len += *input; if (*input++ != 254) break; }

                    const unsigned char * reference = outputStart + value;
                          unsigned char * outputEnd = output + len;

                    if (output - reference < 4)
                    {
                        int offset[4] = {0, 3, 2, 3};
                        
                        *output++ = *reference++;
                        *output++ = *reference++;
                        *output++ = *reference++;
                        *output++ = *reference++;

                        reference -= offset[output - reference];
                    }

                    while (output < outputEnd) { *(unsigned int *)output = *(unsigned int*)reference; output += 4; reference += 4; }

                    output = outputEnd; context = (output[-1] + (output[-2] << 8) + (output[-3] << 16) + (output[-4] << 24));
                }
                else
                {
                    input++; context = (context << 8) | (*output++ = LIBBSC_LZP_MATCH_FLAG);
                }
            }
            else
            {
                context = (context << 8) | (*output++ = *input++);
            }
        }
        
        bsc_free(lookup);
    
        return (int)(output - outputStart);
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-----------------------------------------------------------*/
/* End                                               lzp.cpp */
/*-----------------------------------------------------------*/
