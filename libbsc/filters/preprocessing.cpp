/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Data preprocessing functions                              */
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

#include "../filters.h"

#include "../platform/platform.h"
#include "../libbsc.h"

int bsc_reverse_block(unsigned char * T, int n, int features)
{

#ifdef LIBBSC_OPENMP

    if (features & LIBBSC_FEATURE_MULTITHREADING)
    {
        #pragma omp parallel for
        for (int i = 0; i < n / 2; ++i)
        {
            unsigned char tmp = T[i]; T[i] = T[n - 1 - i]; T[n - 1 - i] = tmp;
        }
    }
    else

#endif

    {
        for (int i = 0, j = n - 1; i < j; ++i, --j)
        {
            unsigned char tmp = T[i]; T[i] = T[j]; T[j] = tmp;
        }
    }

    return LIBBSC_NO_ERROR;
}

int bsc_reorder_forward(unsigned char * T, int n, char recordSize, int features)
{
    if (recordSize <= 0) return LIBBSC_BAD_PARAMETER;
    if (recordSize == 1) return LIBBSC_NO_ERROR;

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n))
    {
        memcpy(buffer, T, n);

        unsigned char * S = buffer;
        unsigned char * D = T;

        int chunk = (n / recordSize);

#ifdef LIBBSC_OPENMP

        if (features & LIBBSC_FEATURE_MULTITHREADING)
        {
            switch (recordSize)
            {
                case 2:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[2 * i]; D[chunk + i] = S[2 * i + 1]; } break;
                case 3:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[3 * i]; D[chunk + i] = S[3 * i + 1]; D[chunk * 2 + i] = S[3 * i + 2]; } break;
                case 4:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[i] = S[4 * i]; D[chunk + i] = S[4 * i + 1]; D[chunk * 2 + i] = S[4 * i + 2]; D[chunk * 3 + i] = S[4 * i + 3]; } break;
                default:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j * chunk + i] = S[recordSize * i + j]; }
            }
        }
        else

#endif

        {
            switch (recordSize)
            {
                case 2: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D++; S += 2; } break;
                case 3: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D[chunk * 2] = S[2]; D++; S += 3; } break;
                case 4: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[chunk] = S[1]; D[chunk * 2] = S[2]; D[chunk * 3] = S[3]; D++; S += 4; } break;
                default:
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j * chunk] = S[j]; D++; S += recordSize; }
            }
        }

        bsc_free(buffer); return LIBBSC_NO_ERROR;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

int bsc_reorder_reverse(unsigned char * T, int n, char recordSize, int features)
{
    if (recordSize <= 0) return LIBBSC_BAD_PARAMETER;
    if (recordSize == 1) return LIBBSC_NO_ERROR;

    if (unsigned char * buffer = (unsigned char *)bsc_malloc(n))
    {
        memcpy(buffer, T, n);

        unsigned char * S = buffer;
        unsigned char * D = T;

        int chunk = (n / recordSize);

#ifdef LIBBSC_OPENMP

        if (features & LIBBSC_FEATURE_MULTITHREADING)
        {
            switch (recordSize)
            {
                case 2:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[2 * i] = S[i]; D[2 * i + 1] = S[chunk + i]; } break;
                case 3:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[3 * i] = S[i]; D[3 * i + 1] = S[chunk + i]; D[3 * i + 2] = S[chunk * 2 + i]; } break;
                case 4:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { D[4 * i] = S[i]; D[4 * i + 1] = S[chunk + i]; D[4 * i + 2] = S[chunk * 2 + i]; D[4 * i + 3] = S[chunk * 3 + i]; } break;
                default:
                    #pragma omp parallel for
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[recordSize * i + j] = S[j * chunk + i]; }
            }
        }
        else

#endif

        {
            switch (recordSize)
            {
                case 2: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D += 2; S++; } break;
                case 3: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D[2] = S[chunk * 2]; D += 3; S++; } break;
                case 4: for (int i = 0; i < chunk; ++i) { D[0] = S[0]; D[1] = S[chunk]; D[2] = S[chunk * 2]; D[3] = S[chunk * 3]; D += 4; S++; } break;
                default:
                    for (int i = 0; i < chunk; ++i) { for (int j = 0; j < recordSize; ++j) D[j] = S[j * chunk]; D += recordSize; S++; }
            }
        }

        bsc_free(buffer); return LIBBSC_NO_ERROR;
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}

/*-------------------------------------------------*/
/* End                           preprocessing.cpp */
/*-------------------------------------------------*/
