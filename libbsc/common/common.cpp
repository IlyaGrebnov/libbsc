/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Common functions and constants                            */
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

#include "common.h"

void * bsc_malloc(size_t size)
{
#if defined(WIN32) || defined(WIN64)
    return VirtualAlloc(0, size, MEM_COMMIT, PAGE_READWRITE);
#else
    return malloc(size);
#endif
}

void bsc_free(void * address)
{
#if defined(WIN32) || defined(WIN64)
    VirtualFree(address, 0, MEM_RELEASE);
#else
    free(address);
#endif
}

/*-----------------------------------------------------------*/
/* End                                            common.cpp */
/*-----------------------------------------------------------*/
