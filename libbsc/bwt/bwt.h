/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to Burrows Wheeler Transform                    */
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

#ifndef _LIBBSC_BWT_H
#define _LIBBSC_BWT_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * Constructs the burrows wheeler transformed string of a given string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param num_indexes - the length of secondary indexes array, can be NULL.
    * @param indexes - the secondary indexes array, can be NULL.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return the primary index if no error occurred, error code otherwise.
    */
    int bsc_bwt_encode(unsigned char * T, int n, unsigned char * num_indexes, int * indexes, int features);

    /**
    * Reconstructs the original string from burrows wheeler transformed string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param index - the primary index.
    * @param num_indexes - the length of secondary indexes array, can be 0.
    * @param indexes - the secondary indexes array, can be NULL.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_bwt_decode(unsigned char * T, int n, int index, unsigned char num_indexes, int * indexes, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                                 bwt.h */
/*-----------------------------------------------------------*/
