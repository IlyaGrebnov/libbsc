/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to data preprocessing filters                   */
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

#ifndef _LIBBSC_FILTERS_H
#define _LIBBSC_FILTERS_H

#define LIBBSC_CONTEXTS_FOLLOWING    1
#define LIBBSC_CONTEXTS_PRECEDING    2

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * Autodetects segments for better compression of heterogeneous files.
    * @param input - the input memory block of n bytes.
    * @param n - the length of the input memory block.
    * @param segments - the output array of segments of k elements size.
    * @param k - the size of the output segments array.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return The number of segments if no error occurred, error code otherwise.
    */
    int bsc_detect_segments(const unsigned char * input, int n, int * segments, int k, int features);

    /**
    * Autodetects order of contexts for better compression of binary files.
    * @param input - the input memory block of n bytes.
    * @param n - the length of the input memory block.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return The detected contexts order if no error occurred, error code otherwise.
    */
    int bsc_detect_contextsorder(const unsigned char * input, int n, int features);

    /**
    * Reverses memory block to change order of contexts.
    * @param T - the input/output memory block of n bytes.
    * @param n - the length of the memory block.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reverse_block(unsigned char * T, int n, int features);

    /**
    * Autodetects record size for better compression of multimedia files.
    * @param input - the input memory block of n bytes.
    * @param n - the length of the input memory block.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return The size of record if no error occurred, error code otherwise.
    */
    int bsc_detect_recordsize(const unsigned char * input, int n, int features);

    /**
    * Reorders memory block for specific size of record (Forward transform).
    * @param T - the input/output memory block of n bytes.
    * @param n - the length of the memory block.
    * @param recordSize - the size of record.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reorder_forward(unsigned char * T, int n, char recordSize, int features);

    /**
    * Reorders memory block for specific size of record (Reverse transform).
    * @param T - the input/output memory block of n bytes.
    * @param n - the length of the memory block.
    * @param recordSize - the size of record.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_reorder_reverse(unsigned char * T, int n, char recordSize, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------*/
/* End                                   filters.h */
/*-------------------------------------------------*/
