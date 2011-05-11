/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to Sort Transform of order 3, 4 and 5           */
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

#ifndef _LIBBSC_ST_H
#define _LIBBSC_ST_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

    /**
    * Constructs the Sort Transform  of order 3 transformed string of a given string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return the primary index if no error occurred, error code otherwise.
    */
    int bsc_st3_encode(unsigned char * T, int n, int features);

    /**
    * Reconstructs the original string from Sort Transform of order 3 transformed string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param index - the primary index.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_st3_decode(unsigned char * T, int n, int index, int features);

    /**
    * Constructs the Sort Transform  of order 4 transformed string of a given string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return the primary index if no error occurred, error code otherwise.
    */
    int bsc_st4_encode(unsigned char * T, int n, int features);

    /**
    * Reconstructs the original string from Sort Transform of order 4 transformed string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param index - the primary index.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_st4_decode(unsigned char * T, int n, int index, int features);

    /**
    * Constructs the Sort Transform  of order 5 transformed string of a given string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return the primary index if no error occurred, error code otherwise.
    */
    int bsc_st5_encode(unsigned char * T, int n, int features);

    /**
    * Reconstructs the original string from Sort Transform of order 5 transformed string.
    * @param T - the input/output string of n chars.
    * @param n - the length of the given string.
    * @param index - the primary index.
    * @param features - the set of additional features, can be LIBBSC_FEATURE_NONE.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_st5_decode(unsigned char * T, int n, int index, int features);

#endif

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                                  st.h */
/*-----------------------------------------------------------*/
