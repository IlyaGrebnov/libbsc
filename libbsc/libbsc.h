/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to compression/decompression functions          */
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

#ifndef _LIBBSC_LIBBSC_H
#define _LIBBSC_LIBBSC_H

#define LIBBSC_NO_ERROR                0
#define LIBBSC_BAD_PARAMETER          -1
#define LIBBSC_NOT_ENOUGH_MEMORY      -2
#define LIBBSC_NOT_COMPRESSIBLE       -3
#define LIBBSC_NOT_SUPPORTED          -4
#define LIBBSC_UNEXPECTED_EOB         -5
#define LIBBSC_DATA_CORRUPT           -6

#define LIBBSC_GPU_ERROR              -7
#define LIBBSC_GPU_NOT_SUPPORTED      -8
#define LIBBSC_GPU_NOT_ENOUGH_MEMORY  -9

#define LIBBSC_BLOCKSORTER_BWT         1

#ifdef LIBBSC_SORT_TRANSFORM_SUPPORT

  #define LIBBSC_BLOCKSORTER_ST3       2
  #define LIBBSC_BLOCKSORTER_ST4       3
  #define LIBBSC_BLOCKSORTER_ST5       4
  #define LIBBSC_BLOCKSORTER_ST6       5
  #define LIBBSC_BLOCKSORTER_ST7       6
  #define LIBBSC_BLOCKSORTER_ST8       7

#endif

#define LIBBSC_FEATURE_NONE            0
#define LIBBSC_FEATURE_FASTMODE        1
#define LIBBSC_FEATURE_MULTITHREADING  2
#define LIBBSC_FEATURE_LARGEPAGES      4
#define LIBBSC_FEATURE_CUDA            8

#define LIBBSC_DEFAULT_LZPHASHSIZE     16
#define LIBBSC_DEFAULT_LZPMINLEN       128
#define LIBBSC_DEFAULT_BLOCKSORTER     LIBBSC_BLOCKSORTER_BWT
#define LIBBSC_DEFAULT_FEATURES        LIBBSC_FEATURE_MULTITHREADING

#define LIBBSC_HEADER_SIZE             24

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * You should call this function before you call any of the other functions in libbsc.
    * @param features   - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_init(int features);

    /**
    * Compress a memory block.
    * @param input                              - the input memory block of n bytes.
    * @param output                             - the output memory block of n + LIBBSC_HEADER_SIZE bytes.
    * @param n                                  - the length of the input memory block.
    * @param lzpHashSize[0, 10..28]             - the hash table size if LZP enabled, 0 otherwise.
    * @param lzpMinLen[0, 4..255]               - the minimum match length if LZP enabled, 0 otherwise.
    * @param blockSorter[ST3..ST8, BWT]         - block sorting algorithm.
    * @param features                           - the set of additional features.
    * @return the length of compressed memory block if no error occurred, error code otherwise.
    */
    int bsc_compress(const unsigned char * input, unsigned char * output, int n, int lzpHashSize, int lzpMinLen, int blockSorter, int features);

    /**
    * Store a memory block.
    * @param input                              - the input memory block of n bytes.
    * @param output                             - the output memory block of n + LIBBSC_HEADER_SIZE bytes.
    * @param n                                  - the length of the input memory block.
    * @param features                           - the set of additional features.
    * @return the length of stored memory block if no error occurred, error code otherwise.
    */
    int bsc_store(const unsigned char * input, unsigned char * output, int n, int features);

    /**
    * Determinate the sizes of input and output memory blocks for bsc_decompress function.
    * @param blockHeader                        - the header of input(compressed) memory block of headerSize bytes.
    * @param headerSize                         - the length of header, should be at least LIBBSC_HEADER_SIZE bytes.
    * @param pBlockSize[out]                    - the length of the input memory block for bsc_decompress function.
    * @param pDataSize[out]                     - the length of the output memory block for bsc_decompress function.
    * @param features                           - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_block_info(const unsigned char * blockHeader, int headerSize, int * pBlockSize, int * pDataSize, int features);

    /**
    * Decompress a memory block.
    * Note : You should call bsc_block_info function to determinate the sizes of input and output memory blocks.
    * @param input                              - the input memory block of inputSize bytes.
    * @param inputSize                          - the length of the input memory block.
    * @param output                             - the output memory block of outputSize bytes.
    * @param outputSize                         - the length of the output memory block.
    * @param features                           - the set of additional features.
    * @return LIBBSC_NO_ERROR if no error occurred, error code otherwise.
    */
    int bsc_decompress(const unsigned char * input, int inputSize, unsigned char * output, int outputSize, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-------------------------------------------------*/
/* End                                    libbsc.h */
/*-------------------------------------------------*/
