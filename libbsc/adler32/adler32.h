/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Interface to Adler-32 checksum functions                  */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2024 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright information and file AUTHORS
for full list of contributors.

See also the bsc and libbsc web site:
  http://libbsc.com/ for more information.

--*/

#ifndef _LIBBSC_ADLER32_H
#define _LIBBSC_ADLER32_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
    * Calculates Adler-32 checksum for input memory block.
    * @param T          - the input memory block of n bytes.
    * @param n          - the length of the input memory block.
    * @param features   - the set of additional features.
    * @return the value of cyclic redundancy check.
    */
    unsigned int bsc_adler32(const unsigned char * T, int n, int features);

#ifdef __cplusplus
}
#endif

#endif

/*-----------------------------------------------------------*/
/* End                                             adler32.h */
/*-----------------------------------------------------------*/
