/*--

This file is a part of libcubwt, a library for CUDA accelerated
suffix array and burrows wheeler transform construction.

   Copyright (c) 2022-2023 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright and license details.

--*/

#ifndef LIBCUBWT_CUH
#define LIBCUBWT_CUH 1

#define LIBCUBWT_VERSION_MAJOR          1
#define LIBCUBWT_VERSION_MINOR          0
#define LIBCUBWT_VERSION_PATCH          0
#define LIBCUBWT_VERSION_STRING	        "1.0.0"

#define LIBCUBWT_NO_ERROR               0
#define LIBCUBWT_BAD_PARAMETER          -1
#define LIBCUBWT_NOT_ENOUGH_MEMORY      -2

#define LIBCUBWT_GPU_ERROR              -7
#define LIBCUBWT_GPU_NOT_SUPPORTED      -8
#define LIBCUBWT_GPU_NOT_ENOUGH_MEMORY  -9

#ifdef __cplusplus
extern "C" {
#endif

    #include <stdint.h>

    /**
    * Allocates storage on the CUDA device that allows reusing allocated memory with each libcubwt operation.
    * @param max_length The maximum length of string to support.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_allocate_device_storage(void ** device_storage, int64_t max_length);

    /**
    * Destroys the previously allocated storage on the CUDA device.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_free_device_storage(void * device_storage);

    /**
    * Constructs the suffix array (SA) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param SA [0..n-1] The output array of suffixes.
    * @param n The length of the input string.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_sa(void * device_storage, const uint8_t * T, uint32_t * SA, int64_t n);

    /**
    * Constructs the inverse suffix array (ISA) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param ISA [0..n-1] The output inverse array of suffixes.
    * @param n The length of the input string.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_isa(void * device_storage, const uint8_t * T, uint32_t * ISA, int64_t n);

    /**
    * Constructs the suffix array (SA) and inverse suffix array (ISA) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param SA [0..n-1] The output array of suffixes.
    * @param ISA [0..n-1] The output inverse array of suffixes.
    * @param n The length of the input string.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_sa_isa(void * device_storage, const uint8_t * T, uint32_t * SA, uint32_t * ISA, int64_t n);

    /**
    * Constructs the Burrows-Wheeler Transform (BWT) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param L [0..n-1] The output string (can be T).
    * @param n The length of the input string.
    * @return The primary index if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_bwt(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n);

    /**
    * Constructs the Burrows-Wheeler Transform (BWT) and inverse suffix array (ISA) of a given string.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param L [0..n-1] The output string (can be T).
    * @param ISA [0..n-1] The output inverse array of suffixes.
    * @param n The length of the input string.
    * @return The primary index if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_bwt_isa(void * device_storage, const uint8_t * T, uint8_t * L, uint32_t * ISA, int64_t n);

    /**
    * Constructs the Burrows-Wheeler Transform (BWT) of a given string with auxiliary indexes.
    * @param device_storage The previously allocated storage on the CUDA device.
    * @param T [0..n-1] The input string.
    * @param L [0..n-1] The output string (can be T).
    * @param n The length of the input string.
    * @param r The sampling rate for auxiliary indexes (must be power of 2).
    * @param I [0..(n-1)/r] The output auxiliary indexes.
    * @return LIBCUBWT_NO_ERROR if no error occurred, libcubwt error code otherwise.
    */
    int64_t libcubwt_bwt_aux(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n, int64_t r, uint32_t * I);

#ifdef __cplusplus
}
#endif

#endif
