/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Sort Transform (GPU version)                              */
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

#if defined(LIBBSC_SORT_TRANSFORM_SUPPORT) && defined(LIBBSC_CUDA_SUPPORT)

#if defined(_MSC_VER)
  #pragma warning(disable : 4267)
#endif

#include <stdlib.h>
#include <memory.h>

#include "st.cuh"

#include "../libbsc.h"
#include "../platform/platform.h"

#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <b40c/util/ping_pong_storage.cuh>
#include <b40c/radix_sort/enactor.cuh>

#ifdef LIBBSC_OPENMP

omp_lock_t cuda_lock;
int bsc_st_cuda_init(int features)
{
    omp_init_lock(&cuda_lock);
    return LIBBSC_NO_ERROR;
}

#else

int bsc_st_cuda_init(int features)
{
    return LIBBSC_NO_ERROR;
}

#endif

#ifndef __CUDA_ARCH__
  #define CUDA_DEVICE_ARCH              0
#else
  #define CUDA_DEVICE_ARCH              __CUDA_ARCH__
#endif

#define CUDA_DEVICE_PADDING             256
#define CUDA_NUM_THREADS_IN_BLOCK       192

#define CUDA_CTA_OCCUPANCY_SM20         8
#define CUDA_CTA_OCCUPANCY_SM12         5
#define CUDA_CTA_OCCUPANCY_SM10         4
#define CUDA_CTA_OCCUPANCY(v)           (v >= 200 ? CUDA_CTA_OCCUPANCY_SM20 : v >= 120 ? CUDA_CTA_OCCUPANCY_SM12 : CUDA_CTA_OCCUPANCY_SM10)

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK, CUDA_CTA_OCCUPANCY(CUDA_DEVICE_ARCH))
void bsc_st567_encode_cuda_presort(unsigned char * T_device, unsigned long long * K_device, int n)
{
    __shared__ unsigned int staging[1 + CUDA_NUM_THREADS_IN_BLOCK + 7];

    unsigned int * thread_staging = &staging[threadIdx.x];
    for (int grid_size = gridDim.x * CUDA_NUM_THREADS_IN_BLOCK, block_start = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK; block_start < n; block_start += grid_size)
    {
        int index = block_start + threadIdx.x;

        {
                                  thread_staging[1                            ] = T_device[index                            ];
            if (threadIdx.x < 7 ) thread_staging[1 + CUDA_NUM_THREADS_IN_BLOCK] = T_device[index + CUDA_NUM_THREADS_IN_BLOCK]; else
            if (threadIdx.x == 7) thread_staging[-7                           ] = T_device[index - 8                        ];

            syncthreads();
        }

        {
            #if CUDA_DEVICE_ARCH >= 200
                unsigned int lo = __byte_perm(thread_staging[4], thread_staging[5], 0x0411) | __byte_perm(thread_staging[6], thread_staging[7], 0x1104);
                unsigned int hi = __byte_perm(thread_staging[0], thread_staging[1], 0x0411) | __byte_perm(thread_staging[2], thread_staging[3], 0x1104);
            #else
                unsigned int lo = (thread_staging[4] << 24) | (thread_staging[5] << 16) | (thread_staging[6] << 8) | thread_staging[7];
                unsigned int hi = (thread_staging[0] << 24) | (thread_staging[1] << 16) | (thread_staging[2] << 8) | thread_staging[3];
            #endif

            K_device[index] = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);

            syncthreads();
        }
    }
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK, CUDA_CTA_OCCUPANCY(CUDA_DEVICE_ARCH))
void bsc_st8_encode_cuda_presort(unsigned char * T_device, unsigned long long * K_device, unsigned char * V_device, int n)
{
    __shared__ unsigned int staging[1 + CUDA_NUM_THREADS_IN_BLOCK + 8];

    unsigned int * thread_staging = &staging[threadIdx.x];
    for (int grid_size = gridDim.x * CUDA_NUM_THREADS_IN_BLOCK, block_start = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK; block_start < n; block_start += grid_size)
    {
        int index = block_start + threadIdx.x;

        {
                                  thread_staging[1                            ] = T_device[index                            ];
            if (threadIdx.x < 8 ) thread_staging[1 + CUDA_NUM_THREADS_IN_BLOCK] = T_device[index + CUDA_NUM_THREADS_IN_BLOCK]; else
            if (threadIdx.x == 8) thread_staging[-8                           ] = T_device[index - 9                        ];

            syncthreads();
        }

        {
            #if CUDA_DEVICE_ARCH >= 200
                unsigned int lo = __byte_perm(thread_staging[5], thread_staging[6], 0x0411) | __byte_perm(thread_staging[7], thread_staging[8], 0x1104);
                unsigned int hi = __byte_perm(thread_staging[1], thread_staging[2], 0x0411) | __byte_perm(thread_staging[3], thread_staging[4], 0x1104);
            #else
                unsigned int lo = (thread_staging[5] << 24) | (thread_staging[6] << 16) | (thread_staging[7] << 8) | thread_staging[8];
                unsigned int hi = (thread_staging[1] << 24) | (thread_staging[2] << 16) | (thread_staging[3] << 8) | thread_staging[4];
            #endif

            K_device[index] = (((unsigned long long)hi) << 32) | ((unsigned long long)lo); V_device[index] = thread_staging[0];

            syncthreads();
        }
    }
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK, CUDA_CTA_OCCUPANCY(CUDA_DEVICE_ARCH))
void bsc_st567_encode_cuda_postsort(unsigned char * T_device, unsigned long long * K_device, int n, unsigned long long lookup, int * I_device)
{
    int min_index = n;
    for (int grid_size = gridDim.x * CUDA_NUM_THREADS_IN_BLOCK, block_start = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK; block_start < n; block_start += grid_size)
    {
        int index = block_start + threadIdx.x;
        {
            unsigned long long value = K_device[index];
            {
                if (value == lookup && index < min_index) min_index = index;
                T_device[index] = (unsigned char)(value >> 56);
            }
        }
    }

    if (min_index != n) atomicMin(I_device, min_index);
}

__global__ __launch_bounds__(CUDA_NUM_THREADS_IN_BLOCK, CUDA_CTA_OCCUPANCY(CUDA_DEVICE_ARCH))
void bsc_st8_encode_cuda_postsort(unsigned long long * K_device, int n, unsigned long long lookup, int * I_device)
{
    int min_index = n;
    for (int grid_size = gridDim.x * CUDA_NUM_THREADS_IN_BLOCK, block_start = blockIdx.x * CUDA_NUM_THREADS_IN_BLOCK; block_start < n; block_start += grid_size)
    {
        int index = block_start + threadIdx.x;
        {
            if (K_device[index] == lookup && index < min_index) min_index = index;
        }
    }

    if (min_index != n) atomicMin(I_device, min_index);
}

int bsc_st567_encode_cuda(unsigned char * T, unsigned char * T_device, int n, int num_blocks, int k)
{
    #ifdef LIBBSC_OPENMP
        omp_set_lock(&cuda_lock);
    #endif

    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned long long * K_device = NULL;
        if (cudaMalloc((void **)&K_device, (n + CUDA_DEVICE_PADDING) * sizeof(unsigned long long)) == cudaSuccess)
        {
            index = LIBBSC_GPU_ERROR;

            bsc_st567_encode_cuda_presort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK>>>(T_device, K_device, n);

            cudaError_t status = cudaSuccess;
            {
                b40c::util::PingPongStorage<unsigned long long> storage(K_device);

                b40c::radix_sort::Enactor enactor;
                if (k == 5) status = enactor.Sort<16, 40, b40c::radix_sort::LARGE_SIZE>(storage, n);
                if (k == 6) status = enactor.Sort< 8, 48, b40c::radix_sort::LARGE_SIZE>(storage, n);
                if (k == 7) status = enactor.Sort< 0, 56, b40c::radix_sort::LARGE_SIZE>(storage, n);

                if (status == cudaErrorMemoryAllocation) index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;

                if (status == cudaSuccess && storage.selector == 1)
                {
                    cudaMemcpy(K_device, storage.d_keys[1], n * sizeof(unsigned long long), cudaMemcpyDeviceToDevice);
                }

                if (storage.d_keys[1] != NULL) cudaFree(storage.d_keys[1]);
            }

            if (status == cudaSuccess)
            {
                unsigned long long lookup;
                {
                    unsigned int lo = (T[3    ] << 24) | (T[4] << 16) | (T[5] << 8) | T[6];
                    unsigned int hi = (T[n - 1] << 24) | (T[0] << 16) | (T[1] << 8) | T[2];

                    lookup = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);

                    cudaMemcpy(T_device - sizeof(int), &n, sizeof(int), cudaMemcpyHostToDevice);
                }

                bsc_st567_encode_cuda_postsort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK>>>(T_device, K_device, n, lookup, (int *)(T_device - sizeof(int)));

                cudaFree(K_device);

                #ifdef LIBBSC_OPENMP
                    omp_unset_lock(&cuda_lock);
                #endif

                cudaMemcpy(T_device + n, T_device - sizeof(int), sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(T, T_device, n + sizeof(int), cudaMemcpyDeviceToHost);

                if (cudaGetLastError() == cudaSuccess)
                {
                    index = *(int *)(T + n);
                }

                return index;
            }
            cudaFree(K_device);
        }
    }

    #ifdef LIBBSC_OPENMP
        omp_unset_lock(&cuda_lock);
    #endif

    return index;
}

int bsc_st8_encode_cuda(unsigned char * T, unsigned char * T_device, int n, int num_blocks)
{
    #ifdef LIBBSC_OPENMP
        omp_set_lock(&cuda_lock);
    #endif

    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned char * V_device = NULL;
        if (cudaMalloc((void **)&V_device, (n + CUDA_DEVICE_PADDING) * sizeof(unsigned char)) == cudaSuccess)
        {
            unsigned long long * K_device = NULL;
            if (cudaMalloc((void **)&K_device, (n + CUDA_DEVICE_PADDING) * sizeof(unsigned long long)) == cudaSuccess)
            {
                index = LIBBSC_GPU_ERROR;

                bsc_st8_encode_cuda_presort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK>>>(T_device, K_device, V_device, n);

                cudaError_t status = cudaSuccess;
                {
                    b40c::util::PingPongStorage<unsigned long long, unsigned char> storage(K_device, V_device);

                    b40c::radix_sort::Enactor enactor;

                    status = enactor.Sort<b40c::radix_sort::LARGE_SIZE>(storage, n);

                    if (status == cudaErrorMemoryAllocation) index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;

                    if (status == cudaSuccess && storage.selector == 1)
                    {
                        cudaMemcpy(K_device, storage.d_keys[1]  , n * sizeof(unsigned long long), cudaMemcpyDeviceToDevice);
                        cudaMemcpy(V_device, storage.d_values[1], n * sizeof(unsigned char     ), cudaMemcpyDeviceToDevice);
                    }

                    if (storage.d_keys[1]   != NULL) cudaFree(storage.d_keys[1]  );
                    if (storage.d_values[1] != NULL) cudaFree(storage.d_values[1]);
                }

                if (status == cudaSuccess)
                {
                    unsigned long long lookup;
                    {
                        unsigned int lo = (T[4] << 24) | (T[5] << 16) | (T[6] << 8) | T[7];
                        unsigned int hi = (T[0] << 24) | (T[1] << 16) | (T[2] << 8) | T[3];

                        lookup = (((unsigned long long)hi) << 32) | ((unsigned long long)lo);

                        cudaMemcpy(V_device + n, &n, sizeof(int), cudaMemcpyHostToDevice);
                    }

                    bsc_st8_encode_cuda_postsort<<<num_blocks, CUDA_NUM_THREADS_IN_BLOCK>>>(K_device, n, lookup, (int *)(V_device + n));

                    cudaFree(K_device);

                    #ifdef LIBBSC_OPENMP
                        omp_unset_lock(&cuda_lock);
                    #endif

                    cudaMemcpy(T, V_device, n + sizeof(int), cudaMemcpyDeviceToHost);

                    cudaFree(V_device);

                    if (cudaGetLastError() == cudaSuccess)
                    {
                        index = *(int *)(T + n);
                    }

                    return index;
                }
                cudaFree(K_device);
            }
            cudaFree(V_device);
        }
    }

    #ifdef LIBBSC_OPENMP
        omp_unset_lock(&cuda_lock);
    #endif

    return index;
}

int bsc_st_encode_cuda(unsigned char * T, int n, int k, int features)
{
    if ((T == NULL) || (n < 0)) return LIBBSC_BAD_PARAMETER;
    if ((k < 5) || (k > 8))     return LIBBSC_BAD_PARAMETER;
    if (n <= 1)                 return 0;

    int num_blocks = 1;
    {
        cudaDeviceProp deviceProperties;
        {
            int deviceId; if (cudaGetDevice(&deviceId) != cudaSuccess || cudaGetDeviceProperties(&deviceProperties, deviceId) != cudaSuccess)
            {
                return LIBBSC_GPU_NOT_SUPPORTED;
            }
        }

        if (deviceProperties.major * 10 + deviceProperties.minor <= 10) return LIBBSC_GPU_NOT_SUPPORTED;
        num_blocks = CUDA_CTA_OCCUPANCY(deviceProperties.major * 100 + deviceProperties.minor * 10) * deviceProperties.multiProcessorCount;

        if (num_blocks > ((n + CUDA_NUM_THREADS_IN_BLOCK - 1) / CUDA_NUM_THREADS_IN_BLOCK)) num_blocks = (n + CUDA_NUM_THREADS_IN_BLOCK - 1) / CUDA_NUM_THREADS_IN_BLOCK;
        if (num_blocks <= 0) num_blocks = 1;
    }

    int index = LIBBSC_GPU_NOT_ENOUGH_MEMORY;
    {
        unsigned char * T_device = NULL;
        if (cudaMalloc((void **)&T_device, n + 2 * CUDA_DEVICE_PADDING) == cudaSuccess)
        {
            cudaMemcpy(T_device + CUDA_DEVICE_PADDING    , T                             , n                  , cudaMemcpyHostToDevice  );
            cudaMemcpy(T_device + CUDA_DEVICE_PADDING + n, T_device + CUDA_DEVICE_PADDING, CUDA_DEVICE_PADDING, cudaMemcpyDeviceToDevice);
            cudaMemcpy(T_device                          , T_device + n                  , CUDA_DEVICE_PADDING, cudaMemcpyDeviceToDevice);

            if (k >= 5 && k <= 7) index = bsc_st567_encode_cuda(T, T_device + CUDA_DEVICE_PADDING, n, num_blocks, k);
            if (k == 8)           index = bsc_st8_encode_cuda  (T, T_device + CUDA_DEVICE_PADDING, n, num_blocks   );

            cudaFree(T_device);
        }
    }

    return index;
}

#endif

/*-----------------------------------------------------------*/
/* End                                                 st.cu */
/*-----------------------------------------------------------*/
