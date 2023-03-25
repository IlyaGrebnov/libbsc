/*--

This file is a part of libcubwt, a library for CUDA accelerated
burrows wheeler transform construction.

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

#include "libcubwt.cuh"

#if defined(_MSC_VER) && defined(__INTELLISENSE__)
    #define __launch_bounds__(block_size) /* */
    #define __CUDACC__

    #include <vector_functions.h>
    #include <device_functions.h>
    #include <device_launch_parameters.h>
#endif

#include <cub/cub.cuh>
#include <cuda.h>

#include <utility>

#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#else
    #define RESTRICT /* */
#endif

#ifndef __CUDA_ARCH__
    #define CUDA_DEVICE_ARCH                0
#else
    #define CUDA_DEVICE_ARCH                __CUDA_ARCH__
#endif

#if CUDA_DEVICE_ARCH == 750
    #define CUDA_SM_THREADS                 (1024)
#elif CUDA_DEVICE_ARCH == 860 || CUDA_DEVICE_ARCH == 870 || CUDA_DEVICE_ARCH == 890
    #define CUDA_SM_THREADS                 (1536)
#else
    #define CUDA_SM_THREADS                 (2048)
#endif

#if CUDA_DEVICE_ARCH == 860 || CUDA_DEVICE_ARCH == 870 || CUDA_DEVICE_ARCH == 890
    #define CUDA_BLOCK_THREADS              (768)
#else
    #define CUDA_BLOCK_THREADS              (512)
#endif

#define CUDA_WARP_THREADS                   (32)
#define CUDA_DEVICE_PADDING                 (12 * 768)

typedef struct LIBCUBWT_DEVICE_STORAGE
{
    void *          device_rsort_temp_storage;
    size_t          device_rsort_temp_storage_size;

    void *          device_ssort_temp_storage;
    size_t          device_ssort_temp_storage_size;

    uint8_t *       device_T;
    uint8_t *       device_heads;

    uint32_t *      device_SA;
    uint32_t *      device_ISA;

    uint32_t *      device_keys;
    uint32_t *      device_offsets;

    uint32_t *      device_temp_SA;
    uint32_t *      device_temp_ISA;
    uint32_t *      device_temp_keys;

    uint64_t *      device_SA_temp_SA;
    uint64_t *      device_keys_temp_keys;
    uint64_t *      device_offsets_ISA;

    uint4 *         device_descriptors_large;
    uint4 *         device_descriptors_copy;
    uint2 *         device_descriptors_small;

    void *          device_storage;
    int32_t         device_L2_cache_bits;

    void *          host_pinned_storage;
    size_t          host_pinned_storage_size;

    int64_t         max_length;
    uint32_t        num_unsorted_segments;
    uint32_t        num_unsorted_suffixes;
    
    uint32_t        cuda_block_threads;
    cudaStream_t    cuda_stream;
} LIBCUBWT_DEVICE_STORAGE;

static int64_t libcubwt_get_error_code(cudaError_t status)
{
    return
        status == cudaErrorMemoryAllocation     ? LIBCUBWT_GPU_NOT_ENOUGH_MEMORY :
        status == cudaErrorDevicesUnavailable   ? LIBCUBWT_GPU_NOT_SUPPORTED :
        status == cudaErrorNoDevice             ? LIBCUBWT_GPU_NOT_SUPPORTED :
        LIBCUBWT_GPU_ERROR;
}

static cudaError_t libcubwt_cuda_safe_call(const char * filename, int32_t line, cudaError_t result, cudaError_t status = cudaSuccess)
{
#if !defined(NDEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "%s(%d): libcubwt_cuda_safe_call failed %d: '%s'.\n", filename, line, result, cudaGetErrorString(result));
        fflush(stderr);
    }
#else
    (void)(filename); (void)(line);
#endif

    return result != cudaSuccess ? result : status;
}

template <typename T>
__device__ __forceinline__ T libcubwt_warp_reduce_sum(T value) 
{
#if CUDA_DEVICE_ARCH >= 800 && !defined(__CUDA__)
    return __reduce_add_sync((uint32_t)-1, value);
#else

    #pragma unroll
    for (uint32_t mask = CUDA_WARP_THREADS / 2; mask > 0; mask >>= 1)
    {
        value = cub::Sum()(value, __shfl_xor_sync((uint32_t)-1, value, mask, CUDA_WARP_THREADS));
    }

    return value;
#endif
}

template <typename T>
__device__ __forceinline__ T libcubwt_warp_reduce_max(T value) 
{
#if CUDA_DEVICE_ARCH >= 800 && !defined(__CUDA__)
    return __reduce_max_sync((uint32_t)-1, value);
#else

    #pragma unroll
    for (uint32_t mask = CUDA_WARP_THREADS / 2; mask > 0; mask >>= 1)
    {
        value = cub::Max()(value, __shfl_xor_sync((uint32_t)-1, value, mask, CUDA_WARP_THREADS));
    }

    return value;
#endif
}

template <typename T>
__device__ __forceinline__ void libcubwt_delay_or_prevent_hoisting(T delay)
{
#if CUDA_DEVICE_ARCH >= 700
    __nanosleep(delay);
#else
    __threadfence_block(); (void)(delay);
#endif
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_gather_values_uint32_kernel(const uint32_t * device_idx, const uint32_t * RESTRICT device_src, uint32_t * device_dst, uint32_t m)
{
    const uint32_t block_index = blockIdx.x * CUDA_BLOCK_THREADS * 4;

    device_idx += block_index; device_dst += block_index; m -= block_index;

    if (m >= CUDA_BLOCK_THREADS * 4)
    {
        const uint4 indexes = *(uint4 *)(device_idx + threadIdx.x * 4);

        *(uint4 *)(device_dst + threadIdx.x * 4) = make_uint4(
            __ldg(device_src + indexes.x),
            __ldg(device_src + indexes.y),
            __ldg(device_src + indexes.z),
            __ldg(device_src + indexes.w));
    }
    else
    {
        for (uint32_t thread_index = threadIdx.x; thread_index < m; thread_index += CUDA_BLOCK_THREADS)
        {
            device_dst[thread_index] = __ldg(device_src + device_idx[thread_index]);
        }
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_scatter_values_uint32_kernel(const uint32_t * RESTRICT device_idx, const uint32_t * RESTRICT device_src, uint32_t * RESTRICT device_dst, uint32_t m)
{
    const uint32_t block_index = blockIdx.x * CUDA_BLOCK_THREADS * 4;

    device_idx += block_index; device_src += block_index; m -= block_index;

    if (m >= CUDA_BLOCK_THREADS * 4)
    {
        const uint4 indexes = __ldg((uint4 *)(device_idx + threadIdx.x * 4));
        const uint4 values  = __ldg((uint4 *)(device_src + threadIdx.x * 4));

        device_dst[indexes.x] = values.x;
        device_dst[indexes.y] = values.y;
        device_dst[indexes.z] = values.z;
        device_dst[indexes.w] = values.w;
    }
    else
    {
        for (uint32_t thread_index = threadIdx.x; thread_index < m; thread_index += CUDA_BLOCK_THREADS)
        {
            device_dst[__ldg(device_idx + thread_index)] = __ldg(device_src + thread_index);
        }
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_permute_block_values_uint32_kernel(const uint32_t * RESTRICT device_idx, const uint32_t * RESTRICT device_src, uint32_t * RESTRICT device_dst, uint32_t n)
{
    __shared__ __align__(32) uint32_t cache[16 * CUDA_BLOCK_THREADS];

    const uint32_t block_index = blockIdx.x * CUDA_BLOCK_THREADS * 16;

    device_idx += block_index; device_src += block_index; device_dst += block_index; n -= block_index;

    if (n >= CUDA_BLOCK_THREADS * 16)
    {
        {
            const uint32_t * RESTRICT thread_idx   = device_idx + threadIdx.x * 4;
            const uint32_t * RESTRICT thread_src   = device_src + threadIdx.x * 4;
                  uint32_t * RESTRICT thread_cache = cache - block_index;

            #pragma unroll
            for (uint32_t round = 0; round < 4; round += 1)
            {
                const uint4 indexes = __ldg((uint4 *)(thread_idx));
                const uint4 values  = __ldg((uint4 *)(thread_src));

                thread_cache[indexes.x] = values.x;
                thread_cache[indexes.y] = values.y;
                thread_cache[indexes.z] = values.z;
                thread_cache[indexes.w] = values.w;

                thread_idx += 4 * CUDA_BLOCK_THREADS; thread_src += 4 * CUDA_BLOCK_THREADS;
            }
        }

        __syncthreads();

        {
            const uint32_t * RESTRICT thread_cache = cache      + threadIdx.x * 4;
                  uint32_t * RESTRICT thread_dst   = device_dst + threadIdx.x * 4;

            #pragma unroll
            for (uint32_t round = 0; round < 4; round += 1)
            {
                *(uint4 *)(thread_dst) = *(uint4 *)(thread_cache);

                thread_cache += 4 * CUDA_BLOCK_THREADS; thread_dst += 4 * CUDA_BLOCK_THREADS;
            }
        }
    }
    else
    {
        {
            uint32_t * RESTRICT thread_cache = cache - block_index;

            for (uint32_t thread_index = threadIdx.x; thread_index < n; thread_index += CUDA_BLOCK_THREADS)
            {
                thread_cache[__ldg(device_idx + thread_index)] = __ldg(device_src + thread_index);
            }
        }

        __syncthreads();

        {
            for (uint32_t thread_index = threadIdx.x; thread_index < n; thread_index += CUDA_BLOCK_THREADS)
            {
                device_dst[thread_index] = cache[thread_index];
            }
        }
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_scatter_values_uint64_kernel(const uint32_t * RESTRICT device_idx, const uint64_t * RESTRICT device_src, uint64_t * RESTRICT device_dst, uint32_t m)
{
    const uint32_t block_index = blockIdx.x * CUDA_BLOCK_THREADS * 2;

    device_idx += block_index; device_src += block_index; m -= block_index;

    if (m >= CUDA_BLOCK_THREADS * 2)
    {
        const uint2      indexes = __ldg((uint2      *)(device_idx + threadIdx.x * 2));
        const ulonglong2 values  = __ldg((ulonglong2 *)(device_src + threadIdx.x * 2));

        device_dst[indexes.x] = values.x;
        device_dst[indexes.y] = values.y;
    }
    else
    {
        for (uint32_t thread_index = threadIdx.x; thread_index < m; thread_index += CUDA_BLOCK_THREADS)
        {
            device_dst[__ldg(device_idx + thread_index)] = __ldg(device_src + thread_index);
        }
    }
}

static cudaError_t libcubwt_gather_scatter_values_uint32(LIBCUBWT_DEVICE_STORAGE * storage, uint32_t * device_src_idx, uint32_t * device_src, uint32_t * device_dst_idx, uint32_t * device_dst, int64_t m, int64_t n, uint32_t * device_temp1, uint32_t * device_temp2)
{
    cudaError_t status = cudaSuccess;

    cub::DoubleBuffer<uint32_t> db_src_index_value(device_src_idx, device_temp1);
    cub::DoubleBuffer<uint32_t> db_dst_index(device_dst_idx, device_temp2);

    int32_t sort_end_bit        = 0; while ((n - 1) >= ((int64_t)1 << sort_end_bit)) { sort_end_bit += 1; }
    int32_t sort_aligned_bits   = (sort_end_bit > storage->device_L2_cache_bits - 2) ? (sort_end_bit - storage->device_L2_cache_bits + 2 + 7) & (-8) : 0;
    int32_t sort_start_bit      = std::max(0, sort_end_bit - sort_aligned_bits);

    if (sort_start_bit < sort_end_bit)
    {
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
            storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
            db_src_index_value, db_dst_index,
            (uint32_t)m,
            sort_start_bit, sort_end_bit,
            storage->cuda_stream));
    }

    if (status == cudaSuccess)
    {
        int64_t n_gather_scatter_blocks = (m + storage->cuda_block_threads * 4 - 1) / (storage->cuda_block_threads * 4);

        libcubwt_gather_values_uint32_kernel<<<(uint32_t)n_gather_scatter_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_src_index_value.Current(), device_src, db_src_index_value.Current(), (uint32_t)m);

        if (sort_start_bit < sort_end_bit)
        {
            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
                storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
                db_dst_index, db_src_index_value,
                (uint32_t)m,
                sort_start_bit, sort_end_bit,
                storage->cuda_stream));
        }

        if (status == cudaSuccess)
        {
            libcubwt_scatter_values_uint32_kernel<<<(uint32_t)n_gather_scatter_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_dst_index.Current(), db_src_index_value.Current(), device_dst, (uint32_t)m);
        }
    }

    return status;
}

static cudaError_t libcubwt_scatter_values_uint32(LIBCUBWT_DEVICE_STORAGE * storage, uint32_t * device_idx, uint32_t * device_src, uint32_t * device_dst, int64_t m, int64_t n, uint32_t * device_temp1, uint32_t * device_temp2)
{
    cudaError_t status = cudaSuccess;

    cub::DoubleBuffer<uint32_t> db_index(device_idx, device_temp1);
    cub::DoubleBuffer<uint32_t> db_value(device_src, device_temp2);

    int32_t sort_end_bit        = 0; while ((n - 1) >= ((int64_t)1 << sort_end_bit)) { sort_end_bit += 1; }
    int32_t sort_aligned_bits   = (sort_end_bit > storage->device_L2_cache_bits - 2) ? (sort_end_bit - storage->device_L2_cache_bits + 2 + 7) & (-8) : 0;
    int32_t sort_start_bit      = std::max(0, sort_end_bit - sort_aligned_bits);

    if (sort_start_bit < sort_end_bit)
    {
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
            storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
            db_index, db_value,
            (uint32_t)m,
            sort_start_bit, sort_end_bit,
            storage->cuda_stream));
    }

    if (status == cudaSuccess)
    {
        int64_t n_scatter_blocks = (m + storage->cuda_block_threads * 4 - 1) / (storage->cuda_block_threads * 4);

        libcubwt_scatter_values_uint32_kernel<<<(uint32_t)n_scatter_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_index.Current(), db_value.Current(), device_dst, (uint32_t)m);
    }

    return status;
}

static cudaError_t libcubwt_permute_values_uint32(LIBCUBWT_DEVICE_STORAGE * storage, uint32_t * device_idx, uint32_t * device_src, uint32_t * device_dst, int64_t n, uint32_t * device_temp1, uint32_t * device_temp2)
{
    cudaError_t status = cudaSuccess;

    cub::DoubleBuffer<uint32_t> db_index(device_idx, device_temp1);
    cub::DoubleBuffer<uint32_t> db_value(device_src, device_temp2);

    int32_t sort_end_bit        = 0; while ((n - 1) >= ((int64_t)1 << sort_end_bit)) { sort_end_bit += 1; }
    int32_t sort_aligned_bits   = (sort_end_bit > storage->device_L2_cache_bits - 2) ? (sort_end_bit - storage->device_L2_cache_bits + 2 + 7) & (-8) : 0;
    int32_t sort_start_bit      = std::max(0, sort_end_bit - sort_aligned_bits);

    if (sort_start_bit < sort_end_bit)
    {
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
            storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
            db_index, db_value,
            (uint32_t)n,
            sort_start_bit, sort_end_bit,
            storage->cuda_stream));
    }

    if (status == cudaSuccess)
    {
        if (((storage->cuda_block_threads * 16) % ((int64_t)1 << sort_start_bit)) == 0)
        {
            int64_t n_permute_blocks = (n + storage->cuda_block_threads * 16 - 1) / (storage->cuda_block_threads * 16);

            libcubwt_permute_block_values_uint32_kernel<<<(uint32_t)n_permute_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_index.Current(), db_value.Current(), device_dst, (uint32_t)n);
        }
        else
        {
            int64_t n_scatter_blocks = (n + storage->cuda_block_threads * 4 - 1) / (storage->cuda_block_threads * 4);

            libcubwt_scatter_values_uint32_kernel<<<(uint32_t)n_scatter_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_index.Current(), db_value.Current(), device_dst, (uint32_t)n);
        }
    }

    return status;
}

static cudaError_t libcubwt_scatter_values_uint64(LIBCUBWT_DEVICE_STORAGE * storage, cub::DoubleBuffer<uint32_t> & db_index, cub::DoubleBuffer<uint64_t> & db_value, int64_t m, int64_t n, int64_t k = 0)
{
    cudaError_t status = cudaSuccess;

    int32_t sort_end_bit        = 0; while ((n - 1) >= ((int64_t)1 << sort_end_bit)) { sort_end_bit += 1; }
    int32_t sort_aligned_bits   = (sort_end_bit > storage->device_L2_cache_bits - 3) ? (sort_end_bit - storage->device_L2_cache_bits + 3 + 7) & (-8) : 0;
    int32_t sort_start_bit      = std::max(0, sort_end_bit - sort_aligned_bits);

    if (sort_start_bit < sort_end_bit)
    {
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
            storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
            db_index, db_value,
            (uint32_t)m,
            sort_start_bit, sort_end_bit,
            storage->cuda_stream));
    }

    if (status == cudaSuccess)
    {
        int64_t n_scatter_blocks = (m + storage->cuda_block_threads * 2 - 1) / (storage->cuda_block_threads * 2);

        libcubwt_scatter_values_uint64_kernel<<<(uint32_t)n_scatter_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(db_index.Current(), db_value.Current(), db_value.Alternate() - k, (uint32_t)m);

        db_index.selector ^= 1;
        db_value.selector ^= 1;
    }

    return status;
}

template <bool extra_sentinel_bits>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_initialize_device_arrays_kernel(const uint8_t * RESTRICT device_T, uint32_t * RESTRICT device_SA, uint64_t * RESTRICT device_keys)
{
    __shared__ __align__(32) uint4 prefixes[4 * CUDA_BLOCK_THREADS];

    {
        device_T += blockIdx.x * CUDA_BLOCK_THREADS * 12 + threadIdx.x * 16;
        if (threadIdx.x < (12 * CUDA_BLOCK_THREADS + 8 + 15) / 16) { prefixes[threadIdx.x] = __ldg((uint4 *)device_T); }

        __syncthreads();
    }

    {
        uint32_t * RESTRICT thread_cache    = ((uint32_t *)prefixes) + threadIdx.x * 3;
        uint4 *    RESTRICT thread_prefixes = ((uint4 *   )prefixes) + threadIdx.x * 4;

        const uint32_t b0 = thread_cache[0];
        const uint32_t b1 = thread_cache[1];
        const uint32_t b2 = thread_cache[2];
        const uint32_t b3 = thread_cache[3];
        const uint32_t b4 = thread_cache[4];

        __syncthreads();

        thread_prefixes[0] = make_uint4
        (
            __byte_perm(b1, b2, 0x1234) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b0, b1, 0x1234),
            __byte_perm(b1, b2, 0x2345) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b0, b1, 0x2345)
        );

        thread_prefixes[1] = make_uint4
        (
            __byte_perm(b2, b3, 0x0123) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b1, b2, 0x0123),
            __byte_perm(b2, b3, 0x1234) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b1, b2, 0x1234)
        );

        thread_prefixes[2] = make_uint4
        (
            __byte_perm(b2, b3, 0x3456) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b1, b2, 0x3456),
            __byte_perm(b3, b4, 0x0123) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b2, b3, 0x0123)
        );

        thread_prefixes[3] = make_uint4
        (
            __byte_perm(b3, b4, 0x2345) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b2, b3, 0x2345),
            __byte_perm(b3, b4, 0x3456) | (extra_sentinel_bits ? (uint32_t)7 : (uint32_t)1), __byte_perm(b2, b3, 0x3456)
        );

        __syncwarp();
    }

    {
        const uint32_t block_index = blockIdx.x * CUDA_BLOCK_THREADS * 8;

        {
            uint32_t thread_index = block_index + threadIdx.x * 4; device_SA += thread_index;
            ((uint4 *)device_SA)[0] = make_uint4(thread_index + 0, thread_index + 1, thread_index + 2, thread_index + 3);

            thread_index += CUDA_BLOCK_THREADS * 4; device_SA += CUDA_BLOCK_THREADS * 4;
            ((uint4 *)device_SA)[0] = make_uint4(thread_index + 0, thread_index + 1, thread_index + 2, thread_index + 3);
        }

        {
            device_keys += block_index;

            uint4 * RESTRICT thread_prefixes = (uint4 *)prefixes    + ((threadIdx.x / CUDA_WARP_THREADS) * CUDA_WARP_THREADS * 4) + (threadIdx.x % CUDA_WARP_THREADS);
            uint4 * RESTRICT thread_keys     = (uint4 *)device_keys + ((threadIdx.x / CUDA_WARP_THREADS) * CUDA_WARP_THREADS * 4) + (threadIdx.x % CUDA_WARP_THREADS);

            thread_keys[0] = thread_prefixes[0]; thread_keys += CUDA_WARP_THREADS; thread_prefixes += CUDA_WARP_THREADS;
            thread_keys[0] = thread_prefixes[0]; thread_keys += CUDA_WARP_THREADS; thread_prefixes += CUDA_WARP_THREADS;
            thread_keys[0] = thread_prefixes[0]; thread_keys += CUDA_WARP_THREADS; thread_prefixes += CUDA_WARP_THREADS;
            thread_keys[0] = thread_prefixes[0];
        }
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, 1)
static void libcubwt_set_sentinel_values_kernel(uint8_t * RESTRICT device_T_end, uint64_t * RESTRICT device_keys_end, uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t k4, uint64_t k5, uint64_t k6, uint64_t k7)
{
    device_T_end[0] = 0;
    device_T_end[1] = 0;
    device_T_end[2] = 0;

    device_keys_end[-8] = k0;
    device_keys_end[-7] = k1;
    device_keys_end[-6] = k2;
    device_keys_end[-5] = k3;
    device_keys_end[-4] = k4;
    device_keys_end[-3] = k5;
    device_keys_end[-2] = k6;
    device_keys_end[-1] = k7;
}

static cudaError_t libcubwt_initialize_device_arrays(LIBCUBWT_DEVICE_STORAGE * storage, const uint8_t * T, int64_t reduced_n, int64_t expanded_n, int64_t input_n)
{
    cudaError_t status = cudaSuccess;

    if ((status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->device_T, T, (size_t)input_n, cudaMemcpyHostToDevice, storage->cuda_stream))) == cudaSuccess)
    {
        int64_t n_initialize_blocks = 1 + (expanded_n / (storage->cuda_block_threads * 12));

        bool extra_sentinel_bits = (expanded_n - input_n >= 2) || (T[input_n - 1] == 0);
        if (extra_sentinel_bits)
        {
            libcubwt_initialize_device_arrays_kernel<true><<<(uint32_t)n_initialize_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(storage->device_T, storage->device_SA, storage->device_keys_temp_keys);
        }
        else
        {
            libcubwt_initialize_device_arrays_kernel<false><<<(uint32_t)n_initialize_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(storage->device_T, storage->device_SA, storage->device_keys_temp_keys);
        }

        {
            uint64_t c0 = (expanded_n - 11 < input_n) ? T[expanded_n - 11] : (uint64_t)0;
            uint64_t c1 = (expanded_n - 10 < input_n) ? T[expanded_n - 10] : (uint64_t)0;
            uint64_t c2 = (expanded_n -  9 < input_n) ? T[expanded_n -  9] : (uint64_t)0;
            uint64_t c3 = (expanded_n -  8 < input_n) ? T[expanded_n -  8] : (uint64_t)0;
            uint64_t c4 = (expanded_n -  7 < input_n) ? T[expanded_n -  7] : (uint64_t)0;
            uint64_t c5 = (expanded_n -  6 < input_n) ? T[expanded_n -  6] : (uint64_t)0;
            uint64_t c6 = (expanded_n -  5 < input_n) ? T[expanded_n -  5] : (uint64_t)0;
            uint64_t c7 = (expanded_n -  4 < input_n) ? T[expanded_n -  4] : (uint64_t)0;
            uint64_t c8 = (expanded_n -  3 < input_n) ? T[expanded_n -  3] : (uint64_t)0;
            uint64_t c9 = (expanded_n -  2 < input_n) ? T[expanded_n -  2] : (uint64_t)0;
            uint64_t ca = (expanded_n -  1 < input_n) ? T[expanded_n -  1] : (uint64_t)0;

            uint64_t k0 = (c0 << 56) | (c1 << 48) | (c2 << 40) | (c3 << 32) | (c4 << 24) | (c5 << 16) | (c6 << 8) | (c7 << 0) | (extra_sentinel_bits ? 7 : 1);
            uint64_t k1 = (c1 << 56) | (c2 << 48) | (c3 << 40) | (c4 << 32) | (c5 << 24) | (c6 << 16) | (c7 << 8) | (c8 << 0) | (extra_sentinel_bits ? 7 : 1);

            uint64_t k2 = (c3 << 56) | (c4 << 48) | (c5 << 40) | (c6 << 32) | (c7 << 24) | (c8 << 16) | (c9 << 8) | (ca << 0) | (extra_sentinel_bits ? 7 : 0);
            uint64_t k3 = (c4 << 56) | (c5 << 48) | (c6 << 40) | (c7 << 32) | (c8 << 24) | (c9 << 16) | (ca << 8) | (extra_sentinel_bits ? 6 : 0);
            
            uint64_t k4 = (c6 << 56) | (c7 << 48) | (c8 << 40) | (c9 << 32) | (ca << 24) | (extra_sentinel_bits ? 4 : 0);
            uint64_t k5 = (c7 << 56) | (c8 << 48) | (c9 << 40) | (ca << 32) | (extra_sentinel_bits ? 3 : 0);

            uint64_t k6 = (c9 << 56) | (ca << 48) | (extra_sentinel_bits ? 1 : 0);
            uint64_t k7 = (ca << 56);

            libcubwt_set_sentinel_values_kernel<<<1, 1, 0, storage->cuda_stream>>>(storage->device_T + input_n, storage->device_keys_temp_keys + reduced_n, k0, k1, k2, k3, k4, k5, k6, k7);
        }

        storage->num_unsorted_segments      = (uint32_t)1;
        storage->num_unsorted_suffixes      = (uint32_t)reduced_n;
    }

    return status;
}

static cudaError_t libcubwt_sort_suffixes_by_prefix(LIBCUBWT_DEVICE_STORAGE * storage, int64_t n)
{
    cub::DoubleBuffer<uint64_t> db_keys(storage->device_keys_temp_keys, storage->device_offsets_ISA);
    cub::DoubleBuffer<uint32_t> db_SA(storage->device_SA, storage->device_temp_SA);

    cudaError_t status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
        storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
        db_keys, db_SA,
        (uint32_t)n,
        0, 64,
        storage->cuda_stream));

    if (db_keys.selector) 
    { 
        std::swap(storage->device_keys_temp_keys, storage->device_offsets_ISA);

        std::swap(storage->device_keys, storage->device_offsets); 
        std::swap(storage->device_temp_keys, storage->device_ISA);
    }

    if (db_SA.selector)
    {
        std::swap(storage->device_SA, storage->device_temp_SA);
    }

    return status;
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_rank_and_segment_suffixes_initialization_kernel(uint32_t * RESTRICT device_SA, uint64_t * RESTRICT device_keys, uint8_t * RESTRICT device_heads, uint4 * RESTRICT device_descriptors_large, uint2 * RESTRICT device_descriptors_small, uint32_t n)
{
    const uint32_t thread_index = blockIdx.x * CUDA_BLOCK_THREADS + threadIdx.x;

    device_descriptors_large += thread_index;
    device_descriptors_small += thread_index;

    device_descriptors_large[0] = make_uint4(0, 0, 0, 0);
    device_descriptors_small[0] = make_uint2(0, 0);

    if (blockIdx.x == 0)
    {
        if (threadIdx.x < CUDA_WARP_THREADS)
        {
            device_descriptors_large[-CUDA_WARP_THREADS] = make_uint4((uint32_t)-1, 0, 0, 0);
            device_descriptors_small[-CUDA_WARP_THREADS] = make_uint2((uint32_t)-1, 0);
        }

        {
            uint64_t key = (threadIdx.x % 2 == 0) ? 0 : (uint64_t)-1;

            device_SA += threadIdx.x; device_keys += threadIdx.x; device_heads += threadIdx.x;

            if (threadIdx.x < 2)
            {
                device_keys [-2] = key;
                device_heads[-2] = 1;
            }

            device_SA += n; device_keys += n; device_heads += n;

            device_SA   [0 * CUDA_BLOCK_THREADS] = n + threadIdx.x + 0 * CUDA_BLOCK_THREADS;
            device_SA   [1 * CUDA_BLOCK_THREADS] = n + threadIdx.x + 1 * CUDA_BLOCK_THREADS;
            device_SA   [2 * CUDA_BLOCK_THREADS] = n + threadIdx.x + 2 * CUDA_BLOCK_THREADS;
            device_SA   [3 * CUDA_BLOCK_THREADS] = n + threadIdx.x + 3 * CUDA_BLOCK_THREADS;

            device_keys [0 * CUDA_BLOCK_THREADS] = key;
            device_keys [1 * CUDA_BLOCK_THREADS] = key;
            device_keys [2 * CUDA_BLOCK_THREADS] = key;
            device_keys [3 * CUDA_BLOCK_THREADS] = key;

            device_heads[0 * CUDA_BLOCK_THREADS] = 1;
            device_heads[1 * CUDA_BLOCK_THREADS] = 1;
            device_heads[2 * CUDA_BLOCK_THREADS] = 1;
            device_heads[3 * CUDA_BLOCK_THREADS] = 1;
        }
    }
}

template <bool scatter_ranks_directly>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_rank_and_segment_suffixes_initiatory_kernel(
    const uint32_t *    RESTRICT device_SA,
    const uint64_t *    RESTRICT device_keys,
    uint8_t *           RESTRICT device_heads,
    uint32_t *          RESTRICT device_ISA,
    uint32_t *          RESTRICT device_offsets_begin,
    uint32_t *          RESTRICT device_offsets_end,
    uint4 *             RESTRICT device_descriptors
)
{
    __shared__ __align__(32) uint2 warp_state[1 + CUDA_WARP_THREADS];

    uint32_t    thread_exclusive_suffix_rank;
    uint32_t    thread_suffix_rank[4];

    uint32_t    thread_exclusive_segment_index;
    uint32_t    thread_segment_index[4];

    {
        __shared__ __align__(32) ulonglong2 cache[1 + 2 * CUDA_BLOCK_THREADS];

        {
            device_keys += blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 2;

            if (threadIdx.x == 0) { cache[0] = __ldg((ulonglong2 *)(device_keys - 2)); }
            cache[1 + threadIdx.x + 0 * CUDA_BLOCK_THREADS] = __ldg((ulonglong2 *)(device_keys + 0 * CUDA_BLOCK_THREADS));
            cache[1 + threadIdx.x + 1 * CUDA_BLOCK_THREADS] = __ldg((ulonglong2 *)(device_keys + 2 * CUDA_BLOCK_THREADS));
        }

        __syncthreads();

        {
            const uint32_t block_index  = blockIdx.x * CUDA_BLOCK_THREADS * 4;
            const uint32_t thread_index = block_index + threadIdx.x * 4;

            ulonglong2 key_a = cache[2 * threadIdx.x + 0];
            ulonglong2 key_b = cache[2 * threadIdx.x + 1];
            ulonglong2 key_c = cache[2 * threadIdx.x + 2];

            uchar4 thread_new_heads = make_uchar4(
                (key_a.y != key_b.x) ? (uint8_t)1 : (uint8_t)0,
                (key_b.x != key_b.y) ? (uint8_t)1 : (uint8_t)0,
                (key_b.y != key_c.x) ? (uint8_t)1 : (uint8_t)0,
                (key_c.x != key_c.y) ? (uint8_t)1 : (uint8_t)0);

            *(uchar4 *)(device_heads + thread_index) = thread_new_heads;

            thread_suffix_rank[0] = (thread_new_heads.x != 0) ? (thread_index + 0) : 0;
            thread_suffix_rank[1] = (thread_new_heads.y != 0) ? (thread_index + 1) : thread_suffix_rank[0];
            thread_suffix_rank[2] = (thread_new_heads.z != 0) ? (thread_index + 2) : thread_suffix_rank[1];
            thread_suffix_rank[3] = (thread_new_heads.w != 0) ? (thread_index + 3) : thread_suffix_rank[2];

            thread_segment_index[0] = ((thread_new_heads.x != 0) && (key_a.x == key_a.y));
            thread_segment_index[1] = thread_segment_index[0] + ((thread_new_heads.y != 0) && (thread_new_heads.x == 0));
            thread_segment_index[2] = thread_segment_index[1] + ((thread_new_heads.z != 0) && (thread_new_heads.y == 0));
            thread_segment_index[3] = thread_segment_index[2] + ((thread_new_heads.w != 0) && (thread_new_heads.z == 0));
        }
    }

    {
        uint32_t thread_inclusive_suffix_rank;
        uint32_t thread_inclusive_segment_index;

        typedef cub::WarpScan<uint32_t> WarpScan;

        __shared__ typename WarpScan::TempStorage warp_scan_storage[CUDA_WARP_THREADS];

        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_suffix_rank[3]  , thread_inclusive_suffix_rank  , thread_exclusive_suffix_rank  , (uint32_t)0, cub::Max());
        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_segment_index[3], thread_inclusive_segment_index, thread_exclusive_segment_index, (uint32_t)0, cub::Sum());

        if ((threadIdx.x % CUDA_WARP_THREADS) == (CUDA_WARP_THREADS - 1))
        {
            warp_state[threadIdx.x / CUDA_WARP_THREADS] = make_uint2(thread_inclusive_suffix_rank, thread_inclusive_segment_index);
        }

        __syncthreads();
    }

    {
        if (threadIdx.x < CUDA_WARP_THREADS)
        {
            uint32_t            block_exclusive_suffix_rank   = 0;
            uint32_t            block_exclusive_segment_index = 0;

            uint32_t            warp_inclusive_suffix_rank;
            uint32_t            warp_inclusive_segment_index;

            {
                typedef cub::WarpScan<uint32_t> WarpScan;

                __shared__ typename WarpScan::TempStorage warp_scan_storage;

                uint2 warp_inclusive_state = warp_state[threadIdx.x];

                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state.x, warp_inclusive_suffix_rank  , cub::Max());
                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state.y, warp_inclusive_segment_index, cub::Sum());
            }

            {
                const uint32_t descriptor_status_aggregate_not_ready        = 0;
                const uint32_t descriptor_status_partial_aggregate_ready    = 1;
                const uint32_t descriptor_status_full_aggregate_ready       = 4;

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint4(descriptor_status_partial_aggregate_ready, 0, warp_inclusive_suffix_rank, warp_inclusive_segment_index));
                }

                {
                    uint4 * RESTRICT descriptors_lookback = device_descriptors + blockIdx.x + threadIdx.x;

                    int32_t full_aggregate_lane, delay = 8;
                    do
                    {
                        descriptors_lookback -= CUDA_WARP_THREADS;

                        uint4 block_descriptor;
                        do
                        {
                            libcubwt_delay_or_prevent_hoisting(delay <<= 1);

                            block_descriptor = cub::ThreadLoad<cub::LOAD_CG>(descriptors_lookback);
                        } while (__any_sync((uint32_t)-1, block_descriptor.x == descriptor_status_aggregate_not_ready));

                        delay = 0;

                        {
                            full_aggregate_lane     = 31 - __clz((int32_t)__ballot_sync((uint32_t)-1, block_descriptor.x != descriptor_status_partial_aggregate_ready));
                            block_descriptor.z      = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.z : 0;
                            block_descriptor.w      = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.w : 0;
                        }

                        {
                            block_exclusive_suffix_rank      = cub::Max()(block_exclusive_suffix_rank  , libcubwt_warp_reduce_max(block_descriptor.z));
                            block_exclusive_segment_index    = cub::Sum()(block_exclusive_segment_index, libcubwt_warp_reduce_sum(block_descriptor.w));
                        }

                    } while (full_aggregate_lane == -1);

                    warp_inclusive_suffix_rank      = cub::Max()(warp_inclusive_suffix_rank  , block_exclusive_suffix_rank  );
                    warp_inclusive_segment_index    = cub::Sum()(warp_inclusive_segment_index, block_exclusive_segment_index);
                }

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint4(descriptor_status_full_aggregate_ready, 0, warp_inclusive_suffix_rank, warp_inclusive_segment_index));
                }
            }

            {
                if (threadIdx.x == 0)
                {
                    warp_state[0] = make_uint2(block_exclusive_suffix_rank, block_exclusive_segment_index);
                }

                warp_state[1 + threadIdx.x] = make_uint2(warp_inclusive_suffix_rank, warp_inclusive_segment_index);
            }
        }

        __syncthreads();
    }

    {
        uint2 warp_exclusive_state              = warp_state[threadIdx.x / CUDA_WARP_THREADS];
        
        thread_exclusive_suffix_rank            = cub::Max()(thread_exclusive_suffix_rank  , warp_exclusive_state.x);
        thread_exclusive_segment_index          = cub::Sum()(thread_exclusive_segment_index, warp_exclusive_state.y);

        thread_suffix_rank[0]                   = cub::Max()(thread_suffix_rank[0], thread_exclusive_suffix_rank);
        thread_suffix_rank[1]                   = cub::Max()(thread_suffix_rank[1], thread_exclusive_suffix_rank);
        thread_suffix_rank[2]                   = cub::Max()(thread_suffix_rank[2], thread_exclusive_suffix_rank);
        thread_suffix_rank[3]                   = cub::Max()(thread_suffix_rank[3], thread_exclusive_suffix_rank);

        thread_segment_index[0]                 = cub::Sum()(thread_segment_index[0], thread_exclusive_segment_index);
        thread_segment_index[1]                 = cub::Sum()(thread_segment_index[1], thread_exclusive_segment_index);
        thread_segment_index[2]                 = cub::Sum()(thread_segment_index[2], thread_exclusive_segment_index);
        thread_segment_index[3]                 = cub::Sum()(thread_segment_index[3], thread_exclusive_segment_index);

        const uint32_t thread_index             = blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;

        if (thread_exclusive_segment_index != thread_segment_index[0]) { device_offsets_begin[thread_segment_index[0]] = thread_exclusive_suffix_rank; device_offsets_end[thread_segment_index[0]] = thread_index + 0; }
        if (thread_segment_index[0]        != thread_segment_index[1]) { device_offsets_begin[thread_segment_index[1]] = thread_suffix_rank[0];        device_offsets_end[thread_segment_index[1]] = thread_index + 1; }
        if (thread_segment_index[1]        != thread_segment_index[2]) { device_offsets_begin[thread_segment_index[2]] = thread_suffix_rank[1];        device_offsets_end[thread_segment_index[2]] = thread_index + 2; }
        if (thread_segment_index[2]        != thread_segment_index[3]) { device_offsets_begin[thread_segment_index[3]] = thread_suffix_rank[2];        device_offsets_end[thread_segment_index[3]] = thread_index + 3; }

        if (scatter_ranks_directly)
        {
            const uint4 indexes                 = __ldg((uint4 *)(device_SA + thread_index));

            device_ISA[indexes.x]               = thread_suffix_rank[0];
            device_ISA[indexes.y]               = thread_suffix_rank[1];
            device_ISA[indexes.z]               = thread_suffix_rank[2];
            device_ISA[indexes.w]               = thread_suffix_rank[3];
        }
        else
        {
            *(uint4 *)(device_ISA + thread_index) = make_uint4(thread_suffix_rank[0], thread_suffix_rank[1], thread_suffix_rank[2], thread_suffix_rank[3]);
        }
    }
}

template <bool alternate_block_descriptor_statuses, bool scatter_ranks_directly>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_rank_and_segment_suffixes_incremental_kernel(
    const uint32_t *    RESTRICT device_SA,
    const uint32_t *    RESTRICT device_keys,
    uint8_t *           RESTRICT device_heads,
    uint32_t *          RESTRICT device_out_SA,
    uint32_t *          RESTRICT device_out_ISA,
    uint32_t *          RESTRICT device_offsets_begin,
    uint32_t *          RESTRICT device_offsets_end,
    uint4 *             RESTRICT device_descriptors,
    const uint4 *       RESTRICT device_descriptors_copy
)
{
    __shared__ __align__(32) uint4    warp_state1[1 + CUDA_WARP_THREADS];
    __shared__ __align__(32) uint32_t warp_state2[1 + CUDA_WARP_THREADS];

    uchar4      thread_old_heads;
    uint32_t    thread_exclusive_suffix_old_rank;

    uchar4      thread_new_heads;
    uint32_t    thread_exclusive_suffix_new_rank;

    uint32_t    thread_exclusive_segment_index;
    uint32_t    thread_segment_index[4];

    uint32_t    thread_exclusive_suffix_index;
    uint32_t    thread_suffix_index[4];

    {
        const uint32_t block_index  = blockIdx.x * CUDA_BLOCK_THREADS * 4;
        const uint32_t thread_index = block_index + threadIdx.x * 4;

        device_keys += thread_index; device_heads += thread_index;

        uint2 key_a                 = __ldg((uint2 *)(device_keys - 2));
        uint4 key_b                 = __ldg((uint4 *)(device_keys));
        thread_old_heads            = *(uchar4 *)(device_heads);

        thread_new_heads = make_uchar4(
            (key_a.y != key_b.x) ? (uint8_t)1 : (uint8_t)thread_old_heads.x,
            (key_b.x != key_b.y) ? (uint8_t)1 : (uint8_t)thread_old_heads.y,
            (key_b.y != key_b.z) ? (uint8_t)1 : (uint8_t)thread_old_heads.z,
            (key_b.z != key_b.w) ? (uint8_t)1 : (uint8_t)thread_old_heads.w);

        *(uchar4 *)(device_heads) = thread_new_heads;

        thread_exclusive_suffix_old_rank = (thread_old_heads.x != 0) ? (thread_index + 0) : 0;
        thread_exclusive_suffix_old_rank = (thread_old_heads.y != 0) ? (thread_index + 1) : thread_exclusive_suffix_old_rank;
        thread_exclusive_suffix_old_rank = (thread_old_heads.z != 0) ? (thread_index + 2) : thread_exclusive_suffix_old_rank;
        thread_exclusive_suffix_old_rank = (thread_old_heads.w != 0) ? (thread_index + 3) : thread_exclusive_suffix_old_rank;

        thread_exclusive_suffix_new_rank = (thread_new_heads.x != 0) ? (thread_index + 0) : 0;
        thread_exclusive_suffix_new_rank = (thread_new_heads.y != 0) ? (thread_index + 1) : thread_exclusive_suffix_new_rank;
        thread_exclusive_suffix_new_rank = (thread_new_heads.z != 0) ? (thread_index + 2) : thread_exclusive_suffix_new_rank;
        thread_exclusive_suffix_new_rank = (thread_new_heads.w != 0) ? (thread_index + 3) : thread_exclusive_suffix_new_rank;

        thread_segment_index[0] = ((thread_new_heads.x != 0) && (key_a.x == key_a.y) && (device_heads[-1] == 0));
        thread_segment_index[1] = thread_segment_index[0] + ((thread_new_heads.y != 0) && (thread_new_heads.x == 0));
        thread_segment_index[2] = thread_segment_index[1] + ((thread_new_heads.z != 0) && (thread_new_heads.y == 0));
        thread_segment_index[3] = thread_segment_index[2] + ((thread_new_heads.w != 0) && (thread_new_heads.z == 0));
    }

    {
        uint32_t thread_inclusive_suffix_old_rank;
        uint32_t thread_inclusive_suffix_new_rank;
        uint32_t thread_inclusive_segment_index;

        typedef cub::WarpScan<uint32_t> WarpScan;

        __shared__ typename WarpScan::TempStorage warp_scan_storage[CUDA_BLOCK_THREADS];

        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_exclusive_suffix_old_rank, thread_inclusive_suffix_old_rank, thread_exclusive_suffix_old_rank, (uint32_t)0, cub::Max());
        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_exclusive_suffix_new_rank, thread_inclusive_suffix_new_rank, thread_exclusive_suffix_new_rank, (uint32_t)0, cub::Max());
        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_segment_index[3]         , thread_inclusive_segment_index  , thread_exclusive_segment_index  , (uint32_t)0, cub::Sum());

        if ((threadIdx.x % CUDA_WARP_THREADS) == (CUDA_WARP_THREADS - 1))
        {
            warp_state1[threadIdx.x / CUDA_WARP_THREADS] = make_uint4(0, thread_inclusive_suffix_old_rank, thread_inclusive_suffix_new_rank, thread_inclusive_segment_index);
        }

        __syncthreads();
    }

    {
        if (threadIdx.x < CUDA_WARP_THREADS)
        {
            uint32_t            block_exclusive_suffix_new_rank = 0;
            uint32_t            block_exclusive_segment_index   = 0;

            uint32_t            warp_inclusive_suffix_old_rank;
            uint32_t            warp_inclusive_suffix_new_rank;
            uint32_t            warp_inclusive_segment_index;

            {
                typedef cub::WarpScan<uint32_t> WarpScan;

                __shared__ typename WarpScan::TempStorage warp_scan_storage;

                uint4 warp_inclusive_state = warp_state1[threadIdx.x];

                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state.y, warp_inclusive_suffix_old_rank, cub::Max());
                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state.z, warp_inclusive_suffix_new_rank, cub::Max());
                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state.w, warp_inclusive_segment_index  , cub::Sum());
            }

            {
                const uint32_t descriptor_status_aggregate_not_ready        = alternate_block_descriptor_statuses ? 4 : 0;
                const uint32_t descriptor_status_partial_aggregate_ready    = alternate_block_descriptor_statuses ? 3 : 1;
                const uint32_t descriptor_status_full_aggregate_ready       = scatter_ranks_directly ? (alternate_block_descriptor_statuses ? 0 : 4) : 2;

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint4(descriptor_status_partial_aggregate_ready, 0, warp_inclusive_suffix_new_rank, warp_inclusive_segment_index));
                }

                {
                    uint4 * RESTRICT descriptors_lookback = device_descriptors + blockIdx.x + threadIdx.x;

                    int32_t full_aggregate_lane, delay = 8;
                    do
                    {
                        descriptors_lookback -= CUDA_WARP_THREADS;

                        uint4 block_descriptor;
                        do
                        {
                            libcubwt_delay_or_prevent_hoisting(delay <<= 1);

                            block_descriptor = cub::ThreadLoad<cub::LOAD_CG>(descriptors_lookback);
                        } while (__any_sync((uint32_t)-1, block_descriptor.x == descriptor_status_aggregate_not_ready));

                        delay = 0;

                        {
                            full_aggregate_lane     = 31 - __clz((int32_t)__ballot_sync((uint32_t)-1, block_descriptor.x != descriptor_status_partial_aggregate_ready));
                            block_descriptor.z      = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.z : 0;
                            block_descriptor.w      = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.w : 0;
                        }

                        {
                            block_exclusive_suffix_new_rank     = cub::Max()(block_exclusive_suffix_new_rank , libcubwt_warp_reduce_max(block_descriptor.z));
                            block_exclusive_segment_index       = cub::Sum()(block_exclusive_segment_index   , libcubwt_warp_reduce_sum(block_descriptor.w));
                        }

                    } while (full_aggregate_lane == -1);

                    warp_inclusive_suffix_new_rank  = cub::Max()(warp_inclusive_suffix_new_rank, block_exclusive_suffix_new_rank);
                    warp_inclusive_segment_index    = cub::Sum()(warp_inclusive_segment_index  , block_exclusive_segment_index  );
                }

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint4(descriptor_status_full_aggregate_ready, 0, warp_inclusive_suffix_new_rank, warp_inclusive_segment_index));
                }
            }

            {
                uint32_t block_exclusive_suffix_old_rank    = __ldg((uint32_t *)(device_descriptors_copy + blockIdx.x - 1) + 2);
                warp_inclusive_suffix_old_rank              = cub::Max()(warp_inclusive_suffix_old_rank, block_exclusive_suffix_old_rank);

                if (threadIdx.x == 0)
                {
                    warp_state1[0] = make_uint4(0, block_exclusive_suffix_old_rank, block_exclusive_suffix_new_rank, block_exclusive_segment_index);
                }

                warp_state1[1 + threadIdx.x] = make_uint4(0, warp_inclusive_suffix_old_rank, warp_inclusive_suffix_new_rank, warp_inclusive_segment_index);
            }
        }

        __syncthreads();
    }

    {
        uint32_t thread_suffix_old_rank[4];
        uint32_t thread_suffix_new_rank[4];

        uint4 warp_exclusive_state              = warp_state1[threadIdx.x / CUDA_WARP_THREADS];
        
        thread_exclusive_suffix_old_rank        = cub::Max()(thread_exclusive_suffix_old_rank, warp_exclusive_state.y);
        thread_exclusive_suffix_new_rank        = cub::Max()(thread_exclusive_suffix_new_rank, warp_exclusive_state.z);
        thread_exclusive_segment_index          = cub::Sum()(thread_exclusive_segment_index  , warp_exclusive_state.w);

        const uint32_t thread_index             = blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;

        thread_suffix_old_rank[0]               = (thread_old_heads.x != 0) ? (thread_index + 0) : thread_exclusive_suffix_old_rank;
        thread_suffix_old_rank[1]               = (thread_old_heads.y != 0) ? (thread_index + 1) : thread_suffix_old_rank[0];
        thread_suffix_old_rank[2]               = (thread_old_heads.z != 0) ? (thread_index + 2) : thread_suffix_old_rank[1];
        thread_suffix_old_rank[3]               = (thread_old_heads.w != 0) ? (thread_index + 3) : thread_suffix_old_rank[2];

        thread_suffix_new_rank[0]               = (thread_new_heads.x != 0) ? (thread_index + 0) : thread_exclusive_suffix_new_rank;
        thread_suffix_new_rank[1]               = (thread_new_heads.y != 0) ? (thread_index + 1) : thread_suffix_new_rank[0];
        thread_suffix_new_rank[2]               = (thread_new_heads.z != 0) ? (thread_index + 2) : thread_suffix_new_rank[1];
        thread_suffix_new_rank[3]               = (thread_new_heads.w != 0) ? (thread_index + 3) : thread_suffix_new_rank[2];

        thread_segment_index[0]                 = cub::Sum()(thread_segment_index[0], thread_exclusive_segment_index);
        thread_segment_index[1]                 = cub::Sum()(thread_segment_index[1], thread_exclusive_segment_index);
        thread_segment_index[2]                 = cub::Sum()(thread_segment_index[2], thread_exclusive_segment_index);
        thread_segment_index[3]                 = cub::Sum()(thread_segment_index[3], thread_exclusive_segment_index);

        if (thread_exclusive_segment_index != thread_segment_index[0]) { device_offsets_begin[thread_segment_index[0]] = thread_exclusive_suffix_new_rank; device_offsets_end[thread_segment_index[0]] = thread_index + 0; }
        if (thread_segment_index[0]        != thread_segment_index[1]) { device_offsets_begin[thread_segment_index[1]] = thread_suffix_new_rank[0];        device_offsets_end[thread_segment_index[1]] = thread_index + 1; }
        if (thread_segment_index[1]        != thread_segment_index[2]) { device_offsets_begin[thread_segment_index[2]] = thread_suffix_new_rank[1];        device_offsets_end[thread_segment_index[2]] = thread_index + 2; }
        if (thread_segment_index[2]        != thread_segment_index[3]) { device_offsets_begin[thread_segment_index[3]] = thread_suffix_new_rank[2];        device_offsets_end[thread_segment_index[3]] = thread_index + 3; }

        if (scatter_ranks_directly)
        {
            const uint4    indexes              = __ldg((uint4 *)(device_SA + thread_index));

            if (thread_suffix_old_rank[0] != thread_suffix_new_rank[0])  { device_out_ISA[indexes.x] = thread_suffix_new_rank[0]; }
            if (thread_suffix_old_rank[1] != thread_suffix_new_rank[1])  { device_out_ISA[indexes.y] = thread_suffix_new_rank[1]; }
            if (thread_suffix_old_rank[2] != thread_suffix_new_rank[2])  { device_out_ISA[indexes.z] = thread_suffix_new_rank[2]; }
            if (thread_suffix_old_rank[3] != thread_suffix_new_rank[3])  { device_out_ISA[indexes.w] = thread_suffix_new_rank[3]; }
        }
        else
        {
            thread_suffix_index[0]              = (thread_suffix_old_rank[0] != thread_suffix_new_rank[0]);
            thread_suffix_index[1]              = thread_suffix_index[0] + (thread_suffix_old_rank[1] != thread_suffix_new_rank[1]);
            thread_suffix_index[2]              = thread_suffix_index[1] + (thread_suffix_old_rank[2] != thread_suffix_new_rank[2]);
            thread_suffix_index[3]              = thread_suffix_index[2] + (thread_suffix_old_rank[3] != thread_suffix_new_rank[3]);
        }
    }

    if (!scatter_ranks_directly)
    {
        {
            uint32_t thread_inclusive_suffix_index;

            typedef cub::WarpScan<uint32_t> WarpScan;

            __shared__ typename WarpScan::TempStorage warp_scan_storage[CUDA_WARP_THREADS];

            WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_suffix_index[3], thread_inclusive_suffix_index, thread_exclusive_suffix_index, (uint32_t)0, cub::Sum());

            if ((threadIdx.x % CUDA_WARP_THREADS) == (CUDA_WARP_THREADS - 1))
            {
                warp_state2[threadIdx.x / CUDA_WARP_THREADS] = thread_inclusive_suffix_index;
            }

            __syncthreads();
        }

        {
            if (threadIdx.x < CUDA_WARP_THREADS)
            {
                uint32_t            block_exclusive_suffix_index = 0;
                uint32_t            warp_inclusive_suffix_index;

                {
                    typedef cub::WarpScan<uint32_t> WarpScan;

                    __shared__ typename WarpScan::TempStorage warp_scan_storage;

                    uint32_t warp_inclusive_state = warp_state2[threadIdx.x];

                    WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state, warp_inclusive_suffix_index, cub::Sum());
                }

                {
                    const uint32_t descriptor_status_aggregate_not_ready        = alternate_block_descriptor_statuses ? 2 : 2;
                    const uint32_t descriptor_status_partial_aggregate_ready    = alternate_block_descriptor_statuses ? 1 : 3;
                    const uint32_t descriptor_status_full_aggregate_ready       = alternate_block_descriptor_statuses ? 0 : 4;

                    if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                    {
                        cub::ThreadStore<cub::STORE_CG>((uint2 *)(device_descriptors + blockIdx.x), make_uint2(descriptor_status_partial_aggregate_ready, warp_inclusive_suffix_index));
                    }

                    {
                        uint4 * RESTRICT descriptors_lookback = device_descriptors + blockIdx.x + threadIdx.x;

                        int32_t full_aggregate_lane, delay = 8;
                        do
                        {
                            descriptors_lookback -= CUDA_WARP_THREADS;

                            uint2 block_descriptor;
                            do
                            {
                                libcubwt_delay_or_prevent_hoisting(delay <<= 1);

                                block_descriptor = cub::ThreadLoad<cub::LOAD_CG>((uint2 *)descriptors_lookback);
                            } while (__any_sync((uint32_t)-1, alternate_block_descriptor_statuses 
                                ? ((int32_t )block_descriptor.x >= (int32_t )descriptor_status_aggregate_not_ready)
                                : ((uint32_t)block_descriptor.x <= (uint32_t)descriptor_status_aggregate_not_ready)));

                            delay = 0;

                            {
                                full_aggregate_lane = 31 - __clz((int32_t)__ballot_sync((uint32_t)-1, block_descriptor.x != descriptor_status_partial_aggregate_ready));
                                block_descriptor.y  = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.y : 0;
                            }

                            {
                                block_exclusive_suffix_index = cub::Sum()(block_exclusive_suffix_index, libcubwt_warp_reduce_sum(block_descriptor.y));
                            }

                        } while (full_aggregate_lane == -1);

                        warp_inclusive_suffix_index = cub::Sum()(warp_inclusive_suffix_index, block_exclusive_suffix_index);
                    }

                    if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                    {
                        cub::ThreadStore<cub::STORE_CG>((uint2 *)(device_descriptors + blockIdx.x), make_uint2(descriptor_status_full_aggregate_ready, warp_inclusive_suffix_index));
                    }
                }

                {
                    if (threadIdx.x == 0)
                    {
                        warp_state2[0] = block_exclusive_suffix_index;
                    }

                    warp_state2[1 + threadIdx.x] = warp_inclusive_suffix_index;
                }
            }

            __syncthreads();
        }

        {
            if (thread_suffix_index[3] > 0)
            {
                uint32_t thread_suffix_new_rank[4];

                uint32_t warp_exclusive_state           = warp_state2[threadIdx.x / CUDA_WARP_THREADS];
                thread_exclusive_suffix_index           = cub::Sum()(thread_exclusive_suffix_index, warp_exclusive_state);

                thread_suffix_index[0]                  = cub::Sum()(thread_suffix_index[0], thread_exclusive_suffix_index);
                thread_suffix_index[1]                  = cub::Sum()(thread_suffix_index[1], thread_exclusive_suffix_index);
                thread_suffix_index[2]                  = cub::Sum()(thread_suffix_index[2], thread_exclusive_suffix_index);
                thread_suffix_index[3]                  = cub::Sum()(thread_suffix_index[3], thread_exclusive_suffix_index);

                const uint32_t thread_index             = blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;
                const uint4    indexes                  = __ldg((uint4 *)(device_SA + thread_index));

                thread_suffix_new_rank[0]               = (thread_new_heads.x != 0) ? (thread_index + 0) : thread_exclusive_suffix_new_rank;
                thread_suffix_new_rank[1]               = (thread_new_heads.y != 0) ? (thread_index + 1) : thread_suffix_new_rank[0];
                thread_suffix_new_rank[2]               = (thread_new_heads.z != 0) ? (thread_index + 2) : thread_suffix_new_rank[1];
                thread_suffix_new_rank[3]               = (thread_new_heads.w != 0) ? (thread_index + 3) : thread_suffix_new_rank[2];

                if (thread_exclusive_suffix_index != thread_suffix_index[0])  { device_out_SA[thread_suffix_index[0]] = indexes.x; device_out_ISA[thread_suffix_index[0]] = thread_suffix_new_rank[0]; }
                if (thread_suffix_index[0]        != thread_suffix_index[1])  { device_out_SA[thread_suffix_index[1]] = indexes.y; device_out_ISA[thread_suffix_index[1]] = thread_suffix_new_rank[1]; }
                if (thread_suffix_index[1]        != thread_suffix_index[2])  { device_out_SA[thread_suffix_index[2]] = indexes.z; device_out_ISA[thread_suffix_index[2]] = thread_suffix_new_rank[2]; }
                if (thread_suffix_index[2]        != thread_suffix_index[3])  { device_out_SA[thread_suffix_index[3]] = indexes.w; device_out_ISA[thread_suffix_index[3]] = thread_suffix_new_rank[3]; }
            }
        }
    }
}

static cudaError_t libcubwt_rank_and_segment_suffixes(LIBCUBWT_DEVICE_STORAGE * storage, int64_t n, int64_t iteration)
{
    cudaError_t status                      = cudaSuccess;
    int64_t     n_segmentation_blocks       = 1 + (n / (storage->cuda_block_threads * 4));
    int64_t     n_initialization_blocks     = (n_segmentation_blocks + storage->cuda_block_threads - 1) / storage->cuda_block_threads;
    bool        scatter_ranks_directly      = (n <= ((int64_t)1 << (storage->device_L2_cache_bits - 3)));

    if (iteration == 0)
    {
        libcubwt_rank_and_segment_suffixes_initialization_kernel<<<(uint32_t)n_initialization_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
            storage->device_SA,
            storage->device_keys_temp_keys,
            storage->device_heads,
            storage->device_descriptors_large,
            storage->device_descriptors_small,
            (uint32_t)n);

        if (scatter_ranks_directly)
        {
            libcubwt_rank_and_segment_suffixes_initiatory_kernel<true><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                storage->device_SA,
                storage->device_keys_temp_keys,
                storage->device_heads,
                storage->device_ISA,
                storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                storage->device_descriptors_large);
        }
        else
        {
            libcubwt_rank_and_segment_suffixes_initiatory_kernel<false><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                NULL,
                storage->device_keys_temp_keys,
                storage->device_heads,
                storage->device_temp_ISA,
                storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                storage->device_descriptors_large);
        }

        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->host_pinned_storage, &storage->device_descriptors_large[n_segmentation_blocks - 1], sizeof(uint4), cudaMemcpyDeviceToHost, storage->cuda_stream));
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(storage->cuda_stream), status);

        if (status == cudaSuccess)
        {
            storage->num_unsorted_segments = ((uint4 *)storage->host_pinned_storage)->w;

            if (!scatter_ranks_directly)
            {
                if ((status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->device_temp_SA, storage->device_SA, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, storage->cuda_stream))) == cudaSuccess)
                {
                    status = libcubwt_permute_values_uint32(storage, storage->device_temp_SA, storage->device_temp_ISA, storage->device_ISA, n, storage->device_keys, storage->device_temp_keys);
                }
            }
        }
    }
    else
    {
        if ((status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->device_descriptors_copy - 1, storage->device_descriptors_large - 1, n_segmentation_blocks * sizeof(uint4), cudaMemcpyDeviceToDevice, storage->cuda_stream))) == cudaSuccess)
        {
            if (scatter_ranks_directly)
            {
                if ((iteration % 2) == 0)
                {
                    libcubwt_rank_and_segment_suffixes_incremental_kernel<false, true><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        storage->device_SA,
                        storage->device_keys,
                        storage->device_heads,
                        NULL, storage->device_ISA,
                        storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                        storage->device_descriptors_large, storage->device_descriptors_copy);
                }
                else
                {
                    libcubwt_rank_and_segment_suffixes_incremental_kernel<true, true><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        storage->device_SA,
                        storage->device_keys,
                        storage->device_heads,
                        NULL, storage->device_ISA,
                        storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                        storage->device_descriptors_large, storage->device_descriptors_copy);
                }
            }
            else
            {
                if ((iteration % 2) == 0)
                {
                    libcubwt_rank_and_segment_suffixes_incremental_kernel<false, false><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        storage->device_SA,
                        storage->device_keys,
                        storage->device_heads,
                        storage->device_temp_SA - 1, storage->device_temp_ISA - 1,
                        storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                        storage->device_descriptors_large, storage->device_descriptors_copy);
                }
                else
                {
                    libcubwt_rank_and_segment_suffixes_incremental_kernel<true, false><<<(uint32_t)n_segmentation_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        storage->device_SA,
                        storage->device_keys,
                        storage->device_heads,
                        storage->device_temp_SA - 1, storage->device_temp_ISA - 1,
                        storage->device_offsets - 1, storage->device_offsets + (n / 2) - 1,
                        storage->device_descriptors_large, storage->device_descriptors_copy);
                }
            }

            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->host_pinned_storage, &storage->device_descriptors_large[n_segmentation_blocks - 1], sizeof(uint4), cudaMemcpyDeviceToHost, storage->cuda_stream));
            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(storage->cuda_stream), status);

            if (status == cudaSuccess)
            {
                storage->num_unsorted_segments = ((uint4 *)storage->host_pinned_storage)->w;

                if (!scatter_ranks_directly)
                {
                    uint32_t num_updated_suffixes = ((uint4 *)storage->host_pinned_storage)->y;

                    if (num_updated_suffixes > 0)
                    {
                        status = libcubwt_scatter_values_uint32(storage, storage->device_temp_SA, storage->device_temp_ISA, storage->device_ISA, num_updated_suffixes, n, storage->device_keys, storage->device_temp_keys);
                    }
                }
            }
        }
    }

    return status;
}

template <bool alternate_block_descriptor_statuses>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_gather_unsorted_suffixes_kernel(
    const uint8_t *     RESTRICT device_heads, 
    const uint32_t *    RESTRICT device_SA,
    uint32_t *          RESTRICT device_out_keys,
    uint32_t *          RESTRICT device_out_SA,
    uint2 *             RESTRICT device_descriptors)
{
    __shared__ __align__(32) uint32_t warp_state[1 + CUDA_WARP_THREADS];

    uint32_t    thread_exclusive_suffix_index;
    uint32_t    thread_suffix_index[4];

    {
        device_heads += blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;

        const uchar4    current_heads   = __ldg((uchar4 *)(device_heads));
        const uint8_t   next_head       = current_heads.w > 0 ? __ldg(device_heads + 4) : 0;

        thread_suffix_index[0]          = (current_heads.x + current_heads.y < 2);
        thread_suffix_index[1]          = thread_suffix_index[0] + (current_heads.y + current_heads.z < 2);
        thread_suffix_index[2]          = thread_suffix_index[1] + (current_heads.z + current_heads.w < 2);
        thread_suffix_index[3]          = thread_suffix_index[2] + (current_heads.w +       next_head < 2);
    }

    {
        uint32_t thread_inclusive_suffix_index;

        typedef cub::WarpScan<uint32_t> WarpScan;

        __shared__ typename WarpScan::TempStorage warp_scan_storage[CUDA_WARP_THREADS];

        WarpScan(warp_scan_storage[threadIdx.x / CUDA_WARP_THREADS]).Scan(thread_suffix_index[3], thread_inclusive_suffix_index, thread_exclusive_suffix_index, (uint32_t)0, cub::Sum());

        if ((threadIdx.x % CUDA_WARP_THREADS) == (CUDA_WARP_THREADS - 1))
        {
            warp_state[threadIdx.x / CUDA_WARP_THREADS] = thread_inclusive_suffix_index;
        }

        __syncthreads();
    }

    {
        if (threadIdx.x < CUDA_WARP_THREADS)
        {
            uint32_t block_exclusive_suffix_index = 0;
            uint32_t warp_inclusive_suffix_index;

            {
                typedef cub::WarpScan<uint32_t> WarpScan;

                __shared__ typename WarpScan::TempStorage warp_scan_storage;

                uint32_t warp_inclusive_state = warp_state[threadIdx.x];

                WarpScan(warp_scan_storage).InclusiveScan(warp_inclusive_state, warp_inclusive_suffix_index, cub::Sum());
            }

            {
                const uint32_t descriptor_status_aggregate_not_ready        = alternate_block_descriptor_statuses ? 2 : 0;
                const uint32_t descriptor_status_partial_aggregate_ready    = alternate_block_descriptor_statuses ? 1 : 1;
                const uint32_t descriptor_status_full_aggregate_ready       = alternate_block_descriptor_statuses ? 0 : 2;

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint2(descriptor_status_partial_aggregate_ready, warp_inclusive_suffix_index));
                }

                {
                    uint2 * RESTRICT descriptors_lookback = device_descriptors + blockIdx.x + threadIdx.x;

                    int32_t full_aggregate_lane, delay = 8;
                    do
                    {
                        descriptors_lookback -= CUDA_WARP_THREADS;

                        uint2 block_descriptor;
                        do
                        {
                            libcubwt_delay_or_prevent_hoisting(delay <<= 1);

                            block_descriptor = cub::ThreadLoad<cub::LOAD_CG>(descriptors_lookback);
                        } while (__any_sync((uint32_t)-1, block_descriptor.x == descriptor_status_aggregate_not_ready));

                        delay = 0;

                        {
                            full_aggregate_lane = 31 - __clz((int32_t)__ballot_sync((uint32_t)-1, block_descriptor.x != descriptor_status_partial_aggregate_ready));
                            block_descriptor.y  = (((int32_t)threadIdx.x) >= full_aggregate_lane) ? block_descriptor.y : 0;
                        }

                        {
                            block_exclusive_suffix_index = cub::Sum()(block_exclusive_suffix_index, libcubwt_warp_reduce_sum(block_descriptor.y));
                        }

                    } while (full_aggregate_lane == -1);

                    warp_inclusive_suffix_index = cub::Sum()(warp_inclusive_suffix_index, block_exclusive_suffix_index);
                }

                if (threadIdx.x == ((CUDA_BLOCK_THREADS / CUDA_WARP_THREADS) - 1))
                {
                    cub::ThreadStore<cub::STORE_CG>(device_descriptors + blockIdx.x, make_uint2(descriptor_status_full_aggregate_ready, warp_inclusive_suffix_index));
                }
            }

            {
                if (threadIdx.x == 0)
                {
                    warp_state[0] = block_exclusive_suffix_index;
                }

                warp_state[1 + threadIdx.x] = warp_inclusive_suffix_index;
            }
        }

        __syncthreads();
    }

    {
        if (thread_suffix_index[3] > 0)
        {
            uint32_t warp_exclusive_state           = warp_state[threadIdx.x / CUDA_WARP_THREADS];
        
            thread_exclusive_suffix_index           = cub::Sum()(thread_exclusive_suffix_index, warp_exclusive_state);

            thread_suffix_index[0]                  = cub::Sum()(thread_suffix_index[0], thread_exclusive_suffix_index);
            thread_suffix_index[1]                  = cub::Sum()(thread_suffix_index[1], thread_exclusive_suffix_index);
            thread_suffix_index[2]                  = cub::Sum()(thread_suffix_index[2], thread_exclusive_suffix_index);
            thread_suffix_index[3]                  = cub::Sum()(thread_suffix_index[3], thread_exclusive_suffix_index);

            const uint32_t thread_index             = blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;
            const uint4    indexes                  = __ldg((uint4 *)(device_SA + thread_index));

            if (thread_exclusive_suffix_index != thread_suffix_index[0]) { device_out_keys[thread_suffix_index[0]] = thread_index + 0; device_out_SA[thread_suffix_index[0]] = indexes.x; }
            if (thread_suffix_index[0]        != thread_suffix_index[1]) { device_out_keys[thread_suffix_index[1]] = thread_index + 1; device_out_SA[thread_suffix_index[1]] = indexes.y; }
            if (thread_suffix_index[1]        != thread_suffix_index[2]) { device_out_keys[thread_suffix_index[2]] = thread_index + 2; device_out_SA[thread_suffix_index[2]] = indexes.z; }
            if (thread_suffix_index[2]        != thread_suffix_index[3]) { device_out_keys[thread_suffix_index[3]] = thread_index + 3; device_out_SA[thread_suffix_index[3]] = indexes.w; }
        }
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_update_suffix_sorting_keys_kernel(const uint8_t * RESTRICT device_heads, const uint32_t * RESTRICT device_SA, const uint32_t * RESTRICT device_ISA, uint32_t * RESTRICT device_keys)
{
    const uint32_t  thread_index    = blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;

    device_heads += thread_index;

    const uchar4    current_heads   = __ldg((uchar4 *)(device_heads));
    const uint8_t   next_head       = current_heads.w > 0 ? __ldg(device_heads + 4) : 0;

    if (current_heads.x + current_heads.y + current_heads.z + current_heads.w + next_head < 5)
    {
        device_SA += thread_index; device_keys += thread_index;

        const uint4 current_SA = __ldg((uint4 *)(device_SA));

        ((uint4 *)device_keys)[0] = make_uint4(
            (current_heads.x + current_heads.y < 2) ? __ldg(device_ISA + current_SA.x) : (uint32_t)-1,
            (current_heads.y + current_heads.z < 2) ? __ldg(device_ISA + current_SA.y) : (uint32_t)-2,
            (current_heads.z + current_heads.w < 2) ? __ldg(device_ISA + current_SA.z) : (uint32_t)-3,
            (current_heads.w +       next_head < 2) ? __ldg(device_ISA + current_SA.w) : (uint32_t)-4);
    }
}

static cudaError_t libcubwt_update_suffix_sorting_keys(LIBCUBWT_DEVICE_STORAGE * storage, int64_t n, int64_t iteration, int64_t depth)
{
    cudaError_t status                  = cudaSuccess;
    int64_t     n_ranking_blocks        = (n + storage->cuda_block_threads * 4 - 1) / (storage->cuda_block_threads * 4);
    bool        gather_keys_directly    = (n <= ((int64_t)1 << (storage->device_L2_cache_bits - 2))) || (n > ((int64_t)1 << (storage->device_L2_cache_bits - 2 + 8)));

    if (gather_keys_directly || (storage->num_unsorted_suffixes <= (n / 4)))
    {
        libcubwt_update_suffix_sorting_keys_kernel<<<(uint32_t)n_ranking_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(storage->device_heads, storage->device_SA, storage->device_ISA + depth, storage->device_keys);
    }
    else
    {
        if ((iteration % 2) == 0)
        {
            libcubwt_gather_unsorted_suffixes_kernel<false><<<(uint32_t)n_ranking_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                storage->device_heads,
                storage->device_SA,
                storage->device_temp_keys - 1, storage->device_temp_SA - 1,
                storage->device_descriptors_small);
        }
        else
        {
            libcubwt_gather_unsorted_suffixes_kernel<true><<<(uint32_t)n_ranking_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                storage->device_heads,
                storage->device_SA,
                storage->device_temp_keys - 1, storage->device_temp_SA - 1,
                storage->device_descriptors_small);
        }

        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->host_pinned_storage, &storage->device_descriptors_small[n_ranking_blocks - 1], sizeof(uint2), cudaMemcpyDeviceToHost, storage->cuda_stream));
        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(storage->cuda_stream), status);

        if (status == cudaSuccess)
        {
            storage->num_unsorted_suffixes = ((uint2 *)storage->host_pinned_storage)->y;

            if (storage->num_unsorted_suffixes > 0)
            {
                status = libcubwt_gather_scatter_values_uint32(storage, storage->device_temp_SA, storage->device_ISA + depth, storage->device_temp_keys, storage->device_keys, storage->num_unsorted_suffixes, n, storage->device_temp_ISA, storage->device_keys);
            }
        }
    }

    return status;
}

static cudaError_t libcubwt_sort_segmented_suffixes_by_rank(LIBCUBWT_DEVICE_STORAGE * storage, int64_t n)
{
    cub::DoubleBuffer<uint32_t> d_keys(storage->device_keys, storage->device_temp_keys);
    cub::DoubleBuffer<uint32_t> d_values(storage->device_SA, storage->device_temp_SA);

    cudaError_t status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceSegmentedSort::SortPairs(
        storage->device_ssort_temp_storage, storage->device_ssort_temp_storage_size,
        d_keys, d_values,
        (int)storage->num_unsorted_suffixes, (int)storage->num_unsorted_segments,
        storage->device_offsets, storage->device_offsets + (n / 2),
        storage->cuda_stream));

    if (d_keys.selector) { std::swap(storage->device_keys, storage->device_temp_keys); }
    if (d_values.selector) { std::swap(storage->device_SA, storage->device_temp_SA); }

    return status;
}

template <bool process_auxiliary_indexes>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_prepare_mod12_suffixes_kernel(const uint8_t * RESTRICT device_T, const uint32_t * RESTRICT device_ISA, const uint64_t * RESTRICT device_suffixes, const uint32_t rm, const uint32_t rs)
{
    __shared__ union
    {
        struct
        {
            __align__(32) uint32_t bytes[4 * CUDA_BLOCK_THREADS];
            __align__(32) uint4    ranks[3 * CUDA_BLOCK_THREADS];
        } stage1;

        struct
        {
            __align__(32) uint4 suffixes[4 * CUDA_BLOCK_THREADS];
        } stage2;

    } shared_storage;

    {
        device_T   += blockIdx.x * CUDA_BLOCK_THREADS * 12 + threadIdx.x * 16;
        device_ISA += blockIdx.x * CUDA_BLOCK_THREADS * 8  + threadIdx.x * 4;

        uint4 * RESTRICT thread_bytes = (uint4 *)shared_storage.stage1.bytes + threadIdx.x;
        uint4 * RESTRICT thread_ranks = (uint4 *)shared_storage.stage1.ranks + threadIdx.x;

        if (threadIdx.x < (12 * CUDA_BLOCK_THREADS + 4 + 15) / 16) { thread_bytes[0] = __ldg((uint4 *)device_T); }

        thread_ranks[0] = __ldg((uint4 *)device_ISA); thread_ranks += CUDA_BLOCK_THREADS; device_ISA += CUDA_BLOCK_THREADS * 4;
        thread_ranks[0] = __ldg((uint4 *)device_ISA); thread_ranks += CUDA_BLOCK_THREADS; device_ISA += CUDA_BLOCK_THREADS * 4;
        if (threadIdx.x == 0) { thread_ranks[0] = __ldg((uint4 *)device_ISA); }
    }

    {
        __syncthreads();

        uint32_t bytes0 = shared_storage.stage1.bytes[threadIdx.x * 3 + 0];
        uint32_t bytes1 = shared_storage.stage1.bytes[threadIdx.x * 3 + 1];
        uint32_t bytes2 = shared_storage.stage1.bytes[threadIdx.x * 3 + 2];
        uint32_t bytes3 = shared_storage.stage1.bytes[threadIdx.x * 3 + 3];

        uint4    ranks0 = shared_storage.stage1.ranks[threadIdx.x * 2 + 0];
        uint4    ranks1 = shared_storage.stage1.ranks[threadIdx.x * 2 + 1];
        uint4    ranks2 = shared_storage.stage1.ranks[threadIdx.x * 2 + 2];

        __syncthreads();

        uint32_t v4 = 0, v8 = 0;

        if (process_auxiliary_indexes)
        {
            const uint32_t i4 = blockIdx.x * CUDA_BLOCK_THREADS * 12 + threadIdx.x * 12 + 4 + rm + 1;
            const uint32_t i8 = blockIdx.x * CUDA_BLOCK_THREADS * 12 + threadIdx.x * 12 + 8 + rm + 1;

            if ((i4 & rm) == 0) { v4 = (i4 >> rs) << 24; }
            if ((i8 & rm) == 0) { v8 = (i8 >> rs) << 24; }
        }

        shared_storage.stage2.suffixes[threadIdx.x * 4 + 0] = make_uint4
        (
            ranks0.y, __byte_perm(bytes0, 0, 0x4021),
            ranks0.z | (uint32_t)INT32_MIN, __byte_perm(bytes0, 0, 0x4132)
        );

        shared_storage.stage2.suffixes[threadIdx.x * 4 + 1] = make_uint4
        (
            ranks0.w, (__byte_perm(bytes0, bytes1, 0x0354) & 0xffffffu) | v4,
            ranks1.x | (uint32_t)INT32_MIN, __byte_perm(bytes1, 0, 0x4021)
        );

        shared_storage.stage2.suffixes[threadIdx.x * 4 + 2] = make_uint4
        (
            ranks1.y, __byte_perm(bytes1, bytes2, 0x0243) & 0xffffffu,
            ranks1.z | (uint32_t)INT32_MIN, (__byte_perm(bytes1, bytes2, 0x0354) & 0xffffffu) | v8
        );

        shared_storage.stage2.suffixes[threadIdx.x * 4 + 3] = make_uint4
        (
            ranks1.w, __byte_perm(bytes2, 0, 0x4132),
            ranks2.x | (uint32_t)INT32_MIN, __byte_perm(bytes2, bytes3, 0x0243) & 0xffffffu
        );

        __syncwarp();
    }

    {
        device_suffixes += blockIdx.x * CUDA_BLOCK_THREADS * 8;

        uint4 * RESTRICT thread_src = shared_storage.stage2.suffixes + ((threadIdx.x / CUDA_WARP_THREADS) * CUDA_WARP_THREADS * 4) + (threadIdx.x % CUDA_WARP_THREADS);
        uint4 * RESTRICT thread_dst = (uint4 *)device_suffixes       + ((threadIdx.x / CUDA_WARP_THREADS) * CUDA_WARP_THREADS * 4) + (threadIdx.x % CUDA_WARP_THREADS);

        thread_dst[0] = thread_src[0]; thread_src += CUDA_WARP_THREADS; thread_dst += CUDA_WARP_THREADS;
        thread_dst[0] = thread_src[0]; thread_src += CUDA_WARP_THREADS; thread_dst += CUDA_WARP_THREADS;
        thread_dst[0] = thread_src[0]; thread_src += CUDA_WARP_THREADS; thread_dst += CUDA_WARP_THREADS;
        thread_dst[0] = thread_src[0];
    }
}

template <bool process_auxiliary_indexes>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_prepare_mod0_suffixes_kernel(const uint8_t * RESTRICT device_T, const uint32_t * RESTRICT device_ISA, const uint64_t * RESTRICT device_suffixes_lh, const uint32_t * RESTRICT device_suffixes_hh, const uint32_t rm, const uint32_t rs)
{
    __shared__ __align__(32) uint16_t bytes[3 * CUDA_BLOCK_THREADS + 8];

    {
        device_T += blockIdx.x * CUDA_BLOCK_THREADS * 6 + threadIdx.x * 16;

        uint4 * RESTRICT thread_bytes = (uint4 *)bytes + threadIdx.x;

        if (threadIdx.x <= (6 * CUDA_BLOCK_THREADS) / 16) { thread_bytes[0] = __ldg((uint4 *)(device_T - 16)); }
    }

    {
        device_ISA         += blockIdx.x * CUDA_BLOCK_THREADS * 4 + threadIdx.x * 4;
        device_suffixes_lh += blockIdx.x * CUDA_BLOCK_THREADS * 2 + threadIdx.x * 2;
        device_suffixes_hh += blockIdx.x * CUDA_BLOCK_THREADS * 2 + threadIdx.x * 2;

        __syncthreads();

        uint32_t bytes0 = bytes[threadIdx.x * 3 + 7 ];
        uint32_t bytes1 = bytes[threadIdx.x * 3 + 8 ];
        uint32_t bytes2 = bytes[threadIdx.x * 3 + 9 ];
        uint32_t bytes3 = bytes[threadIdx.x * 3 + 10];
        uint4    ranks  = __ldg((uint4 *)(device_ISA));

        uint32_t v0 = 0;

        if (process_auxiliary_indexes)
        {
            const uint32_t i0 = blockIdx.x * CUDA_BLOCK_THREADS * 6 + threadIdx.x * 6 + 0 + rm + 1;

            if ((i0 & rm) == 0) { v0 = (i0 >> rs) << 24; }
        }
        else if ((blockIdx.x | threadIdx.x) == 0)
        {
            v0 = 1u << 24;
        }

        *(uint4 *)(device_suffixes_lh) = make_uint4
        (
            ranks.x, __byte_perm(bytes0, bytes1, 0x3154) | v0,
            ranks.z, __byte_perm(bytes2, bytes3, 0x3041)
        );

        *(uint2 *)(device_suffixes_hh) = make_uint2(ranks.y | (uint32_t)INT32_MIN, ranks.w | (uint32_t)INT32_MIN);
    }
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, 1)
static void libcubwt_set_sentinel_suffixes_kernel(uint64_t * RESTRICT device_mod0l_suffixes_end, uint32_t * RESTRICT device_mod0h_suffixes_end,uint64_t * RESTRICT device_mod12_suffixes_end)
{
    uint32_t thread_index = blockIdx.x * CUDA_BLOCK_THREADS + threadIdx.x;

    device_mod0l_suffixes_end += thread_index; 
    device_mod0h_suffixes_end += thread_index;
    device_mod12_suffixes_end += thread_index;

    *(uint2    *)(device_mod0l_suffixes_end) = make_uint2(0x7fffffffu - 12 * CUDA_BLOCK_THREADS + 2 * thread_index + 0, 0x00ffffffu);
    *(uint32_t *)(device_mod0h_suffixes_end) = (uint32_t)(0xffffffffu - 12 * CUDA_BLOCK_THREADS + 2 * thread_index + 0             );
    *(uint2    *)(device_mod12_suffixes_end) = make_uint2(0x7fffffffu - 12 * CUDA_BLOCK_THREADS + 2 * thread_index + 1, 0x00ffffffu);
}

__device__ __forceinline__
bool libcubwt_compare_suffixes_kernel(const uint2 mod0l_suffix, const uint32_t mod0h_suffix, const uint2 mod12_suffix)
{
    uint32_t difference = __byte_perm(mod0l_suffix.y, 0, 0x4401) - __byte_perm(mod12_suffix.y, 0, 0x4401);
    if (difference == 0) { difference = (((int32_t)mod12_suffix.x < 0) ? mod0h_suffix : mod0l_suffix.x) - mod12_suffix.x; }

    return (int32_t)difference <= 0;
}

__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_suffixes_merge_initialization_kernel(
    const uint64_t * RESTRICT device_mod0l_suffixes,
    const uint32_t * RESTRICT device_mod0h_suffixes,
    const uint32_t num_mod0_suffixes,

    const uint64_t * RESTRICT device_mod12_suffixes,
    const uint32_t num_mod12_suffixes,

    uint32_t * RESTRICT device_suffixes_merge_path,
    uint32_t num_merging_blocks)
{
    uint32_t thread_index = blockIdx.x * CUDA_BLOCK_THREADS + threadIdx.x;

    if (thread_index <= num_merging_blocks)
    {
        uint32_t diagonal   = thread_index * CUDA_BLOCK_THREADS * 5;
        uint32_t begin      = (diagonal > num_mod12_suffixes) ? (diagonal - num_mod12_suffixes) : 0;
        uint32_t end        = (diagonal > num_mod0_suffixes ) ? (num_mod0_suffixes            ) : diagonal;

        while (begin < end)
        {
            uint32_t pivot = begin + ((end - begin) >> 1);

            bool predicate = libcubwt_compare_suffixes_kernel(
                __ldg((uint2    *)(device_mod0l_suffixes + pivot)),
                __ldg((uint32_t *)(device_mod0h_suffixes + pivot)),
                __ldg((uint2    *)(device_mod12_suffixes + diagonal - pivot - 1)));

            begin = predicate ? (pivot + 1) : begin;
            end   = predicate ? (end      ) : pivot;
        }

        __syncwarp();

        device_suffixes_merge_path[thread_index] = begin;
    }
}

template <bool process_auxiliary_indexes>
__global__ __launch_bounds__(CUDA_BLOCK_THREADS, CUDA_SM_THREADS / CUDA_BLOCK_THREADS)
static void libcubwt_merge_suffixes_kernel(
    const uint64_t * RESTRICT device_mod0l_suffixes,
    const uint32_t * RESTRICT device_mod0h_suffixes,
    const uint64_t * RESTRICT device_mod12_suffixes,
    const uint32_t * RESTRICT device_suffixes_merge_path,

    uint32_t * RESTRICT device_auxiliary_indexes,
    uint8_t  * RESTRICT device_L)
{
    __shared__ union
    {
        struct
        {
            __align__(32) uint2    suffixes_l[CUDA_BLOCK_THREADS * 5 + 12];
            __align__(32) uint32_t suffixes_h[CUDA_BLOCK_THREADS * 5 + 12];
        } stage1;

        struct
        {
            __align__(32) uint8_t  bwt[CUDA_BLOCK_THREADS * 5];
        } stage2;

    } shared_storage;

    uint32_t num_mod0_suffixes;
    uint32_t num_mod12_suffixes;

    {
        const uint32_t block_mod0_path_begin = (device_suffixes_merge_path + blockIdx.x)[0];
        const uint32_t block_mod0_path_end   = (device_suffixes_merge_path + blockIdx.x)[1];
        
        num_mod0_suffixes   = block_mod0_path_end - block_mod0_path_begin + 6; 
        num_mod12_suffixes  = CUDA_BLOCK_THREADS * 5 + 12 - num_mod0_suffixes;

        device_mod0l_suffixes += block_mod0_path_begin; 
        device_mod0h_suffixes += block_mod0_path_begin;

        device_mod12_suffixes += (blockIdx.x * CUDA_BLOCK_THREADS * 5 - block_mod0_path_begin);
        device_mod12_suffixes -= num_mod0_suffixes;

        #pragma unroll
        for (uint32_t thread_index = threadIdx.x; thread_index < CUDA_BLOCK_THREADS * 5 + 12; thread_index += CUDA_BLOCK_THREADS)
        {
            if (thread_index < num_mod0_suffixes) { shared_storage.stage1.suffixes_h[thread_index] = __ldg(device_mod0h_suffixes + thread_index); }
            shared_storage.stage1.suffixes_l[thread_index] = __ldg((uint2 *)(thread_index < num_mod0_suffixes ? device_mod0l_suffixes : device_mod12_suffixes) + thread_index);
        }

        __syncthreads();
    }

    {
        uint32_t diagonal   = threadIdx.x * 5;
        uint32_t begin      = (diagonal > num_mod12_suffixes) ? (diagonal - num_mod12_suffixes) : 0;
        uint32_t end        = (diagonal > num_mod0_suffixes ) ? (num_mod0_suffixes            ) : diagonal;

        while (begin < end)
        {
            uint32_t pivot = (begin + end) >> 1;

            bool predicate = libcubwt_compare_suffixes_kernel(
                shared_storage.stage1.suffixes_l[pivot],
                shared_storage.stage1.suffixes_h[pivot],
                shared_storage.stage1.suffixes_l[num_mod0_suffixes + diagonal - pivot - 1]);

            begin = predicate ? (pivot + 1) : begin;
            end   = predicate ? (end      ) : pivot;
        }

        __syncwarp();

        uint32_t suffixes[5];

        {
            uint32_t mod0_index     = begin;
            uint32_t mod12_index    = num_mod0_suffixes + diagonal - begin;
            uint2    mod0l_suffix   = shared_storage.stage1.suffixes_l[mod0_index];
            uint32_t mod0h_suffix   = shared_storage.stage1.suffixes_h[mod0_index];
            uint2    mod12_suffix   = shared_storage.stage1.suffixes_l[mod12_index];

            #pragma unroll
            for (uint32_t item = 0; item < 5; ++item)
            {
                bool predicate = libcubwt_compare_suffixes_kernel(mod0l_suffix, mod0h_suffix, mod12_suffix);
           
                suffixes[item] = predicate ? mod0l_suffix.y : mod12_suffix.y;

                if ( predicate) { mod0_index  += 1; mod0l_suffix = shared_storage.stage1.suffixes_l[mod0_index]; mod0h_suffix = shared_storage.stage1.suffixes_h[mod0_index]; }
                if (!predicate) { mod12_index += 1; mod12_suffix = shared_storage.stage1.suffixes_l[mod12_index]; }
            }

            __syncthreads();
        }

        {
            #pragma unroll
            for (uint32_t item = 0; item < 5; ++item)
            {
                if (suffixes[item] >= 0x01000000u)
                {
                    device_auxiliary_indexes[process_auxiliary_indexes ? suffixes[item] >> 24 : 1] = blockIdx.x * CUDA_BLOCK_THREADS * 5 + diagonal + item;
                }

                shared_storage.stage2.bwt[diagonal + item] = (uint8_t)(suffixes[item] >> 16);
            }

            __syncthreads();
        }
    }

    {
        device_L += blockIdx.x * CUDA_BLOCK_THREADS * 5 + threadIdx.x * 16;
        if (threadIdx.x < (CUDA_BLOCK_THREADS * 5 / 16)) { ((uint4 *)device_L)[0] = ((uint4 *)shared_storage.stage2.bwt)[threadIdx.x]; }
    }
}

static cudaError_t libcubwt_compute_burrows_wheeler_transform(LIBCUBWT_DEVICE_STORAGE * storage, const uint8_t * T, int64_t input_n, int64_t r, uint32_t * I)
{
    cudaError_t status  = cudaSuccess;

    int64_t reduced_n   = (input_n   / 3) * 2 + 2;
    int64_t expanded_n  = (reduced_n / 2) * 3 + 0;
    int64_t num_indexes = (input_n + r - 1) / r;

    if ((status = libcubwt_initialize_device_arrays(storage, T, reduced_n, expanded_n, input_n)) == cudaSuccess)
    {
        status = libcubwt_sort_suffixes_by_prefix(storage, reduced_n);
    }

    if (status == cudaSuccess)
    {
        for (int64_t iteration = 0, depth = 4; true; iteration += 1, depth *= 2)
        {
            if ((status = libcubwt_rank_and_segment_suffixes(storage, reduced_n, iteration)) != cudaSuccess)
            {
                break;
            }

            if (storage->num_unsorted_segments == 0)
            {
                break;
            }

            if ((status = libcubwt_update_suffix_sorting_keys(storage, reduced_n, iteration, depth)) != cudaSuccess)
            {
                break;
            }

            if ((status = libcubwt_sort_segmented_suffixes_by_rank(storage, reduced_n)) != cudaSuccess)
            {
                break;
            }
        }
    }

    if (status == cudaSuccess)
    {
        int64_t num_mod0_suffixes  = (input_n / 3) * 1 + ((input_n % 3) != 0);
        int64_t num_mod12_suffixes = (input_n / 3) * 2 + ((input_n % 3) == 2);

        if ((status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(storage->device_temp_ISA, storage->device_ISA, reduced_n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, storage->cuda_stream))) == cudaSuccess)
        {
            cub::DoubleBuffer<uint64_t> db_mod12_suffixes(storage->device_keys_temp_keys, storage->device_SA_temp_SA);

            if (status == cudaSuccess)
            {
                {
                    int64_t n_preparing_blocks = (num_mod12_suffixes + storage->cuda_block_threads * 8 - 1) / (storage->cuda_block_threads * 8);

                    if (num_indexes > 1)
                    {
                        uint32_t rm = (uint32_t)(r - 1), rs = 0; while (rm >= ((uint32_t)1 << rs)) { rs += 1; }

                        libcubwt_prepare_mod12_suffixes_kernel<true><<<(uint32_t)n_preparing_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            storage->device_T, storage->device_ISA,
                            db_mod12_suffixes.Current(),
                            rm, rs);
                    }
                    else
                    {
                        libcubwt_prepare_mod12_suffixes_kernel<false><<<(uint32_t)n_preparing_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            storage->device_T, storage->device_ISA,
                            db_mod12_suffixes.Current(),
                            0, 0);
                    }
                }

                {
                    cub::DoubleBuffer<uint32_t> db_index(storage->device_ISA, storage->device_offsets);
                    status = libcubwt_scatter_values_uint64(storage, db_index, db_mod12_suffixes, num_mod12_suffixes, reduced_n, reduced_n - num_mod12_suffixes);
                }
            }

            cub::DoubleBuffer<uint32_t> db_mod0h_suffixes(storage->device_ISA, storage->device_offsets);
            cub::DoubleBuffer<uint64_t> db_mod0l_suffixes = db_mod12_suffixes.Current() == storage->device_keys_temp_keys
                ? cub::DoubleBuffer<uint64_t>((uint64_t *)storage->device_SA, (uint64_t *)storage->device_temp_SA)
                : cub::DoubleBuffer<uint64_t>((uint64_t *)storage->device_keys, (uint64_t *)storage->device_temp_keys);

            if (status == cudaSuccess)
            {
                {
                    int64_t n_preparing_blocks = (num_mod0_suffixes + storage->cuda_block_threads * 2 - 1) / (storage->cuda_block_threads * 2);

                    if (num_indexes > 1)
                    {
                        uint32_t rm = (uint32_t)(r - 1), rs = 0; while (rm >= ((uint32_t)1 << rs)) { rs += 1; }

                        libcubwt_prepare_mod0_suffixes_kernel<true><<<(uint32_t)n_preparing_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            storage->device_T, storage->device_temp_ISA,
                            db_mod0l_suffixes.Current(), db_mod0h_suffixes.Current(),
                            rm, rs);
                    }
                    else
                    {
                        libcubwt_prepare_mod0_suffixes_kernel<false><<<(uint32_t)n_preparing_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            storage->device_T, storage->device_temp_ISA,
                            db_mod0l_suffixes.Current(), db_mod0h_suffixes.Current(),
                            0, 0);
                    }
                }

                if (reduced_n <= (1 << 24))
                {
                    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
                        storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
                        db_mod0l_suffixes, db_mod0h_suffixes,
                        (uint32_t)num_mod0_suffixes,
                        0, 24,
                        storage->cuda_stream));

                    if (status == cudaSuccess)
                    {
                        status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
                            storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
                            db_mod0l_suffixes, db_mod0h_suffixes,
                            (uint32_t)num_mod0_suffixes,
                            32, 40,
                            storage->cuda_stream));
                    }
                }
                else
                {
                    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(
                        storage->device_rsort_temp_storage, storage->device_rsort_temp_storage_size,
                        db_mod0l_suffixes, db_mod0h_suffixes,
                        (uint32_t)num_mod0_suffixes,
                        0, 40,
                        storage->cuda_stream));
                }
            }

            if (status == cudaSuccess)
            {
                int64_t n_merging_blocks = (input_n + storage->cuda_block_threads * 5 - 1) / (storage->cuda_block_threads * 5);

                {
                    libcubwt_set_sentinel_suffixes_kernel<<<6, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        db_mod0l_suffixes.Current() + num_mod0_suffixes,
                        db_mod0h_suffixes.Current() + num_mod0_suffixes,
                        db_mod12_suffixes.Current() + num_mod12_suffixes);
                }

                {
                    int64_t n_merge_initialization_blocks = 1 + (n_merging_blocks / storage->cuda_block_threads);

                    libcubwt_suffixes_merge_initialization_kernel<<<(uint32_t)n_merge_initialization_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                        db_mod0l_suffixes.Current(), db_mod0h_suffixes.Current(), (uint32_t)(num_mod0_suffixes + 6 * storage->cuda_block_threads),
                        db_mod12_suffixes.Current(), (uint32_t)(num_mod12_suffixes + 6 * storage->cuda_block_threads),
                        (uint32_t *)storage->device_descriptors_large, (uint32_t)n_merging_blocks);
                }

                {
                    if (num_indexes > 1)
                    {
                        libcubwt_merge_suffixes_kernel<true><<<(uint32_t)n_merging_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            db_mod0l_suffixes.Current(), db_mod0h_suffixes.Current(), db_mod12_suffixes.Current(),
                            (uint32_t *)storage->device_descriptors_large,
                            (uint32_t *)storage->device_descriptors_small - 1,
                            storage->device_T);
                    }
                    else
                    {
                        libcubwt_merge_suffixes_kernel<false><<<(uint32_t)n_merging_blocks, storage->cuda_block_threads, 0, storage->cuda_stream>>>(
                            db_mod0l_suffixes.Current(), db_mod0h_suffixes.Current(), db_mod12_suffixes.Current(),
                            (uint32_t *)storage->device_descriptors_large,
                            (uint32_t *)storage->device_descriptors_small - 1,
                            storage->device_T);
                    }
                }
            }

            if (status == cudaSuccess)
            {
                uint32_t * buffer = ((sizeof(uint32_t) * num_indexes) <= storage->host_pinned_storage_size) ? (uint32_t *)storage->host_pinned_storage : I;

                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(buffer, storage->device_descriptors_small, sizeof(uint32_t) * num_indexes, cudaMemcpyDeviceToHost, storage->cuda_stream), status);

                if ((status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(storage->cuda_stream), status)) == cudaSuccess)
                {
                    if (I != buffer) { memcpy(I, buffer, sizeof(uint32_t) * num_indexes); }

                    for (int64_t index = 0; index < num_indexes; index += 1) { I[index] += 1; }
                }
            }
        }
    }

    return status;
}

static cudaError_t libcubwt_copy_burrows_wheeler_transform(LIBCUBWT_DEVICE_STORAGE * storage, const uint8_t * T, uint8_t * L, int64_t input_n, int64_t index)
{
    cudaError_t status = cudaSuccess;

    L[0] = T[input_n - 1];

    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(L + 1, storage->device_T, (size_t)(index - 1), cudaMemcpyDeviceToHost, storage->cuda_stream), status);
    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMemcpyAsync(L + index, storage->device_T + index, (size_t)(input_n - index), cudaMemcpyDeviceToHost, storage->cuda_stream), status);

    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamSynchronize(storage->cuda_stream), status);

    return status;
}

int64_t libcubwt_allocate_device_storage(void ** device_storage, int64_t max_length)
{
    int64_t max_reduced_length  = ((max_length         / 3) * 2 + 2 + 1023) & (-1024);
    int64_t max_expanded_length = ((max_reduced_length / 2) * 3 + 0 + 1023) & (-1024);

    if ((device_storage == NULL) || (max_expanded_length >= INT32_MAX))
    {
        return LIBCUBWT_BAD_PARAMETER;
    }

    *device_storage = NULL;

    LIBCUBWT_DEVICE_STORAGE * storage = (LIBCUBWT_DEVICE_STORAGE *)malloc(sizeof(LIBCUBWT_DEVICE_STORAGE));
    if (storage != NULL)
    {
        memset(storage, 0, sizeof(LIBCUBWT_DEVICE_STORAGE));

        cudaError_t status = cudaSuccess;

        {
            int32_t cuda_device_ordinal;
            int32_t cuda_device_L2_cache_size;
            int32_t cuda_device_capability;

            libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaGetDevice(&cuda_device_ordinal), status);
            libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaDeviceGetAttribute(&cuda_device_L2_cache_size, cudaDevAttrL2CacheSize, cuda_device_ordinal), status);
            libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::PtxVersion(cuda_device_capability, cuda_device_ordinal), status);

            if (status == cudaSuccess)
            {
                storage->device_L2_cache_bits = 0; while (cuda_device_L2_cache_size >>= 1) { storage->device_L2_cache_bits += 1; };

                storage->cuda_block_threads = (cuda_device_capability == 860 || cuda_device_capability == 870 || cuda_device_capability == 890) ? 768 : 512;
            }
        }
               
        if (status == cudaSuccess)
        {
            int64_t num_descriptors = ((max_reduced_length / (storage->cuda_block_threads * 4)) + 1024) & (-1024);

            {
                cub::DoubleBuffer<uint8_t> uint8_db;
                cub::DoubleBuffer<uint32_t> uint32_db;
                cub::DoubleBuffer<uint64_t> uint64_db;

                size_t temp_radix_segmented_sort_k32v32 = 0;
                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceSegmentedSort::SortPairs(NULL, temp_radix_segmented_sort_k32v32, uint32_db, uint32_db, (int)max_reduced_length, (int)max_reduced_length / 2, uint32_db.Current(), uint32_db.Current()), status);

                size_t temp_radix_sort_k32v32 = 0;
                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(NULL, temp_radix_sort_k32v32, uint32_db, uint32_db, (uint32_t)max_reduced_length), status);

                size_t temp_radix_sort_k64v32 = 0;
                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(NULL, temp_radix_sort_k64v32, uint64_db, uint32_db, (uint32_t)max_reduced_length), status);

                size_t temp_radix_sort_k32v64 = 0;
                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cub::DeviceRadixSort::SortPairs(NULL, temp_radix_sort_k32v64, uint32_db, uint64_db, (uint32_t)max_reduced_length), status);

                storage->device_ssort_temp_storage_size = std::max(temp_radix_segmented_sort_k32v32, (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint32_t));
                storage->device_rsort_temp_storage_size = std::max(std::max(temp_radix_sort_k32v32, temp_radix_sort_k64v32), temp_radix_sort_k32v64);

                storage->device_ssort_temp_storage_size = (storage->device_ssort_temp_storage_size + (size_t)1023) & (size_t)(-1024);
                storage->device_rsort_temp_storage_size = (storage->device_rsort_temp_storage_size + (size_t)1023) & (size_t)(-1024);
            }

            if (status == cudaSuccess)
            {
                size_t device_storage_size = 0;

                device_storage_size += storage->device_ssort_temp_storage_size;
                device_storage_size += storage->device_rsort_temp_storage_size;

                device_storage_size += (max_expanded_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint8_t);

                device_storage_size += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint8_t);
                device_storage_size += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);
                device_storage_size += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);
                device_storage_size += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);

                device_storage_size += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint4);
                device_storage_size += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint4);
                device_storage_size += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint2);

                status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMalloc((void **)&storage->device_storage, device_storage_size), status);

                if (status == cudaSuccess)
                {
                    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaMallocHost((void **)&storage->host_pinned_storage, storage->host_pinned_storage_size = 256 * sizeof(uint32_t)), status);
                    status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamCreate(&storage->cuda_stream), status);
                }
            }

            if (status == cudaSuccess)
            {
                uint8_t * device_alloc              = (uint8_t *)storage->device_storage;

                storage->device_ssort_temp_storage  = (void *)device_alloc; device_alloc += storage->device_ssort_temp_storage_size;
                storage->device_rsort_temp_storage  = (void *)device_alloc; device_alloc += storage->device_rsort_temp_storage_size;

                storage->device_T                   = (uint8_t  *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (max_expanded_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint8_t);

                storage->device_heads               = (uint8_t  *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint8_t);
                storage->device_SA_temp_SA          = (uint64_t *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);
                storage->device_keys_temp_keys      = (uint64_t *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);
                storage->device_offsets_ISA         = (uint64_t *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (max_reduced_length + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint64_t);

                storage->device_descriptors_large   = (uint4    *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint4);
                storage->device_descriptors_copy    = (uint4    *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint4);
                storage->device_descriptors_small   = (uint2    *)(void *)device_alloc + CUDA_DEVICE_PADDING; device_alloc += (num_descriptors + (int64_t)2 * CUDA_DEVICE_PADDING) * sizeof(uint2);

                storage->device_temp_ISA            = (uint32_t *)(void *)storage->device_ssort_temp_storage + CUDA_DEVICE_PADDING;

                storage->device_SA                  = (uint32_t *)(void *)(storage->device_SA_temp_SA       - CUDA_DEVICE_PADDING) + 1 * CUDA_DEVICE_PADDING;
                storage->device_keys                = (uint32_t *)(void *)(storage->device_keys_temp_keys   - CUDA_DEVICE_PADDING) + 1 * CUDA_DEVICE_PADDING;
                storage->device_offsets             = (uint32_t *)(void *)(storage->device_offsets_ISA      - CUDA_DEVICE_PADDING) + 1 * CUDA_DEVICE_PADDING;

                storage->device_temp_SA             = (uint32_t *)(void *)(storage->device_SA_temp_SA       - CUDA_DEVICE_PADDING) + 3 * CUDA_DEVICE_PADDING + max_reduced_length;
                storage->device_temp_keys           = (uint32_t *)(void *)(storage->device_keys_temp_keys   - CUDA_DEVICE_PADDING) + 3 * CUDA_DEVICE_PADDING + max_reduced_length;
                storage->device_ISA                 = (uint32_t *)(void *)(storage->device_offsets_ISA      - CUDA_DEVICE_PADDING) + 3 * CUDA_DEVICE_PADDING + max_reduced_length;

                storage->max_length                 = max_length;

                *device_storage = storage;
                return LIBCUBWT_NO_ERROR;
            }
        }

        libcubwt_free_device_storage(storage);

        return libcubwt_get_error_code(status);
    }

    return LIBCUBWT_NOT_ENOUGH_MEMORY;
}

int64_t libcubwt_free_device_storage(void * device_storage)
{
    cudaError_t status = cudaSuccess;

    LIBCUBWT_DEVICE_STORAGE * storage = (LIBCUBWT_DEVICE_STORAGE *)device_storage;
    if (storage != NULL)
    {
        if (storage->device_storage != NULL)
        {
            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaStreamDestroy(storage->cuda_stream), status);
            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaFreeHost((void *)storage->host_pinned_storage), status);
            status = libcubwt_cuda_safe_call(__FILE__, __LINE__, cudaFree((void *)storage->device_storage), status);
        }

        free(storage);
    }

    return status != cudaSuccess ? libcubwt_get_error_code(status) : LIBCUBWT_NO_ERROR;
}

int64_t libcubwt_bwt(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n)
{
    LIBCUBWT_DEVICE_STORAGE * storage = (LIBCUBWT_DEVICE_STORAGE *)device_storage;

    if ((storage == NULL) || (T == NULL) || (L == NULL) || (n < 16) || (n > storage->max_length))
    {
        return LIBCUBWT_BAD_PARAMETER;
    }

    cudaError_t status; uint32_t index;
    if ((status = libcubwt_compute_burrows_wheeler_transform(storage, T, n, n, &index)) == cudaSuccess &&
        (status = libcubwt_copy_burrows_wheeler_transform(storage, T, L, n, index)) == cudaSuccess)
    {
        return index;
    }

    return libcubwt_get_error_code(status);
}

int64_t libcubwt_bwt_aux(void * device_storage, const uint8_t * T, uint8_t * L, int64_t n, int64_t r, uint32_t * I)
{
    LIBCUBWT_DEVICE_STORAGE * storage = (LIBCUBWT_DEVICE_STORAGE *)device_storage;

    if ((storage == NULL) || (T == NULL) || (L == NULL) || (n < 16) || (n > storage->max_length) || (r < 4) || ((r & (r - 1)) != 0) || ((n + r - 1) / r > 255) || (I == NULL))
    {
        return LIBCUBWT_BAD_PARAMETER;
    }

    cudaError_t status;
    if ((status = libcubwt_compute_burrows_wheeler_transform(storage, T, n, r, I)) == cudaSuccess &&
        (status = libcubwt_copy_burrows_wheeler_transform(storage, T, L, n, I[0])) == cudaSuccess)
    {
        return LIBCUBWT_NO_ERROR;
    }

    return libcubwt_get_error_code(status);
}
