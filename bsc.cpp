/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Block Sorting Compressor                                  */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

   Copyright (c) 2009-2026 Ilya Grebnov <ilya.grebnov@gmail.com>

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

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <memory.h>

#include "libbsc/libbsc.h"
#include "libbsc/filters.h"
#include "libbsc/platform/platform.h"

#pragma pack(push, 1)

#define LIBBSC_CONTEXTS_AUTODETECT   3

unsigned char bscFileSign[4] = {'b', 's', 'c', 0x31};

typedef struct BSC_BLOCK_HEADER
{
    long long       blockOffset;
    signed char     recordSize;
    signed char     sortingContexts;
} BSC_BLOCK_HEADER;

#pragma pack(pop)

int paramBlockSize                = 25 * 1024 * 1024;
int paramBlockSorter              = LIBBSC_BLOCKSORTER_BWT;
int paramCoder                    = LIBBSC_CODER_QLFC_STATIC;
int paramSortingContexts          = LIBBSC_CONTEXTS_FOLLOWING;

int paramEnableParallelProcessing = 1;
int paramEnableMultiThreading     = 1;
int paramEnableFastMode           = 1;
int paramEnableLargePages         = 0;
int paramEnableCUDA               = 0;
int paramEnableSegmentation       = 0;
int paramEnableReordering         = 0;
int paramEnableLZP                = 1;
int paramLZPHashSize              = LIBBSC_DEFAULT_LZPHASHSIZE;
int paramLZPMinLen                = LIBBSC_DEFAULT_LZPMINLEN;

int paramFeatures()
{
    int features =
        (paramEnableFastMode       ? LIBBSC_FEATURE_FASTMODE       : LIBBSC_FEATURE_NONE) |
        (paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE) |
        (paramEnableLargePages     ? LIBBSC_FEATURE_LARGEPAGES     : LIBBSC_FEATURE_NONE) |
        (paramEnableCUDA           ? LIBBSC_FEATURE_CUDA           : LIBBSC_FEATURE_NONE)
    ;

    return features;
}

#if defined(__GNUC__) && (defined(_GLIBCXX_USE_LFS) || defined(__MINGW32__))
    #define BSC_FSEEK fseeko64
    #define BSC_FTELL ftello64
    #define BSC_FILEOFFSET off64_t
#elif defined(_MSC_VER) && _MSC_VER >= 1400
    #define BSC_FSEEK _fseeki64
    #define BSC_FTELL _ftelli64
    #define BSC_FILEOFFSET __int64
#else
    #define BSC_FSEEK fseek
    #define BSC_FTELL ftell
    #define BSC_FILEOFFSET long
#endif

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>

    static unsigned long long bsc_now_ns() 
    {
        static LARGE_INTEGER freq = { 0 };
        if (freq.QuadPart == 0)
        {
            QueryPerformanceFrequency(&freq);
        }

        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);

        unsigned long long ticks = (unsigned long long)counter.QuadPart;
        unsigned long long frequency = (unsigned long long)freq.QuadPart;
        unsigned long long whole_seconds = ticks / frequency;
        unsigned long long remainder_ticks = ticks % frequency;
        unsigned long long ns = whole_seconds * 1000000000ull;
        unsigned long long fractional_ns = (remainder_ticks * 1000000000ull) / frequency;

        return ns + fractional_ns;
    }

    typedef HANDLE BSC_FILEHANDLE;

    #define BSC_INVALID_FILEHANDLE INVALID_HANDLE_VALUE

    static BSC_FILEHANDLE bsc_open_positional(FILE * file, bool write)
    {
        HANDLE handle = (HANDLE)_get_osfhandle(_fileno(file));
        if (handle == INVALID_HANDLE_VALUE || GetFileType(handle) != FILE_TYPE_DISK) return BSC_INVALID_FILEHANDLE;

        return ReOpenFile(handle, write ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_FLAG_OVERLAPPED);
    }

    static void bsc_close_positional(BSC_FILEHANDLE handle)
    {
        if (handle != BSC_INVALID_FILEHANDLE) CloseHandle(handle);
    }

    static bool bsc_positional_io(BSC_FILEHANDLE handle, unsigned char * buffer, int size, BSC_FILEOFFSET offset, bool write)
    {
        if (offset < 0) return false;

        OVERLAPPED overlapped; memset(&overlapped, 0, sizeof(overlapped));

        overlapped.Offset       = (DWORD)((unsigned long long)offset >>  0);
        overlapped.OffsetHigh   = (DWORD)((unsigned long long)offset >> 32);
        overlapped.hEvent       = CreateEvent(NULL, TRUE, FALSE, NULL);

        if (overlapped.hEvent == NULL) return false;

        DWORD bytes = 0;
        BOOL  result = write ? WriteFile(handle, buffer, (DWORD)size, NULL, &overlapped) : ReadFile(handle, buffer, (DWORD)size, NULL, &overlapped);
        if (result || GetLastError() == ERROR_IO_PENDING) result = GetOverlappedResult(handle, &overlapped, &bytes, TRUE);

        CloseHandle(overlapped.hEvent);

        return result && bytes == (DWORD)size;
    }
#else
    #include <time.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <errno.h>
    #include <sys/stat.h>

    static unsigned long long bsc_now_ns()
    {
        struct timespec ts;
        struct timeval  tv;

#ifdef CLOCK_UPTIME_RAW
        if (clock_gettime(CLOCK_UPTIME_RAW, &ts) == 0)
        {
            return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
        }
#endif

#ifdef CLOCK_MONOTONIC_RAW
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0)
        {
            return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
        }
#endif

#ifdef CLOCK_MONOTONIC
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        {
            return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
        }
#endif

        gettimeofday(&tv, NULL);

        return (unsigned long long)tv.tv_sec * 1000000000ull + (unsigned long long)tv.tv_usec * 1000ull;
     }

    typedef int BSC_FILEHANDLE;

    #define BSC_INVALID_FILEHANDLE (-1)

    static BSC_FILEHANDLE bsc_open_positional(FILE * file, bool write)
    {
        struct stat st;
        if (sizeof(off_t) < sizeof(BSC_FILEOFFSET) || fstat(fileno(file), &st) != 0 || !S_ISREG(st.st_mode)) return BSC_INVALID_FILEHANDLE;

        return fileno(file);
    }

    static void bsc_close_positional(BSC_FILEHANDLE handle)
    {
    }

    static bool bsc_positional_io(BSC_FILEHANDLE handle, unsigned char * buffer, int size, BSC_FILEOFFSET offset, bool write)
    {
        if (offset < 0) return false;

        while (size > 0)
        {
            ssize_t bytes = write ? pwrite(handle, buffer, (size_t)size, (off_t)offset) : pread(handle, buffer, (size_t)size, (off_t)offset);
            if (bytes < 0 && errno == EINTR) continue;
            if (bytes <= 0) return false;

            buffer += bytes; size -= (int)bytes; offset += (BSC_FILEOFFSET)bytes;
        }

        return true;
    }
#endif

#ifdef LIBBSC_OPENMP

#ifdef _WIN32
    #include <process.h>

    typedef HANDLE  BSC_THREAD;
    typedef SRWLOCK BSC_MUTEX;

    #define BSC_MUTEX_INITIALIZER       SRWLOCK_INIT
    #define BSC_THREAD_ROUTINE(name)    static unsigned __stdcall name(void * argument)

    static void bsc_mutex_lock(BSC_MUTEX * mutex)
    {
        AcquireSRWLockExclusive(mutex);
    }

    static void bsc_mutex_unlock(BSC_MUTEX * mutex)
    {
        ReleaseSRWLockExclusive(mutex);
    }

    static bool bsc_thread_start(BSC_THREAD * thread, unsigned (__stdcall * routine)(void *), void * argument)
    {
        *thread = (HANDLE)_beginthreadex(NULL, 0, routine, argument, 0, NULL);

        return *thread != NULL;
    }

    static void bsc_thread_join(BSC_THREAD thread)
    {
        WaitForSingleObject(thread, INFINITE); CloseHandle(thread);
    }
#else
    #include <pthread.h>

    typedef pthread_t       BSC_THREAD;
    typedef pthread_mutex_t BSC_MUTEX;

    #define BSC_MUTEX_INITIALIZER       PTHREAD_MUTEX_INITIALIZER
    #define BSC_THREAD_ROUTINE(name)    static void * name(void * argument)

    static void bsc_mutex_lock(BSC_MUTEX * mutex)
    {
        pthread_mutex_lock(mutex);
    }

    static void bsc_mutex_unlock(BSC_MUTEX * mutex)
    {
        pthread_mutex_unlock(mutex);
    }

    static bool bsc_thread_start(BSC_THREAD * thread, void * (* routine)(void *), void * argument)
    {
        return pthread_create(thread, NULL, routine, argument) == 0;
    }

    static void bsc_thread_join(BSC_THREAD thread)
    {
        pthread_join(thread, NULL);
    }
#endif

static int bsc_block_threads(int blockSize, int maxThreads)
{
    int numThreads = 1 + blockSize / (256 * 1024);

    return numThreads < maxThreads ? numThreads : maxThreads;
}

enum { BSC_LOCK_INPUT, BSC_LOCK_OUTPUT, BSC_LOCK_PRINT };

static BSC_MUTEX bscLocks[3] = { BSC_MUTEX_INITIALIZER, BSC_MUTEX_INITIALIZER, BSC_MUTEX_INITIALIZER };

struct BSC_LOCK_GUARD
{
    BSC_MUTEX * mutex;

    BSC_LOCK_GUARD(BSC_MUTEX * mutex) : mutex(mutex) { bsc_mutex_lock(mutex); }
    ~BSC_LOCK_GUARD()                                { bsc_mutex_unlock(mutex); }
};

#define BSC_LOCK(id) BSC_LOCK_GUARD bscLockGuard(&bscLocks[id])

#else

#define BSC_LOCK(id)

#endif

int segmentedBlock[256];

static void bsc_exit(int code)
{
    fflush(NULL); _Exit(code);
}

void compression_omp(char * input_name, FILE * fInput, BSC_FILEHANDLE hInput, BSC_FILEOFFSET& inFileSize, char * output_name,  FILE * fOutput, BSC_FILEOFFSET& outFileSize, int& segmentationStart, int& segmentationEnd, bool showProgress)
{
    unsigned char * buffer = (unsigned char *)bsc_malloc(paramBlockSize + LIBBSC_HEADER_SIZE);
    if (buffer == NULL)
    {
        {
            BSC_LOCK(BSC_LOCK_PRINT);

            fprintf(stderr, "Not enough memory! Please check README file for more information.\n");
            bsc_exit(2);
        }
    }

    while (true)
    {
        BSC_FILEOFFSET  blockOffset     = 0;
        int             dataSize        = 0;

        {
            BSC_LOCK(BSC_LOCK_INPUT);

            if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != inFileSize))
            {
                if (showProgress)
                {
                    double progress = (100.0 * (double)BSC_FTELL(fInput)) / inFileSize;
                    fprintf(stdout, "\rCompressing %.55s(%02d%%)", input_name, (int)progress);
                    fflush(stdout);
                }

                blockOffset = BSC_FTELL(fInput);

                int currentBlockSize = paramBlockSize;
                if (paramEnableSegmentation)
                {
                    if (segmentationEnd - segmentationStart > 1) currentBlockSize = segmentedBlock[segmentationStart];
                }

                if (hInput != BSC_INVALID_FILEHANDLE)
                {
                    dataSize = (int)(inFileSize - blockOffset < currentBlockSize ? inFileSize - blockOffset : currentBlockSize);
                    if (BSC_FSEEK(fInput, blockOffset + dataSize, SEEK_SET)) dataSize = 0;
                }
                else
                {
                    dataSize = (int)fread(buffer, 1, currentBlockSize, fInput);
                }

                if (dataSize <= 0)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", input_name);
                    bsc_exit(1);
                }

                if (paramEnableSegmentation)
                {
                    bool bSegmentation = false;

                    if (segmentationStart == segmentationEnd) bSegmentation = true;
                    if ((segmentationEnd - segmentationStart == 1) && (dataSize != segmentedBlock[segmentationStart])) bSegmentation = true;

                    if (bSegmentation)
                    {
                        segmentationStart = 0; segmentationEnd = bsc_detect_segments(buffer, dataSize, segmentedBlock, 256, paramFeatures());
                        if (segmentationEnd <= LIBBSC_NO_ERROR)
                        {
                            switch (segmentationEnd)
                            {
                                case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                                default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                            }
                            bsc_exit(2);
                        }
                    }

                    int newDataSize = segmentedBlock[segmentationStart++];
                    if (dataSize != newDataSize)
                    {
                        BSC_FILEOFFSET pos = BSC_FTELL(fInput) - dataSize + newDataSize;
                        BSC_FSEEK(fInput, pos, SEEK_SET);
                        dataSize = newDataSize;
                    }
                }
            }
        }

        if (dataSize == 0) break;

        if ((hInput != BSC_INVALID_FILEHANDLE) && !bsc_positional_io(hInput, buffer, dataSize, blockOffset, false))
        {
            {
                BSC_LOCK(BSC_LOCK_PRINT);

                fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                bsc_exit(1);
            }
        }

        signed char recordSize = 1;
        if (paramEnableReordering)
        {
            recordSize = bsc_detect_recordsize(buffer, dataSize, paramFeatures());
            if (recordSize < LIBBSC_NO_ERROR)
            {
                {
                    BSC_LOCK(BSC_LOCK_PRINT);

                    switch (recordSize)
                    {
                        case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                        default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    bsc_exit(2);
                }
            }
            if (recordSize > 1)
            {
                int result = bsc_reorder_forward(buffer, dataSize, recordSize, paramFeatures());
                if (result != LIBBSC_NO_ERROR)
                {
                    {
                        BSC_LOCK(BSC_LOCK_PRINT);

                        switch (result)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        bsc_exit(2);
                    }
                }
            }
        }

        signed char sortingContexts = paramSortingContexts;
        if (paramSortingContexts == LIBBSC_CONTEXTS_AUTODETECT)
        {
            sortingContexts = bsc_detect_contextsorder(buffer, dataSize, paramFeatures());
            if (sortingContexts < LIBBSC_NO_ERROR)
            {
                {
                    BSC_LOCK(BSC_LOCK_PRINT);

                    switch (sortingContexts)
                    {
                        case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                        default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    bsc_exit(2);
                }
            }
        }
        if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
        {
            int result = bsc_reverse_block(buffer, dataSize, paramFeatures());
            if (result != LIBBSC_NO_ERROR)
            {
                {
                    BSC_LOCK(BSC_LOCK_PRINT);

                    fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    bsc_exit(2);
                }
            }
        }

        int blockSize = bsc_compress(buffer, buffer, dataSize, paramLZPHashSize, paramLZPMinLen, paramBlockSorter, paramCoder, paramFeatures());
        if (blockSize == LIBBSC_NOT_COMPRESSIBLE)
        {
            {
                BSC_LOCK(BSC_LOCK_INPUT);

                sortingContexts = LIBBSC_CONTEXTS_FOLLOWING; recordSize = 1;

                BSC_FILEOFFSET pos = BSC_FTELL(fInput);
                {
                    BSC_FSEEK(fInput, blockOffset, SEEK_SET);
                    if (dataSize != (int)fread(buffer, 1, dataSize, fInput))
                    {
                        fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        bsc_exit(2);
                    }
                }
                BSC_FSEEK(fInput, pos, SEEK_SET);
            }

            blockSize = bsc_store(buffer, buffer, dataSize, paramFeatures());
        }
        if (blockSize < LIBBSC_NO_ERROR)
        {
            {
                BSC_LOCK(BSC_LOCK_PRINT);

                switch (blockSize)
                {
                    case LIBBSC_NOT_ENOUGH_MEMORY       : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                    case LIBBSC_NOT_SUPPORTED           : fprintf(stderr, "\nSpecified compression method is not supported on this platform!\n"); break;
                    case LIBBSC_GPU_ERROR               : fprintf(stderr, "\nGeneral GPU failure! Please check README file for more information.\n"); break;
                    case LIBBSC_GPU_NOT_SUPPORTED       : fprintf(stderr, "\nYour GPU is not supported! Please check README file for more information.\n"); break;
                    case LIBBSC_GPU_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough GPU memory! Please check README file for more information.\n"); break;

                    default                             : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                }
                bsc_exit(2);
            }
        }

        {
            BSC_LOCK(BSC_LOCK_OUTPUT);

            BSC_BLOCK_HEADER header = {blockOffset, recordSize, sortingContexts};

            if (fwrite(&header, sizeof(BSC_BLOCK_HEADER), 1, fOutput) != 1)
            {
                fprintf(stderr, "\nIO error on file: %s!\n", output_name);
                bsc_exit(1);
            }

            outFileSize += (int)(sizeof(BSC_BLOCK_HEADER));

            if ((int)fwrite(buffer, 1, blockSize, fOutput) != blockSize)
            {
                fprintf(stderr, "\nIO error on file: %s!\n", output_name);
                bsc_exit(1);
            }

            outFileSize += (int)(blockSize);
        }

    }

    bsc_free(buffer);
}

#ifdef LIBBSC_OPENMP

struct BSC_COMPRESSION_WORKER
{
    int numThreads; char * input_name; FILE * fInput; BSC_FILEHANDLE hInput; BSC_FILEOFFSET * inFileSize; char * output_name; FILE * fOutput; BSC_FILEOFFSET * outFileSize; int * segmentationStart; int * segmentationEnd;
};

BSC_THREAD_ROUTINE(compression_worker)
{
    BSC_COMPRESSION_WORKER * worker = (BSC_COMPRESSION_WORKER *)argument;

    omp_set_num_threads(worker->numThreads); compression_omp(worker->input_name, worker->fInput, worker->hInput, *worker->inFileSize, worker->output_name, worker->fOutput, *worker->outFileSize, *worker->segmentationStart, *worker->segmentationEnd, false);

    return 0;
}

#endif

void Compression(char * argv[])
{
    if (!paramEnableLZP)
    {
        paramLZPHashSize = 0;
        paramLZPMinLen = 0;
    }

    FILE * fInput = fopen(argv[2], "rb");
    if (fInput == NULL)
    {
        fprintf(stderr, "Can't open input file: %s!\n", argv[2]);
        exit(1);
    }

    FILE * fOutput = fopen(argv[3], "wb");
    if (fOutput == NULL)
    {
        fprintf(stderr, "Can't create output file: %s!\n", argv[3]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    BSC_FILEOFFSET inFileSize = BSC_FTELL(fInput);
    BSC_FILEOFFSET outFileSize = 0;

    if (inFileSize < 0)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_SET))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (paramBlockSize > inFileSize)
    {
        paramBlockSize = (int)inFileSize;
    }

    if (fwrite(bscFileSign, sizeof(bscFileSign), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    outFileSize += (int)(sizeof(bscFileSign));

    int nBlocks = paramBlockSize > 0 ? (int)((inFileSize + paramBlockSize - 1) / paramBlockSize) : 0;
    if (fwrite(&nBlocks, sizeof(nBlocks), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    outFileSize += (int)(sizeof(nBlocks));

    unsigned long long startTime = bsc_now_ns();

    int segmentationStart = 0, segmentationEnd = 0;

#ifdef LIBBSC_OPENMP
    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();

        if (numThreads / 2 < nBlocks) paramEnableMultiThreading = 0;
        if (numThreads > 1 * nBlocks) numThreads = nBlocks;
        if (numThreads < 1 * nBlocks)
        {
            int nRounds = (nBlocks + numThreads - 1) / numThreads;
            numThreads  = (nBlocks + nRounds - 1) / nRounds;
        }
    }

    if (numThreads > 1)
    {
        BSC_FILEHANDLE hInput = paramEnableSegmentation ? BSC_INVALID_FILEHANDLE : bsc_open_positional(fInput, false);

        BSC_COMPRESSION_WORKER worker = { omp_get_max_threads() / numThreads, argv[2], fInput, hInput, &inFileSize, argv[3], fOutput, &outFileSize, &segmentationStart, &segmentationEnd };

        int numStarted = 0; BSC_THREAD * threads = (BSC_THREAD *)malloc(numThreads * sizeof(BSC_THREAD));
        while ((threads != NULL) && (numStarted < numThreads - 1) && bsc_thread_start(&threads[numStarted], compression_worker, &worker)) { numStarted++; }

        omp_set_num_threads(worker.numThreads); compression_omp(argv[2], fInput, hInput, inFileSize, argv[3], fOutput, outFileSize, segmentationStart, segmentationEnd, true);

        while (numStarted > 0) { bsc_thread_join(threads[--numStarted]); }
        free(threads);

        bsc_close_positional(hInput);
    }
    else
#endif
    {
        compression_omp(argv[2], fInput, BSC_INVALID_FILEHANDLE, inFileSize, argv[3], fOutput, outFileSize, segmentationStart, segmentationEnd, true);
    }

    if (fflush(fOutput))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    startTime = bsc_now_ns() - startTime;

    fprintf(stdout, "\r%.55s encoded %.0f => %.0f in %.3fs (%.2f MB/s)\n", argv[2], (double)inFileSize, (double)outFileSize, (double)startTime / 1e9, ((double)inFileSize) * 1000.0 / (double)startTime);

    fclose(fInput); fclose(fOutput);
}

void decompression_omp(char * input_name, FILE * fInput, BSC_FILEHANDLE hInput, BSC_FILEOFFSET& inFileSize, char * output_name, FILE * fOutput, BSC_FILEHANDLE hOutput, BSC_FILEOFFSET& outFileSize, BSC_FILEOFFSET& outFileEnd, int& nBlocksRead, bool showProgress)
{
    int bufferSize = -1; unsigned char * buffer = NULL;

#ifdef LIBBSC_OPENMP
    int maxThreads = omp_get_max_threads(), numThreads = 0;
#endif

    bool positional = (hInput != BSC_INVALID_FILEHANDLE) && (hOutput != BSC_INVALID_FILEHANDLE);

    while (true)
    {
        BSC_FILEOFFSET  blockOffset     = 0;
        BSC_FILEOFFSET  payloadOffset   = 0;

        signed char     sortingContexts = 0;
        signed char     recordSize      = 0;
        int             blockSize       = 0;
        int             dataSize        = 0;

        {
            BSC_LOCK(BSC_LOCK_INPUT);

            if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != inFileSize))
            {
                if (showProgress)
                {
                    double progress = (100.0 * (double)BSC_FTELL(fInput)) / inFileSize;
                    fprintf(stdout, "\rDecompressing %.55s(%02d%%)", input_name, (int)progress);
                    fflush(stdout);
                }

                BSC_BLOCK_HEADER header = {0, 0, 0};
                if (fread(&header, sizeof(BSC_BLOCK_HEADER), 1, fInput) != 1)
                {
                    fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                    bsc_exit(1);
                }

                recordSize = header.recordSize;
                if (recordSize < 1)
                {
                    fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                    bsc_exit(2);
                }

                sortingContexts = header.sortingContexts;
                if ((sortingContexts != LIBBSC_CONTEXTS_FOLLOWING) && (sortingContexts != LIBBSC_CONTEXTS_PRECEDING))
                {
                    fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                    bsc_exit(2);
                }

                blockOffset = (BSC_FILEOFFSET)header.blockOffset;

                unsigned char bscBlockHeader[LIBBSC_HEADER_SIZE];

                if (fread(bscBlockHeader, LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                {
                    fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                    bsc_exit(1);
                }

                if (bsc_block_info(bscBlockHeader, LIBBSC_HEADER_SIZE, &blockSize, &dataSize, paramFeatures()) != LIBBSC_NO_ERROR)
                {
                    fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                    bsc_exit(2);
                }

                nBlocksRead++; if (outFileEnd < blockOffset + dataSize) outFileEnd = blockOffset + dataSize;

                if ((blockSize > bufferSize) || (dataSize > bufferSize))
                {
                    if (blockSize > bufferSize) bufferSize = blockSize;
                    if (dataSize  > bufferSize) bufferSize = dataSize;

                    if (buffer != NULL) { bsc_free(buffer); } buffer = (unsigned char *)bsc_malloc(bufferSize);
                }

                if (buffer == NULL)
                {
                    fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n");
                    bsc_exit(2);
                }

                memcpy(buffer, bscBlockHeader, LIBBSC_HEADER_SIZE);

                if (positional)
                {
                    payloadOffset = BSC_FTELL(fInput);
                    if ((blockSize == LIBBSC_HEADER_SIZE) || (blockSize - LIBBSC_HEADER_SIZE > inFileSize - payloadOffset) || BSC_FSEEK(fInput, payloadOffset + (blockSize - LIBBSC_HEADER_SIZE), SEEK_SET))
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                        bsc_exit(1);
                    }
                }
                else if (fread(buffer + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                {
                    fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                    bsc_exit(1);
                }
            }
        }

        if (dataSize == 0) break;

        if (positional && !bsc_positional_io(hInput, buffer + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, payloadOffset, false))
        {
            {
                BSC_LOCK(BSC_LOCK_PRINT);

                fprintf(stderr, "\nUnexpected end of file: %s!\n", input_name);
                bsc_exit(1);
            }
        }

#ifdef LIBBSC_OPENMP
        if (numThreads < bsc_block_threads(dataSize, maxThreads)) omp_set_num_threads(numThreads = bsc_block_threads(dataSize, maxThreads));
#endif

        int result = bsc_decompress(buffer, blockSize, buffer, dataSize, paramFeatures());
        if (result < LIBBSC_NO_ERROR)
        {
            {
                BSC_LOCK(BSC_LOCK_PRINT);

                switch (result)
                {
                    case LIBBSC_DATA_CORRUPT            : fprintf(stderr, "\nThe compressed data is corrupted!\n"); break;
                    case LIBBSC_NOT_ENOUGH_MEMORY       : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                    case LIBBSC_GPU_ERROR               : fprintf(stderr, "\nGeneral GPU failure! Please check README file for more information.\n"); break;
                    case LIBBSC_GPU_NOT_SUPPORTED       : fprintf(stderr, "\nYour GPU is not supported! Please check README file for more information.\n"); break;
                    case LIBBSC_GPU_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough GPU memory! Please check README file for more information.\n"); break;

                    default                             : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                }
                bsc_exit(2);
            }
        }

        if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
        {
            result = bsc_reverse_block(buffer, dataSize, paramFeatures());
            if (result != LIBBSC_NO_ERROR)
            {
                {
                    BSC_LOCK(BSC_LOCK_PRINT);

                    fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    bsc_exit(2);
                }
            }
        }

        if (recordSize > 1)
        {
            result = bsc_reorder_reverse(buffer, dataSize, recordSize, paramFeatures());
            if (result != LIBBSC_NO_ERROR)
            {
                {
                    BSC_LOCK(BSC_LOCK_PRINT);

                    switch (result)
                    {
                        case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory! Please check README file for more information.\n"); break;
                        default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    bsc_exit(2);
                }
            }
        }

        if (positional && !bsc_positional_io(hOutput, buffer, dataSize, blockOffset, true))
        {
            {
                BSC_LOCK(BSC_LOCK_PRINT);

                fprintf(stderr, "\nIO error on file: %s!\n", output_name);
                bsc_exit(1);
            }
        }

        {
            BSC_LOCK(BSC_LOCK_OUTPUT);

            if (!positional)
            {
                if (BSC_FSEEK(fOutput, blockOffset, SEEK_SET))
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", output_name);
                    bsc_exit(1);
                }

                if ((int)fwrite(buffer, 1, dataSize, fOutput) != dataSize)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", output_name);
                    bsc_exit(1);
                }
            }

            outFileSize += (int)(dataSize);
        }
    }

    if (buffer != NULL) bsc_free(buffer);
}

#ifdef LIBBSC_OPENMP

struct BSC_DECOMPRESSION_WORKER
{
    int numThreads; char * input_name; FILE * fInput; BSC_FILEHANDLE hInput; BSC_FILEOFFSET * inFileSize; char * output_name; FILE * fOutput; BSC_FILEHANDLE hOutput; BSC_FILEOFFSET * outFileSize; BSC_FILEOFFSET * outFileEnd; int * nBlocksRead;
};

BSC_THREAD_ROUTINE(decompression_worker)
{
    BSC_DECOMPRESSION_WORKER * worker = (BSC_DECOMPRESSION_WORKER *)argument;

    omp_set_num_threads(worker->numThreads); decompression_omp(worker->input_name, worker->fInput, worker->hInput, *worker->inFileSize, worker->output_name, worker->fOutput, worker->hOutput, *worker->outFileSize, *worker->outFileEnd, *worker->nBlocksRead, false);

    return 0;
}

#endif

void Decompression(char * argv[])
{
    FILE * fInput = fopen(argv[2], "rb");
    if (fInput == NULL)
    {
        fprintf(stderr, "Can't open input file: %s!\n", argv[2]);
        exit(1);
    }

    FILE * fOutput = fopen(argv[3], "wb");
    if (fOutput == NULL)
    {
        fprintf(stderr, "Can't create output file: %s!\n", argv[3]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    BSC_FILEOFFSET inFileSize = BSC_FTELL(fInput);
    BSC_FILEOFFSET outFileSize = 0, outFileEnd = 0;

    if (inFileSize < 0)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_SET))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    unsigned char inputFileSign[sizeof(bscFileSign)];

    if (fread(inputFileSign, sizeof(bscFileSign), 1, fInput) != 1)
    {
        fprintf(stderr, "This is not bsc archive!\n");
        exit(1);
    }

    if (memcmp(inputFileSign, bscFileSign, sizeof(bscFileSign)) != 0)
    {
        fprintf(stderr, "This is not bsc archive or invalid compression method!\n");
        exit(2);
    }

    int nBlocks = 0, nBlocksRead = 0;
    if (fread(&nBlocks, sizeof(nBlocks), 1, fInput) != 1)
    {
        fprintf(stderr, "This is not bsc archive!\n");
        exit(1);
    }

    BSC_FILEHANDLE hInput  = bsc_open_positional(fInput, false);
    BSC_FILEHANDLE hOutput = bsc_open_positional(fOutput, true);

    unsigned long long startTime = bsc_now_ns();

#ifdef LIBBSC_OPENMP
    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();

        if (numThreads / 2 < nBlocks) paramEnableMultiThreading = 0;
        if (numThreads > 1 * nBlocks) numThreads = nBlocks;
    }

    if (numThreads > 1)
    {
        BSC_DECOMPRESSION_WORKER worker = { omp_get_max_threads() / numThreads, argv[2], fInput, hInput, &inFileSize, argv[3], fOutput, hOutput, &outFileSize, &outFileEnd, &nBlocksRead };

        int numStarted = 0; BSC_THREAD * threads = (BSC_THREAD *)malloc(numThreads * sizeof(BSC_THREAD));
        while ((threads != NULL) && (numStarted < numThreads - 1) && bsc_thread_start(&threads[numStarted], decompression_worker, &worker)) { numStarted++; }

        omp_set_num_threads(worker.numThreads); decompression_omp(argv[2], fInput, hInput, inFileSize, argv[3], fOutput, hOutput, outFileSize, outFileEnd, nBlocksRead, true);

        while (numStarted > 0) { bsc_thread_join(threads[--numStarted]); }
        free(threads);
    }
    else
#endif
    {
        decompression_omp(argv[2], fInput, hInput, inFileSize, argv[3], fOutput, hOutput, outFileSize, outFileEnd, nBlocksRead, true);
    }

    if ((nBlocksRead < nBlocks) || (outFileSize != outFileEnd))
    {
        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
        exit(1);
    }

    bsc_close_positional(hInput); bsc_close_positional(hOutput);

    if (BSC_FSEEK(fOutput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    startTime = bsc_now_ns() - startTime;

    fprintf(stdout, "\r%.55s decoded %.0f => %.0f in %.3fs (%.2f MB/s)\n", argv[2], (double)inFileSize, (double)outFileSize, (double)startTime / 1e9, ((double)outFileSize) * 1000.0 / (double)startTime);

    fclose(fInput); fclose(fOutput);
}

void ShowUsage(void)
{
#if !defined(BSC_DECOMPRESSION_ONLY)
    fprintf(stdout, "Usage: bsc <e|d> inputfile outputfile <options>\n\n");
#elif defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout, "Usage: bsc d inputfile outputfile <options>\n\n");
#else
    fprintf(stdout, "Usage: bsc d inputfile outputfile\n\n");
#endif

#if !defined(BSC_DECOMPRESSION_ONLY)
    fprintf(stdout, "Block sorting options:\n");
    fprintf(stdout, "  -b<size> Block size in megabytes or bytes (for size 10000+), default: -b25\n");
    fprintf(stdout, "             minimum: -b1, maximum: -b2047\n");
    fprintf(stdout, "  -m<algo> Block sorting algorithm, default: -m0\n");
    fprintf(stdout, "             -m0 Burrows Wheeler Transform (default)\n");
#ifndef LIBBSC_NO_SORT_TRANSFORM
    fprintf(stdout, "             -m3..8 Sort Transform of order n\n");
#endif
    fprintf(stdout, "  -c<ctx>  Contexts for sorting, default: -cf\n");
    fprintf(stdout, "             -cf Following contexts (default)\n");
    fprintf(stdout, "             -cp Preceding contexts\n");
    fprintf(stdout, "             -ca Autodetect (experimental)\n");
    fprintf(stdout, "  -e<algo> Entropy encoding algorithm, default: -e1\n");
    fprintf(stdout, "             -e0 Fast Quantized Local Frequency Coding\n");
    fprintf(stdout, "             -e1 Static Quantized Local Frequency Coding (default)\n");
    fprintf(stdout, "             -e2 Adaptive Quantized Local Frequency Coding (best compression)\n");
   
    fprintf(stdout, "\nPreprocessing options:\n");
    fprintf(stdout, "  -p       Disable all preprocessing techniques\n");
    fprintf(stdout, "  -s       Enable segmentation (adaptive block size), default: disable\n");
    fprintf(stdout, "  -r       Enable structured data reordering, default: disable\n");
    fprintf(stdout, "  -l       Enable Lempel-Ziv preprocessing, default: enable\n");
    fprintf(stdout, "  -H<size> LZP dictionary size in bits, default: -H%d\n", LIBBSC_DEFAULT_LZPHASHSIZE);
    fprintf(stdout, "             minimum: -H10, maximum: -H28\n");
    fprintf(stdout, "  -M<size> LZP minimum match length, default: -M%d\n", LIBBSC_DEFAULT_LZPMINLEN);
    fprintf(stdout, "             minimum: -M4, maximum: -M255\n\n");
#endif

#if defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout, "Platform specific options:\n");
#ifdef LIBBSC_CUDA_SUPPORT
    fprintf(stdout, "  -G       Enable Sort Transform acceleration on NVIDIA GPU, default: disable\n");
#endif
#ifdef _WIN32
    fprintf(stdout, "  -P       Enable large 2MB RAM pages, default: disable\n");
#endif
#ifdef LIBBSC_OPENMP
    fprintf(stdout, "  -t       Disable parallel blocks processing, default: enable\n");
    fprintf(stdout, "  -T       Disable multithreading (-T<num> sets num threads), default: enable\n");
#endif
    fprintf(stdout, "\n");
#endif

#if !defined(BSC_DECOMPRESSION_ONLY) || defined(LIBBSC_CUDA_SUPPORT) || defined(_WIN32) || defined(LIBBSC_OPENMP)
    fprintf(stdout,"Options may be combined into one, like -b128p -m5e1\n");
#endif

    exit(1);
}

int ParseNumber(const char * strNum, const char * strEnd)
{
    long long number = 0;
    for (; strNum < strEnd; ++strNum)
    {
        number = number * 10 + (*strNum - '0'); if (number > 2147483647) number = 2147483647;
    }
    return (int)number;
}

void ProcessSwitch(char * s)
{
    if (*s == 0)
    {
        ShowUsage();
    }

    for (; *s != 0; )
    {
        switch (*s++)
        {
            case 'b':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                long long blockSize = ParseNumber(strNum, s); if (blockSize < 10000) blockSize *= 1024 * 1024;
                if ((blockSize < 10000) || (blockSize > 2047 * 1024 * 1024)) ShowUsage();
                paramBlockSize = (int)blockSize;
                break;
            }

            case 'm':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                switch (ParseNumber(strNum, s))
                {
                    case 0   : paramBlockSorter = LIBBSC_BLOCKSORTER_BWT; break;

#ifndef LIBBSC_NO_SORT_TRANSFORM
                    case 3   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST3; break;
                    case 4   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST4; break;
                    case 5   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST5; break;
                    case 6   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST6; break;
                    case 7   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST7; paramEnableCUDA = 1; break;
                    case 8   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST8; paramEnableCUDA = 1; break;
#endif

                    default  : ShowUsage();
                }
                break;
            }

            case 'c':
            {
                switch (*s++)
                {
                    case 'f' : paramSortingContexts = LIBBSC_CONTEXTS_FOLLOWING;  break;
                    case 'p' : paramSortingContexts = LIBBSC_CONTEXTS_PRECEDING;  break;
                    case 'a' : paramSortingContexts = LIBBSC_CONTEXTS_AUTODETECT; break;
                    default  : ShowUsage();
                }
                break;
            }

            case 'e':
            {
                switch (*s++)
                {
                    case '0' : paramCoder = LIBBSC_CODER_QLFC_FAST;     break;
                    case '1' : paramCoder = LIBBSC_CODER_QLFC_STATIC;   break;
                    case '2' : paramCoder = LIBBSC_CODER_QLFC_ADAPTIVE; break;
                    default  : ShowUsage();
                }
                break;
            }

            case 'H':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                paramLZPHashSize = ParseNumber(strNum, s);
                if ((paramLZPHashSize < 10) || (paramLZPHashSize > 28)) ShowUsage();
                break;
            }

            case 'M':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                paramLZPMinLen = ParseNumber(strNum, s);
                if ((paramLZPMinLen < 4) || (paramLZPMinLen > 255)) ShowUsage();
                break;
            }

            case 'l': paramEnableLZP            = 1; break;
            case 's': paramEnableSegmentation   = 1; break;
            case 'r': paramEnableReordering     = 1; break;

            case 'p': paramEnableLZP = paramEnableSegmentation = paramEnableReordering = 0; break;

#ifdef LIBBSC_OPENMP
            case 't': paramEnableParallelProcessing = 0; break;
            case 'T':
            {
                char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                int numThreads = ParseNumber(strNum, s);
                if (numThreads > 1) omp_set_num_threads(numThreads);
                if (numThreads < 2) paramEnableParallelProcessing = paramEnableMultiThreading = 0;
                break;
            }
#endif

#ifdef LIBBSC_CUDA_SUPPORT
            case 'G': paramEnableCUDA           = 1; break;
#endif

#ifdef _WIN32
            case 'P': paramEnableLargePages     = 1; break;
#endif

            default : ShowUsage();
        }
    }
}

void ProcessCommandline(int argc, char * argv[])
{
    if (argc < 4 || strlen(argv[1]) != 1)
    {
        ShowUsage();
    }

    for (int i = 4; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            ProcessSwitch(&argv[i][1]);
        }
        else
        {
            ShowUsage();
        }
    }
}

int main(int argc, char * argv[])
{
#if defined(LIBBSC_OPENMP)
    omp_set_dynamic(0);
#if defined(__INTEL_COMPILER)
    kmp_set_warnings_off();
#endif
#endif

    fprintf(stdout, "This is bsc, Block Sorting Compressor. Version %s, 18 November 2026.\n", LIBBSC_VERSION_STRING);
    fprintf(stdout, "Copyright (c) 2009-2026 Ilya Grebnov <Ilya.Grebnov@gmail.com>.\n\n");

    ProcessCommandline(argc, argv);

    if (bsc_init(paramFeatures()) != LIBBSC_NO_ERROR)
    {
        fprintf(stderr, "\nInternal program error, please contact the author!\n");
        exit(2);
    }

    switch (*argv[1])
    {
#if !defined(BSC_DECOMPRESSION_ONLY)
        case 'e' : case 'E' : Compression(argv); break;
#endif
        case 'd' : case 'D' : Decompression(argv); break;
        default  : ShowUsage();
    }

    return 0;
}

/*-----------------------------------------------------------*/
/* End                                               bsc.cpp */
/*-----------------------------------------------------------*/
