/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Block Sorting Compressor                                  */
/*-----------------------------------------------------------*/

/*--

This file is a part of bsc and/or libbsc, a program and a library for
lossless, block-sorting data compression.

Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@libbsc.com>

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

#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <math.h>
#include <memory.h>

#include "../libbsc/libbsc.h"
#include "../libbsc/filters.h"

#pragma pack(push, 1)

#define LIBBSC_CONTEXTS_AUTODETECT   3

unsigned char bscFileSign[4] = {'b', 's', 'c', 0x26};

typedef struct BSC_BLOCK_HEADER
{
    long long   blockOffset;
    char        recordSize;
    char        sortingContexts;
} BSC_BLOCK_HEADER;

int paramBlockSize                = 25 * 1024 * 1024;
int paramEnableSegmentation       = 0;
int paramEnableReordering         = 0;
int paramEnableFastMode           = 0;
int paramEnableLZP                = 1;
int paramLZPHashSize              = 16;
int paramLZPMinLen                = 128;
int paramBlockSorter              = LIBBSC_BLOCKSORTER_BWT;
int paramSortingContexts          = LIBBSC_CONTEXTS_FOLLOWING;

int paramEnableParallelProcessing = 1;
int paramEnableMultiThreading     = 1;

#pragma pack(pop)

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

#if defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__) || defined(_MSC_VER)
  #include <windows.h>
  double BSC_CLOCK() { return 0.001 * GetTickCount(); }
#elif defined (__unix) || defined (__linux__) || defined (__QNX__) || defined (_AIX)  || defined (__NetBSD__) || defined(macintosh) || defined (_MAC)
  #include <sys/time.h>
  double BSC_CLOCK() { timeval tv; gettimeofday(&tv, 0); return tv.tv_sec + tv.tv_usec * 0.000001; }
#else
  double BSC_CLOCK() { return (double)clock() / CLOCKS_PER_SEC; }
#endif

int segmentedBlock[256];

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

    BSC_FILEOFFSET fileSize = BSC_FTELL(fInput);
    if (fileSize < 0)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (BSC_FSEEK(fInput, 0, SEEK_SET))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[2]);
        exit(1);
    }

    if (paramBlockSize > fileSize)
    {
        paramBlockSize = (int)fileSize;
    }

    if (fwrite(bscFileSign, sizeof(bscFileSign), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    int nBlocks = (int)((fileSize + paramBlockSize - 1) / paramBlockSize);
    if (fwrite(&nBlocks, sizeof(nBlocks), 1, fOutput) != 1)
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    double startTime = BSC_CLOCK();

#ifdef _OPENMP

    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();
        if (numThreads > nBlocks)
        {
            numThreads = nBlocks;
        }
    }

#endif

    int segmentationStart = 0, segmentationEnd = 0;

#ifdef _OPENMP
    #pragma omp parallel default(shared) num_threads(numThreads)
#endif
    {
        unsigned char * buffer = (unsigned char *)malloc(paramBlockSize + LIBBSC_HEADER_SIZE);
        if (buffer == NULL)
        {
#ifdef _OPENMP
            #pragma omp critical(print)
#endif
            {

                fprintf(stderr, "Not enough memory!\n");
                exit(2);
            }
        }

        while (true)
        {
            BSC_FILEOFFSET  blockOffset     = 0;
            int             dataSize        = 0;

#ifdef _OPENMP
            #pragma omp critical(input)
#endif
            {
                if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != fileSize))
                {
#ifdef _OPENMP
                    #pragma omp master
#endif
                    {
                        double progress = (100.0 * (double)BSC_FTELL(fInput)) / fileSize;
                        fprintf(stdout, "\rCompressing %.55s(%02d%%)", argv[2], (int)progress);
                        fflush(stdout);
                    }

                    blockOffset = BSC_FTELL(fInput);

                    int currentBlockSize = paramBlockSize;
                    if (paramEnableSegmentation)
                    {
                        if (segmentationEnd - segmentationStart > 1) currentBlockSize = segmentedBlock[segmentationStart];
                    }

                    dataSize = (int)fread(buffer, 1, currentBlockSize, fInput);
                    if (dataSize <= 0)
                    {
                        fprintf(stderr, "\nIO error on file: %s!\n", argv[2]);
                        exit(1);
                    }

                    if (paramEnableSegmentation)
                    {
                        bool bSegmentation = false;

                        if (segmentationStart == segmentationEnd) bSegmentation = true;
                        if ((segmentationEnd - segmentationStart == 1) && (dataSize != segmentedBlock[segmentationStart])) bSegmentation = true;

                        if (bSegmentation)
                        {
                            segmentationStart = 0; segmentationEnd = bsc_detect_segments(buffer, dataSize, segmentedBlock, 256, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
                            if (segmentationEnd <= LIBBSC_NO_ERROR)
                            {
                                switch (segmentationEnd)
                                {
                                    case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                                    default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                                }
                                exit(2);
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

            char recordSize = 1;
            if (paramEnableReordering)
            {
                recordSize = bsc_detect_recordsize(buffer, dataSize, LIBBSC_FEATURE_FASTMODE);
                if (recordSize < LIBBSC_NO_ERROR)
                {
#ifdef _OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (recordSize)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
                if (recordSize > 1)
                {
                    int result = bsc_reorder_forward(buffer, dataSize, recordSize, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
                    if (result != LIBBSC_NO_ERROR)
                    {
#ifdef _OPENMP
                        #pragma omp critical(print)
#endif
                        {
                            switch (result)
                            {
                                case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                                default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                            }
                            exit(2);
                        }
                    }
                }
            }

            char sortingContexts = paramSortingContexts;
            if (paramSortingContexts == LIBBSC_CONTEXTS_AUTODETECT)
            {
                sortingContexts = bsc_detect_contextsorder(buffer, dataSize, LIBBSC_FEATURE_FASTMODE);
                if (sortingContexts < LIBBSC_NO_ERROR)
                {
#ifdef _OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (sortingContexts)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
            }
            if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
            {
                int result = bsc_reverse_block(buffer, dataSize, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef _OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        exit(2);
                    }
                }
            }

            int features =
                (paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE) |
                (paramEnableFastMode ? LIBBSC_FEATURE_FASTMODE : LIBBSC_FEATURE_NONE)
            ;

            int blockSize = bsc_compress(buffer, buffer, dataSize, paramLZPHashSize, paramLZPMinLen, paramBlockSorter, features);
            if (blockSize == LIBBSC_NOT_COMPRESSIBLE)
            {
#ifdef _OPENMP
                #pragma omp critical(input)
#endif
                {
                    sortingContexts = LIBBSC_CONTEXTS_FOLLOWING; recordSize = 1;

                    BSC_FILEOFFSET pos = BSC_FTELL(fInput);
                    {
                        BSC_FSEEK(fInput, blockOffset, SEEK_SET);
                        if (dataSize != (int)fread(buffer, 1, dataSize, fInput))
                        {
                            fprintf(stderr, "\nInternal program error, please contact the author!\n");
                            exit(2);
                        }
                    }
                    BSC_FSEEK(fInput, pos, SEEK_SET);
                }

                blockSize = bsc_store(buffer, buffer, dataSize);
            }
            if (blockSize < LIBBSC_NO_ERROR)
            {
#ifdef _OPENMP
                #pragma omp critical(print)
#endif
                {
                    switch (blockSize)
                    {
                        case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                        default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    exit(2);
                }
            }

#ifdef _OPENMP
            #pragma omp critical(output)
#endif
            {
                BSC_BLOCK_HEADER header = {blockOffset, recordSize, sortingContexts};

                if (fwrite(&header, sizeof(BSC_BLOCK_HEADER), 1, fOutput) != 1)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }

                if ((int)fwrite(buffer, 1, blockSize, fOutput) != blockSize)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }
            }

        }

        free(buffer);
    }

    fprintf(stdout, "\r%.55s compressed %.0f into %.0f in %.3f seconds.\n", argv[2], (double)fileSize, (double)BSC_FTELL(fOutput), BSC_CLOCK() - startTime);

    fclose(fInput); fclose(fOutput);
}

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

    BSC_FILEOFFSET fileSize = BSC_FTELL(fInput);
    if (fileSize < 0)
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

    int nBlocks = 0;
    if (fread(&nBlocks, sizeof(nBlocks), 1, fInput) != 1)
    {
        fprintf(stderr, "This is not bsc archive!\n");
        exit(1);
    }

    double startTime = BSC_CLOCK();

#ifdef _OPENMP

    int numThreads = 1;
    if (paramEnableParallelProcessing)
    {
        numThreads = omp_get_max_threads();
        if (numThreads > nBlocks)
        {
            numThreads = nBlocks;
        }
    }

    #pragma omp parallel default(shared) num_threads(numThreads)
#endif
    {
        int bufferSize = 1024;
        unsigned char * buffer = (unsigned char *)malloc(bufferSize);
        if (buffer == NULL)
        {
#ifdef _OPENMP
            #pragma omp critical(print)
#endif
            {
                fprintf(stderr, "Not enough memory!\n");
                exit(2);
            }
        }

        while (true)
        {
            BSC_FILEOFFSET  blockOffset     = 0;

            char            sortingContexts = 0;
            char            recordSize      = 0;
            int             blockSize       = 0;
            int             dataSize        = 0;

#ifdef _OPENMP
            #pragma omp critical(input)
#endif
            {
                if ((feof(fInput) == 0) && (BSC_FTELL(fInput) != fileSize))
                {
#ifdef _OPENMP
                    #pragma omp master
#endif
                    {
                        double progress = (100.0 * (double)BSC_FTELL(fInput)) / fileSize;
                        fprintf(stdout, "\rDecompressing %.55s(%02d%%)", argv[2], (int)progress);
                        fflush(stdout);
                    }

                    BSC_BLOCK_HEADER header = {0, 0, 0};
                    if (fread(&header, sizeof(BSC_BLOCK_HEADER), 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }

                    recordSize = header.recordSize;
                    if (recordSize < 1)
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    sortingContexts = header.sortingContexts;
                    if ((sortingContexts != LIBBSC_CONTEXTS_FOLLOWING) && (sortingContexts != LIBBSC_CONTEXTS_PRECEDING))
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    blockOffset = (BSC_FILEOFFSET)header.blockOffset;

                    unsigned char bscBlockHeader[LIBBSC_HEADER_SIZE];

                    if (fread(bscBlockHeader, LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }

                    if (bsc_block_info(bscBlockHeader, LIBBSC_HEADER_SIZE, &blockSize, &dataSize) != LIBBSC_NO_ERROR)
                    {
                        fprintf(stderr, "\nThis is not bsc archive or invalid compression method!\n");
                        exit(2);
                    }

                    if (blockSize > bufferSize)
                    {
                        free(buffer); buffer = (unsigned char *)malloc(blockSize);
                        bufferSize = blockSize;
                    }

                    if (dataSize > bufferSize)
                    {
                        free(buffer); buffer = (unsigned char *)malloc(dataSize);
                        bufferSize = dataSize;
                    }

                    if (buffer == NULL)
                    {
                        fprintf(stderr, "\nNot enough memory!\n");
                        exit(2);
                    }

                    memcpy(buffer, bscBlockHeader, LIBBSC_HEADER_SIZE);

                    if (fread(buffer + LIBBSC_HEADER_SIZE, blockSize - LIBBSC_HEADER_SIZE, 1, fInput) != 1)
                    {
                        fprintf(stderr, "\nUnexpected end of file: %s!\n", argv[2]);
                        exit(1);
                    }
                }
            }

            if (dataSize == 0) break;

            int result = bsc_decompress(buffer, blockSize, buffer, dataSize, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
            if (result < LIBBSC_NO_ERROR)
            {
#ifdef _OPENMP
                #pragma omp critical(print)
#endif
                {
                    switch (result)
                    {
                        case LIBBSC_DATA_CORRUPT        : fprintf(stderr, "\nThe compressed data is corrupted!\n"); break;
                        case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                        default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                    }
                    exit(2);
                }
            }

            if (sortingContexts == LIBBSC_CONTEXTS_PRECEDING)
            {
                result = bsc_reverse_block(buffer, dataSize, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef _OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        exit(2);
                    }
                }
            }

            if (recordSize > 1)
            {
                result = bsc_reorder_reverse(buffer, dataSize, recordSize, paramEnableMultiThreading ? LIBBSC_FEATURE_MULTITHREADING : LIBBSC_FEATURE_NONE);
                if (result != LIBBSC_NO_ERROR)
                {
#ifdef _OPENMP
                    #pragma omp critical(print)
#endif
                    {
                        switch (result)
                        {
                            case LIBBSC_NOT_ENOUGH_MEMORY   : fprintf(stderr, "\nNot enough memory!\n"); break;
                            default                         : fprintf(stderr, "\nInternal program error, please contact the author!\n");
                        }
                        exit(2);
                    }
                }
            }

#ifdef _OPENMP
            #pragma omp critical(output)
#endif
            {
                if (BSC_FSEEK(fOutput, blockOffset, SEEK_SET))
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }

                if ((int)fwrite(buffer, 1, dataSize, fOutput) != dataSize)
                {
                    fprintf(stderr, "\nIO error on file: %s!\n", argv[3]);
                    exit(1);
                }
            }
        }

        free(buffer);
    }

    if (BSC_FSEEK(fOutput, 0, SEEK_END))
    {
        fprintf(stderr, "IO error on file: %s!\n", argv[3]);
        exit(1);
    }

    fprintf(stdout, "\r%.55s decompressed %.0f into %.0f in %.3f seconds.\n", argv[2], (double)fileSize, (double)BSC_FTELL(fOutput), BSC_CLOCK() - startTime);

    fclose(fInput); fclose(fOutput);
}

void ShowUsage(void)
{
    fprintf(stdout, "Usage: bsc <e|d> inputfile outputfile <switches>\n\n");
    fprintf(stdout, "Switches:\n");
    fprintf(stdout, "  -b<size> Block size in megabytes, default: -b25\n");
    fprintf(stdout, "             minimum: -b1, maximum: -b1024\n");
    fprintf(stdout, "  -m<algo> Block sorting algorithm, default: -m0\n");
    fprintf(stdout, "             -m0 Burrows Wheeler Transform\n");

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5) && defined(LIBBSC_BLOCKSORTER_ST6)

    fprintf(stdout, "             -m3 Sort Transform of order 3\n");
    fprintf(stdout, "             -m4 Sort Transform of order 4\n");
    fprintf(stdout, "             -m5 Sort Transform of order 5\n");
    fprintf(stdout, "             -m6 Sort Transform of order 6\n");

#endif

    fprintf(stdout, "  -c<ctx>  Contexts for sorting, default: -cf\n");
    fprintf(stdout, "             -cf Following contexts\n");
    fprintf(stdout, "             -cp Preceding contexts\n");
    fprintf(stdout, "             -ca Autodetect (experimental)\n");
    fprintf(stdout, "  -H<size> LZP hash table size in bits, default: -H16\n");
    fprintf(stdout, "             minimum: -H10, maximum: -H28\n");
    fprintf(stdout, "  -M<size> LZP minimum match length, default: -M128\n");
    fprintf(stdout, "             minimum: -M4, maximum: -M255\n");
    fprintf(stdout, "  -f       Enable fast compression mode, default: disable\n");
    fprintf(stdout, "  -l       Enable LZP, default: enable\n");
    fprintf(stdout, "  -r       Enable Reordering, default: disable\n");
    fprintf(stdout, "  -s       Enable Segmentation, default: disable\n");
    fprintf(stdout, "  -p       Disable all preprocessing techniques\n");

#ifdef _OPENMP

    fprintf(stdout, "  -t       Disable parallel blocks processing, default: enable\n");
    fprintf(stdout, "  -T       Disable multi-core systems support, default: enable\n");

#endif

    fprintf(stdout,"\nSwitches may be combined into one, like -b128p\n");
    exit(0);
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
                    paramBlockSize = atoi(strNum) * 1024 * 1024;
                    if ((paramBlockSize < 1024 * 1024) || (paramBlockSize > 1024 * 1024 * 1024)) ShowUsage();
                    break;
                }

            case 'm':
                {
                    char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                    switch (atoi(strNum))
                    {
                        case 0   : paramBlockSorter = LIBBSC_BLOCKSORTER_BWT; break;

#if defined(LIBBSC_BLOCKSORTER_ST3) && defined(LIBBSC_BLOCKSORTER_ST4) && defined(LIBBSC_BLOCKSORTER_ST5) && defined(LIBBSC_BLOCKSORTER_ST6)

                        case 3   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST3; break;
                        case 4   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST4; break;
                        case 5   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST5; break;
                        case 6   : paramBlockSorter = LIBBSC_BLOCKSORTER_ST6; break;
#endif

                        default  : ShowUsage();
                    }
                    break;
                }

            case 'c':
                {
                    switch (*s++)
                    {
                        case 'f' : paramSortingContexts = LIBBSC_CONTEXTS_FOLLOWING; break;
                        case 'p' : paramSortingContexts = LIBBSC_CONTEXTS_PRECEDING; break;
                        case 'a' : paramSortingContexts = LIBBSC_CONTEXTS_AUTODETECT; break;
                        default  : ShowUsage();
                    }
                    break;
                }

            case 'H':
                {
                    char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                    paramLZPHashSize = atoi(strNum);
                    if ((paramLZPHashSize < 10) || (paramLZPHashSize > 28)) ShowUsage();
                    break;
                }

            case 'M':
                {
                    char * strNum = s; while ((*s >= '0') && (*s <= '9')) s++;
                    paramLZPMinLen = atoi(strNum);
                    if ((paramLZPMinLen < 4) || (paramLZPMinLen > 255)) ShowUsage();
                    break;
                }

#ifdef _OPENMP
            case 't': paramEnableParallelProcessing = 0; break;
            case 'T': paramEnableParallelProcessing = paramEnableMultiThreading = 0; break;
#endif

            case 'f': paramEnableFastMode       = 1; break;
            case 'l': paramEnableLZP            = 1; break;
            case 's': paramEnableSegmentation   = 1; break;
            case 'r': paramEnableReordering     = 1; break;

            case 'p': paramEnableLZP = paramEnableSegmentation = paramEnableReordering = 0; break;

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
    fprintf(stdout, "This is bsc, Block Sorting Compressor. Version 2.6.1. 6 May 2011.\n");
    fprintf(stdout, "Copyright (c) 2009-2011 Ilya Grebnov <Ilya.Grebnov@libbsc.com>.\n\n");

#if defined(_OPENMP) && defined(__INTEL_COMPILER)

    kmp_set_warnings_off();

#endif

    if (bsc_init(LIBBSC_FEATURE_NONE) != LIBBSC_NO_ERROR)
    {
        fprintf(stderr, "\nInternal program error, please contact the author!\n");
        exit(2);
    }

    ProcessCommandline(argc, argv);
    switch (*argv[1])
    {
        case 'e' : case 'E' : Compression(argv); break;
        case 'd' : case 'D' : Decompression(argv); break;
        default  : ShowUsage();
    }

    return 0;
}

/*-----------------------------------------------------------*/
/* End                                               bsc.cpp */
/*-----------------------------------------------------------*/
