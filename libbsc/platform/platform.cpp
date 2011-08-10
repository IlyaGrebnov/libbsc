/*-----------------------------------------------------------*/
/* Block Sorting, Lossless Data Compression Library.         */
/* Platform specific functions and constants                 */
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

#include <stdlib.h>
#include <string.h>
#include <memory.h>

#include "platform.h"

#include "../libbsc.h"

#if defined(_WIN32)
  #include <windows.h>
#endif

long long g_LargePageSize = 0;

int bsc_platform_init(int features)
{

#if defined(_WIN32)

    if (features & LIBBSC_FEATURE_LARGEPAGES)
    {
        HANDLE hToken = 0;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken))
        {
            LUID luid;
            if (LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &luid))
            {
                TOKEN_PRIVILEGES tp;

                tp.PrivilegeCount = 1;
                tp.Privileges[0].Luid = luid;
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

                AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), 0, 0);
            }

            CloseHandle(hToken);
        }

        {
            typedef SIZE_T (WINAPI * GetLargePageMinimumProcT)();

            GetLargePageMinimumProcT largePageMinimumProc = (GetLargePageMinimumProcT)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "GetLargePageMinimum");
            if (largePageMinimumProc != NULL)
            {
                SIZE_T largePageSize = largePageMinimumProc();

                if ((largePageSize & (largePageSize - 1)) != 0) largePageSize = 0;

                g_LargePageSize = largePageSize;
            }
        }
    }

#endif

    return LIBBSC_NO_ERROR;
}

void * bsc_malloc(size_t size)
{
#if defined(_WIN32)
    if ((g_LargePageSize != 0) && (size >= 256 * 1024))
    {
        void * address = VirtualAlloc(0, (size + g_LargePageSize - 1) & (~(g_LargePageSize - 1)), MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (address != NULL) return address;
    }
    return VirtualAlloc(0, size, MEM_COMMIT, PAGE_READWRITE);
#else
    return malloc(size);
#endif
}

void * bsc_zero_malloc(size_t size)
{
#if defined(_WIN32)
    if ((g_LargePageSize != 0) && (size >= 256 * 1024))
    {
        void * address = VirtualAlloc(0, (size + g_LargePageSize - 1) & (~(g_LargePageSize - 1)), MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (address != NULL) return address;
    }
    return VirtualAlloc(0, size, MEM_COMMIT, PAGE_READWRITE);
#else
    return calloc(1, size);
#endif
}

void bsc_free(void * address)
{
#if defined(_WIN32)
    VirtualFree(address, 0, MEM_RELEASE);
#else
    free(address);
#endif
}

/*-----------------------------------------------------------*/
/* End                                            common.cpp */
/*-----------------------------------------------------------*/
