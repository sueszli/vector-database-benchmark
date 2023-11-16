/*
 * Copyright 2014-2023 Real Logic Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "util/aeron_platform.h"

#if defined(AERON_COMPILER_MSVC)
#include "aeron_windows.h"
#include "util/aeron_error.h"

#include <WinSock2.h>
#include <Windows.h>
#include <time.h>
#include <intrin.h>

#include "concurrent/aeron_thread.h"
#include "aeron_alloc.h"

#define __builtin_bswap32 _byteswap_ulong
#define __builtin_bswap64 _byteswap_uint64
#define __builtin_popcount __popcnt
#define __builtin_popcountll __popcnt64

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    switch (fdwReason)
    {
        case DLL_PROCESS_ATTACH:
            if (!aeron_error_dll_process_attach())
            {
                return FALSE;
            }
            break;

        case DLL_THREAD_DETACH:
            aeron_error_dll_thread_detach();
            break;

        case DLL_PROCESS_DETACH:
            aeron_error_dll_process_detach();
            break;

        default:
            break;
    }

    return TRUE;
}

typedef struct { UINT64 q[2]; } aeron_uint128_t;

aeron_uint128_t make_aeron_uint128_t(UINT64 x)
{
    aeron_uint128_t result;
    result.q[0] = x;
    result.q[1] = 0;
    return result;
}

aeron_uint128_t aeron_uint128_bitwise_negate(aeron_uint128_t x)
{
    aeron_uint128_t r;
    r.q[0] = ~x.q[0];
    r.q[1] = ~x.q[1];
    return r;
}

BOOL aeron_uint128_equals(const aeron_uint128_t lhs, const aeron_uint128_t rhs)
{
    return lhs.q[0] == rhs.q[0] && lhs.q[1] == rhs.q[1];
}

aeron_uint128_t aeron_uint128_bitwise_shift_left(const aeron_uint128_t lhs, size_t n)
{
    aeron_uint128_t result = lhs;

    if (n >= 128)
    {
        result.q[1] = 0;
        result.q[0] = 0;
    }
    else
    {
        const unsigned int halfsize = 128 / 2;

        if (n >= halfsize)
        {
            n -= halfsize;
            result.q[1] = result.q[0];
            result.q[0] = 0;
        }

        if (n != 0)
        {
            // shift high half
            result.q[1] <<= n;

            const UINT64 mask = (~((UINT64)-1) >> n);

            // and add them to high half
            result.q[1] |= (result.q[0] & mask) >> (halfsize - n);

            // and finally shift also low half
            result.q[0] <<= n;
        }
    }

    return result;
}

aeron_uint128_t aeron_uint128_sub(const aeron_uint128_t lhs, const aeron_uint128_t rhs)
{
    aeron_uint128_t result = lhs;

    result.q[1] -= rhs.q[1];
    result.q[0] -= rhs.q[0];

    if (rhs.q[0] >= lhs.q[0])
    {
        result.q[1] -= 1;
    }

    return result;
}

aeron_uint128_t aeron_uint128_bitwise_and(const aeron_uint128_t lhs, const aeron_uint128_t rhs)
{
    aeron_uint128_t result;
    result.q[0] = lhs.q[0] & rhs.q[0];
    result.q[1] = lhs.q[1] & rhs.q[1];
    return result;
}

aeron_uint128_t aeron_ipv6_netmask_from_prefixlen(size_t prefixlen)
{
    aeron_uint128_t netmask;

    if (0 == prefixlen)
    {
        netmask.q[1] = 0;
        netmask.q[0] = 0;
    }
    else
    {
        netmask = aeron_uint128_bitwise_negate(aeron_uint128_sub(aeron_uint128_bitwise_shift_left(
            make_aeron_uint128_t(1), (128 - prefixlen)), make_aeron_uint128_t(1)));
    }

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    UINT64 lo = netmask.q[0];
    netmask.q[0] = __builtin_bswap64(netmask.q[1]);
    netmask.q[1] = __builtin_bswap64(lo);
#endif

    return netmask;
}

BOOL aeron_ipv6_does_prefix_match(struct in6_addr *in6_addr1, struct in6_addr *in6_addr2, size_t prefixlen)
{
    aeron_uint128_t addr1;
    aeron_uint128_t addr2;
    const aeron_uint128_t netmask = aeron_ipv6_netmask_from_prefixlen(prefixlen);

    memcpy(&addr1, in6_addr1, sizeof(addr1));
    memcpy(&addr2, in6_addr2, sizeof(addr2));

    return aeron_uint128_equals(aeron_uint128_bitwise_and(addr1, netmask), aeron_uint128_bitwise_and(addr2, netmask));
}

void aeron_srand48(uint64_t aeron_nano_clock)
{
    srand((unsigned int)aeron_nano_clock);
}

double aeron_drand48()
{
    return rand() / (double)(RAND_MAX + 1);
}

double aeron_erand48(unsigned short xsubi[3])
{
    return rand() / (double)(RAND_MAX + 1);
}

void localtime_r(const time_t *timep, struct tm *result)
{
    localtime_s(result, timep);
}

char *aeron_strndup(const char *value, size_t length)
{
    size_t str_length = strlen(value);
    char *dup = NULL;

    str_length = (str_length > length) ? length : str_length;
    if (aeron_alloc((void **)&dup, str_length + 1) < 0)
    {
        errno = ENOMEM;
        return NULL;
    }

    strncpy(dup, value, str_length);
    dup[str_length] = '\0';
    return dup;
}

#else

typedef int aeron_make_into_non_empty_translation_unit_t;

#endif
