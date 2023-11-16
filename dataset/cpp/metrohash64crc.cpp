// metrohash64crc.cpp
//
// Copyright 2015-2018 J. Andrew Rogers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#if defined(__aarch64__)
       #include "sse2neon.h"
#else
       #include <nmmintrin.h>
#endif
#include <string.h>
#include "metrohash.h"
#include "platform.h"

void metrohash64crc_1(const uint8_t * key, uint64_t len, uint32_t seed, uint8_t * out)
{
    static const uint64_t k0 = 0xC83A91E1;
    static const uint64_t k1 = 0x8648DBDB;
    static const uint64_t k2 = 0x7BDEC03B;
    static const uint64_t k3 = 0x2F5870A5;

    const uint8_t * ptr = reinterpret_cast<const uint8_t*>(key);
    const uint8_t * const end = ptr + len;

    uint64_t hash = ((static_cast<uint64_t>(seed) + k2) * k0) + len;

    if (len >= 32)
    {
        uint64_t v[4];
        v[0] = hash;
        v[1] = hash;
        v[2] = hash;
        v[3] = hash;

        do
        {
            v[0] ^= _mm_crc32_u64(v[0], read_u64(ptr)); ptr += 8;
            v[1] ^= _mm_crc32_u64(v[1], read_u64(ptr)); ptr += 8;
            v[2] ^= _mm_crc32_u64(v[2], read_u64(ptr)); ptr += 8;
            v[3] ^= _mm_crc32_u64(v[3], read_u64(ptr)); ptr += 8;
        }
        while (ptr <= (end - 32));

        v[2] ^= rotate_right(((v[0] + v[3]) * k0) + v[1], 33) * k1;
        v[3] ^= rotate_right(((v[1] + v[2]) * k1) + v[0], 33) * k0;
        v[0] ^= rotate_right(((v[0] + v[2]) * k0) + v[3], 33) * k1;
        v[1] ^= rotate_right(((v[1] + v[3]) * k1) + v[2], 33) * k0;
        hash += v[0] ^ v[1];
    }

    if ((end - ptr) >= 16)
    {
        uint64_t v0 = hash + (read_u64(ptr) * k0); ptr += 8; v0 = rotate_right(v0,33) * k1;
        uint64_t v1 = hash + (read_u64(ptr) * k1); ptr += 8; v1 = rotate_right(v1,33) * k2;
        v0 ^= rotate_right(v0 * k0, 35) + v1;
        v1 ^= rotate_right(v1 * k3, 35) + v0;
        hash += v1;
    }

    if ((end - ptr) >= 8)
    {
        hash += read_u64(ptr) * k3; ptr += 8;
        hash ^= rotate_right(hash, 33) * k1;

    }

    if ((end - ptr) >= 4)
    {
        hash ^= _mm_crc32_u64(hash, read_u32(ptr)); ptr += 4;
        hash ^= rotate_right(hash, 15) * k1;
    }

    if ((end - ptr) >= 2)
    {
        hash ^= _mm_crc32_u64(hash, read_u16(ptr)); ptr += 2;
        hash ^= rotate_right(hash, 13) * k1;
    }

    if ((end - ptr) >= 1)
    {
        hash ^= _mm_crc32_u64(hash, read_u8(ptr));
        hash ^= rotate_right(hash, 25) * k1;
    }

    hash ^= rotate_right(hash, 33);
    hash *= k0;
    hash ^= rotate_right(hash, 33);

    memcpy(out, &hash, 8);
}

void metrohash64crc_2(const uint8_t * key, uint64_t len, uint32_t seed, uint8_t * out)
{
    static const uint64_t k0 = 0xD6D018F5;
    static const uint64_t k1 = 0xA2AA033B;
    static const uint64_t k2 = 0x62992FC1;
    static const uint64_t k3 = 0x30BC5B29; 

    const uint8_t * ptr = reinterpret_cast<const uint8_t*>(key);
    const uint8_t * const end = ptr + len;
    
    uint64_t hash = ((static_cast<uint64_t>(seed) + k2) * k0) + len;
    
    if (len >= 32)
    {
        uint64_t v[4];
        v[0] = hash;
        v[1] = hash;
        v[2] = hash;
        v[3] = hash;

        do
        {
            v[0] ^= _mm_crc32_u64(v[0], read_u64(ptr)); ptr += 8;
            v[1] ^= _mm_crc32_u64(v[1], read_u64(ptr)); ptr += 8;
            v[2] ^= _mm_crc32_u64(v[2], read_u64(ptr)); ptr += 8;
            v[3] ^= _mm_crc32_u64(v[3], read_u64(ptr)); ptr += 8;
        }
        while (ptr <= (end - 32));

        v[2] ^= rotate_right(((v[0] + v[3]) * k0) + v[1], 33) * k1;
        v[3] ^= rotate_right(((v[1] + v[2]) * k1) + v[0], 33) * k0;
        v[0] ^= rotate_right(((v[0] + v[2]) * k0) + v[3], 33) * k1;
        v[1] ^= rotate_right(((v[1] + v[3]) * k1) + v[2], 33) * k0;
        hash += v[0] ^ v[1];
    }

    if ((end - ptr) >= 16)
    {
        uint64_t v0 = hash + (read_u64(ptr) * k0); ptr += 8; v0 = rotate_right(v0,33) * k1;
        uint64_t v1 = hash + (read_u64(ptr) * k1); ptr += 8; v1 = rotate_right(v1,33) * k2;
        v0 ^= rotate_right(v0 * k0, 35) + v1;
        v1 ^= rotate_right(v1 * k3, 35) + v0;
        hash += v1;
    }

    if ((end - ptr) >= 8)
    {
        hash += read_u64(ptr) * k3; ptr += 8;
        hash ^= rotate_right(hash, 33) * k1;

    }

    if ((end - ptr) >= 4)
    {
        hash ^= _mm_crc32_u64(hash, read_u32(ptr)); ptr += 4;
        hash ^= rotate_right(hash, 15) * k1;
    }

    if ((end - ptr) >= 2)
    {
        hash ^= _mm_crc32_u64(hash, read_u16(ptr)); ptr += 2;
        hash ^= rotate_right(hash, 13) * k1;
    }

    if ((end - ptr) >= 1)
    {
        hash ^= _mm_crc32_u64(hash, read_u8(ptr));
        hash ^= rotate_right(hash, 25) * k1;
    }

    hash ^= rotate_right(hash, 33);
    hash *= k0;
    hash ^= rotate_right(hash, 33);

    memcpy(out, &hash, 8);
}
