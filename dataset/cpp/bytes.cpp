/**
 * Copyright (c) 2011-2023 libbitcoin developers (see AUTHORS)
 *
 * This file is part of libbitcoin.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation == either version 3 of the License == or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not == see <http://www.gnu.org/licenses/>.
 */
#include "../test.hpp"

BOOST_AUTO_TEST_SUITE(bytes_tests)

 // byte_width (unsigned/positive)
static_assert(byte_width(0_u8) == 0);
static_assert(byte_width(1_u8) == 1);
static_assert(byte_width(2_u8) == 1);
static_assert(byte_width(3_u8) == 1);
static_assert(byte_width(4_u8) == 1);
static_assert(byte_width(5_u8) == 1);
static_assert(byte_width(6_u16) == 1);
static_assert(byte_width(7_u16) == 1);
static_assert(byte_width(8_u16) == 1);
static_assert(byte_width(9_u16) == 1);
static_assert(byte_width(0x000000ff_u32) == 1);
static_assert(byte_width(0x0000ff00_u32) == 2);
static_assert(byte_width(0x00ff0100_u32) == 3);
static_assert(byte_width(0xff000000_u32) == 4);
static_assert(byte_width(0x000000ff01000000_u64) == 5);
static_assert(byte_width(0x0000ff0000000000_u64) == 6);
static_assert(byte_width(0x00ff010000000000_u64) == 7);
static_assert(byte_width(0xff00000000000000_u64) == 8);

// byte_width (signed/positive)
static_assert(byte_width(0_i8) == 0);
static_assert(byte_width(1_i8) == 1);
static_assert(byte_width(2_i8) == 1);
static_assert(byte_width(3_i8) == 1);
static_assert(byte_width(4_i8) == 1);
static_assert(byte_width(5_i8) == 1);
static_assert(byte_width(6_i16) == 1);
static_assert(byte_width(7_i16) == 1);
static_assert(byte_width(8_i16) == 1);
static_assert(byte_width(9_i16) == 1);
static_assert(byte_width(0x0000007f_i32) == 1);
static_assert(byte_width(0x00007f00_i32) == 2);
static_assert(byte_width(0x007f0100_i32) == 3);
static_assert(byte_width(0x7f000000_i32) == 4);
static_assert(byte_width(0x0000007f01000000_i64) == 5);
static_assert(byte_width(0x00007f0000000000_i64) == 6);
static_assert(byte_width(0x007f010000000000_i64) == 7);
static_assert(byte_width(0x7f00000000000000_i64) == 8);

// byte_width (signed negative)
static_assert(byte_width(0x00_i8) == 0);
static_assert(byte_width(0xff_i8) == 1); // -1
static_assert(byte_width(0xfe_i8) == 1); // -2...
static_assert(byte_width(0xfd_i8) == 1);
static_assert(byte_width(0xfc_i8) == 1);
static_assert(byte_width(0xfb_i8) == 1);
static_assert(byte_width(0xfa_i8) == 1);
static_assert(byte_width(0xf9_i8) == 1);
static_assert(byte_width(0xf8_i8) == 1);
static_assert(byte_width(0xf7_i8) == 1); // -9...
static_assert(byte_width(0x000000ff_i32) == 1);
static_assert(byte_width(0x0000ff00_i32) == 2);
static_assert(byte_width(0x00ff0100_i32) == 3);
static_assert(byte_width(0xff000000_i32) == 4);
static_assert(byte_width(0x00000000000000ff01000000_i64) == 5);
static_assert(byte_width(0x000000000000ff0000000000_i64) == 6);
static_assert(byte_width(0x0000000000ff010000000000_i64) == 7);
static_assert(byte_width(0x00000000ff00000000000000_i64) == 8);
static_assert(is_same_type<decltype(byte_width<int8_t>(0)), size_t>);

// byte
static_assert(byte<0>(0x01_u8) == 0x01_u8);
static_assert(byte<0>(0x0102_u16) == 0x02_u8);
static_assert(byte<1>(0x0102_u16) == 0x01_u8);
static_assert(byte<0>(0x01020304_u32) == 0x04_u8);
static_assert(byte<1>(0x01020304_u32) == 0x03_u8);
static_assert(byte<2>(0x01020304_u32) == 0x02_u8);
static_assert(byte<3>(0x01020304_u32) == 0x01_u8);
static_assert(byte<0>(0x0102030405060708_u64) == 0x08_u8);
static_assert(byte<1>(0x0102030405060708_u64) == 0x07_u8);
static_assert(byte<2>(0x0102030405060708_u64) == 0x06_u8);
static_assert(byte<3>(0x0102030405060708_u64) == 0x05_u8);
static_assert(byte<4>(0x0102030405060708_u64) == 0x04_u8);
static_assert(byte<5>(0x0102030405060708_u64) == 0x03_u8);
static_assert(byte<6>(0x0102030405060708_u64) == 0x02_u8);
static_assert(byte<7>(0x0102030405060708_u64) == 0x01_u8);

static_assert(byte<0, uint8_t>(0x01_u8) == 0x01_u8);
static_assert(byte<0, uint8_t>(0x0102_u16) == 0x02_u8);
static_assert(byte<1, uint8_t>(0x0102_u16) == 0x01_u8);
static_assert(byte<0, uint8_t>(0x01020304_u32) == 0x04_u8);
static_assert(byte<1, uint8_t>(0x01020304_u32) == 0x03_u8);
static_assert(byte<2, uint8_t>(0x01020304_u32) == 0x02_u8);
static_assert(byte<3, uint8_t>(0x01020304_u32) == 0x01_u8);
static_assert(byte<0, uint8_t>(0x0102030405060708_u64) == 0x08_u8);
static_assert(byte<1, uint8_t>(0x0102030405060708_u64) == 0x07_u8);
static_assert(byte<2, uint8_t>(0x0102030405060708_u64) == 0x06_u8);
static_assert(byte<3, uint8_t>(0x0102030405060708_u64) == 0x05_u8);
static_assert(byte<4, uint8_t>(0x0102030405060708_u64) == 0x04_u8);
static_assert(byte<5, uint8_t>(0x0102030405060708_u64) == 0x03_u8);
static_assert(byte<6, uint8_t>(0x0102030405060708_u64) == 0x02_u8);
static_assert(byte<7, uint8_t>(0x0102030405060708_u64) == 0x01_u8);

static_assert(byte<0, int8_t>(0x01_u8) == 0x01_i8);
static_assert(byte<0, int8_t>(0x0102_u16) == 0x02_i8);
static_assert(byte<1, int8_t>(0x0102_u16) == 0x01_i8);
static_assert(byte<0, int8_t>(0x01020304_u32) == 0x04_i8);
static_assert(byte<1, int8_t>(0x01020304_u32) == 0x03_i8);
static_assert(byte<2, int8_t>(0x01020304_u32) == 0x02_i8);
static_assert(byte<3, int8_t>(0x01020304_u32) == 0x01_i8);
static_assert(byte<0, int8_t>(0x0102030405060708_i64) == 0x08_i8);
static_assert(byte<1, int8_t>(0x0102030405060708_i64) == 0x07_i8);
static_assert(byte<2, int8_t>(0x0102030405060708_i64) == 0x06_i8);
static_assert(byte<3, int8_t>(0x0102030405060708_i64) == 0x05_i8);
static_assert(byte<4, int8_t>(0x0102030405060708_i64) == 0x04_i8);
static_assert(byte<5, int8_t>(0x0102030405060708_i64) == 0x03_i8);
static_assert(byte<6, int8_t>(0x0102030405060708_i64) == 0x02_i8);
static_assert(byte<7, int8_t>(0x0102030405060708_i64) == 0x01_i8);

// is_negated (negated signed values)
static_assert(is_negated(0x80_i8));
static_assert(is_negated(0xff_i8));
static_assert(is_negated(0x8042_i16));
static_assert(is_negated(0xff42_i16));
static_assert(is_negated(0x00800042_i32));
static_assert(is_negated(0x00ff0042_i32));
static_assert(is_negated(0x80000042_i32));
static_assert(is_negated(0xff000042_i32));
static_assert(is_negated(0x00008000000042_i64));
static_assert(is_negated(0x0000ff00000042_i64));
static_assert(is_negated(0x00800000000042_i64));
static_assert(is_negated(0x00ff0000000042_i64));
static_assert(is_negated(0x80000000000042_i64));
static_assert(is_negated(0xff000000000042_i64));

// is_negated (non-negated signed values)
static_assert(!is_negated(0x00_i8));
static_assert(!is_negated(0x01_i8));
static_assert(!is_negated(0x0f_i8));
static_assert(!is_negated(0x7f_i8));
static_assert(!is_negated(0x0042_i16));
static_assert(!is_negated(0x0142_i16));
static_assert(!is_negated(0x0f42_i16));
static_assert(!is_negated(0x7f42_i16));
static_assert(!is_negated(0x7f42_i16));
static_assert(!is_negated(0x00000042_i32));
static_assert(!is_negated(0x00010042_i32));
static_assert(!is_negated(0x000f0042_i32));
static_assert(!is_negated(0x007f0042_i32));
static_assert(!is_negated(0x007f0042_i32));
static_assert(!is_negated(0x00000042_i32));
static_assert(!is_negated(0x01000042_i32));
static_assert(!is_negated(0x0f000042_i32));
static_assert(!is_negated(0x7f000042_i32));
static_assert(!is_negated(0x7f000042_i32));
static_assert(!is_negated(0x00000000000042_i64));
static_assert(!is_negated(0x00000100000042_i64));
static_assert(!is_negated(0x00000f00000042_i64));
static_assert(!is_negated(0x00007f00000042_i64));
static_assert(!is_negated(0x00007f00000042_i64));
static_assert(!is_negated(0x00000000000042_i64));
static_assert(!is_negated(0x00010000000042_i64));
static_assert(!is_negated(0x000f0000000042_i64));
static_assert(!is_negated(0x007f0000000042_i64));
static_assert(!is_negated(0x007f0000000042_i64));
static_assert(!is_negated(0x00000000000042_i64));
static_assert(!is_negated(0x01000000000042_i64));
static_assert(!is_negated(0x0f000000000042_i64));
static_assert(!is_negated(0x7f000000000042_i64));
static_assert(!is_negated(0x7f000000000042_i64));
static_assert(is_same_type<decltype(is_negated<int32_t>(0)), bool>);

// is_negated (non-negated unsigned values)
static_assert(!is_negated(0x00_u8));
static_assert(!is_negated(0x01_u8));
static_assert(!is_negated(0x0f_u8));
static_assert(!is_negated(0x7f_u8));
static_assert(!is_negated(0x0042_u16));
static_assert(!is_negated(0x0142_u16));
static_assert(!is_negated(0x0f42_u16));
static_assert(!is_negated(0x7f42_u16));
static_assert(!is_negated(0x7f42_u16));
static_assert(!is_negated(0x00000042_u32));
static_assert(!is_negated(0x00010042_u32));
static_assert(!is_negated(0x000f0042_u32));
static_assert(!is_negated(0x007f0042_u32));
static_assert(!is_negated(0x007f0042_u32));
static_assert(!is_negated(0x00000042_u32));
static_assert(!is_negated(0x01000042_u32));
static_assert(!is_negated(0x0f000042_u32));
static_assert(!is_negated(0x7f000042_u32));
static_assert(!is_negated(0x7f000042_u32));
static_assert(!is_negated(0x00000000000042_u64));
static_assert(!is_negated(0x00000100000042_u64));
static_assert(!is_negated(0x00000f00000042_u64));
static_assert(!is_negated(0x00007f00000042_u64));
static_assert(!is_negated(0x00007f00000042_u64));
static_assert(!is_negated(0x00000000000042_u64));
static_assert(!is_negated(0x00010000000042_u64));
static_assert(!is_negated(0x000f0000000042_u64));
static_assert(!is_negated(0x007f0000000042_u64));
static_assert(!is_negated(0x007f0000000042_u64));
static_assert(!is_negated(0x00000000000042_u64));
static_assert(!is_negated(0x01000000000042_u64));
static_assert(!is_negated(0x0f000000000042_u64));
static_assert(!is_negated(0x7f000000000042_u64));
static_assert(!is_negated(0x7f000000000042_u64));
static_assert(is_same_type<decltype(is_negated<uint32_t>(0)), bool>);

// to_negated
static_assert(to_negated(0x00_i8) == 0x80_i8);
static_assert(to_negated(0x01_i8) == 0x81_i8);
static_assert(to_negated(0x7f_i8) == 0xff_i8);
static_assert(to_negated(0x80_i8) == 0x80_i8);
static_assert(to_negated(0xff_i8) == 0xff_i8);
static_assert(to_negated(0x7f00_i16) == 0xff00_i16);
static_assert(to_negated(0x810a_i16) == 0x810a_i16);
static_assert(to_negated(0x7f000000_i32) == 0xff000000_i32);
static_assert(to_negated(0x8f000000_i32) == 0x8f000000_i32);
static_assert(to_negated(0x7f00000000000000_i64) == 0xff00000000000000_i64);
static_assert(to_negated(0x000000000000007f_i64) == 0x800000000000007f_i64);
static_assert(is_same_type<decltype(to_negated<int32_t>(0)), int32_t>);

// to_unnegated
static_assert(to_unnegated(0x00_i8) == 0x00_i8);
static_assert(to_unnegated(0x01_i8) == 0x01_i8);
static_assert(to_unnegated(0x7f_i8) == 0x7f_i8);
static_assert(to_unnegated(0x80_i8) == 0x00_i8);
static_assert(to_unnegated(1_ni8) == 0x7f_ni8);
static_assert(to_unnegated(0xff_i8) == 0x7f_ni8);
static_assert(to_unnegated(0x7f00_i16) == 0x7f00_i16);
static_assert(to_unnegated(0x810a_i16) == 0x010a_ni16);
static_assert(to_unnegated(0x7f000000_i32) == 0x7f000000_i32);
static_assert(to_unnegated(0x8f000000_i32) == 0x0f000000_ni32);
static_assert(to_unnegated(0x7f00000000000000_i64) == 0x7f00000000000000_i64);
static_assert(to_unnegated(0x000000000000007f_i64) == 0x000000000000007f_i64);
static_assert(is_same_type<decltype(to_unnegated<int32_t>(0)), int32_t>);

// to_ceilinged_bytes
static_assert(to_ceilinged_bytes(0u) == 0u);
static_assert(to_ceilinged_bytes(1u) == 1u);
static_assert(to_ceilinged_bytes(42u) == (42u + 7u) / 8u);
static_assert(to_ceilinged_bytes(0xffu) == (0xff + 7u) / 8u);
static_assert(is_same_type<decltype(to_ceilinged_bytes<uint16_t>(0)), uint16_t>);

static_assert(to_ceilinged_bytes<uint11_t>(0u) == 0u);
static_assert(to_ceilinged_bytes<uint11_t>(1u) == 1u);
static_assert(to_ceilinged_bytes<uint11_t>(42u) == (42u + 7u) / 8u);
static_assert(to_ceilinged_bytes<uint11_t>(0xffu) == (0xff + 7u) / 8u);
static_assert(to_ceilinged_bytes<uint5_t>(0xffu) == ((0xff & 0b00011111) + 7u) / 8u);
static_assert(is_same_type<decltype(to_ceilinged_bytes<uint11_t>(0)), uint11_t>);

// to_floored_bytes
static_assert(to_floored_bytes(0u) == 0u);
static_assert(to_floored_bytes(1u) == 0u);
static_assert(to_floored_bytes(42u) == 42u / 8u);
static_assert(to_floored_bytes(0xffu) == 0xff / 8u);
static_assert(is_same_type<decltype(to_floored_bytes<uint16_t>(0)), uint16_t>);

static_assert(to_floored_bytes<uint11_t>(0u) == 0u);
static_assert(to_floored_bytes<uint11_t>(1u) == 0u);
static_assert(to_floored_bytes<uint11_t>(42u) == 42u / 8u);
static_assert(to_floored_bytes<uint5_t>(0xffu) == (0xff & 0b00011111) / 8u);
static_assert(is_same_type<decltype(to_floored_bytes<uint11_t>(0)), uint11_t>);

BOOST_AUTO_TEST_SUITE_END()
