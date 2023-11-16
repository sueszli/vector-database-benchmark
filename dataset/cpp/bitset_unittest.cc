//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/bitset.h"

#include <gtest/gtest.h>

namespace base {

TEST(BitSetTest, Range64)
{
    EXPECT_EQ(BitSet<uint64_t>(0).range(0, 63), 0);
    EXPECT_EQ(BitSet<uint64_t>(0x4020000000000000).range(48, 63), 0x4020);
    EXPECT_EQ(BitSet<uint64_t>(0x00000000000000E0).range(4, 5), 2);
    EXPECT_EQ(BitSet<uint64_t>(0xC0000000000000E0).range(62, 63), 3);
    EXPECT_EQ(BitSet<uint64_t>(0xC0008000000000E0).range(46, 47), 2);
    EXPECT_EQ(BitSet<uint64_t>(0xF0208902001080E0).range(32, 63), 0xF0208902);
    EXPECT_EQ(BitSet<uint64_t>(0xF0208902001080E0).range(0, 31), 0x1080E0);

    EXPECT_EQ(BitSet<uint64_t>(1).range(0, 0), 1);
    EXPECT_EQ(BitSet<uint64_t>(2).range(1, 1), 1);
    EXPECT_EQ(BitSet<uint64_t>(2).range(0, 0), 0);
    EXPECT_EQ(BitSet<uint64_t>(0x2000000000000000).range(61, 61), 1);
    EXPECT_EQ(BitSet<uint64_t>(0x1000000000000000).range(60, 60), 1);

    const BitSet<uint64_t> set1(0xFFFFFFFFFFFFFFFF);

    for (size_t i = 0; i < set1.size(); ++i)
    {
        EXPECT_EQ(set1.range(i, i), 1);
        EXPECT_TRUE(set1.test(i));
    }

    const BitSet<uint64_t> set2(0x0000000000000000);

    for (size_t i = 0; i < set2.size(); ++i)
    {
        EXPECT_EQ(set2.range(i, i), 0);
        EXPECT_FALSE(set2.test(i));
    }

    const BitSet<uint64_t> set3(0xAAAAAAAAAAAAAAAA);

    for (size_t i = 0; i < set3.size(); ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_EQ(set3.range(i, i), 0);
            EXPECT_FALSE(set3.test(i));
        }
        else
        {
            EXPECT_EQ(set3.range(i, i), 1);
            EXPECT_TRUE(set3.test(i));
        }
    }
}

TEST(BitSetTest, Range32)
{
    EXPECT_EQ(BitSet<uint32_t>(0).range(0, 31), 0);
    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(12, 15), 0);
    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(0, 15), 192);
    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(0, 31), 0x404400C0);
    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(6, 7), 3);

    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(30, 30), 1);
    EXPECT_EQ(BitSet<uint32_t>(0x404400C0).range(6, 6), 1);
    EXPECT_EQ(BitSet<uint32_t>(1).range(0, 0), 1);
    EXPECT_EQ(BitSet<uint32_t>(3).range(1, 1), 1);
    EXPECT_EQ(BitSet<uint32_t>(0x80000000).range(31, 31), 1);

    const BitSet<uint32_t> set1(0xFFFFFFFF);

    for (size_t i = 0; i < set1.size(); ++i)
    {
        EXPECT_EQ(set1.range(i, i), 1);
        EXPECT_TRUE(set1.test(i));
    }

    const BitSet<uint32_t> set2(0x00000000);

    for (size_t i = 0; i < set2.size(); ++i)
    {
        EXPECT_EQ(set2.range(i, i), 0);
        EXPECT_FALSE(set2.test(i));
    }

    const BitSet<uint32_t> set3(0xAAAAAAAA);

    for (size_t i = 0; i < set3.size(); ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_EQ(set3.range(i, i), 0);
            EXPECT_FALSE(set3.test(i));
        }
        else
        {
            EXPECT_EQ(set3.range(i, i), 1);
            EXPECT_TRUE(set3.test(i));
        }
    }
}

TEST(BitSetTest, Range16)
{
    EXPECT_EQ(BitSet<uint16_t>(0).range(0, 15), 0);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(0, 15), 0xC081);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(5, 7), 4);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(13, 15), 6);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(0, 3), 1);

    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(11, 11), 0);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(15, 15), 1);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(14, 14), 1);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(0, 0), 1);
    EXPECT_EQ(BitSet<uint16_t>(0xC081).range(1, 1), 0);

    const BitSet<uint16_t> set1(0xFFFF);

    for (size_t i = 0; i < set1.size(); ++i)
    {
        EXPECT_EQ(set1.range(i, i), 1);
        EXPECT_TRUE(set1.test(i));
    }

    const BitSet<uint16_t> set2(0x0000);

    for (size_t i = 0; i < set2.size(); ++i)
    {
        EXPECT_EQ(set2.range(i, i), 0);
        EXPECT_FALSE(set2.test(i));
    }

    const BitSet<uint16_t> set3(0xAAAA);

    for (size_t i = 0; i < set3.size(); ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_EQ(set3.range(i, i), 0);
            EXPECT_FALSE(set3.test(i));
        }
        else
        {
            EXPECT_EQ(set3.range(i, i), 1);
            EXPECT_TRUE(set3.test(i));
        }
    }
}

TEST(BitSetTest, Range8)
{
    EXPECT_EQ(BitSet<uint8_t>(0).range(0, 7), 0);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(0, 2), 2);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(0, 3), 10);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(0, 7), 106);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(6, 7), 1);

    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(6, 6), 1);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(0, 0), 0);
    EXPECT_EQ(BitSet<uint8_t>(0x6A).range(4, 4), 0);

    const BitSet<uint8_t> set1(0xFF);

    for (size_t i = 0; i < set1.size(); ++i)
    {
        EXPECT_EQ(set1.range(i, i), 1);
        EXPECT_TRUE(set1.test(i));
    }

    const BitSet<uint8_t> set2(0x00);

    for (size_t i = 0; i < set2.size(); ++i)
    {
        EXPECT_EQ(set2.range(i, i), 0);
        EXPECT_FALSE(set2.test(i));
    }

    const BitSet<uint8_t> set3(0xAA);

    for (size_t i = 0; i < set3.size(); ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_EQ(set3.range(i, i), 0);
            EXPECT_FALSE(set3.test(i));
        }
        else
        {
            EXPECT_EQ(set3.range(i, i), 1);
            EXPECT_TRUE(set3.test(i));
        }
    }
}

} // namespace base
