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

#include "base/memory/aligned_memory.h"
#include "desktop/diff_block_32bpp_c.h"

#include <gtest/gtest.h>

namespace base {

namespace {

using AlignedBuffer = std::unique_ptr<uint8_t, AlignedFreeDeleter>;

// Run 900 times to mimic 1280x720.
const int kTimesToRun = 900;
const int kBytesPerPixel = 4;
const int kAlignment = 16;

void generateData(uint8_t* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = i;
}

int fullBlockSize(int block_size)
{
    return block_size * block_size * kBytesPerPixel;
}

void prepareBuffers(AlignedBuffer* block1, AlignedBuffer* block2, int block_size, int alignment)
{
    int full_block_size = fullBlockSize(block_size);

    block1->reset(reinterpret_cast<uint8_t*>(alignedAlloc(full_block_size, alignment)));
    block2->reset(reinterpret_cast<uint8_t*>(alignedAlloc(full_block_size, alignment)));

    generateData(block1->get(), full_block_size);

    memcpy(block2->get(), block1->get(), full_block_size);
}

} // namespace

TEST(diff_block_c, block_difference_test_same)
{
    AlignedBuffer block1;
    AlignedBuffer block2;

    {
        static const int kBlockSize = 32;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);

        // These blocks should match.
        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_32x32_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(0, result);
        }
    }

    {
        static const int kBlockSize = 16;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);

        // These blocks should match.
        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_16x16_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(0, result);
        }
    }
}

TEST(diff_block_c, block_difference_test_last)
{
    AlignedBuffer block1;
    AlignedBuffer block2;

    {
        static const int kBlockSize = 32;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[fullBlockSize(kBlockSize) - 2] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_32x32_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }

    {
        static const int kBlockSize = 16;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[fullBlockSize(kBlockSize) - 2] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_16x16_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }
}

TEST(diff_block_c, block_difference_test_mid)
{
    AlignedBuffer block1;
    AlignedBuffer block2;

    {
        static const int kBlockSize = 32;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[fullBlockSize(kBlockSize) / 2 + 1] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_32x32_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }

    {
        static const int kBlockSize = 16;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[fullBlockSize(kBlockSize) / 2 + 1] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_16x16_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }
}

TEST(diff_block_c, block_difference_test_first)
{
    AlignedBuffer block1;
    AlignedBuffer block2;

    {
        static const int kBlockSize = 32;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[0] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_32x32_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }

    {
        static const int kBlockSize = 16;

        prepareBuffers(&block1, &block2, kBlockSize, kAlignment);
        block2.get()[0] += 1;

        for (int i = 0; i < kTimesToRun; ++i)
        {
            int result = diffFullBlock_32bpp_16x16_C(
                block1.get(), block2.get(), kBlockSize * kBytesPerPixel);
            EXPECT_EQ(1, result);
        }
    }
}

} // namespace base
