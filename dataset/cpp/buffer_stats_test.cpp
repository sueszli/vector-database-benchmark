// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/datastore/buffer_stats.h>
#include <vespa/vespalib/datastore/memory_stats.h>
#include <vespa/vespalib/gtest/gtest.h>

using namespace vespalib::datastore;

TEST(BufferStatsTest, buffer_stats_to_memory_stats)
{
    InternalBufferStats buf;
    buf.set_alloc_entries(17);
    buf.pushed_back(7);
    buf.set_dead_entries(5);
    buf.set_hold_entries(3);
    buf.inc_extra_used_bytes(13);
    buf.inc_extra_hold_bytes(11);

    MemoryStats mem;
    constexpr size_t es = 8;
    buf.add_to_mem_stats(es, mem);

    EXPECT_EQ(17, mem._alloc_entries);
    EXPECT_EQ(7, mem._used_entries);
    EXPECT_EQ(5, mem._dead_entries);
    EXPECT_EQ(3, mem._hold_entries);
    EXPECT_EQ(17 * es + 13, mem._allocBytes);
    EXPECT_EQ(7 * es + 13, mem._usedBytes);
    EXPECT_EQ(5 * es, mem._deadBytes);
    EXPECT_EQ(3 * es + 11, mem._holdBytes);
}

GTEST_MAIN_RUN_ALL_TESTS()

