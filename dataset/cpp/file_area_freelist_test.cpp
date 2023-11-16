// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/util/file_area_freelist.h>
#include <vespa/vespalib/gtest/gtest.h>

using vespalib::alloc::FileAreaFreeList;

class FileAreaFreeListTest : public ::testing::Test
{
protected:
    FileAreaFreeList _freelist;
    static constexpr auto bad_offset = FileAreaFreeList::bad_offset;

public:
    FileAreaFreeListTest();
    ~FileAreaFreeListTest();
    void test_merge_area_blocked(bool previous);
};

FileAreaFreeListTest::FileAreaFreeListTest()
    : _freelist()
{
}

FileAreaFreeListTest::~FileAreaFreeListTest() = default;

void
FileAreaFreeListTest::test_merge_area_blocked(bool previous)
{
    _freelist.add_premmapped_area(4, 1);
    _freelist.add_premmapped_area(5, 1);
    EXPECT_EQ(4, _freelist.alloc(1));
    EXPECT_EQ(5, _freelist.alloc(1));
    if (previous) {
        _freelist.free(4, 1);
        _freelist.free(5, 1);
    } else {
        _freelist.free(5, 1);
        _freelist.free(4, 1);
    }
    EXPECT_EQ(bad_offset, _freelist.alloc(2));
    _freelist.remove_premmapped_area(4, 1);
    _freelist.remove_premmapped_area(5, 1);
}

TEST_F(FileAreaFreeListTest, empty_freelist_is_ok)
{
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}

TEST_F(FileAreaFreeListTest, can_reuse_free_area)
{
    _freelist.free(4, 1);
    EXPECT_EQ(4, _freelist.alloc(1));
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}

TEST_F(FileAreaFreeListTest, merge_area_with_next_area)
{
    _freelist.free(5, 1);
    _freelist.free(4, 1);
    EXPECT_EQ(4, _freelist.alloc(2));
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}

TEST_F(FileAreaFreeListTest, merge_area_with_next_area_blocked_by_fence)
{
    test_merge_area_blocked(false);
}

TEST_F(FileAreaFreeListTest, merge_area_with_previous_area)
{
    _freelist.free(3, 1);
    _freelist.free(4, 1);
    EXPECT_EQ(3, _freelist.alloc(2));
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}

TEST_F(FileAreaFreeListTest, merge_area_with_previous_area_blocked_by_fence)
{
    test_merge_area_blocked(true);
}

TEST_F(FileAreaFreeListTest, merge_area_with_previous_and_next_area)
{
    _freelist.free(5, 1);
    _freelist.free(3, 1);
    _freelist.free(4, 1);
    EXPECT_EQ(3, _freelist.alloc(3));
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}

TEST_F(FileAreaFreeListTest, can_use_part_of_free_area)
{
    _freelist.free(4, 2);
    EXPECT_EQ(4, _freelist.alloc(1));
    EXPECT_EQ(5, _freelist.alloc(1));
    EXPECT_EQ(bad_offset, _freelist.alloc(1));
}


GTEST_MAIN_RUN_ALL_TESTS()
