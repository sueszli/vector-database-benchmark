// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Unit tests for diskindexcleaner.

#include <vespa/searchcorespi/index/disk_indexes.h>
#include <vespa/searchcorespi/index/diskindexcleaner.h>
#include <vespa/vespalib/testkit/testapp.h>
#include <vespa/fastos/file.h>
#include <algorithm>
#include <filesystem>

#include <vespa/log/log.h>
LOG_SETUP("diskindexcleaner_test");

using std::string;
using std::vector;
using namespace searchcorespi::index;

namespace {

class Test : public vespalib::TestApp {
    void requireThatAllIndexesOlderThanLastFusionIsRemoved();
    void requireThatIndexesInUseAreNotRemoved();
    void requireThatInvalidFlushIndexesAreRemoved();
    void requireThatInvalidFusionIndexesAreRemoved();
    void requireThatRemoveDontTouchNewIndexes();

public:
    int Main() override;
};

const string index_dir = "diskindexcleaner_test_data";

void removeTestData() {
    std::filesystem::remove_all(std::filesystem::path(index_dir));
}

int
Test::Main()
{
    TEST_INIT("diskindexcleaner_test");

    TEST_DO(removeTestData());

    TEST_DO(requireThatAllIndexesOlderThanLastFusionIsRemoved());
    TEST_DO(requireThatIndexesInUseAreNotRemoved());
    TEST_DO(requireThatInvalidFlushIndexesAreRemoved());
    TEST_DO(requireThatInvalidFusionIndexesAreRemoved());
    TEST_DO(requireThatRemoveDontTouchNewIndexes());

    TEST_DO(removeTestData());

    TEST_DONE();
}

void createIndex(const string &name) {
    std::filesystem::create_directory(std::filesystem::path(index_dir));
    const string dir_name = index_dir + "/" + name;
    std::filesystem::create_directory(std::filesystem::path(dir_name));
    const string serial_file = dir_name + "/serial.dat";
    FastOS_File file(serial_file.c_str());
    file.OpenWriteOnlyTruncate();
}

vector<string> readIndexes() {
    vector<string> indexes;
    std::filesystem::directory_iterator dir_scan(index_dir);
    for (auto& entry : dir_scan) {
        if (entry.is_directory() && entry.path().filename().string().find("index.") == 0) {
            indexes.push_back(entry.path().filename().string());
        }
    }
    return indexes;
}

template <class Container>
bool contains(Container c, typename Container::value_type v) {
    return find(c.begin(), c.end(), v) != c.end();
}

void createIndexes() {
    createIndex("index.flush.0");
    createIndex("index.flush.1");
    createIndex("index.fusion.1");
    createIndex("index.flush.2");
    createIndex("index.fusion.2");
    createIndex("index.flush.3");
    createIndex("index.flush.4");
}

void Test::requireThatAllIndexesOlderThanLastFusionIsRemoved() {
    createIndexes();
    DiskIndexes disk_indexes;
    DiskIndexCleaner::clean(index_dir, disk_indexes);
    vector<string> indexes = readIndexes();
    EXPECT_EQUAL(3u, indexes.size());
    EXPECT_TRUE(contains(indexes, "index.fusion.2"));
    EXPECT_TRUE(contains(indexes, "index.flush.3"));
    EXPECT_TRUE(contains(indexes, "index.flush.4"));
}

void Test::requireThatIndexesInUseAreNotRemoved() {
    createIndexes();
    DiskIndexes disk_indexes;
    disk_indexes.setActive(index_dir + "/index.fusion.1", 0);
    disk_indexes.setActive(index_dir + "/index.flush.2", 0);
    DiskIndexCleaner::clean(index_dir, disk_indexes);
    vector<string> indexes = readIndexes();
    EXPECT_TRUE(contains(indexes, "index.fusion.1"));
    EXPECT_TRUE(contains(indexes, "index.flush.2"));

    disk_indexes.notActive(index_dir + "/index.fusion.1");
    disk_indexes.notActive(index_dir + "/index.flush.2");
    DiskIndexCleaner::clean(index_dir, disk_indexes);
    indexes = readIndexes();
    EXPECT_TRUE(!contains(indexes, "index.fusion.1"));
    EXPECT_TRUE(!contains(indexes, "index.flush.2"));
}

void Test::requireThatInvalidFlushIndexesAreRemoved() {
    createIndexes();
    std::filesystem::remove(std::filesystem::path(index_dir + "/index.flush.4/serial.dat"));
    DiskIndexes disk_indexes;
    DiskIndexCleaner::clean(index_dir, disk_indexes);
    vector<string> indexes = readIndexes();
    EXPECT_EQUAL(2u, indexes.size());
    EXPECT_TRUE(contains(indexes, "index.fusion.2"));
    EXPECT_TRUE(contains(indexes, "index.flush.3"));
}

void Test::requireThatInvalidFusionIndexesAreRemoved() {
    createIndexes();
    std::filesystem::remove(std::filesystem::path(index_dir + "/index.fusion.2/serial.dat"));
    DiskIndexes disk_indexes;
    DiskIndexCleaner::clean(index_dir, disk_indexes);
    vector<string> indexes = readIndexes();
    EXPECT_EQUAL(4u, indexes.size());
    EXPECT_TRUE(contains(indexes, "index.fusion.1"));
    EXPECT_TRUE(contains(indexes, "index.flush.2"));
    EXPECT_TRUE(contains(indexes, "index.flush.3"));
    EXPECT_TRUE(contains(indexes, "index.flush.4"));
}

void Test::requireThatRemoveDontTouchNewIndexes() {
    createIndexes();
    std::filesystem::remove(std::filesystem::path(index_dir + "/index.flush.4/serial.dat"));
    DiskIndexes disk_indexes;
    DiskIndexCleaner::removeOldIndexes(index_dir, disk_indexes);
    vector<string> indexes = readIndexes();
    EXPECT_EQUAL(3u, indexes.size());
    EXPECT_TRUE(contains(indexes, "index.fusion.2"));
    EXPECT_TRUE(contains(indexes, "index.flush.3"));
    EXPECT_TRUE(contains(indexes, "index.flush.4"));
}

}  // namespace

TEST_APPHOOK(Test);
