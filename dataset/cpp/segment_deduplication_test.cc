// Copyright 2023 Redpanda Data, Inc.
//
// Use of this software is governed by the Business Source License
// included in the file licenses/BSL.md
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0

#include "gmock/gmock.h"
#include "random/generators.h"
#include "storage/disk_log_impl.h"
#include "storage/key_offset_map.h"
#include "storage/segment_deduplication_utils.h"
#include "storage/segment_utils.h"
#include "storage/tests/disk_log_builder_fixture.h"
#include "storage/tests/utils/disk_log_builder.h"
#include "test_utils/test.h"

#include <seastar/core/io_priority_class.hh>
#include <seastar/util/defer.hh>

#include <stdexcept>

using namespace storage;

namespace {
ss::abort_source never_abort;
} // anonymous namespace

// Builds a segment layout:
// [0    9][10   19][20    29]...
void build_segments(storage::disk_log_builder& b, int num_segs) {
    b | start();
    auto& disk_log = b.get_disk_log_impl();
    auto records_per_seg = 10;
    for (int i = 0; i < num_segs; i++) {
        auto offset = i * records_per_seg;
        b | add_segment(offset)
          | add_random_batch(
            offset, records_per_seg, maybe_compress_batches::yes);
    }
    for (auto& seg : disk_log.segments()) {
        seg->mark_as_finished_windowed_compaction();
        if (seg->has_appender()) {
            seg->appender().close().get();
            seg->release_appender();
        }
    }
}

TEST(FindSlidingRangeTest, TestCollectSegments) {
    storage::disk_log_builder b;
    build_segments(b, 3);
    auto cleanup = ss::defer([&] { b.stop().get(); });
    auto& disk_log = b.get_disk_log_impl();
    for (int start = 0; start < 30; start += 5) {
        for (int end = start; end < 30; end += 5) {
            compaction_config cfg(
              model::offset{end}, ss::default_priority_class(), never_abort);
            auto segs = disk_log.find_sliding_range(cfg, model::offset{start});
            if (end - start < 10) {
                // If the compactible range isn't a full segment, we can't
                // compact anything. We only care about full segments.
                ASSERT_EQ(segs.size(), 0);
                continue;
            }
            // We can't compact partial segments so we round the end down to
            // the nearest segment boundary.
            ASSERT_EQ((end - (end % 10) - start) / 10, segs.size())
              << ssx::sformat("{} to {}: {}", start, end, segs.size());
        }
    }
}

TEST(FindSlidingRangeTest, TestCollectExcludesPrevious) {
    storage::disk_log_builder b;
    build_segments(b, 3);
    auto cleanup = ss::defer([&] { b.stop().get(); });
    auto& disk_log = b.get_disk_log_impl();
    compaction_config cfg(
      model::offset{30}, ss::default_priority_class(), never_abort);
    auto segs = disk_log.find_sliding_range(cfg);
    ASSERT_EQ(3, segs.size());
    ASSERT_EQ(segs.front()->offsets().base_offset(), 0);

    // Let's pretend the previous compaction indexed offsets [20, 30).
    // Subsequent compaction should ignore that last segment.
    disk_log.set_last_compaction_window_start_offset(model::offset(20));
    segs = disk_log.find_sliding_range(cfg);
    ASSERT_EQ(2, segs.size());
    ASSERT_EQ(segs.front()->offsets().base_offset(), 0);

    disk_log.set_last_compaction_window_start_offset(model::offset(10));
    segs = disk_log.find_sliding_range(cfg);
    ASSERT_EQ(1, segs.size());
    ASSERT_EQ(segs.front()->offsets().base_offset(), 0);
}

TEST(BuildOffsetMap, TestBuildSimpleMap) {
    storage::disk_log_builder b;
    build_segments(b, 3);
    auto cleanup = ss::defer([&] { b.stop().get(); });
    auto& disk_log = b.get_disk_log_impl();
    auto& segs = disk_log.segments();
    compaction_config cfg(
      model::offset{30}, ss::default_priority_class(), never_abort);
    probe pb;

    // Self-compact each segment so we're left with compaction indices. This is
    // a requirement to build the offset map.
    for (auto& seg : segs) {
        storage::internal::self_compact_segment(
          seg,
          disk_log.stm_manager(),
          cfg,
          pb,
          disk_log.readers(),
          disk_log.resources(),
          offset_delta_time::yes)
          .get();
    }

    // Build a map, configuring it to hold too little data for even a single
    // segment.
    simple_key_offset_map too_small_map(5);
    ASSERT_THAT(
      [&] { build_offset_map(cfg, segs, too_small_map).get(); },
      testing::ThrowsMessage<std::runtime_error>(
        testing::HasSubstr("Couldn't index")));

    // Now configure a map to index some segments.
    simple_key_offset_map partial_map(15);
    auto partial_o = build_offset_map(cfg, segs, partial_map).get();
    ASSERT_GT(partial_o(), 0);

    // Now make it large enough to index all segments.
    simple_key_offset_map all_segs_map(100);
    auto all_segs_o = build_offset_map(cfg, segs, all_segs_map).get();
    ASSERT_EQ(all_segs_o(), 0);
}

TEST(BuildOffsetMap, TestBuildMapWithError) {
    storage::disk_log_builder b;
    build_segments(b, 3);
    auto cleanup = ss::defer([&] { b.stop().get(); });
    auto& segs = b.get_disk_log_impl().segments();
    compaction_config cfg(
      model::offset{30}, ss::default_priority_class(), never_abort);

    // Proceed to window compaction without building any compacted indexes.
    // This should indicate a failure to build the map.
    simple_key_offset_map missing_index_map(100);
    ASSERT_THAT(
      [&] { build_offset_map(cfg, segs, missing_index_map).get(); },
      testing::ThrowsMessage<std::runtime_error>(
        testing::HasSubstr("compaction_index size is too small")));
}
