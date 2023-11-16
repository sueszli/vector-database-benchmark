// Copyright 2020 Redpanda Data, Inc.
//
// Use of this software is governed by the Business Source License
// included in the file licenses/BSL.md
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0

#include "config/property.h"
#include "raft/timeout_jitter.h"

#include <seastar/core/thread.hh>
#include <seastar/testing/thread_test_case.hh>

#include <boost/test/unit_test.hpp>

#include <chrono>

using namespace std::chrono_literals; // NOLINT

SEASTAR_THREAD_TEST_CASE(base_jitter_gurantees) {
    raft::timeout_jitter jit(
      config::mock_binding<std::chrono::milliseconds>(100ms));
    auto const low = jit.base_duration();
    auto const high = jit.base_duration() + 50ms;
    BOOST_CHECK_EQUAL(
      std::chrono::duration_cast<std::chrono::milliseconds>(low).count(),
      (100ms).count());
    for (auto i = 0; i < 10; ++i) {
        auto now = raft::clock_type::now();
        auto next = jit();
        BOOST_CHECK(next >= now + low && next <= now + high);
    }
}
