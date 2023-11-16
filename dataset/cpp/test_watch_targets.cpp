#include "snitch/snitch.hpp"
#include "test_watcher/test_watcher.hpp"
#include "wtr/watcher.hpp"
#include <iostream>
#include <tuple>
#include <vector>

/* Test that files are scanned */
TEST_CASE("Event Targets", "[test][file][dir][watch-target]")
{
  using namespace wtr::watcher;
  using namespace wtr::test_watcher;

  check_event_lists_eq(
    watch_gather("Event Targets", test_store_path / "event_targets_store", 10));
};
