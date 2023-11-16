#include "snitch/snitch.hpp"
#include "test_watcher/constant.hpp"
#include "wtr/watcher.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off

namespace ti = std::chrono;

struct BenchCfg {
  int watcher_count;
  int event_count;
};

struct BenchResult {
  struct BenchCfg cfg;
  ti::nanoseconds time_taken;
  unsigned nth;
};

struct Range {
  int start{1};
  int stop{1};
  int step{1};
};

struct RangePair {
  Range watcher_range;
  Range event_range;
};

auto show_result(BenchResult const& res) -> void
{
  auto const ss = static_cast<long long>(ti::duration_cast<ti::seconds>(res.time_taken).count());
  auto const ms = static_cast<long long>(ti::duration_cast<ti::milliseconds>(res.time_taken).count());
  auto const us = static_cast<long long>(ti::duration_cast<ti::microseconds>(res.time_taken).count());
  auto const ns = static_cast<long long>(res.time_taken.count());

  printf(
    "%i|%i|",
    res.cfg.watcher_count, res.cfg.event_count);
  ss > 0
    ? printf("%lld s\n", ss)
    : ms > 0
      ? printf("%lld ms\n", ms)
      : us > 0
        ? printf("%lld us\n", us)
        : printf("%lld ns\n", ns);
};

template<BenchCfg cfg>
class Bench{
public:
  auto concurrent_watchers() -> BenchResult
  {
    namespace tw = wtr::test_watcher;
    namespace fs = std::filesystem;

    static unsigned nth = 0;

    if (! fs::exists(tw::test_store_path))
      fs::create_directory(tw::test_store_path);

    auto start = ti::system_clock{}.now();

    auto watchers = std::array<std::unique_ptr<wtr::watch>, cfg.watcher_count>{};

    for (int i = 0; i < cfg.watcher_count; ++i)
      watchers.at(i) = std::move(
          std::make_unique<wtr::watch>(
            tw::test_store_path,
              [](auto) {}
            ));

    for (int i = 0; i < cfg.event_count; ++i)
      std::ofstream{tw::test_store_path / std::to_string(i)};  // touch

    auto time_taken = duration_cast<ti::nanoseconds>(
        ti::system_clock{}.now() - start);

    if (std::filesystem::exists(tw::test_store_path))
      std::filesystem::remove_all(tw::test_store_path);

    return {cfg, time_taken, nth++};
  };
};

template<RangePair Rp>
constexpr auto bench_range() -> void
{
  // Until the end ...
  if constexpr (
      Rp.watcher_range.start + Rp.watcher_range.step <= Rp.watcher_range.stop
      && Rp.event_range.start + Rp.event_range.step <= Rp.event_range.stop)
  {
    // Run this ...
    auto res =
      Bench<BenchCfg{
        .watcher_count=Rp.watcher_range.start,
        .event_count=Rp.event_range.start}>{}
      .concurrent_watchers();
    show_result(res);

    // Then run the next ...
    return bench_range<
      RangePair{
        .watcher_range = Range{
          .start=Rp.watcher_range.start + Rp.watcher_range.step,
          .stop=Rp.watcher_range.stop,
          .step=Rp.watcher_range.step},
        .event_range = Range{
          .start=Rp.event_range.start + Rp.event_range.step,
          .stop=Rp.event_range.stop,
          .step=Rp.event_range.step}}>();
  }
};

/*  We bench 1..30 watchers on directories with 100..1k events with a
    callback that prints events to stdout.
    These benchmarks should be bound by how fast we can create events
    and write to stdout.
    We want to offload most of the unrelated work onto compile-time
    computations so that we can accurately measure a "common" watcher
    path/callback setup. Things like:
      Allocating, storing and resizing vectors of watchers.
      Complicated iteration logic.
    We use some fancy templates for that, but beware of this (fatal)
    compiler error when testing many watchers or events:
      template instantiation depth exceeds <some number around 1k>
    We bench 1 to 30 watchers on directories with 100 to 1k events.
    The callback prints the events to stdout. These benchmarks should
    ideally be bound by how fast we can create the events and perform
    i/o to stdout. */

TEST_CASE("Bench Concurrent watch Targets", "[bench][concurrent][file][watch-target]")
{
  printf("Watcher Count|Event Count|Time Taken\n");
  bench_range<RangePair{
    .watcher_range={.start=1, .stop=1, .step=0},
    .event_range={.start=100, .stop=1000, .step=100}}>();
};

TEST_CASE("Bench Concurrent watch Targets 2", "[bench][concurrent][file][watch-target]")
{
  printf("Watcher Count|Event Count|Time Taken\n");
  bench_range<RangePair{
    .watcher_range={.start=1, .stop=30, .step=5},
    .event_range={.start=100, .stop=100, .step=0}}>();
};

TEST_CASE("Bench Concurrent watch Targets 3", "[bench][concurrent][file][watch-target]")
{
  printf("Watcher Count|Event Count|Time Taken\n");
  bench_range<RangePair{
    .watcher_range={.start=1, .stop=1, .step=0},
    .event_range={.start=100, .stop=10000, .step=1000}}>();
};

// clang-format on
