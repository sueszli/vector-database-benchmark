// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/util/spin_lock.h>
#include <vespa/vespalib/util/rw_spin_lock.h>
#include <vespa/vespalib/util/time.h>
#include <vespa/vespalib/util/classname.h>
#include <vespa/vespalib/test/thread_meets.h>
#include <vespa/vespalib/test/nexus.h>
#include <vespa/vespalib/gtest/gtest.h>
#include <type_traits>
#include <ranges>
#include <random>
#include <array>
#include <algorithm>

using namespace vespalib;
using namespace vespalib::test;

bool bench = false;
duration budget = 250ms;
constexpr size_t LOOP_CNT = 4096;
size_t thread_safety_work = 1'000'000;
size_t state_loop = 1;

//-----------------------------------------------------------------------------

/**
 * Estimates the 80th percentile by throwing away the 2 best samples
 * in each set of 10 samples, using the best remaining sample as a
 * representative for the set. Representatives are hierarchically
 * matched against representatives from other sample sets. Result
 * extraction is simplified in that it does not try to estimate the
 * actual 80th percentile, but rather tries to drop the best samples
 * if possible.
 *
 * The goal is to have a more robust way of combining repeated
 * micro-benchmark samples than simply using minimum time. With simple
 * single-threaded CPU-bound tasks, minimum time is a good measure of
 * how expensive something is, but when we start benchmarking
 * operations that may conflict with themselves, we do not want to
 * account for being super lucky. However, we still want to account
 * for the benchmark conditions being as good as possible.
 **/
struct Est80P {
    struct Level {
        int cnt;
        std::array<double,3> data;
        Level(double value) noexcept
          : cnt(1), data{value, 0.0, 0.0} {}
        bool empty() const noexcept { return (cnt == 0); }
        bool full() const noexcept { return (cnt == 10); }
        void add(double value) noexcept {
            assert(!full());
            if (cnt < 3 || data[2] > value) {
                size_t i = std::min(cnt, 2);
                while (i > 0 && data[i - 1] > value) {
                    data[i] = data[i - 1];
                    --i;
                }
                data[i] = value;
            }
            ++cnt;
        }
        double get() const noexcept {
            assert(!empty());
            return data[std::min(2, cnt - 1)];
        }
        void clear() noexcept {
            cnt = 0;
        }
    };
    std::vector<Level> levels;
    void add_sample(double value) {
        for (auto &level: levels) {
            level.add(value);
            if (!level.full()) [[likely]] {
                return;
            }
            value = level.get();
            level.clear();
        }
        levels.emplace_back(value);
    }
    double get_result() {
        assert(!levels.empty());
        return levels.back().get();
    }
};

//-----------------------------------------------------------------------------

struct DummyLock {
    constexpr DummyLock() noexcept {}
    // BasicLockable
    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}
    // SharedLockable
    constexpr void lock_shared() noexcept {}
    [[nodiscard]] constexpr bool try_lock_shared() noexcept { return true; }
    constexpr void unlock_shared() noexcept {}
    // rw_upgrade_downgrade_lock
    [[nodiscard]] constexpr bool try_convert_read_to_write() noexcept { return true; }
    constexpr void convert_write_to_read() noexcept {}
};

//-----------------------------------------------------------------------------

struct MyState {
    static constexpr size_t SZ = 5;
    std::array<std::atomic<size_t>,SZ> state = {0,0,0,0,0};
    std::atomic<size_t> inconsistent_reads = 0;
    std::atomic<size_t> expected_writes = 0;
    [[nodiscard]] size_t update() {
        std::array<size_t,SZ> tmp;
        for (size_t i = 0; i < SZ; ++i) {
            tmp[i] = state[i].load(std::memory_order_relaxed);
        }
        for (size_t n = 0; n < state_loop; ++n) {
            for (size_t i = 0; i < SZ; ++i) {
                state[i].store(tmp[i] + 1, std::memory_order_relaxed);
            }
        }
        return 1;
    }
    [[nodiscard]] size_t peek() {
        size_t my_inconsistent_reads = 0;
        std::array<size_t,SZ> tmp;
        for (size_t i = 0; i < SZ; ++i) {
            tmp[i] = state[i].load(std::memory_order_relaxed);
        }
        for (size_t n = 0; n < state_loop; ++n) {
            for (size_t i = 0; i < SZ; ++i) {
                if (state[i].load(std::memory_order_relaxed) != tmp[i]) [[unlikely]] {
                    ++my_inconsistent_reads;
                }
            }
        }
        return my_inconsistent_reads;
    }
    void commit_inconsistent_reads(size_t n) {
        inconsistent_reads.fetch_add(n, std::memory_order_relaxed);
    }
    void commit_expected_writes(size_t n) {
        expected_writes.fetch_add(n, std::memory_order_relaxed);
    }
    [[nodiscard]] bool check() const {
        if (inconsistent_reads > 0) {
            return false;
        }
        for (const auto& value: state) {
            if (value != expected_writes) {
               return false;
            }
        }
        return true;
    }
    void report(const char *name) const {
        if (check()) {
            fprintf(stderr, "%s is thread safe\n", name);
        } else {
            fprintf(stderr, "%s is not thread safe\n", name);
            fprintf(stderr, "    inconsistent reads: %zu\n", inconsistent_reads.load());
            fprintf(stderr, "    expected %zu, got [%zu,%zu,%zu,%zu,%zu]\n",
                    expected_writes.load(), state[0].load(), state[1].load(), state[2].load(), state[3].load(), state[4].load());
        }
    }
};

// random generator used to make per-thread decisions
class Rnd {
private:
    std::mt19937 _engine;
    std::uniform_int_distribution<int> _dist;    
public:
    Rnd(uint32_t seed) : _engine(seed), _dist(0,9999) {}
    bool operator()(int bp) { return _dist(_engine) < bp; }
};

//-----------------------------------------------------------------------------

template<typename T>
concept basic_lockable = requires(T a) {
    { a.lock() } -> std::same_as<void>;
    { a.unlock() } -> std::same_as<void>;
};

template<typename T>
concept lockable = requires(T a) {
    { a.try_lock() } -> std::same_as<bool>;
    { a.lock() } -> std::same_as<void>;
    { a.unlock() } -> std::same_as<void>;
};

template<typename T>
concept shared_lockable = requires(T a) {
    { a.try_lock_shared() } -> std::same_as<bool>;
    { a.lock_shared() } -> std::same_as<void>;
    { a.unlock_shared() } -> std::same_as<void>;
};

template<typename T>
concept can_upgrade = requires(std::shared_lock<T> a, std::unique_lock<T> b) {
    { try_upgrade(std::move(a)) } -> std::same_as<std::unique_lock<T>>;
    { downgrade(std::move(b)) } -> std::same_as<std::shared_lock<T>>;
};

//-----------------------------------------------------------------------------

template <size_t N>
auto run_loop(auto &f) {
    static_assert(N % 4 == 0);
    for (size_t i = 0; i < N / 4; ++i) {
        f(); f(); f(); f();
    }
}

double measure_ns(auto &work) __attribute__((noinline));
double measure_ns(auto &work) {
    constexpr double factor = LOOP_CNT;
    auto t0 = steady_clock::now();
    run_loop<LOOP_CNT>(work);
    return count_ns(steady_clock::now() - t0) / factor;
}

struct BenchmarkResult {
    double cost_ns;
    BenchmarkResult(double cost_ns_in) : cost_ns(cost_ns_in) {}
    void report(vespalib::string desc) {
        fprintf(stderr, "%s: cost_ns: %g\n", desc.c_str(), cost_ns);
    }
    void report(vespalib::string name, vespalib::string desc) {
        report(name + "(" + desc + ")");
    }
};

BenchmarkResult benchmark_ns(auto &&work, size_t num_threads = 1) {
    Est80P collector;
    vespalib::test::ThreadMeets::Avg avg(num_threads);
    auto entry = [&](Nexus &ctx) {
        Timer timer;
        BenchmarkResult result(ctx.num_threads());
        for (bool once_more = true; ctx.vote(once_more); once_more = (timer.elapsed() < budget)) {
            auto cost_ns = avg(measure_ns(work));
            if (ctx.is_main()) {
                collector.add_sample(cost_ns);
            }
        }
    };
    Nexus::run(num_threads, entry);
    auto result = collector.get_result();
    return {result};
}

//-----------------------------------------------------------------------------

template <typename T>
void estimate_cost() {
    T lock;
    auto name = getClassName(lock);
    static_assert(basic_lockable<T>);
    benchmark_ns([&lock]{ lock.lock(); lock.unlock(); }).report(name, "exclusive lock/unlock");
    if constexpr (shared_lockable<T>) {
        benchmark_ns([&lock]{ lock.lock_shared(); lock.unlock_shared(); }).report(name, "shared lock/unlock");
    }
    if constexpr (can_upgrade<T>) {
        auto guard = std::shared_lock(lock);
        benchmark_ns([&lock]{
                         assert(lock.try_convert_read_to_write());
                         lock.convert_write_to_read();
                     }).report(name, "upgrade/downgrade");
    }
}

//-----------------------------------------------------------------------------

template <typename T>
void thread_safety_loop(Nexus &ctx, T &lock, MyState &state, auto &max, int read_bp) {
    Rnd rnd(ctx.thread_id());
    size_t write_cnt = 0;
    size_t bad_reads = 0;
    size_t loop_cnt = thread_safety_work / ctx.num_threads();
    ctx.barrier();
    auto t0 = steady_clock::now();
    for (size_t i = 0; i < loop_cnt; ++i) {
        if (rnd(read_bp)) {
            if constexpr (shared_lockable<T>) {
                std::shared_lock guard(lock);
                bad_reads += state.peek();
            } else {
                std::lock_guard guard(lock);
                bad_reads += state.peek();
            }
        } else {
            {
                std::lock_guard guard(lock);
                write_cnt += state.update();                
            }
        }
    }
    auto my_ms = count_ns(steady_clock::now() - t0) / 1'000'000.0;
    auto actual_ms = max(my_ms);
    if (ctx.is_main()) {
        fprintf(stderr, "---> %s with %2zu threads (%5d bp r): time: %10.2f ms\n",
                getClassName(lock).c_str(), ctx.num_threads(), read_bp, actual_ms);
    }
    state.commit_inconsistent_reads(bad_reads);
    state.commit_expected_writes(write_cnt);
}

//-----------------------------------------------------------------------------

TEST(RWSpinLockTest, different_guards_work_with_rw_spin_lock) {
    static_assert(basic_lockable<RWSpinLock>);
    static_assert(lockable<RWSpinLock>);
    static_assert(shared_lockable<RWSpinLock>);
    static_assert(can_upgrade<RWSpinLock>);
    RWSpinLock lock;
    { auto guard = std::lock_guard(lock); }
    { auto guard = std::unique_lock(lock); }
    { auto guard = std::shared_lock(lock); }
}

TEST(RWSpinLockTest, estimate_basic_costs) {
    Rnd rnd(123);
    MyState state;
    benchmark_ns([&]{ rnd(50); })              .report("   rnd cost");
    benchmark_ns([&]{ (void) state.peek(); })  .report("  peek cost");
    benchmark_ns([&]{ (void) state.update(); }).report("update cost");
}

template <typename T>
void benchmark_lock() {
    auto lock = std::make_unique<T>();
    auto state = std::make_unique<MyState>();
    for (size_t bp: {10000, 9999, 5000, 0}) {
        for (size_t num_threads: {8, 4, 2, 1}) {
            if (bench || (bp == 9999 && num_threads == 8)) {
                vespalib::test::ThreadMeets::Max<double> max(num_threads);
                Nexus::run(num_threads, [&](Nexus &ctx) {
                    thread_safety_loop(ctx, *lock, *state, max, bp);
                });
            }
        }
    }
    state->report(getClassName(*lock).c_str());
    if (!std::same_as<T,DummyLock>) {
        EXPECT_TRUE(state->check());
    }
}

TEST(RWSpinLockTest, benchmark_dummy_lock)   { benchmark_lock<DummyLock>(); }
TEST(RWSpinLockTest, benchmark_rw_spin_lock) { benchmark_lock<RWSpinLock>(); }
TEST(RWSpinLockTest, benchmark_shared_mutex) { benchmark_lock<std::shared_mutex>(); }
TEST(RWSpinLockTest, benchmark_mutex)        { benchmark_lock<std::mutex>(); }
TEST(RWSpinLockTest, benchmark_spin_lock)    { benchmark_lock<SpinLock>(); }

struct MyRefCnt {
    std::atomic<uint32_t> value;
    void fetch_add() noexcept {
        value.fetch_add(1, std::memory_order_acquire);
    }
    void fetch_sub() noexcept {
        value.fetch_sub(1, std::memory_order_release);
    };
    void cmp_add_guess() noexcept {
        uint32_t expected = 0;
        uint32_t desired = 1;
        while (!value.compare_exchange_weak(expected, desired,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed))
        {
            desired = expected + 1;
        }
    }
    void cmp_sub_guess() noexcept {
        uint32_t expected = 1;
        uint32_t desired = 0;
        while (!value.compare_exchange_weak(expected, desired,
                                            std::memory_order_release,
                                            std::memory_order_relaxed))
        {
            desired = expected - 1;
        }
    }
    void cmp_add_load() noexcept {
        uint32_t expected = value.load(std::memory_order_relaxed);
        uint32_t desired = expected + 1;
        while (!value.compare_exchange_weak(expected, desired,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed))
        {
            desired = expected + 1;
        }
    }
    void cmp_sub_load() noexcept {
        uint32_t expected = value.load(std::memory_order_relaxed);
        uint32_t desired = expected - 1;
        while (!value.compare_exchange_weak(expected, desired,
                                            std::memory_order_release,
                                            std::memory_order_relaxed))
        {
            desired = expected - 1;
        }
    }
};

TEST(RWSpinLockTest, benchmark_compare_exchange_vs_fetch_add_sub) {
    if (!bench) {
        fprintf(stderr, "[ SKIPPED  ] this test is only run in benchmarking mode\n");
        return;
    }
    MyRefCnt value;
    auto fetch_add = [&value]{ value.fetch_add(); };
    auto fetch_sub = [&value]{ value.fetch_sub(); };
    auto cmp_add_guess = [&value]{ value.cmp_add_guess(); };
    auto cmp_sub_guess = [&value]{ value.cmp_sub_guess(); };
    auto cmp_add_load = [&value]{ value.cmp_add_load(); };
    auto cmp_sub_load = [&value]{ value.cmp_sub_load(); };

    auto do_fetch = [&]{ fetch_add(); fetch_sub(); };
    auto do_cmp_guess = [&]{ cmp_add_guess(); cmp_sub_guess(); };
    auto do_cmp_load = [&]{ cmp_add_load(); cmp_sub_load(); };

    auto do_4_fetch = [&]{ run_loop<4>(fetch_add); run_loop<4>(fetch_sub); };
    auto do_4_cmp_guess = [&]{ run_loop<4>(cmp_add_guess); run_loop<4>(cmp_sub_guess); };
    auto do_4_cmp_load = [&]{ run_loop<4>(cmp_add_load); run_loop<4>(cmp_sub_load); };

    benchmark_ns(do_fetch, 4).report("fetch_add -> fetch_sub");
    benchmark_ns(do_cmp_guess, 4).report("cmp_add_guess -> cmp_sub_guess");
    benchmark_ns(do_cmp_load, 4).report("cmp_add_load -> cmp_sub_load");
    benchmark_ns(do_4_fetch, 4).report("4fetch_add -> 4fetch_sub");
    benchmark_ns(do_4_cmp_guess, 4).report("4cmp_add_guess -> 4cmp_sub_guess");
    benchmark_ns(do_4_cmp_load, 4).report("4cmp_add_load -> 4cmp_sub_load");
}

TEST(RWSpinLockTest, estimate_single_threaded_costs) {
    estimate_cost<DummyLock>();
    estimate_cost<SpinLock>();
    estimate_cost<std::mutex>();
    estimate_cost<RWSpinLock>();
    estimate_cost<std::shared_mutex>();
}

int main(int argc, char **argv) {
    if (argc > 1 && (argv[1] == std::string("bench"))) {
        bench = true;
        budget = 5s;
        state_loop = 1024;
        fprintf(stderr, "running in benchmarking mode\n");
        ++argv;
        --argc;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
