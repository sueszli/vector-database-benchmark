/*
 * Copyright (C) 2018-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <boost/range/adaptors.hpp>

#include <seastar/core/sleep.hh>
#include <seastar/testing/perf_tests.hh>
#include <seastar/util/closeable.hh>

#include "test/lib/simple_schema.hh"
#include "test/lib/simple_position_reader_queue.hh"
#include "test/perf/perf.hh"

#include "readers/from_mutations_v2.hh"
#include "readers/mutation_fragment_v1_stream.hh"
#include "readers/empty_v2.hh"
#include "readers/combined.hh"
#include "replica/memtable.hh"

namespace tests {

class combined {
    mutable simple_schema _schema;
    perf::reader_concurrency_semaphore_wrapper _semaphore;
    reader_permit _permit;
    std::vector<mutation> _one_row;
    std::vector<mutation> _single;
    std::vector<std::vector<mutation>> _disjoint_interleaved;
    std::vector<std::vector<mutation>> _disjoint_ranges;
    std::vector<std::vector<mutation>> _overlapping_partitions_disjoint_rows;
private:
    static std::vector<mutation> create_one_row(simple_schema&, reader_permit);
    static std::vector<mutation> create_single_stream(simple_schema&, reader_permit);
    static std::vector<std::vector<mutation>> create_disjoint_interleaved_streams(simple_schema&, reader_permit);
    static std::vector<std::vector<mutation>> create_disjoint_ranges_streams(simple_schema&, reader_permit);
    static std::vector<std::vector<mutation>> create_overlapping_partitions_disjoint_rows_streams(simple_schema&, reader_permit);
protected:
    simple_schema& schema() const { return _schema; }
    reader_permit permit() const { return _permit; }
    const std::vector<mutation>& one_row_stream() const { return _one_row; }
    const std::vector<mutation>& single_stream() const { return _single; }
    const std::vector<std::vector<mutation>>& disjoint_interleaved_streams() const {
        return _disjoint_interleaved;
    }
    const std::vector<std::vector<mutation>>& disjoint_ranges_streams() const {
        return _disjoint_ranges;
    }
    const std::vector<std::vector<mutation>>& overlapping_partitions_disjoint_rows_streams() const {
        return _overlapping_partitions_disjoint_rows;
    }
    future<> consume_all(flat_mutation_reader_v2 mr) const;
public:
    combined()
        : _semaphore("combined")
        , _permit(_semaphore.make_permit())
        , _one_row(create_one_row(_schema, _permit))
        , _single(create_single_stream(_schema, _permit))
        , _disjoint_interleaved(create_disjoint_interleaved_streams(_schema, _permit))
        , _disjoint_ranges(create_disjoint_ranges_streams(_schema, _permit))
        , _overlapping_partitions_disjoint_rows(create_overlapping_partitions_disjoint_rows_streams(_schema, _permit))
    { }
};

std::vector<mutation> combined::create_one_row(simple_schema& s, reader_permit permit)
{
    return boost::copy_range<std::vector<mutation>>(
        s.make_pkeys(1)
        | boost::adaptors::transformed([&] (auto& dkey) {
            auto m = mutation(s.schema(), dkey);
            m.apply(s.make_row(permit, s.make_ckey(0), "value"));
            return m;
        })
    );
}

std::vector<mutation> combined::create_single_stream(simple_schema& s, reader_permit permit)
{
    return boost::copy_range<std::vector<mutation>>(
        s.make_pkeys(32)
        | boost::adaptors::transformed([&] (auto& dkey) {
            auto m = mutation(s.schema(), dkey);
            for (auto i = 0; i < 16; i++) {
                m.apply(s.make_row(permit, s.make_ckey(i), "value"));
            }
            return m;
        })
    );
}

std::vector<std::vector<mutation>> combined::create_disjoint_interleaved_streams(simple_schema& s, reader_permit permit)
{
    auto base = create_single_stream(s, permit);
    std::vector<std::vector<mutation>> mss;
    for (auto i = 0; i < 4; i++) {
        mss.emplace_back(boost::copy_range<std::vector<mutation>>(
            base
            | boost::adaptors::sliced(i, base.size())
            | boost::adaptors::strided(4)
        ));
    }
    return mss;
}

std::vector<std::vector<mutation>> combined::create_disjoint_ranges_streams(simple_schema& s, reader_permit permit)
{
    auto base = create_single_stream(s, permit);
    std::vector<std::vector<mutation>> mss;
    auto slice = base.size() / 4;
    for (auto i = 0; i < 4; i++) {
        mss.emplace_back(boost::copy_range<std::vector<mutation>>(
            base
            | boost::adaptors::sliced(i * slice, std::min((i + 1) * slice, base.size()))
        ));
    }
    return mss;
}

std::vector<std::vector<mutation>> combined::create_overlapping_partitions_disjoint_rows_streams(simple_schema& s, reader_permit permit) {
    auto keys = s.make_pkeys(4);
    std::vector<std::vector<mutation>> mss;
    for (int i = 0; i < 4; i++) {
        mss.emplace_back(boost::copy_range<std::vector<mutation>>(
            keys
            | boost::adaptors::transformed([&] (auto& dkey) {
                auto m = mutation(s.schema(), dkey);
                for (int j = 0; j < 32; j++) {
                    m.apply(s.make_row(permit, s.make_ckey(32 * i + j), "value"));
                }
                return m;
            })
        ));
    }
    return mss;
}

future<> combined::consume_all(flat_mutation_reader_v2 mr) const
{
    return with_closeable(mutation_fragment_v1_stream(std::move(mr)), [] (auto& mr) {
        perf_tests::start_measuring_time();
        return mr.consume_pausable([] (mutation_fragment mf) {
            perf_tests::do_not_optimize(mf);
            return stop_iteration::no;
        }).then([] {
            perf_tests::stop_measuring_time();
        });
    });
}

PERF_TEST_F(combined, one_mutation)
{
    std::vector<flat_mutation_reader_v2> mrs;
    mrs.emplace_back(make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), one_row_stream()[0]));
    return consume_all(make_combined_reader(schema().schema(), permit(), std::move(mrs)));
}

PERF_TEST_F(combined, one_row)
{
    std::vector<flat_mutation_reader_v2> mrs;
    mrs.emplace_back(make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), one_row_stream()));
    return consume_all(make_combined_reader(schema().schema(), permit(), std::move(mrs)));
}

PERF_TEST_F(combined, single_active)
{
    std::vector<flat_mutation_reader_v2> mrs;
    mrs.reserve(4);
    mrs.emplace_back(make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), single_stream()));
    for (auto i = 0; i < 3; i++) {
        mrs.emplace_back(make_empty_flat_reader_v2(schema().schema(), permit()));
    }
    return consume_all(make_combined_reader(schema().schema(), permit(), std::move(mrs)));
}

PERF_TEST_F(combined, many_overlapping)
{
    std::vector<flat_mutation_reader_v2> mrs;
    mrs.reserve(4);
    for (auto i = 0; i < 4; i++) {
        mrs.emplace_back(make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), single_stream()));
    }
    return consume_all(make_combined_reader(schema().schema(), permit(), std::move(mrs)));
}

PERF_TEST_F(combined, disjoint_interleaved)
{
    return consume_all(make_combined_reader(schema().schema(), permit(),
        boost::copy_range<std::vector<flat_mutation_reader_v2>>(
            disjoint_interleaved_streams()
            | boost::adaptors::transformed([this] (auto&& ms) {
                return schema().schema(), make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), std::move(ms));
            })
        )
    ));
}

PERF_TEST_F(combined, disjoint_ranges)
{
    return consume_all(make_combined_reader(schema().schema(), permit(),
        boost::copy_range<std::vector<flat_mutation_reader_v2>>(
            disjoint_ranges_streams()
            | boost::adaptors::transformed([this] (auto&& ms) {
                return make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), std::move(ms));
            })
        )
    ));
}

PERF_TEST_F(combined, overlapping_partitions_disjoint_rows)
{
    return consume_all(make_combined_reader(schema().schema(), permit(),
        boost::copy_range<std::vector<flat_mutation_reader_v2>>(
            overlapping_partitions_disjoint_rows_streams()
            | boost::adaptors::transformed([this] (auto&& ms) {
                return make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), std::move(ms));
            })
        )
    ));
}

struct mutation_bounds {
    mutation m;
    position_in_partition lower;
    position_in_partition upper;
};

class clustering_combined {
    mutable simple_schema _schema;
    perf::reader_concurrency_semaphore_wrapper _semaphore;
    reader_permit _permit;
    std::vector<mutation_bounds> _almost_disjoint_ranges;
private:
    static std::vector<mutation_bounds> create_almost_disjoint_ranges(simple_schema&, reader_permit);
protected:
    simple_schema& schema() const { return _schema; }
    reader_permit permit() const { return _permit; }
    const std::vector<mutation_bounds>& almost_disjoint_clustering_ranges() const {
        return _almost_disjoint_ranges;
    }
    future<size_t> consume_all(flat_mutation_reader_v2 mr) const;
public:
    clustering_combined()
        : _semaphore("clustering_combined")
        , _permit(_semaphore.make_permit())
        , _almost_disjoint_ranges(create_almost_disjoint_ranges(_schema, _permit))
    { }
};

std::vector<mutation_bounds> clustering_combined::create_almost_disjoint_ranges(simple_schema& s, reader_permit permit) {
    auto pk = s.make_pkey();
    std::vector<mutation_bounds> mbs;
    for (int i = 0; i < 150; i += 30) {
        auto m = mutation(s.schema(), pk);
        for (int j = 0; j < 32; ++j) {
            m.apply(s.make_row(permit, s.make_ckey(i + j), "value"));
        }
        mbs.push_back(mutation_bounds{std::move(m),
                position_in_partition::for_key(s.make_ckey(i)),
                position_in_partition::for_key(s.make_ckey(i + 31))});
    }
    return mbs;
}

future<size_t> clustering_combined::consume_all(flat_mutation_reader_v2 mr) const
{
    return do_with(mutation_fragment_v1_stream(std::move(mr)), size_t(0), [] (auto& mr, size_t& num_mfs) {
        perf_tests::start_measuring_time();
        return mr.consume_pausable([&num_mfs] (mutation_fragment mf) {
            ++num_mfs;
            perf_tests::do_not_optimize(mf);
            return stop_iteration::no;
        }).then([&num_mfs] {
            perf_tests::stop_measuring_time();
            return num_mfs;
        }).finally([&mr] {
            return mr.close();
        });
    });
}

PERF_TEST_F(clustering_combined, ranges_generic)
{
    return consume_all(make_combined_reader(schema().schema(), permit(),
        boost::copy_range<std::vector<flat_mutation_reader_v2>>(
            almost_disjoint_clustering_ranges()
            | boost::adaptors::transformed([this] (auto&& mb) {
                return make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), std::move(mb.m));
            })
        )
    ));
}

PERF_TEST_F(clustering_combined, ranges_specialized)
{
    auto rbs = boost::copy_range<std::vector<reader_bounds>>(
        almost_disjoint_clustering_ranges() | boost::adaptors::transformed([this] (auto&& mb) {
            return reader_bounds{
                make_flat_mutation_reader_from_mutations_v2(schema().schema(), permit(), std::move(mb.m)),
                std::move(mb.lower), std::move(mb.upper)};
        }));
    auto q = std::make_unique<simple_position_reader_queue>(*schema().schema(), std::move(rbs));
    return consume_all(make_clustering_combined_reader(
        schema().schema(), permit(), streamed_mutation::forwarding::no, std::move(q)));
}

class memtable {
    static constexpr size_t partition_count = 1000;
    static constexpr size_t row_count = 50;
    mutable simple_schema _schema;
    perf::reader_concurrency_semaphore_wrapper _semaphore;
    reader_permit _permit;
    std::vector<dht::decorated_key> _dkeys;
    lw_shared_ptr<replica::memtable> _single_row;
    lw_shared_ptr<replica::memtable> _multi_row;
    lw_shared_ptr<replica::memtable> _large_partition;
    std::optional<dht::partition_range> _partition_range;
public:
    memtable()
        : _semaphore(__FILE__)
        , _permit(_semaphore.make_permit())
        , _dkeys(_schema.make_pkeys(partition_count))
        , _single_row(make_lw_shared<replica::memtable>(_schema.schema()))
        , _multi_row(make_lw_shared<replica::memtable>(_schema.schema()))
        , _large_partition(make_lw_shared<replica::memtable>(_schema.schema()))
    {
        boost::for_each(
            _dkeys
            | boost::adaptors::transformed([&] (auto& dkey) {
                auto m = mutation(_schema.schema(), dkey);
                m.apply(_schema.make_row(_permit, _schema.make_ckey(0), "value"));
                return m;
            }),
            [&] (mutation m) {
                _single_row->apply(m);
            }
        );
        boost::for_each(
            _dkeys
            | boost::adaptors::transformed([&] (auto& dkey) {
                auto m = mutation(_schema.schema(), dkey);
                for (auto i = 0u; i < row_count; i++) {
                    m.apply(_schema.make_row(_permit, _schema.make_ckey(i), "value"));
                }
                return m;
            }),
            [&] (mutation m) {
                _multi_row->apply(m);
            }
        );
        boost::for_each(
            _dkeys
            | boost::adaptors::transformed([&] (auto& dkey) {
                auto m = mutation(_schema.schema(), dkey);
                // Make sure the partition fills buffers in flat mutation reader multiple times
                for (auto i = 0u; i < 8 * 1024; i++) {
                    m.apply(_schema.make_row(_permit, _schema.make_ckey(i), "value"));
                }
                return m;
            }),
            [&] (mutation m) {
                _large_partition->apply(m);
            }
        );
    }
protected:
    schema_ptr schema() const { return _schema.schema(); }
    reader_permit permit() const { return _permit; }

    replica::memtable& single_row_mt() { return *_single_row; }
    replica::memtable& multi_row_mt() { return *_multi_row; }
    replica::memtable& large_partition_mt() { return *_large_partition; }

    const dht::partition_range& single_partition_range() {
        auto& dk = _dkeys[_dkeys.size() / 2];
        _partition_range.emplace(dht::partition_range::make_singular(dk));
        return *_partition_range;
    }

    const dht::partition_range& multi_partition_range(size_t n) {
        auto start_idx = (_dkeys.size() - n) / 2;
        auto& start_dk = _dkeys[start_idx];
        auto& end_dk = _dkeys[start_idx + n];
        _partition_range.emplace(dht::partition_range::make(dht::ring_position(start_dk),
                                                            {dht::ring_position(end_dk), false}));
        return *_partition_range;
    }

    future<> consume_all(flat_mutation_reader_v2 mr) const {
        return with_closeable(std::move(mr), [] (auto& mr) {
            return mr.consume_pausable([] (mutation_fragment_v2 mf) {
                perf_tests::do_not_optimize(mf);
                return stop_iteration::no;
            });
        });
    }
};

PERF_TEST_F(memtable, one_partition_one_row)
{
    return consume_all(single_row_mt().make_flat_reader(schema(), permit(), single_partition_range()));
}

PERF_TEST_F(memtable, one_partition_many_rows)
{
    return consume_all(multi_row_mt().make_flat_reader(schema(), permit(), single_partition_range()));
}

PERF_TEST_F(memtable, one_large_partition)
{
    return consume_all(large_partition_mt().make_flat_reader(schema(), permit(), single_partition_range()));
}

PERF_TEST_F(memtable, many_partitions_one_row)
{
    return consume_all(single_row_mt().make_flat_reader(schema(), permit(), multi_partition_range(25)));
}

PERF_TEST_F(memtable, many_partitions_many_rows)
{
    return consume_all(multi_row_mt().make_flat_reader(schema(), permit(), multi_partition_range(25)));
}

PERF_TEST_F(memtable, many_large_partitions)
{
    return consume_all(large_partition_mt().make_flat_reader(schema(), permit(), multi_partition_range(25)));
}

}
