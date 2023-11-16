/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include "readers/multishard.hh"
#include <seastar/core/gate.hh>

class test_reader_lifecycle_policy
        : public reader_lifecycle_policy_v2
        , public enable_shared_from_this<test_reader_lifecycle_policy> {
    using factory_function = std::function<flat_mutation_reader_v2(
            schema_ptr,
            reader_permit,
            const dht::partition_range&,
            const query::partition_slice&,
            tracing::trace_state_ptr,
            mutation_reader::forwarding)>;

    struct reader_context {
        std::optional<reader_concurrency_semaphore> semaphore;
        lw_shared_ptr<const dht::partition_range> range;
        std::optional<const query::partition_slice> slice;

        reader_context() = default;
        reader_context(dht::partition_range range, query::partition_slice slice)
            : range(make_lw_shared<const dht::partition_range>(std::move(range))), slice(std::move(slice)) {
        }
    };

    factory_function _factory_function;
    std::vector<foreign_ptr<std::unique_ptr<reader_context>>> _contexts;
    std::vector<future<>> _destroy_futures;
    bool _evict_paused_readers = false;

public:
    explicit test_reader_lifecycle_policy(factory_function f, bool evict_paused_readers = false)
        : _factory_function(std::move(f))
        , _contexts(smp::count)
        , _evict_paused_readers(evict_paused_readers) {
    }
    virtual flat_mutation_reader_v2 create_reader(
            schema_ptr schema,
            reader_permit permit,
            const dht::partition_range& range,
            const query::partition_slice& slice,
            tracing::trace_state_ptr trace_state,
            mutation_reader::forwarding fwd_mr) override {
        const auto shard = this_shard_id();
        if (_contexts[shard]) {
            _contexts[shard]->range = make_lw_shared<const dht::partition_range>(range);
            _contexts[shard]->slice.emplace(slice);
        } else {
            _contexts[shard] = make_foreign(std::make_unique<reader_context>(range, slice));
        }
        return _factory_function(std::move(schema), std::move(permit), *_contexts[shard]->range, *_contexts[shard]->slice, std::move(trace_state), fwd_mr);
    }
    virtual const dht::partition_range* get_read_range() const override {
        const auto shard = this_shard_id();
        assert(_contexts[shard]);
        return _contexts[shard]->range.get();
    }
    void update_read_range(lw_shared_ptr<const dht::partition_range> range) override {
        const auto shard = this_shard_id();
        assert(_contexts[shard]);
        _contexts[shard]->range = std::move(range);
    }
    virtual future<> destroy_reader(stopped_reader reader) noexcept override {
        auto& ctx = _contexts[this_shard_id()];
        auto reader_opt = ctx->semaphore->unregister_inactive_read(std::move(reader.handle));
        auto ret = reader_opt ? reader_opt->close() : make_ready_future<>();
        return ret.finally([&ctx] {
            return ctx->semaphore->stop().finally([&ctx] {
                ctx.release();
            });
        });
    }
    virtual reader_concurrency_semaphore& semaphore() override {
        const auto shard = this_shard_id();
        if (!_contexts[shard]) {
            _contexts[shard] = make_foreign(std::make_unique<reader_context>());
        } else if (_contexts[shard]->semaphore) {
            return *_contexts[shard]->semaphore;
        }
        // To support multiple reader life-cycle instances alive at the same
        // time, incorporate `this` into the name, to make their names unique.
        auto name = format("tests::reader_lifecycle_policy@{}@shard_id={}", fmt::ptr(this), shard);
        if (_evict_paused_readers) {
            // Create with no memory, so all inactive reads are immediately evicted.
            _contexts[shard]->semaphore.emplace(reader_concurrency_semaphore::for_tests{}, std::move(name), 1, 0);
        } else {
            _contexts[shard]->semaphore.emplace(reader_concurrency_semaphore::no_limits{}, std::move(name));
        }
        return *_contexts[shard]->semaphore;
    }
    virtual future<reader_permit> obtain_reader_permit(schema_ptr schema, const char* const description, db::timeout_clock::time_point timeout, tracing::trace_state_ptr trace_ptr) override {
        return semaphore().obtain_permit(schema.get(), description, 128 * 1024, timeout, std::move(trace_ptr));
    }
};

