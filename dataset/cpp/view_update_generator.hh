/*
 * Copyright (C) 2018-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include "sstables/shared_sstable.hh"
#include "db/timeout_clock.hh"
#include "utils/chunked_vector.hh"

#include <seastar/core/sharded.hh>
#include <seastar/core/metrics_registration.hh>
#include <seastar/core/abort_source.hh>
#include <seastar/core/condition-variable.hh>
#include <seastar/core/semaphore.hh>

using namespace seastar;

struct frozen_mutation_and_schema;

namespace dht {
class token;
}

namespace tracing {
class trace_state_ptr;
}

namespace replica {
class database;
class table;
struct cf_stats;
}

namespace service {
class storage_proxy;
struct allow_hints_tag;
using allow_hints = bool_class<allow_hints_tag>;
}

namespace db::view {

class stats;
struct wait_for_all_updates_tag {};
using wait_for_all_updates = bool_class<wait_for_all_updates_tag>;

class view_update_generator : public async_sharded_service<view_update_generator> {
public:
    static constexpr size_t registration_queue_size = 100;

private:
    replica::database& _db;
    sharded<service::storage_proxy>& _proxy;
    seastar::abort_source _as;
    future<> _started = make_ready_future<>();
    seastar::condition_variable _pending_sstables;
    named_semaphore _registration_sem{registration_queue_size, named_semaphore_exception_factory{"view update generator"}};
    std::unordered_map<lw_shared_ptr<replica::table>, std::vector<sstables::shared_sstable>> _sstables_with_tables;
    std::unordered_map<lw_shared_ptr<replica::table>, std::vector<sstables::shared_sstable>> _sstables_to_move;
    metrics::metric_groups _metrics;
    class progress_tracker;
    std::unique_ptr<progress_tracker> _progress_tracker;
    optimized_optional<abort_source::subscription> _early_abort_subscription;
    void do_abort() noexcept;
public:
    view_update_generator(replica::database& db, sharded<service::storage_proxy>& proxy, abort_source& as);
    ~view_update_generator();

    future<> start();
    future<> stop();
    future<> register_staging_sstable(sstables::shared_sstable sst, lw_shared_ptr<replica::table> table);

    replica::database& get_db() noexcept { return _db; }

    future<> mutate_MV(
            dht::token base_token,
            utils::chunked_vector<frozen_mutation_and_schema> view_updates,
            db::view::stats& stats,
            replica::cf_stats& cf_stats,
            tracing::trace_state_ptr tr_state,
            db::timeout_semaphore_units pending_view_updates,
            service::allow_hints allow_hints,
            wait_for_all_updates wait_for_all);

    ssize_t available_register_units() const { return _registration_sem.available_units(); }
    size_t queued_batches_count() const { return _sstables_with_tables.size(); }
private:
    bool should_throttle() const;
    void setup_metrics();
    void discover_staging_sstables();
};

}
