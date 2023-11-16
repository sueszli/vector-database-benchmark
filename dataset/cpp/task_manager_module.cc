/*
 * Copyright (C) 2023-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <boost/range/algorithm/min_element.hpp>

#include "compaction/task_manager_module.hh"
#include "compaction/compaction_manager.hh"
#include "replica/database.hh"
#include "sstables/sstables.hh"
#include "sstables/sstable_directory.hh"
#include "utils/error_injection.hh"
#include "utils/pretty_printers.hh"

using namespace std::chrono_literals;

namespace replica {

// Helper structure for resharding.
//
// Describes the sstables (represented by their foreign_sstable_open_info) that are shared and
// need to be resharded. Each shard will keep one such descriptor, that contains the list of
// SSTables assigned to it, and their total size. The total size is used to make sure we are
// fairly balancing SSTables among shards.
struct reshard_shard_descriptor {
    sstables::sstable_directory::sstable_open_info_vector info_vec;
    uint64_t uncompressed_data_size = 0;

    bool total_size_smaller(const reshard_shard_descriptor& rhs) const {
        return uncompressed_data_size < rhs.uncompressed_data_size;
    }

    uint64_t size() const {
        return uncompressed_data_size;
    }
};

} // namespace replica

// Collects shared SSTables from all shards and sstables that require cleanup and returns a vector containing them all.
// This function assumes that the list of SSTables can be fairly big so it is careful to
// manipulate it in a do_for_each loop (which yields) instead of using standard accumulators.
future<sstables::sstable_directory::sstable_open_info_vector>
collect_all_shared_sstables(sharded<sstables::sstable_directory>& dir, sharded<replica::database>& db, sstring ks_name, sstring table_name, compaction::owned_ranges_ptr owned_ranges_ptr) {
    auto info_vec = sstables::sstable_directory::sstable_open_info_vector();

    // We want to make sure that each distributed object reshards about the same amount of data.
    // Each sharded object has its own shared SSTables. We can use a clever algorithm in which they
    // all distributely figure out which SSTables to exchange, but we'll keep it simple and move all
    // their foreign_sstable_open_info to a coordinator (the shard who called this function). We can
    // move in bulk and that's efficient. That shard can then distribute the work among all the
    // others who will reshard.
    auto coordinator = this_shard_id();
    // We will first move all of the foreign open info to temporary storage so that we can sort
    // them. We want to distribute bigger sstables first.
    const auto* sorted_owned_ranges_ptr = owned_ranges_ptr.get();
    co_await dir.invoke_on_all([&] (sstables::sstable_directory& d) -> future<> {
        auto shared_sstables = d.retrieve_shared_sstables();
        sstables::sstable_directory::sstable_open_info_vector need_cleanup;
        if (sorted_owned_ranges_ptr) {
            co_await d.filter_sstables([&] (sstables::shared_sstable sst) -> future<bool> {
                if (needs_cleanup(sst, *sorted_owned_ranges_ptr)) {
                    need_cleanup.push_back(co_await sst->get_open_info());
                    co_return false;
                }
                co_return true;
            });
        }
        if (shared_sstables.empty() && need_cleanup.empty()) {
            co_return;
        }
        co_await smp::submit_to(coordinator, [&] () -> future<> {
            info_vec.reserve(info_vec.size() + shared_sstables.size() + need_cleanup.size());
            for (auto& info : shared_sstables) {
                info_vec.emplace_back(std::move(info));
                co_await coroutine::maybe_yield();
            }
            for (auto& info : need_cleanup) {
                info_vec.emplace_back(std::move(info));
                co_await coroutine::maybe_yield();
            }
        });
    });

    co_return info_vec;
}

// Given a vector of shared sstables to be resharded, distribute it among all shards.
// The vector is first sorted to make sure that we are moving the biggest SSTables first.
//
// Returns a reshard_shard_descriptor per shard indicating the work that each shard has to do.
future<std::vector<replica::reshard_shard_descriptor>>
distribute_reshard_jobs(sstables::sstable_directory::sstable_open_info_vector source) {
    auto destinations = std::vector<replica::reshard_shard_descriptor>(smp::count);

    std::sort(source.begin(), source.end(), [] (const sstables::foreign_sstable_open_info& a, const sstables::foreign_sstable_open_info& b) {
        // Sort on descending SSTable sizes.
        return a.uncompressed_data_size > b.uncompressed_data_size;
    });

    for (auto& info : source) {
        // Choose the stable shard owner with the smallest amount of accumulated work.
        // Note that for sstables that need cleanup via resharding, owners may contain
        // a single shard.
        auto shard_it = boost::min_element(info.owners, [&] (const shard_id& lhs, const shard_id& rhs) {
            return destinations[lhs].total_size_smaller(destinations[rhs]);
        });
        auto& dest = destinations[*shard_it];
        dest.uncompressed_data_size += info.uncompressed_data_size;
        dest.info_vec.push_back(std::move(info));
        co_await coroutine::maybe_yield();
    }

    co_return destinations;
}

// reshards a collection of SSTables.
//
// A reference to the compaction manager must be passed so we can register with it. Knowing
// which table is being processed is a requirement of the compaction manager, so this must be
// passed too.
//
// We will reshard max_sstables_per_job at once.
//
// A creator function must be passed that will create an SSTable object in the correct shard,
// and an I/O priority must be specified.
future<> reshard(sstables::sstable_directory& dir, sstables::sstable_directory::sstable_open_info_vector shared_info, replica::table& table,
                           sstables::compaction_sstable_creator_fn creator, compaction::owned_ranges_ptr owned_ranges_ptr, std::optional<tasks::task_info> parent_info)
{
    // Resharding doesn't like empty sstable sets, so bail early. There is nothing
    // to reshard in this shard.
    if (shared_info.empty()) {
        co_return;
    }

    // We want to reshard many SSTables at a time for efficiency. However if we have too many we may
    // be risking OOM.
    auto max_sstables_per_job = table.schema()->max_compaction_threshold();
    auto num_jobs = (shared_info.size() + max_sstables_per_job - 1) / max_sstables_per_job;
    auto sstables_per_job = shared_info.size() / num_jobs;

    std::vector<std::vector<sstables::shared_sstable>> buckets;
    buckets.reserve(num_jobs);
    buckets.emplace_back();
    co_await coroutine::parallel_for_each(shared_info, [&] (sstables::foreign_sstable_open_info& info) -> future<> {
        auto sst = co_await dir.load_foreign_sstable(info);
        // Last bucket gets leftover SSTables
        if ((buckets.back().size() >= sstables_per_job) && (buckets.size() < num_jobs)) {
            buckets.emplace_back();
        }
        buckets.back().push_back(std::move(sst));
    });
    // There is a semaphore inside the compaction manager in run_resharding_jobs. So we
    // parallel_for_each so the statistics about pending jobs are updated to reflect all
    // jobs. But only one will run in parallel at a time
    auto& t = table.as_table_state();
    co_await coroutine::parallel_for_each(buckets, [&] (std::vector<sstables::shared_sstable>& sstlist) mutable {
        return table.get_compaction_manager().run_custom_job(table.as_table_state(), sstables::compaction_type::Reshard, "Reshard compaction", [&] (sstables::compaction_data& info, sstables::compaction_progress_monitor& progress_monitor) -> future<> {
            auto erm = table.get_effective_replication_map(); // keep alive around compaction.

            sstables::compaction_descriptor desc(sstlist);
            desc.options = sstables::compaction_type_options::make_reshard();
            desc.creator = creator;
            desc.sharder = &erm->get_sharder(*table.schema());
            desc.owned_ranges = owned_ranges_ptr;

            auto result = co_await sstables::compact_sstables(std::move(desc), info, t, progress_monitor);
            // input sstables are moved, to guarantee their resources are released once we're done
            // resharding them.
            co_await when_all_succeed(dir.collect_output_unshared_sstables(std::move(result.new_sstables), sstables::sstable_directory::can_be_remote::yes), dir.remove_sstables(std::move(sstlist))).discard_result();
        }, parent_info, throw_if_stopping::no);
    });
}

namespace compaction {

struct table_tasks_info {
    tasks::task_manager::task_ptr task;
    table_info ti;

    table_tasks_info(tasks::task_manager::task_ptr t, table_info info)
        : task(t)
        , ti(info)
    {}
};

future<> run_on_table(sstring op, replica::database& db, std::string keyspace, table_info ti, std::function<future<> (replica::table&)> func) {
    std::exception_ptr ex;
    tasks::tmlogger.debug("Starting {} on {}.{}", op, keyspace, ti.name);
    try {
        co_await func(db.find_column_family(ti.id));
    } catch (const replica::no_such_column_family& e) {
        tasks::tmlogger.warn("Skipping {} of {}.{}: {}", op, keyspace, ti.name, e.what());
    } catch (...) {
        ex = std::current_exception();
        tasks::tmlogger.error("Failed {} of {}.{}: {}", op, keyspace, ti.name, ex);
    }
    if (ex) {
        co_await coroutine::return_exception_ptr(std::move(ex));
    }
}

// Run on all tables, skipping dropped tables
future<> run_on_existing_tables(sstring op, replica::database& db, std::string keyspace, const std::vector<table_info> local_tables, std::function<future<> (replica::table&)> func) {
    for (const auto& ti : local_tables) {
        co_await run_on_table(op, db, keyspace, ti, func);
    }
}

future<> wait_for_your_turn(seastar::condition_variable& cv, tasks::task_manager::task_ptr& current_task, tasks::task_id id) {
    co_await cv.wait([&] {
        return current_task && current_task->id() == id;
    });
}

future<> run_table_tasks(replica::database& db, std::vector<table_tasks_info> table_tasks, seastar::condition_variable& cv, tasks::task_manager::task_ptr& current_task, bool sort) {
    std::exception_ptr ex;

    // While compaction is run on one table, the size of tables may significantly change.
    // Thus, they are sorted before each invidual compaction and the smallest table is chosen.
    while (!table_tasks.empty()) {
        try {
            if (sort) {
                // Major compact smaller tables first, to increase chances of success if low on space.
                // Tables will be kept in descending order.
                std::ranges::sort(table_tasks, std::greater<>(), [&] (const table_tasks_info& tti) {
                    try {
                        return db.find_column_family(tti.ti.id).get_stats().live_disk_space_used;
                    } catch (const replica::no_such_column_family& e) {
                        return int64_t(-1);
                    }
                });
            }
            // Task responsible for the smallest table.
            current_task = table_tasks.back().task;
            table_tasks.pop_back();
            cv.broadcast();
            co_await current_task->done();
        } catch (...) {
            ex = std::current_exception();
            current_task = nullptr;
            cv.broken(ex);
            break;
        }
    }

    if (ex) {
        // Wait for all tasks even on failure.
        for (auto& tti: table_tasks) {
            co_await tti.task->done();
        }
        co_await coroutine::return_exception_ptr(std::move(ex));
    }
}

future<tasks::task_manager::task::progress> compaction_task_impl::get_progress(const sstables::compaction_data& cdata, const sstables::compaction_progress_monitor& progress_monitor) const {
    if (cdata.compaction_size == 0) {
        co_return get_binary_progress();
    }

    co_return tasks::task_manager::task::progress{
        .completed = is_done() ? cdata.compaction_size : progress_monitor.get_progress(),   // Consider tasks which skip all files.
        .total = cdata.compaction_size
    };
}

tasks::is_abortable compaction_task_impl::is_abortable() const noexcept {
    return tasks::is_abortable{!_parent_id};
}

future<> major_keyspace_compaction_task_impl::run() {
    co_await utils::get_local_injector().inject_with_handler("compaction_major_keyspace_compaction_task_impl_run",
            [] (auto& handler) { return handler.wait_for_message(db::timeout_clock::now() + 10s); });

    co_await _db.invoke_on_all([&] (replica::database& db) -> future<> {
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& module = db.get_compaction_manager().get_task_manager_module();
        auto task = co_await module.make_and_start_task<shard_major_keyspace_compaction_task_impl>(parent_info, _status.keyspace, _status.id, db, _table_infos);
        co_await task->done();
    });
}

future<> shard_major_keyspace_compaction_task_impl::run() {
    seastar::condition_variable cv;
    tasks::task_manager::task_ptr current_task;
    tasks::task_info parent_info{_status.id, _status.shard};
    std::vector<table_tasks_info> table_tasks;
    for (auto& ti : _local_tables) {
        table_tasks.emplace_back(co_await _module->make_and_start_task<table_major_keyspace_compaction_task_impl>(parent_info, _status.keyspace, ti.name, _status.id, _db, ti, cv, current_task), ti);
    }

    co_await run_table_tasks(_db, std::move(table_tasks), cv, current_task, true);
}

future<> table_major_keyspace_compaction_task_impl::run() {
    co_await wait_for_your_turn(_cv, _current_task, _status.id);
    tasks::task_info info{_status.id, _status.shard};
    co_await run_on_table("force_keyspace_compaction", _db, _status.keyspace, _ti, [info] (replica::table& t) {
        return t.compact_all_sstables(info);
    });
}

future<> cleanup_keyspace_compaction_task_impl::run() {
    co_await _db.invoke_on_all([&] (replica::database& db) -> future<> {
        auto& module = db.get_compaction_manager().get_task_manager_module();
        auto task = co_await module.make_and_start_task<shard_cleanup_keyspace_compaction_task_impl>({_status.id, _status.shard}, _status.keyspace, _status.id, db, _table_infos);
        co_await task->done();
    });
}

future<> shard_cleanup_keyspace_compaction_task_impl::run() {
    seastar::condition_variable cv;
    tasks::task_manager::task_ptr current_task;
    tasks::task_info parent_info{_status.id, _status.shard};
    std::vector<table_tasks_info> table_tasks;
    for (auto& ti : _local_tables) {
        table_tasks.emplace_back(co_await _module->make_and_start_task<table_cleanup_keyspace_compaction_task_impl>(parent_info, _status.keyspace, ti.name, _status.id, _db, ti, cv, current_task), ti);
    }

    co_await run_table_tasks(_db, std::move(table_tasks), cv, current_task, true);
}

future<> table_cleanup_keyspace_compaction_task_impl::run() {
    co_await wait_for_your_turn(_cv, _current_task, _status.id);
    auto owned_ranges_ptr = compaction::make_owned_ranges_ptr(_db.get_keyspace_local_ranges(_status.keyspace));
    co_await run_on_table("force_keyspace_cleanup", _db, _status.keyspace, _ti, [&] (replica::table& t) {
        return t.perform_cleanup_compaction(owned_ranges_ptr, tasks::task_info{_status.id, _status.shard});
    });
}

future<> offstrategy_keyspace_compaction_task_impl::run() {
    _needed = co_await _db.map_reduce0([&] (replica::database& db) -> future<bool> {
        bool needed = false;
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& module = db.get_compaction_manager().get_task_manager_module();
        auto task = co_await module.make_and_start_task<shard_offstrategy_keyspace_compaction_task_impl>(parent_info, _status.keyspace, _status.id, db, _table_infos, needed);
        co_await task->done();
        co_return needed;
    }, false, std::plus<bool>());
}

future<> shard_offstrategy_keyspace_compaction_task_impl::run() {
    seastar::condition_variable cv;
    tasks::task_manager::task_ptr current_task;
    tasks::task_info parent_info{_status.id, _status.shard};
    std::vector<table_tasks_info> table_tasks;
    for (auto& ti : _table_infos) {
        table_tasks.emplace_back(co_await _module->make_and_start_task<table_offstrategy_keyspace_compaction_task_impl>(parent_info, _status.keyspace, ti.name, _status.id, _db, ti, cv, current_task, _needed), ti);
    }

    co_await run_table_tasks(_db, std::move(table_tasks), cv, current_task, false);
}

future<> table_offstrategy_keyspace_compaction_task_impl::run() {
    co_await wait_for_your_turn(_cv, _current_task, _status.id);
    tasks::task_info info{_status.id, _status.shard};
    co_await run_on_table("perform_keyspace_offstrategy_compaction", _db, _status.keyspace, _ti, [this, info] (replica::table& t) -> future<> {
        _needed |= co_await t.perform_offstrategy_compaction(info);
    });
}

future<> upgrade_sstables_compaction_task_impl::run() {
    co_await _db.invoke_on_all([&] (replica::database& db) -> future<> {
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& compaction_module = db.get_compaction_manager().get_task_manager_module();
        auto task = co_await compaction_module.make_and_start_task<shard_upgrade_sstables_compaction_task_impl>(parent_info, _status.keyspace, _status.id, db, _table_infos, _exclude_current_version);
        co_await task->done();
    });
}

future<> shard_upgrade_sstables_compaction_task_impl::run() {
    seastar::condition_variable cv;
    tasks::task_manager::task_ptr current_task;
    tasks::task_info parent_info{_status.id, _status.shard};
    std::vector<table_tasks_info> table_tasks;
    for (auto& ti : _table_infos) {
        table_tasks.emplace_back(co_await _module->make_and_start_task<table_upgrade_sstables_compaction_task_impl>(parent_info, _status.keyspace, ti.name, _status.id, _db, ti, cv, current_task, _exclude_current_version), ti);
    }

    co_await run_table_tasks(_db, std::move(table_tasks), cv, current_task, false);
}

future<> table_upgrade_sstables_compaction_task_impl::run() {
    co_await wait_for_your_turn(_cv, _current_task, _status.id);
    auto owned_ranges_ptr = compaction::make_owned_ranges_ptr(_db.get_keyspace_local_ranges(_status.keyspace));
    tasks::task_info info{_status.id, _status.shard};
    co_await run_on_table("upgrade_sstables", _db, _status.keyspace, _ti, [&] (replica::table& t) -> future<> {
        return t.parallel_foreach_table_state([&] (compaction::table_state& ts) -> future<> {
            return t.get_compaction_manager().perform_sstable_upgrade(owned_ranges_ptr, ts, _exclude_current_version, info);
        });
    });
}

future<> scrub_sstables_compaction_task_impl::run() {
    _stats = co_await _db.map_reduce0([&] (replica::database& db) -> future<sstables::compaction_stats> {
        sstables::compaction_stats stats;
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& compaction_module = db.get_compaction_manager().get_task_manager_module();
        auto task = co_await compaction_module.make_and_start_task<shard_scrub_sstables_compaction_task_impl>(parent_info, _status.keyspace, _status.id, db, _column_families, _opts, stats);
        co_await task->done();
        co_return stats;
    }, sstables::compaction_stats{}, std::plus<sstables::compaction_stats>());
}

future<> shard_scrub_sstables_compaction_task_impl::run() {
    _stats = co_await map_reduce(_column_families, [&] (sstring cfname) -> future<sstables::compaction_stats> {
        sstables::compaction_stats stats{};
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& compaction_module = _db.get_compaction_manager().get_task_manager_module();
        auto task = co_await compaction_module.make_and_start_task<table_scrub_sstables_compaction_task_impl>(parent_info, _status.keyspace, cfname, _status.id, _db, _opts, stats);
        co_await task->done();
        co_return stats;
    }, sstables::compaction_stats{}, std::plus<sstables::compaction_stats>());
}

future<> table_scrub_sstables_compaction_task_impl::run() {
    auto& cm = _db.get_compaction_manager();
    auto& cf = _db.find_column_family(_status.keyspace, _status.table);
    tasks::task_info info{_status.id, _status.shard};
    co_await cf.parallel_foreach_table_state([&] (compaction::table_state& ts) mutable -> future<> {
        auto r = co_await cm.perform_sstable_scrub(ts, _opts, info);
        _stats += r.value_or(sstables::compaction_stats{});
    });
}

future<> table_reshaping_compaction_task_impl::run() {
    auto start = std::chrono::steady_clock::now();
    auto total_size = co_await _dir.map_reduce0([&] (sstables::sstable_directory& d) -> future<uint64_t> {
        uint64_t total_shard_size;
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& compaction_module = _db.local().get_compaction_manager().get_task_manager_module();
        auto task = co_await compaction_module.make_and_start_task<shard_reshaping_compaction_task_impl>(parent_info, _status.keyspace, _status.table, _status.id, d, _db, _mode, _creator, _filter, total_shard_size);
        co_await task->done();
        co_return total_shard_size;
    }, uint64_t(0), std::plus<uint64_t>());

    if (total_size > 0) {
        auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::steady_clock::now() - start);
        dblog.info("Reshaped {} in {:.2f} seconds, {}", utils::pretty_printed_data_size(total_size), duration.count(), utils::pretty_printed_throughput(total_size, duration));
    }
}

future<> shard_reshaping_compaction_task_impl::run() {
    auto& table = _db.local().find_column_family(_status.keyspace, _status.table);
    uint64_t reshaped_size = 0;
    tasks::task_info info{_status.id, _status.shard};

    while (true) {
        auto reshape_candidates = boost::copy_range<std::vector<sstables::shared_sstable>>(_dir.get_unshared_local_sstables()
                | boost::adaptors::filtered([&filter = _filter] (const auto& sst) {
            return filter(sst);
        }));
        auto desc = table.get_compaction_strategy().get_reshaping_job(std::move(reshape_candidates), table.schema(), _mode);
        if (desc.sstables.empty()) {
            break;
        }

        if (!reshaped_size) {
            dblog.info("Table {}.{} with compaction strategy {} found SSTables that need reshape. Starting reshape process", table.schema()->ks_name(), table.schema()->cf_name(), table.get_compaction_strategy().name());
        }

        std::vector<sstables::shared_sstable> sstlist;
        for (auto& sst : desc.sstables) {
            reshaped_size += sst->data_size();
            sstlist.push_back(sst);
        }

        desc.creator = _creator;

        std::exception_ptr ex;
        try {
            co_await table.get_compaction_manager().run_custom_job(table.as_table_state(), sstables::compaction_type::Reshape, "Reshape compaction", [&dir = _dir, &table, sstlist = std::move(sstlist), desc = std::move(desc)] (sstables::compaction_data& info, sstables::compaction_progress_monitor& progress_monitor) mutable -> future<> {
                sstables::compaction_result result = co_await sstables::compact_sstables(std::move(desc), info, table.as_table_state(), progress_monitor);
                co_await dir.remove_unshared_sstables(std::move(sstlist));
                co_await dir.collect_output_unshared_sstables(std::move(result.new_sstables), sstables::sstable_directory::can_be_remote::no);
            }, info, throw_if_stopping::yes);
        } catch (...) {
            ex = std::current_exception();
        }

        if (ex != nullptr) {
              try {
                std::rethrow_exception(std::move(ex));
              } catch (sstables::compaction_stopped_exception& e) {
                  dblog.info("Table {}.{} with compaction strategy {} had reshape successfully aborted.", table.schema()->ks_name(), table.schema()->cf_name(), table.get_compaction_strategy().name());
                  break;
              } catch (...) {
                  dblog.info("Reshape failed for Table {}.{} with compaction strategy {} due to {}", table.schema()->ks_name(), table.schema()->cf_name(), table.get_compaction_strategy().name(), std::current_exception());
                  break;
              }
        }

        co_await coroutine::maybe_yield();
    }

    _total_shard_size = reshaped_size;
}

future<> table_resharding_compaction_task_impl::run() {
    auto all_jobs = co_await collect_all_shared_sstables(_dir, _db, _status.keyspace, _status.table, _owned_ranges_ptr);
    auto destinations = co_await distribute_reshard_jobs(std::move(all_jobs));

    uint64_t total_size = boost::accumulate(destinations | boost::adaptors::transformed(std::mem_fn(&replica::reshard_shard_descriptor::size)), uint64_t(0));
    if (total_size == 0) {
        co_return;
    }

    auto start = std::chrono::steady_clock::now();
    dblog.info("Resharding {} for {}.{}", utils::pretty_printed_data_size(total_size), _status.keyspace, _status.table);

    co_await _db.invoke_on_all(coroutine::lambda([&] (replica::database& db) -> future<> {
        tasks::task_info parent_info{_status.id, _status.shard};
        auto& compaction_module = _db.local().get_compaction_manager().get_task_manager_module();
        // make shard-local copy of owned_ranges
        compaction::owned_ranges_ptr local_owned_ranges_ptr;
        if (_owned_ranges_ptr) {
            local_owned_ranges_ptr = make_lw_shared<const dht::token_range_vector>(*_owned_ranges_ptr);
        }
        auto task = co_await compaction_module.make_and_start_task<shard_resharding_compaction_task_impl>(parent_info, _status.keyspace, _status.table, _status.id, _dir, db, _creator, std::move(local_owned_ranges_ptr), destinations);
        co_await task->done();
    }));

    auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::steady_clock::now() - start);
    dblog.info("Resharded {} for {}.{} in {:.2f} seconds, {}", utils::pretty_printed_data_size(total_size), _status.keyspace, _status.table, duration.count(), utils::pretty_printed_throughput(total_size, duration));
}

future<> shard_resharding_compaction_task_impl::run() {
    auto& table = _db.find_column_family(_status.keyspace, _status.table);
    auto info_vec = std::move(_destinations[this_shard_id()].info_vec);
    tasks::task_info info{_status.id, _status.shard};
    co_await reshard(_dir.local(), std::move(info_vec), table, _creator, std::move(_local_owned_ranges_ptr), info);
    co_await _dir.local().move_foreign_sstables(_dir);
}

}
