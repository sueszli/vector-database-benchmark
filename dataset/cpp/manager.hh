/*
 * Modified by ScyllaDB
 * Copyright (C) 2017-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

// Seastar features.
#include <seastar/core/abort_source.hh>
#include <seastar/core/gate.hh>
#include <seastar/core/lowres_clock.hh>
#include <seastar/core/sharded.hh>
#include <seastar/core/shared_mutex.hh>
#include <seastar/core/timer.hh>
#include <seastar/util/noncopyable_function.hh>

// Scylla includes.
#include "db/commitlog/commitlog.hh"
#include "db/hints/internal/common.hh"
#include "db/hints/internal/hint_storage.hh"
#include "db/hints/internal/hint_endpoint_manager.hh"
#include "db/hints/resource_manager.hh"
#include "db/hints/host_filter.hh"
#include "db/hints/sync_point.hh"
#include "locator/abstract_replication_strategy.hh"
#include "utils/loading_shared_values.hh"
#include "inet_address_vectors.hh"

// STD.
#include <chrono>
#include <span>
#include <unordered_map>

class fragmented_temporary_buffer;

namespace utils {
class directories;
} // namespace utils

namespace gms {
class gossiper;
} // namespace gms

namespace db::hints {

/// A helper class which tracks hints directory creation
/// and allows to perform hints directory initialization lazily.
class directory_initializer {
private:
    class impl;
    ::std::shared_ptr<impl> _impl;

    directory_initializer(::std::shared_ptr<impl> impl);

public:
    /// Creates an initializer that does nothing. Useful in tests.
    static directory_initializer make_dummy() noexcept {
        return {nullptr};
    }
    static future<directory_initializer> make(utils::directories& dirs, sstring hints_directory);

    future<> ensure_created_and_verified();
    future<> ensure_rebalanced();
};

class manager {
public:
    using endpoint_id = internal::endpoint_id;
private:
    using hint_stats = internal::hint_stats;
    using drain = internal::drain;

    friend class internal::hint_endpoint_manager;
    friend class internal::hint_sender;

    using hint_endpoint_manager = internal::hint_endpoint_manager;
    using node_to_hint_store_factory_type = internal::node_to_hint_store_factory_type;

    enum class state {
        started,        // Hinting is currently allowed (start() has completed).
        replay_allowed, // Replaying (sending) hints is allowed.
        draining_all,   // Accepting new hints is not allowed. All endpoint managers
                        // are being drained because the node is leaving the cluster.
        stopping        // Accepting new hints is not allowed. Stopping this manager
                        // is in progress (stop() has been called).
    };

    using state_set = enum_set<super_enum<state,
        state::started,
        state::replay_allowed,
        state::draining_all,
        state::stopping>>;

public:
    static inline const std::string FILENAME_PREFIX{"HintsLog" + commitlog::descriptor::SEPARATOR};
    // Non-const - can be modified with an error injection.
    static inline std::chrono::seconds hints_flush_period = std::chrono::seconds(10);
    static constexpr std::chrono::seconds HINT_FILE_WRITE_TIMEOUT = std::chrono::seconds(2);
private:
    static constexpr uint64_t MAX_SIZE_OF_HINTS_IN_PROGRESS = 10 * 1024 * 1024; // 10MB

private:
    state_set _state;
    const fs::path _hints_dir;
    dev_t _hints_dir_device_id = 0;

    node_to_hint_store_factory_type _store_factory;
    host_filter _host_filter;
    service::storage_proxy& _proxy;
    shared_ptr<gms::gossiper> _gossiper_anchor;
    int64_t _max_hint_window_us = 0;
    replica::database& _local_db;

    seastar::gate _draining_eps_gate; // gate used to control the progress of endpoint_managers stopping not in the context of manager::stop() call

    resource_manager& _resource_manager;

    std::unordered_map<endpoint_id, hint_endpoint_manager> _ep_managers;
    hint_stats _stats;
    seastar::metrics::metric_groups _metrics;
    std::unordered_set<endpoint_id> _eps_with_pending_hints;
    seastar::named_semaphore _drain_lock = {1, named_semaphore_exception_factory{"drain lock"}};

public:
    manager(service::storage_proxy& proxy, sstring hints_directory, host_filter filter,
            int64_t max_hint_window_ms, resource_manager& res_manager, sharded<replica::database>& db);
    
    manager(const manager&) = delete;
    manager& operator=(const manager&) = delete;

    manager(manager&&) = delete;
    manager& operator=(manager&&) = delete;
    
    ~manager() noexcept {
        assert(_ep_managers.empty());
    }

public:
    void register_metrics(const sstring& group_name);
    future<> start(shared_ptr<gms::gossiper> gossiper_ptr);
    future<> stop();
    bool store_hint(endpoint_id ep, schema_ptr s, lw_shared_ptr<const frozen_mutation> fm, tracing::trace_state_ptr tr_state) noexcept;

    /// \brief Changes the host_filter currently used, stopping and starting endpoint_managers relevant to the new host_filter.
    /// \param filter the new host_filter
    /// \return A future that resolves when the operation is complete.
    future<> change_host_filter(host_filter filter);

    const host_filter& get_host_filter() const noexcept {
        return _host_filter;
    }

    /// \brief Check if a hint may be generated to the give end point
    /// \param ep end point to check
    /// \return true if we should generate the hint to the given end point if it becomes unavailable
    bool can_hint_for(endpoint_id ep) const noexcept;

    /// \brief Check if there aren't too many in-flight hints
    ///
    /// This function checks if there are too many "in-flight" hints on the current shard - hints that are being stored
    /// and which storing is not complete yet. This is meant to stabilize the memory consumption of the hints storing path
    /// which is initialed from the storage_proxy WRITE flow. storage_proxy is going to check this condition and if it
    /// returns TRUE it won't attempt any new WRITEs thus eliminating the possibility of new hints generation. If new hints
    /// are not generated the amount of in-flight hints amount and thus the memory they are consuming is going to drop eventualy
    /// because the hints are going to be either stored or dropped. After that the things are going to get back to normal again.
    ///
    /// Note that we can't consider the disk usage consumption here because the disk usage is not promissed to drop down shortly
    /// because it requires the remote node to be UP.
    ///
    /// \param ep end point to check
    /// \return TRUE if we are allowed to generate hint to the given end point but there are too many in-flight hints
    bool too_many_in_flight_hints_for(endpoint_id ep) const noexcept;

    /// \brief Check if DC \param ep belongs to is "hintable"
    /// \param ep End point identificator
    /// \return TRUE if hints are allowed to be generated to \param ep.
    bool check_dc_for(endpoint_id ep) const noexcept;

    /// Execute a given functor while having an endpoint's file update mutex locked.
    ///
    /// The caller must ensure that the passed endpoint_id is valid, i.e. this manager instance
    /// really manages an endpoint manager corresponding to it. See @ref have_ep_manager.
    ///
    /// \param ep endpoint whose file update mutex should be locked
    /// \param func functor to be executed
    future<> with_file_update_mutex_for(endpoint_id ep, noncopyable_function<future<> ()> func);

    /// \brief Checks if hints are disabled for all endpoints
    /// \return TRUE if hints are disabled.
    bool is_disabled_for_all() const noexcept {
        return _host_filter.is_disabled_for_all();
    }

    /// \return Size of mutations of hints in-flight (to the disk) at the moment.
    uint64_t size_of_hints_in_progress() const noexcept {
        return _stats.size_of_hints_in_progress;
    }

    /// \brief Get the number of in-flight (to the disk) hints to a given end point.
    /// \param ep End point identificator
    /// \return Number of hints in-flight to \param ep.
    uint64_t hints_in_progress_for(endpoint_id ep) const noexcept {
        auto it = _ep_managers.find(ep);
        if (it == _ep_managers.end()) {
            return 0;
        }
        return it->second.hints_in_progress();
    }

    void add_ep_with_pending_hints(endpoint_id key) {
        _eps_with_pending_hints.insert(key);
    }

    void clear_eps_with_pending_hints() {
        _eps_with_pending_hints.clear();
        _eps_with_pending_hints.reserve(_ep_managers.size());
    }

    bool has_ep_with_pending_hints(endpoint_id key) const {
        return _eps_with_pending_hints.contains(key);
    }

    size_t ep_managers_size() const {
        return _ep_managers.size();
    }

    const fs::path& hints_dir() const {
        return _hints_dir;
    }

    dev_t hints_dir_device_id() const {
        return _hints_dir_device_id;
    }

    void allow_hints();
    void forbid_hints();
    void forbid_hints_for_eps_with_pending_hints();

    void allow_replaying() noexcept {
        _state.set(state::replay_allowed);
    }

    /// \brief Returns a set of replay positions for hint queues towards endpoints from the `target_eps`.
    sync_point::shard_rps calculate_current_sync_point(std::span<const endpoint_id> target_eps) const;

    /// \brief Waits until hint replay reach replay positions described in `rps`.
    future<> wait_for_sync_point(abort_source& as, const sync_point::shard_rps& rps);

private:
    future<> compute_hints_dir_device_id();

    node_to_hint_store_factory_type& store_factory() noexcept {
        return _store_factory;
    }

    service::storage_proxy& local_storage_proxy() const noexcept {
        return _proxy;
    }

    gms::gossiper& local_gossiper() const noexcept {
        return *_gossiper_anchor;
    }

    replica::database& local_db() noexcept {
        return _local_db;
    }

    hint_endpoint_manager& get_ep_manager(endpoint_id ep);

public:
    bool have_ep_manager(endpoint_id ep) const noexcept;

public:
    /// \brief Initiate the draining when we detect that the node has left the cluster.
    ///
    /// If the node that has left is the current node - drains all pending hints to all nodes.
    /// Otherwise drains hints to the node that has left.
    ///
    /// In both cases - removes the corresponding hints' directories after all hints have been drained and erases the
    /// corresponding hint_endpoint_manager objects.
    ///
    /// \param endpoint node that left the cluster
    future<> drain_for(endpoint_id endpoint) noexcept;

    void update_backlog(size_t backlog, size_t max_backlog);

private:
    bool stopping() const noexcept {
        return _state.contains(state::stopping);
    }

    void set_stopping() noexcept {
        _state.set(state::stopping);
    }

    bool started() const noexcept {
        return _state.contains(state::started);
    }

    void set_started() noexcept {
        _state.set(state::started);
    }

    bool replay_allowed() const noexcept {
        return _state.contains(state::replay_allowed);
    }

    void set_draining_all() noexcept {
        _state.set(state::draining_all);
    }

    bool draining_all() noexcept {
        return _state.contains(state::draining_all);
    }
};

} // namespace db::hints
