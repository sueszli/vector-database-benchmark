/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
#pragma once

#include <seastar/core/future.hh>
#include <seastar/core/sharded.hh>
#include <seastar/core/gate.hh>

#include "message/messaging_service_fwd.hh"
#include "raft/raft.hh"
#include "raft/server.hh"
#include "utils/recent_entries_map.hh"
#include "direct_failure_detector/failure_detector.hh"
#include "service/raft/group0_fwd.hh"

namespace gms {
class gossiper;
}

namespace db {
class system_keyspace;
}

namespace service {

class raft_rpc;
class raft_sys_table_storage;
using raft_ticker_type = seastar::timer<lowres_clock>;

struct raft_group_not_found: public raft::error {
    raft::group_id gid;
    raft_group_not_found(raft::group_id gid_arg)
            : raft::error(format("Raft group {} not found", gid_arg)), gid(gid_arg)
    {}
};

struct raft_destination_id_not_correct: public raft::error {
    raft_destination_id_not_correct(raft::server_id my_id, raft::server_id dst_id)
            : raft::error(format("Got message for server {}, but my id is {}", dst_id, my_id))
    {}
};

// An entry in the group registry
struct raft_server_for_group {
    raft::group_id gid;
    std::unique_ptr<raft::server> server;
    std::unique_ptr<raft_ticker_type> ticker;
    raft_rpc& rpc;
    raft_sys_table_storage& persistence;
    std::optional<seastar::future<>> aborted;
};

class direct_fd_pinger;
class direct_fd_proxy;
class gossiper_state_change_subscriber_proxy;

// This class is responsible for creating, storing and accessing raft servers.
// It also manages the raft rpc verbs initialization.
//
// `peering_sharded_service` inheritance is used to forward requests
// to the owning shard for a given raft group_id.
class raft_group_registry : public seastar::peering_sharded_service<raft_group_registry> {
private:
    // True if the feature is enabled
    bool _is_enabled;

    netw::messaging_service& _ms;
    gms::gossiper& _gossiper;
    // A proxy class representing subscription to on_change
    // events, and updating the address map on this events.
    shared_ptr<gossiper_state_change_subscriber_proxy> _gossiper_proxy;
    // Raft servers along with the corresponding timers to tick each instance.
    // Currently ticking every 100ms.
    std::unordered_map<raft::group_id, raft_server_for_group> _servers;
    // inet_address:es for remote raft servers known to us
    raft_address_map& _address_map;

    direct_failure_detector::failure_detector& _direct_fd;
    // Listens to notifications from direct failure detector.
    // Implements the `raft::failure_detector` interface. Used by all raft groups to check server liveness.
    seastar::shared_ptr<direct_fd_proxy> _direct_fd_proxy;
    // Direct failure detector listener subscription for `_direct_fd_proxy`.
    std::optional<direct_failure_detector::subscription> _direct_fd_subscription;

    void init_rpc_verbs();
    seastar::future<> uninit_rpc_verbs();
    seastar::future<> stop_servers() noexcept;

    raft_server_for_group& server_for_group(raft::group_id id);

    // Group 0 id, valid only on shard 0 after boot/upgrade is over
    std::optional<raft::group_id> _group0_id;

    // My Raft ID. Shared between different Raft groups.
    raft::server_id _my_id;

public:
    // `is_enabled` must be `true` iff the local RAFT feature is enabled.
    raft_group_registry(bool is_enabled, raft::server_id my_id, raft_address_map&,
            netw::messaging_service& ms, gms::gossiper& gs, direct_failure_detector::failure_detector& fd);
    ~raft_group_registry();

    // If is_enabled(),
    // Called manually at start on every shard.
    seastar::future<> start();
    // Called by sharded<>::stop()
    seastar::future<> stop();

    // Stop the server for the given group and remove it from the registry.
    // It differs from abort_server in that it waits for the server to stop
    // and removes it from the registry.
    seastar::future<> stop_server(raft::group_id gid, sstring reason);

    // Must not be called before `start`.
    const raft::server_id& get_my_raft_id();

    // Called by before stopping the database.
    // May be called multiple times.
    seastar::future<> drain_on_shutdown() noexcept;

    raft_rpc& get_rpc(raft::group_id gid);

    // Find server for group by group id. Throws exception if
    // there is no such group.
    raft::server& get_server(raft::group_id gid);

    // Find server for the given group.
    // Returns `nullptr` if there is no such group.
    raft::server* find_server(raft::group_id);

    // Returns the list of all Raft groups on this shard by their IDs.
    std::vector<raft::group_id> all_groups() const;

    // Return an instance of group 0. Valid only on shard 0,
    // after boot/upgrade is complete
    raft::server& group0();

    // Start raft server instance, store in the map of raft servers and
    // arm the associated timer to tick the server.
    future<> start_server_for_group(raft_server_for_group grp);
    void abort_server(raft::group_id gid, sstring reason = "");
    unsigned shard_for_group(const raft::group_id& gid) const;
    shared_ptr<raft::failure_detector> failure_detector();
    raft_address_map& address_map() { return _address_map; }
    direct_failure_detector::failure_detector& direct_fd() { return _direct_fd; }

    // Is the RAFT local feature enabled?
    // Note: do not confuse with the SUPPORTS_RAFT cluster feature.
    bool is_enabled() const { return _is_enabled; }
};

// Implementation of `direct_failure_detector::pinger` which uses DIRECT_FD_PING verb for pinging.
// Translates `raft::server_id`s to `gms::inet_address`es before pinging.
class direct_fd_pinger : public seastar::peering_sharded_service<direct_fd_pinger>, public direct_failure_detector::pinger {
    netw::messaging_service& _ms;
    raft_address_map& _address_map;

    using rate_limits = utils::recent_entries_map<direct_failure_detector::pinger::endpoint_id, logger::rate_limit>;
    rate_limits _rate_limits;

public:
    direct_fd_pinger(netw::messaging_service& ms, raft_address_map& address_map)
            : _ms(ms), _address_map(address_map) {}

    direct_fd_pinger(const direct_fd_pinger&) = delete;
    direct_fd_pinger(direct_fd_pinger&&) = delete;

    future<bool> ping(direct_failure_detector::pinger::endpoint_id id, abort_source& as) override;
};

// XXX: find a better place to put this?
struct direct_fd_clock : public direct_failure_detector::clock {
    using base = std::chrono::steady_clock;

    direct_failure_detector::clock::timepoint_t now() noexcept override;
    future<> sleep_until(direct_failure_detector::clock::timepoint_t tp, abort_source& as) override;
};

} // end of namespace service
