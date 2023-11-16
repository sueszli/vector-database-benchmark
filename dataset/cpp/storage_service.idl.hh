/*
 * Copyright 2022-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

namespace locator {

struct tablet_id final {
    uint64_t value();
};

struct global_tablet_id final {
    ::table_id table;
    locator::tablet_id tablet;
};

}

namespace service {
struct fencing_token {
    service::topology::version_t topology_version;
};

struct raft_topology_cmd {
    enum class command: uint8_t {
        barrier,
        barrier_and_drain,
        stream_ranges,
        fence
    };
    service::raft_topology_cmd::command cmd;
};

struct raft_topology_cmd_result {
    enum class command_status: uint8_t {
        fail,
        success
    };
    service::raft_topology_cmd_result::command_status status;
};

struct raft_topology_snapshot {
    std::vector<canonical_mutation> topology_mutations;
    std::vector<canonical_mutation> cdc_generation_mutations;
};

struct raft_topology_pull_params {};

verb raft_topology_cmd (raft::server_id dst_id, raft::term_t term, uint64_t cmd_index, service::raft_topology_cmd) -> service::raft_topology_cmd_result;
verb [[cancellable]] raft_pull_topology_snapshot (raft::server_id dst_id, service::raft_topology_pull_params) -> service::raft_topology_snapshot;
verb [[cancellable]] tablet_stream_data (raft::server_id dst_id, locator::global_tablet_id);
verb [[cancellable]] tablet_cleanup (raft::server_id dst_id, locator::global_tablet_id);
}
