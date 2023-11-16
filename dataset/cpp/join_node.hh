/*
 * Copyright 2023-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <set>
#include <seastar/core/sstring.hh>
#include "raft/raft.hh"

namespace service {

struct join_node_request_params {
    raft::server_id host_id;
    std::optional<raft::server_id> replaced_id;
    std::vector<sstring> ignore_nodes;
    sstring cluster_name;
    sstring snitch_name;
    sstring datacenter;
    sstring rack;
    sstring release_version;
    uint32_t num_tokens;
    uint32_t shard_count;
    uint32_t ignore_msb;
    std::vector<sstring> supported_features;
};

struct join_node_request_result {
    // Request was successfully placed and will be processed
    // by the topology coordinator.
    struct ok {};

    // The request was immediately rejected, most likely due to some
    // parameters being incorrect or incompatible with the cluster.
    struct rejected {
        sstring reason;
    };

    std::variant<ok, rejected> result;
};

struct join_node_response_params {
    // The topology coordinator accepts and wants to add the joining node
    // to group 0 and to the cluster in general.
    struct accepted {};

    // The topology coordinator rejects the node, most likely due to some
    // parameters being incorrect or incompatible with the cluster.
    struct rejected {
        sstring reason;
    };

    std::variant<accepted, rejected> response;
};

struct join_node_response_result {};

}
