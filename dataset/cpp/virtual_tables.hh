/*
 * Modified by ScyllaDB
 * Copyright (C) 2023-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#pragma once

#include <seastar/core/distributed.hh>
#include "schema/schema_fwd.hh"

namespace replica {
class database;
}

namespace service {
class storage_service;
class raft_group_registry;
}

namespace gms {
class gossiper;
}

namespace db {

class config;
class system_keyspace;

future<> initialize_virtual_tables(
    distributed<replica::database>&,
    distributed<service::storage_service>&,
    sharded<gms::gossiper>&,
    sharded<service::raft_group_registry>&,
    sharded<db::system_keyspace>& sys_ks,
    db::config&);

} // namespace db
