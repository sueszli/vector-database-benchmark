/*
 * Copyright (C) 2021-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include "gms/inet_address.hh"
#include "utils/small_vector.hh"

using inet_address_vector_replica_set = utils::small_vector<gms::inet_address, 3>;

using inet_address_vector_topology_change = utils::small_vector<gms::inet_address, 1>;
