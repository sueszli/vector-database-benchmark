/*
 * Copyright 2022 Redpanda Data, Inc.
 *
 * Licensed as a Redpanda Enterprise file under the Redpanda Community
 * License (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * https://github.com/redpanda-data/redpanda/blob/master/licenses/rcl.md
 */

#include "cloud_storage/base_manifest.h"

namespace cloud_storage {

base_manifest::~base_manifest() = default;

std::ostream& operator<<(std::ostream& s, manifest_type t) {
    switch (t) {
    case manifest_type::topic:
        s << "topic";
        break;
    case manifest_type::partition:
        s << "partition";
        break;
    case manifest_type::tx_range:
        s << "tx-range";
        break;
    case manifest_type::cluster_metadata:
        s << "cluster-metadata";
        break;
    case manifest_type::spillover:
        s << "spillover";
        break;
    }
    return s;
}

} // namespace cloud_storage
