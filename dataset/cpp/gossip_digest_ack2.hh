/*
 *
 * Modified by ScyllaDB
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#pragma once

#include "utils/serialization.hh"
#include "gms/gossip_digest.hh"
#include "gms/inet_address.hh"
#include "gms/endpoint_state.hh"

namespace gms {
/**
 * This ack gets sent out as a result of the receipt of a GossipDigestAckMessage. This the
 * last stage of the 3 way messaging of the Gossip protocol.
 */
class gossip_digest_ack2 {
private:
    using inet_address = gms::inet_address;
    std::map<inet_address, endpoint_state> _map;
public:
    gossip_digest_ack2() {
    }

    gossip_digest_ack2(std::map<inet_address, endpoint_state> m)
        : _map(std::move(m)) {
    }

    std::map<inet_address, endpoint_state>& get_endpoint_state_map() {
        return _map;
    }

    const std::map<inet_address, endpoint_state>& get_endpoint_state_map() const {
        return _map;
    }

    friend std::ostream& operator<<(std::ostream& os, const gossip_digest_ack2& ack2);
};

} // gms
