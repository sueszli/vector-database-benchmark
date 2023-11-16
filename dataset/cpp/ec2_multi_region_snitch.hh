/*
 *
 * Modified by ScyllaDB
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#pragma once

#include "locator/ec2_snitch.hh"

namespace locator {
class ec2_multi_region_snitch : public ec2_snitch {
public:
    ec2_multi_region_snitch(const snitch_config&);
    virtual std::list<std::pair<gms::application_state, gms::versioned_value>> get_app_states() const override;
    virtual future<> start() override;
    virtual void set_local_private_addr(const sstring& addr_str) override;
    virtual sstring get_name() const override {
        return "org.apache.cassandra.locator.Ec2MultiRegionSnitch";
    }
private:
    sstring _local_private_address;
    bool _broadcast_rpc_address_specified_by_user;
};
} // namespace locator
