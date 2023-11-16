/*
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */


#include <boost/test/unit_test.hpp>
#include "locator/gossiping_property_file_snitch.hh"
#include "utils/fb_utilities.hh"
#include "test/lib/scylla_test_case.hh"
#include <seastar/util/std-compat.hh>
#include <seastar/core/reactor.hh>
#include <vector>
#include <string>
#include <tuple>

namespace fs = std::filesystem;

static fs::path test_files_subdir("test/resource/snitch_property_files");

future<> one_test(const std::string& property_fname, bool exp_result) {
    using namespace locator;
    using namespace std::filesystem;

    printf("Testing %s property file: %s\n",
           (exp_result ? "well-formed" : "ill-formed"),
           property_fname.c_str());

    path fname(test_files_subdir);
    fname /= path(property_fname);

    utils::fb_utilities::set_broadcast_address(gms::inet_address("localhost"));
    utils::fb_utilities::set_broadcast_rpc_address(gms::inet_address("localhost"));

    engine().set_strict_dma(false);

    snitch_config cfg;
    cfg.name = "org.apache.cassandra.locator.GossipingPropertyFileSnitch";
    cfg.properties_file_name = fname.string();
    auto snitch_i = std::make_unique<sharded<locator::snitch_ptr>>();
    auto& snitch = *snitch_i;

    return snitch.start(cfg).then([&snitch] {
        return snitch.invoke_on_all(&locator::snitch_ptr::start);
    }).then_wrapped([&snitch, exp_result] (auto&& f) -> future<> {
            try {
                f.get();
                if (!exp_result) {
                    BOOST_ERROR("Failed to catch an error in a malformed "
                                "configuration file");
                    return snitch.stop();
                }
                auto cpu0_dc = make_lw_shared<sstring>();
                auto cpu0_rack = make_lw_shared<sstring>();
                auto res = make_lw_shared<bool>(true);

                return snitch.invoke_on(0,
                        [cpu0_dc, cpu0_rack,
                         res] (snitch_ptr& inst) {
                    *cpu0_dc =inst->get_datacenter();
                    *cpu0_rack = inst->get_rack();
                }).then([&snitch, cpu0_dc, cpu0_rack, res] {
                    return snitch.invoke_on_all(
                            [cpu0_dc, cpu0_rack,
                             res] (snitch_ptr& inst) {
                        if (*cpu0_dc != inst->get_datacenter() ||
                            *cpu0_rack != inst->get_rack()) {
                            *res = false;
                        }
                    }).then([res] {
                        if (!*res) {
                            BOOST_ERROR("Data center or Rack do not match on "
                                        "different shards");
                        } else {
                            BOOST_CHECK(true);
                        }
                        return make_ready_future<>();
                    });
                });
            } catch (std::exception& e) {
                BOOST_CHECK(!exp_result);
                return make_ready_future<>();
            }
        }).finally([ snitch_i = std::move(snitch_i) ] () mutable {
            return snitch_i->stop().finally([snitch_i = std::move(snitch_i)] {});
        });
}

#define GOSSIPING_TEST_CASE(tag, exp_res) \
SEASTAR_TEST_CASE(tag) { \
    return one_test(#tag".property", exp_res); \
}

////////////////////////////////////////////////////////////////////////////////
GOSSIPING_TEST_CASE(bad_double_dc,             false);
GOSSIPING_TEST_CASE(bad_double_rack,           false);
GOSSIPING_TEST_CASE(bad_double_prefer_local,   false);
GOSSIPING_TEST_CASE(bad_missing_dc,            false);
GOSSIPING_TEST_CASE(bad_missing_rack,          false);
GOSSIPING_TEST_CASE(good_missing_prefer_local, true);
GOSSIPING_TEST_CASE(bad_format_1,              false);
GOSSIPING_TEST_CASE(bad_format_2,              false);
GOSSIPING_TEST_CASE(bad_format_3,              false);
GOSSIPING_TEST_CASE(bad_format_4,              false);
GOSSIPING_TEST_CASE(bad_format_5,              false);
GOSSIPING_TEST_CASE(bad_format_6,              false);
GOSSIPING_TEST_CASE(good_1,                    true);
GOSSIPING_TEST_CASE(good_2,                    true);
