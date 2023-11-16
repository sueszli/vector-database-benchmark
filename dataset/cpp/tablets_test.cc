/*
 * Copyright (C) 2023-present-2020 ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */



#include "test/lib/scylla_test_case.hh"
#include "test/lib/random_utils.hh"
#include <seastar/testing/thread_test_case.hh>
#include "test/lib/cql_test_env.hh"
#include "test/lib/log.hh"
#include "db/config.hh"
#include "schema/schema_builder.hh"

#include "replica/tablets.hh"
#include "replica/tablet_mutation_builder.hh"
#include "locator/tablets.hh"
#include "service/tablet_allocator.hh"
#include "locator/tablet_sharder.hh"
#include "locator/load_sketch.hh"
#include "locator/tablet_replication_strategy.hh"
#include "utils/fb_utilities.hh"
#include "utils/UUID_gen.hh"
#include "utils/error_injection.hh"

using namespace locator;
using namespace replica;
using namespace service;

static api::timestamp_type next_timestamp = api::new_timestamp();

static utils::UUID next_uuid() {
    static uint64_t counter = 1;
    return utils::UUID_gen::get_time_UUID(std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                    std::chrono::seconds(counter++))));
}

static
void verify_tablet_metadata_persistence(cql_test_env& env, const tablet_metadata& tm) {
    save_tablet_metadata(env.local_db(), tm, next_timestamp++).get();
    auto tm2 = read_tablet_metadata(env.local_qp()).get0();
    BOOST_REQUIRE_EQUAL(tm, tm2);
}

static
cql_test_config tablet_cql_test_config() {
    cql_test_config c;
    c.db_config->experimental_features({
            db::experimental_features_t::feature::TABLETS,
        }, db::config::config_source::CommandLine);
    c.db_config->consistent_cluster_management(true);
    return c;
}

static
future<table_id> add_table(cql_test_env& e) {
    auto id = table_id(utils::UUID_gen::get_time_UUID());
    co_await e.create_table([id] (std::string_view ks_name) {
        return *schema_builder(ks_name, id.to_sstring(), id)
                .with_column("p1", utf8_type, column_kind::partition_key)
                .with_column("r1", int32_type)
                .build();
    });
    co_return id;
}

SEASTAR_TEST_CASE(test_tablet_metadata_persistence) {
    return do_with_cql_env_thread([] (cql_test_env& e) {
        auto h1 = host_id(utils::UUID_gen::get_time_UUID());
        auto h2 = host_id(utils::UUID_gen::get_time_UUID());
        auto h3 = host_id(utils::UUID_gen::get_time_UUID());

        auto table1 = add_table(e).get0();
        auto table2 = add_table(e).get0();

        {
            tablet_metadata tm;

            // Empty
            verify_tablet_metadata_persistence(e, tm);

            // Add table1
            {
                tablet_map tmap(1);
                tmap.set_tablet(tmap.first_tablet(), tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 0},
                        tablet_replica {h2, 3},
                        tablet_replica {h3, 1},
                    }
                });
                tm.set_tablet_map(table1, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);

            // Add table2
            {
                tablet_map tmap(4);
                auto tb = tmap.first_tablet();
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 0},
                    }
                });
                tb = *tmap.next_tablet(tb);
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h3, 3},
                    }
                });
                tb = *tmap.next_tablet(tb);
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h2, 2},
                    }
                });
                tb = *tmap.next_tablet(tb);
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 1},
                    }
                });
                tm.set_tablet_map(table2, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);

            // Increase RF of table2
            {
                auto&& tmap = tm.get_tablet_map(table2);
                auto tb = tmap.first_tablet();
                tb = *tmap.next_tablet(tb);

                tmap.set_tablet_transition_info(tb, tablet_transition_info{
                    tablet_transition_stage::allow_write_both_read_old,
                    tablet_replica_set {
                        tablet_replica {h3, 3},
                        tablet_replica {h1, 7},
                    },
                    tablet_replica {h1, 7}
                });

                tb = *tmap.next_tablet(tb);
                tmap.set_tablet_transition_info(tb, tablet_transition_info{
                    tablet_transition_stage::use_new,
                    tablet_replica_set {
                        tablet_replica {h1, 4},
                        tablet_replica {h2, 2},
                    },
                    tablet_replica {h1, 4}
                });
            }

            verify_tablet_metadata_persistence(e, tm);

            // Reduce tablet count in table2
            {
                tablet_map tmap(2);
                auto tb = tmap.first_tablet();
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 0},
                    }
                });
                tb = *tmap.next_tablet(tb);
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h3, 3},
                    }
                });
                tm.set_tablet_map(table2, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);

            // Reduce RF for table1, increasing tablet count
            {
                tablet_map tmap(2);
                auto tb = tmap.first_tablet();
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h3, 7},
                    }
                });
                tb = *tmap.next_tablet(tb);
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 3},
                    }
                });
                tm.set_tablet_map(table1, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);

            // Reduce tablet count for table1
            {
                tablet_map tmap(1);
                auto tb = tmap.first_tablet();
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h1, 3},
                    }
                });
                tm.set_tablet_map(table1, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);

            // Change replica of table1
            {
                tablet_map tmap(1);
                auto tb = tmap.first_tablet();
                tmap.set_tablet(tb, tablet_info {
                    tablet_replica_set {
                        tablet_replica {h3, 7},
                    }
                });
                tm.set_tablet_map(table1, std::move(tmap));
            }

            verify_tablet_metadata_persistence(e, tm);
        }
    }, tablet_cql_test_config());
}

SEASTAR_TEST_CASE(test_get_shard) {
    return do_with_cql_env_thread([] (cql_test_env& e) {
        auto h1 = host_id(utils::UUID_gen::get_time_UUID());
        auto h2 = host_id(utils::UUID_gen::get_time_UUID());
        auto h3 = host_id(utils::UUID_gen::get_time_UUID());

        auto table1 = table_id(utils::UUID_gen::get_time_UUID());

        tablet_metadata tm;
        tablet_id tid(0);
        tablet_id tid1(0);

        {
            tablet_map tmap(2);
            tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h1, 0},
                    tablet_replica {h3, 5},
                }
            });
            tid1 = *tmap.next_tablet(tid);
            tmap.set_tablet(tid1, tablet_info {
                tablet_replica_set {
                    tablet_replica {h1, 2},
                    tablet_replica {h3, 1},
                }
            });
            tmap.set_tablet_transition_info(tid, tablet_transition_info {
                tablet_transition_stage::allow_write_both_read_old,
                tablet_replica_set {
                    tablet_replica {h1, 0},
                    tablet_replica {h2, 3},
                },
                tablet_replica {h2, 3}
            });
            tm.set_tablet_map(table1, std::move(tmap));
        }

        auto&& tmap = tm.get_tablet_map(table1);

        BOOST_REQUIRE_EQUAL(tmap.get_shard(tid1, h1), std::make_optional(shard_id(2)));
        BOOST_REQUIRE(!tmap.get_shard(tid1, h2));
        BOOST_REQUIRE_EQUAL(tmap.get_shard(tid1, h3), std::make_optional(shard_id(1)));

        BOOST_REQUIRE_EQUAL(tmap.get_shard(tid, h1), std::make_optional(shard_id(0)));
        BOOST_REQUIRE_EQUAL(tmap.get_shard(tid, h2), std::make_optional(shard_id(3)));
        BOOST_REQUIRE_EQUAL(tmap.get_shard(tid, h3), std::make_optional(shard_id(5)));

    }, tablet_cql_test_config());
}

SEASTAR_TEST_CASE(test_mutation_builder) {
    return do_with_cql_env_thread([] (cql_test_env& e) {
        auto h1 = host_id(utils::UUID_gen::get_time_UUID());
        auto h2 = host_id(utils::UUID_gen::get_time_UUID());
        auto h3 = host_id(utils::UUID_gen::get_time_UUID());

        auto table1 = add_table(e).get0();

        tablet_metadata tm;
        tablet_id tid(0);
        tablet_id tid1(0);

        {
            tablet_map tmap(2);
            tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h1, 0},
                    tablet_replica {h3, 5},
                }
            });
            tid1 = *tmap.next_tablet(tid);
            tmap.set_tablet(tid1, tablet_info {
                tablet_replica_set {
                    tablet_replica {h1, 2},
                    tablet_replica {h3, 1},
                }
            });
            tm.set_tablet_map(table1, std::move(tmap));
        }

        save_tablet_metadata(e.local_db(), tm, next_timestamp++).get();

        {
            tablet_mutation_builder b(next_timestamp++, "ks", table1);
            auto last_token = tm.get_tablet_map(table1).get_last_token(tid1);
            b.set_new_replicas(last_token, tablet_replica_set {
                    tablet_replica {h1, 2},
                    tablet_replica {h2, 3},
            });
            b.set_stage(last_token, tablet_transition_stage::write_both_read_new);
            e.local_db().apply({freeze(b.build())}, db::no_timeout).get();
        }

        {
            tablet_map expected_tmap(2);
            tid = expected_tmap.first_tablet();
            expected_tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 0},
                            tablet_replica {h3, 5},
                    }
            });
            tid1 = *expected_tmap.next_tablet(tid);
            expected_tmap.set_tablet(tid1, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 2},
                            tablet_replica {h3, 1},
                    }
            });
            expected_tmap.set_tablet_transition_info(tid1, tablet_transition_info {
                    tablet_transition_stage::write_both_read_new,
                    tablet_replica_set {
                            tablet_replica {h1, 2},
                            tablet_replica {h2, 3},
                    },
                    tablet_replica {h2, 3}
            });

            auto tm_from_disk = read_tablet_metadata(e.local_qp()).get0();
            BOOST_REQUIRE_EQUAL(expected_tmap, tm_from_disk.get_tablet_map(table1));
        }

        {
            tablet_mutation_builder b(next_timestamp++, "ks", table1);
            auto last_token = tm.get_tablet_map(table1).get_last_token(tid1);
            b.set_stage(last_token, tablet_transition_stage::use_new);
            e.local_db().apply({freeze(b.build())}, db::no_timeout).get();
        }

        {
            tablet_map expected_tmap(2);
            tid = expected_tmap.first_tablet();
            expected_tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 0},
                            tablet_replica {h3, 5},
                    }
            });
            tid1 = *expected_tmap.next_tablet(tid);
            expected_tmap.set_tablet(tid1, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 2},
                            tablet_replica {h3, 1},
                    }
            });
            expected_tmap.set_tablet_transition_info(tid1, tablet_transition_info {
                    tablet_transition_stage::use_new,
                    tablet_replica_set {
                            tablet_replica {h1, 2},
                            tablet_replica {h2, 3},
                    },
                    tablet_replica {h2, 3}
            });

            auto tm_from_disk = read_tablet_metadata(e.local_qp()).get0();
            BOOST_REQUIRE_EQUAL(expected_tmap, tm_from_disk.get_tablet_map(table1));
        }

        {
            tablet_mutation_builder b(next_timestamp++, "ks", table1);
            auto last_token = tm.get_tablet_map(table1).get_last_token(tid1);
            b.set_replicas(last_token, tablet_replica_set {
                tablet_replica {h1, 2},
                tablet_replica {h2, 3},
            });
            b.del_transition(last_token);
            e.local_db().apply({freeze(b.build())}, db::no_timeout).get();
        }

        {
            tablet_map expected_tmap(2);
            tid = expected_tmap.first_tablet();
            expected_tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 0},
                            tablet_replica {h3, 5},
                    }
            });
            tid1 = *expected_tmap.next_tablet(tid);
            expected_tmap.set_tablet(tid1, tablet_info {
                    tablet_replica_set {
                            tablet_replica {h1, 2},
                            tablet_replica {h2, 3},
                    }
            });

            auto tm_from_disk = read_tablet_metadata(e.local_qp()).get0();
            BOOST_REQUIRE_EQUAL(expected_tmap, tm_from_disk.get_tablet_map(table1));
        }
    }, tablet_cql_test_config());
}

SEASTAR_TEST_CASE(test_sharder) {
    return do_with_cql_env_thread([] (cql_test_env& e) {
        auto h1 = host_id(utils::UUID_gen::get_time_UUID());
        auto h2 = host_id(utils::UUID_gen::get_time_UUID());
        auto h3 = host_id(utils::UUID_gen::get_time_UUID());

        auto table1 = table_id(utils::UUID_gen::get_time_UUID());

        token_metadata tokm(token_metadata::config{ .topo_cfg{ .this_host_id = h1 } });
        tokm.get_topology().add_or_update_endpoint(utils::fb_utilities::get_broadcast_address(), h1);

        std::vector<tablet_id> tablet_ids;
        {
            tablet_map tmap(4);
            auto tid = tmap.first_tablet();

            tablet_ids.push_back(tid);
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h1, 3},
                    tablet_replica {h3, 5},
                }
            });

            tid = *tmap.next_tablet(tid);
            tablet_ids.push_back(tid);
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h2, 3},
                    tablet_replica {h3, 1},
                }
            });

            tid = *tmap.next_tablet(tid);
            tablet_ids.push_back(tid);
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h3, 2},
                    tablet_replica {h1, 1},
                }
            });
            tmap.set_tablet_transition_info(tid, tablet_transition_info {
                tablet_transition_stage::use_new,
                tablet_replica_set {
                    tablet_replica {h1, 1},
                    tablet_replica {h2, 3},
                },
                tablet_replica {h2, 3}
            });

            tid = *tmap.next_tablet(tid);
            tablet_ids.push_back(tid);
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {h3, 7},
                    tablet_replica {h2, 3},
                }
            });

            tablet_metadata tm;
            tm.set_tablet_map(table1, std::move(tmap));
            tokm.set_tablets(std::move(tm));
        }

        auto& tm = tokm.tablets().get_tablet_map(table1);
        tablet_sharder sharder(tokm, table1);
        BOOST_REQUIRE_EQUAL(sharder.shard_of(tm.get_last_token(tablet_ids[0])), 3);
        BOOST_REQUIRE_EQUAL(sharder.shard_of(tm.get_last_token(tablet_ids[1])), 0); // missing
        BOOST_REQUIRE_EQUAL(sharder.shard_of(tm.get_last_token(tablet_ids[2])), 1);
        BOOST_REQUIRE_EQUAL(sharder.shard_of(tm.get_last_token(tablet_ids[3])), 0); // missing

        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_last_token(tablet_ids[1]), 0), tm.get_first_token(tablet_ids[3]));
        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_last_token(tablet_ids[1]), 1), tm.get_first_token(tablet_ids[2]));
        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_last_token(tablet_ids[1]), 3), dht::maximum_token());

        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_first_token(tablet_ids[1]), 0), tm.get_first_token(tablet_ids[3]));
        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_first_token(tablet_ids[1]), 1), tm.get_first_token(tablet_ids[2]));
        BOOST_REQUIRE_EQUAL(sharder.token_for_next_shard(tm.get_first_token(tablet_ids[1]), 3), dht::maximum_token());

        {
            auto shard_opt = sharder.next_shard(tm.get_last_token(tablet_ids[0]));
            BOOST_REQUIRE(shard_opt);
            BOOST_REQUIRE_EQUAL(shard_opt->shard, 0);
            BOOST_REQUIRE_EQUAL(shard_opt->token, tm.get_first_token(tablet_ids[1]));
        }

        {
            auto shard_opt = sharder.next_shard(tm.get_last_token(tablet_ids[1]));
            BOOST_REQUIRE(shard_opt);
            BOOST_REQUIRE_EQUAL(shard_opt->shard, 1);
            BOOST_REQUIRE_EQUAL(shard_opt->token, tm.get_first_token(tablet_ids[2]));
        }

        {
            auto shard_opt = sharder.next_shard(tm.get_last_token(tablet_ids[2]));
            BOOST_REQUIRE(shard_opt);
            BOOST_REQUIRE_EQUAL(shard_opt->shard, 0);
            BOOST_REQUIRE_EQUAL(shard_opt->token, tm.get_first_token(tablet_ids[3]));
        }

        {
            auto shard_opt = sharder.next_shard(tm.get_last_token(tablet_ids[3]));
            BOOST_REQUIRE(!shard_opt);
        }
    }, tablet_cql_test_config());
}

SEASTAR_TEST_CASE(test_large_tablet_metadata) {
    return do_with_cql_env_thread([] (cql_test_env& e) {
        tablet_metadata tm;

        auto h1 = host_id(utils::UUID_gen::get_time_UUID());
        auto h2 = host_id(utils::UUID_gen::get_time_UUID());
        auto h3 = host_id(utils::UUID_gen::get_time_UUID());

        const int nr_tables = 1'00;
        const int tablets_per_table = 1024;

        for (int i = 0; i < nr_tables; ++i) {
            tablet_map tmap(tablets_per_table);

            for (tablet_id j : tmap.tablet_ids()) {
                tmap.set_tablet(j, tablet_info {
                    tablet_replica_set {{h1, 0}, {h2, 1}, {h3, 2},}
                });
            }

            auto id = add_table(e).get0();
            tm.set_tablet_map(id, std::move(tmap));
        }

        verify_tablet_metadata_persistence(e, tm);
    }, tablet_cql_test_config());
}

SEASTAR_THREAD_TEST_CASE(test_token_ownership_splitting) {
    const auto real_min_token = dht::token(dht::token_kind::key, std::numeric_limits<int64_t>::min() + 1);
    const auto real_max_token = dht::token(dht::token_kind::key, std::numeric_limits<int64_t>::max());

    for (auto&& tmap : {
        tablet_map(1),
        tablet_map(2),
        tablet_map(4),
        tablet_map(16),
        tablet_map(1024),
    }) {
        testlog.debug("tmap: {}", tmap);

        BOOST_REQUIRE_EQUAL(real_min_token, tmap.get_first_token(tmap.first_tablet()));
        BOOST_REQUIRE_EQUAL(real_max_token, tmap.get_last_token(tmap.last_tablet()));

        std::optional<tablet_id> prev_tb;
        for (tablet_id tb : tmap.tablet_ids()) {
            testlog.debug("first: {}, last: {}", tmap.get_first_token(tb), tmap.get_last_token(tb));
            BOOST_REQUIRE_EQUAL(tb, tmap.get_tablet_id(tmap.get_first_token(tb)));
            BOOST_REQUIRE_EQUAL(tb, tmap.get_tablet_id(tmap.get_last_token(tb)));
            if (prev_tb) {
                BOOST_REQUIRE_EQUAL(dht::next_token(tmap.get_last_token(*prev_tb)), tmap.get_first_token(tb));
            }
            prev_tb = tb;
        }
    }
}

// Reflects the plan in a given token metadata as if the migrations were fully executed.
static
void apply_plan(token_metadata& tm, const migration_plan& plan) {
    for (auto&& mig : plan.migrations()) {
        tablet_map& tmap = tm.tablets().get_tablet_map(mig.tablet.table);
        auto tinfo = tmap.get_tablet_info(mig.tablet.tablet);
        tinfo.replicas = replace_replica(tinfo.replicas, mig.src, mig.dst);
        tmap.set_tablet(mig.tablet.tablet, tinfo);
    }
}

static
tablet_transition_info migration_to_transition_info(const tablet_migration_info& mig, const tablet_info& ti) {
    return tablet_transition_info {
            tablet_transition_stage::allow_write_both_read_old,
            replace_replica(ti.replicas, mig.src, mig.dst),
            mig.dst
    };
}

// Reflects the plan in a given token metadata as if the migrations were started but not yet executed.
static
void apply_plan_as_in_progress(token_metadata& tm, const migration_plan& plan) {
    for (auto&& mig : plan.migrations()) {
        tablet_map& tmap = tm.tablets().get_tablet_map(mig.tablet.table);
        auto tinfo = tmap.get_tablet_info(mig.tablet.tablet);
        tmap.set_tablet_transition_info(mig.tablet.tablet, migration_to_transition_info(mig, tinfo));
    }
}

static
void rebalance_tablets(tablet_allocator& talloc, shared_token_metadata& stm) {
    while (true) {
        auto plan = talloc.balance_tablets(stm.get()).get0();
        if (plan.empty()) {
            break;
        }
        stm.mutate_token_metadata([&] (token_metadata& tm) {
            apply_plan(tm, plan);
            return make_ready_future<>();
        }).get();
    }
}

static
void rebalance_tablets_as_in_progress(tablet_allocator& talloc, shared_token_metadata& stm) {
    while (true) {
        auto plan = talloc.balance_tablets(stm.get()).get0();
        if (plan.empty()) {
            break;
        }
        stm.mutate_token_metadata([&] (token_metadata& tm) {
            apply_plan_as_in_progress(tm, plan);
            return make_ready_future<>();
        }).get();
    }
}

// Completes any in progress tablet migrations.
static
void execute_transitions(shared_token_metadata& stm) {
    stm.mutate_token_metadata([&] (token_metadata& tm) {
        for (auto&& [tablet, tmap_] : tm.tablets().all_tables()) {
            auto& tmap = tmap_;
            for (auto&& [tablet, trinfo]: tmap.transitions()) {
                auto ti = tmap.get_tablet_info(tablet);
                ti.replicas = trinfo.next;
                tmap.set_tablet(tablet, ti);
            }
            tmap.clear_transitions();
        }
        return make_ready_future<>();
    }).get();
}

SEASTAR_THREAD_TEST_CASE(test_load_balancing_with_empty_node) {
  do_with_cql_env_thread([] (auto& e) {
    // Tests the scenario of bootstrapping a single node
    // Verifies that load balancer sees it and moves tablets to that node.

    inet_address ip1("192.168.0.1");
    inet_address ip2("192.168.0.2");
    inet_address ip3("192.168.0.3");

    auto host1 = host_id(next_uuid());
    auto host2 = host_id(next_uuid());
    auto host3 = host_id(next_uuid());

    auto table1 = table_id(next_uuid());

    unsigned shard_count = 2;

    semaphore sem(1);
    shared_token_metadata stm([&sem] () noexcept { return get_units(sem, 1); }, locator::token_metadata::config{
        locator::topology::config{
            .this_endpoint = ip1,
            .local_dc_rack = locator::endpoint_dc_rack::default_location
        }
    });

    stm.mutate_token_metadata([&] (auto& tm) {
        tm.update_host_id(host1, ip1);
        tm.update_host_id(host2, ip2);
        tm.update_host_id(host3, ip3);
        tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
        tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
        tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);

        tablet_map tmap(4);
        auto tid = tmap.first_tablet();
        tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                        tablet_replica {host1, 0},
                        tablet_replica {host2, 1},
                }
        });
        tid = *tmap.next_tablet(tid);
        tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                        tablet_replica {host1, 0},
                        tablet_replica {host2, 1},
                }
        });
        tid = *tmap.next_tablet(tid);
        tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                        tablet_replica {host1, 0},
                        tablet_replica {host2, 0},
                }
        });
        tid = *tmap.next_tablet(tid);
        tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                        tablet_replica {host1, 1},
                        tablet_replica {host2, 0},
                }
        });
        tablet_metadata tmeta;
        tmeta.set_tablet_map(table1, std::move(tmap));
        tm.set_tablets(std::move(tmeta));
        return make_ready_future<>();
    }).get();

    // Sanity check
    {
        load_sketch load(stm.get());
        load.populate().get();
        BOOST_REQUIRE_EQUAL(load.get_load(host1), 4);
        BOOST_REQUIRE_EQUAL(load.get_avg_shard_load(host1), 2);
        BOOST_REQUIRE_EQUAL(load.get_load(host2), 4);
        BOOST_REQUIRE_EQUAL(load.get_avg_shard_load(host2), 2);
        BOOST_REQUIRE_EQUAL(load.get_load(host3), 0);
        BOOST_REQUIRE_EQUAL(load.get_avg_shard_load(host3), 0);
    }

    rebalance_tablets(e.get_tablet_allocator().local(), stm);

    {
        load_sketch load(stm.get());
        load.populate().get();

        for (auto h : {host1, host2, host3}) {
            testlog.debug("Checking host {}", h);
            BOOST_REQUIRE(load.get_load(h) <= 3);
            BOOST_REQUIRE(load.get_load(h) > 1);
            BOOST_REQUIRE(load.get_avg_shard_load(h) <= 2);
            BOOST_REQUIRE(load.get_avg_shard_load(h) > 0);
        }
    }
  }).get();
}

SEASTAR_THREAD_TEST_CASE(test_decommission_rf_met) {
    // Verifies that load balancer moves tablets out of the decommissioned node.
    // The scenario is such that replication factor of tablets can be satisfied after decommission.
    do_with_cql_env_thread([](auto& e) {
        inet_address ip1("192.168.0.1");
        inet_address ip2("192.168.0.2");
        inet_address ip3("192.168.0.3");

        auto host1 = host_id(next_uuid());
        auto host2 = host_id(next_uuid());
        auto host3 = host_id(next_uuid());

        auto table1 = table_id(next_uuid());

        semaphore sem(1);
        shared_token_metadata stm([&sem]() noexcept { return get_units(sem, 1); }, locator::token_metadata::config {
                locator::topology::config {
                        .this_endpoint = ip1,
                        .local_dc_rack = locator::endpoint_dc_rack::default_location
                }
        });

        stm.mutate_token_metadata([&](auto& tm) {
            const unsigned shard_count = 2;

            tm.update_host_id(host1, ip1);
            tm.update_host_id(host2, ip2);
            tm.update_host_id(host3, ip3);
            tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
            tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
            tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, node::state::being_decommissioned,
                               shard_count);

            tablet_map tmap(4);
            auto tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 1},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 1},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host3, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host2, 1},
                            tablet_replica {host3, 1},
                    }
            });
            tablet_metadata tmeta;
            tmeta.set_tablet_map(table1, std::move(tmap));
            tm.set_tablets(std::move(tmeta));
            return make_ready_future<>();
        }).get();

        rebalance_tablets(e.get_tablet_allocator().local(), stm);

        {
            load_sketch load(stm.get());
            load.populate().get();
            BOOST_REQUIRE(load.get_avg_shard_load(host1) == 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host2) == 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host3) == 0);
        }

        stm.mutate_token_metadata([&](auto& tm) {
            tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, node::state::left);
            return make_ready_future<>();
        }).get();

        rebalance_tablets(e.get_tablet_allocator().local(), stm);

        {
            load_sketch load(stm.get());
            load.populate().get();
            BOOST_REQUIRE(load.get_avg_shard_load(host1) == 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host2) == 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host3) == 0);
        }
    }).get();
}

SEASTAR_THREAD_TEST_CASE(test_decommission_two_racks) {
    // Verifies that load balancer moves tablets out of the decommissioned node.
    // The scenario is such that replication constraints of tablets can be satisfied after decommission.
    do_with_cql_env_thread([](auto& e) {
        inet_address ip1("192.168.0.1");
        inet_address ip2("192.168.0.2");
        inet_address ip3("192.168.0.3");
        inet_address ip4("192.168.0.4");

        auto host1 = host_id(next_uuid());
        auto host2 = host_id(next_uuid());
        auto host3 = host_id(next_uuid());
        auto host4 = host_id(next_uuid());

        std::vector<endpoint_dc_rack> racks = {
                endpoint_dc_rack{ "dc1", "rack-1" },
                endpoint_dc_rack{ "dc1", "rack-2" }
        };

        auto table1 = table_id(next_uuid());

        semaphore sem(1);
        shared_token_metadata stm([&sem]() noexcept { return get_units(sem, 1); }, locator::token_metadata::config {
                locator::topology::config {
                        .this_endpoint = ip1,
                        .local_dc_rack = racks[0]
                }
        });

        stm.mutate_token_metadata([&](auto& tm) {
            const unsigned shard_count = 1;

            tm.update_host_id(host1, ip1);
            tm.update_host_id(host2, ip2);
            tm.update_host_id(host3, ip3);
            tm.update_host_id(host4, ip4);
            tm.update_topology(ip1, racks[0], std::nullopt, shard_count);
            tm.update_topology(ip2, racks[1], std::nullopt, shard_count);
            tm.update_topology(ip3, racks[0], std::nullopt, shard_count);
            tm.update_topology(ip4, racks[1], node::state::being_decommissioned,
                               shard_count);

            tablet_map tmap(4);
            auto tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host2, 0},
                            tablet_replica {host3, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host3, 0},
                            tablet_replica {host4, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 0},
                    }
            });
            tablet_metadata tmeta;
            tmeta.set_tablet_map(table1, std::move(tmap));
            tm.set_tablets(std::move(tmeta));
            return make_ready_future<>();
        }).get();

        rebalance_tablets(e.get_tablet_allocator().local(), stm);

        {
            load_sketch load(stm.get());
            load.populate().get();
            BOOST_REQUIRE(load.get_avg_shard_load(host1) >= 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host2) >= 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host3) >= 2);
            BOOST_REQUIRE(load.get_avg_shard_load(host4) == 0);
        }

        // Verify replicas are not collocated on racks
        {
            auto tm = stm.get();
            auto& tmap = tm->tablets().get_tablet_map(table1);
            tmap.for_each_tablet([&](auto tid, auto& tinfo) {
                auto rack1 = tm->get_topology().get_rack(tinfo.replicas[0].host);
                auto rack2 = tm->get_topology().get_rack(tinfo.replicas[1].host);
                BOOST_REQUIRE(rack1 != rack2);
            }).get();
        }
    }).get();
}

SEASTAR_THREAD_TEST_CASE(test_decommission_rack_load_failure) {
    // Verifies that load balancer moves tablets out of the decommissioned node.
    // The scenario is such that it is impossible to distribute replicas without violating rack uniqueness.
    do_with_cql_env_thread([](auto& e) {
        inet_address ip1("192.168.0.1");
        inet_address ip2("192.168.0.2");
        inet_address ip3("192.168.0.3");
        inet_address ip4("192.168.0.4");

        auto host1 = host_id(next_uuid());
        auto host2 = host_id(next_uuid());
        auto host3 = host_id(next_uuid());
        auto host4 = host_id(next_uuid());

        std::vector<endpoint_dc_rack> racks = {
                endpoint_dc_rack{ "dc1", "rack-1" },
                endpoint_dc_rack{ "dc1", "rack-2" }
        };

        auto table1 = table_id(next_uuid());

        semaphore sem(1);
        shared_token_metadata stm([&sem]() noexcept { return get_units(sem, 1); }, locator::token_metadata::config {
                locator::topology::config {
                        .this_endpoint = ip1,
                        .local_dc_rack = racks[0]
                }
        });

        stm.mutate_token_metadata([&](auto& tm) {
            const unsigned shard_count = 1;

            tm.update_host_id(host1, ip1);
            tm.update_host_id(host2, ip2);
            tm.update_host_id(host3, ip3);
            tm.update_host_id(host4, ip4);
            tm.update_topology(ip1, racks[0], std::nullopt, shard_count);
            tm.update_topology(ip2, racks[0], std::nullopt, shard_count);
            tm.update_topology(ip3, racks[0], std::nullopt, shard_count);
            tm.update_topology(ip4, racks[1], node::state::being_decommissioned,
                               shard_count);

            tablet_map tmap(4);
            auto tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host4, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host2, 0},
                            tablet_replica {host4, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host3, 0},
                            tablet_replica {host4, 0},
                    }
            });
            tid = *tmap.next_tablet(tid);
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host4, 0},
                    }
            });
            tablet_metadata tmeta;
            tmeta.set_tablet_map(table1, std::move(tmap));
            tm.set_tablets(std::move(tmeta));
            return make_ready_future<>();
        }).get();

        BOOST_REQUIRE_THROW(rebalance_tablets(e.get_tablet_allocator().local(), stm), std::runtime_error);
    }).get();
}

SEASTAR_THREAD_TEST_CASE(test_decommission_rf_not_met) {
    // Verifies that load balancer moves tablets out of the decommissioned node.
    // The scenario is such that replication factor of tablets can be satisfied after decommission.
    do_with_cql_env_thread([](auto& e) {
        inet_address ip1("192.168.0.1");
        inet_address ip2("192.168.0.2");
        inet_address ip3("192.168.0.3");

        auto host1 = host_id(next_uuid());
        auto host2 = host_id(next_uuid());
        auto host3 = host_id(next_uuid());

        auto table1 = table_id(next_uuid());

        semaphore sem(1);
        shared_token_metadata stm([&sem]() noexcept { return get_units(sem, 1); }, locator::token_metadata::config {
                locator::topology::config {
                        .this_endpoint = ip1,
                        .local_dc_rack = locator::endpoint_dc_rack::default_location
                }
        });

        stm.mutate_token_metadata([&](auto& tm) {
            const unsigned shard_count = 2;

            tm.update_host_id(host1, ip1);
            tm.update_host_id(host2, ip2);
            tm.update_host_id(host3, ip3);
            tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
            tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
            tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, node::state::being_decommissioned,
                               shard_count);

            tablet_map tmap(1);
            auto tid = tmap.first_tablet();
            tmap.set_tablet(tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 0},
                            tablet_replica {host3, 0},
                    }
            });
            tablet_metadata tmeta;
            tmeta.set_tablet_map(table1, std::move(tmap));
            tm.set_tablets(std::move(tmeta));
            return make_ready_future<>();
        }).get();

        BOOST_REQUIRE_THROW(rebalance_tablets(e.get_tablet_allocator().local(), stm), std::runtime_error);
    }).get();
}

SEASTAR_THREAD_TEST_CASE(test_load_balancing_works_with_in_progress_transitions) {
  do_with_cql_env_thread([] (auto& e) {
    // Tests the scenario of bootstrapping a single node.
    // Verifies that the load balancer balances tablets on that node
    // even though there is already an active migration.
    // The test verifies that the load balancer creates a plan
    // which when executed will achieve perfect balance,
    // which is a proof that it doesn't stop due to active migrations.

    inet_address ip1("192.168.0.1");
    inet_address ip2("192.168.0.2");
    inet_address ip3("192.168.0.3");

    auto host1 = host_id(next_uuid());
    auto host2 = host_id(next_uuid());
    auto host3 = host_id(next_uuid());

    auto table1 = table_id(next_uuid());

    semaphore sem(1);
    shared_token_metadata stm([&sem] () noexcept { return get_units(sem, 1); }, locator::token_metadata::config{
        locator::topology::config{
            .this_endpoint = ip1,
            .local_dc_rack = locator::endpoint_dc_rack::default_location
        }
    });

    stm.mutate_token_metadata([&] (auto& tm) {
        tm.update_host_id(host1, ip1);
        tm.update_host_id(host2, ip2);
        tm.update_host_id(host3, ip3);
        tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, 1);
        tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, 1);
        tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, std::nullopt, 2);

        tablet_map tmap(4);
        std::optional<tablet_id> tid = tmap.first_tablet();
        for (int i = 0; i < 4; ++i) {
            tmap.set_tablet(*tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 0},
                    }
            });
            tid = tmap.next_tablet(*tid);
        }
        tmap.set_tablet_transition_info(tmap.first_tablet(), tablet_transition_info {
                tablet_transition_stage::allow_write_both_read_old,
                tablet_replica_set {
                        tablet_replica {host3, 0},
                        tablet_replica {host2, 0},
                },
                tablet_replica {host3, 0}
        });
        tablet_metadata tmeta;
        tmeta.set_tablet_map(table1, std::move(tmap));
        tm.set_tablets(std::move(tmeta));
        return make_ready_future<>();
    }).get();

    rebalance_tablets_as_in_progress(e.get_tablet_allocator().local(), stm);
    execute_transitions(stm);

    {
        load_sketch load(stm.get());
        load.populate().get();

        for (auto h : {host1, host2, host3}) {
            testlog.debug("Checking host {}", h);
            BOOST_REQUIRE(load.get_avg_shard_load(h) == 2);
        }
    }
  }).get();
}

#ifdef SCYLLA_ENABLE_ERROR_INJECTION
SEASTAR_THREAD_TEST_CASE(test_load_balancer_shuffle_mode) {
  do_with_cql_env_thread([] (auto& e) {
    inet_address ip1("192.168.0.1");
    inet_address ip2("192.168.0.2");
    inet_address ip3("192.168.0.3");

    auto host1 = host_id(next_uuid());
    auto host2 = host_id(next_uuid());
    auto host3 = host_id(next_uuid());

    auto table1 = table_id(next_uuid());

    semaphore sem(1);
    shared_token_metadata stm([&sem] () noexcept { return get_units(sem, 1); }, locator::token_metadata::config{
        locator::topology::config{
            .this_endpoint = ip1,
            .local_dc_rack = locator::endpoint_dc_rack::default_location
        }
    });

    stm.mutate_token_metadata([&] (auto& tm) {
        tm.update_host_id(host1, ip1);
        tm.update_host_id(host2, ip2);
        tm.update_host_id(host3, ip3);
        tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, 1);
        tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, 1);
        tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, std::nullopt, 2);

        tablet_map tmap(4);
        std::optional<tablet_id> tid = tmap.first_tablet();
        for (int i = 0; i < 4; ++i) {
            tmap.set_tablet(*tid, tablet_info {
                    tablet_replica_set {
                            tablet_replica {host1, 0},
                            tablet_replica {host2, 0},
                    }
            });
            tid = tmap.next_tablet(*tid);
        }
        tablet_metadata tmeta;
        tmeta.set_tablet_map(table1, std::move(tmap));
        tm.set_tablets(std::move(tmeta));
        return make_ready_future<>();
    }).get();

    rebalance_tablets(e.get_tablet_allocator().local(), stm);

    BOOST_REQUIRE(e.get_tablet_allocator().local().balance_tablets(stm.get()).get0().empty());

    utils::get_local_injector().enable("tablet_allocator_shuffle");
    auto disable_injection = seastar::defer([&] {
        utils::get_local_injector().disable("tablet_allocator_shuffle");
    });

    BOOST_REQUIRE(!e.get_tablet_allocator().local().balance_tablets(stm.get()).get0().empty());
  }).get();
}
#endif

SEASTAR_THREAD_TEST_CASE(test_load_balancing_with_two_empty_nodes) {
  do_with_cql_env_thread([] (auto& e) {
    inet_address ip1("192.168.0.1");
    inet_address ip2("192.168.0.2");
    inet_address ip3("192.168.0.3");
    inet_address ip4("192.168.0.4");

    auto host1 = host_id(next_uuid());
    auto host2 = host_id(next_uuid());
    auto host3 = host_id(next_uuid());
    auto host4 = host_id(next_uuid());

    auto table1 = table_id(next_uuid());

    unsigned shard_count = 2;

    semaphore sem(1);
    shared_token_metadata stm([&sem] () noexcept { return get_units(sem, 1); }, locator::token_metadata::config{
        locator::topology::config{
            .this_endpoint = ip1,
            .local_dc_rack = locator::endpoint_dc_rack::default_location
        }
    });

    stm.mutate_token_metadata([&] (auto& tm) {
        tm.update_host_id(host1, ip1);
        tm.update_host_id(host2, ip2);
        tm.update_host_id(host3, ip3);
        tm.update_host_id(host4, ip4);
        tm.update_topology(ip1, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
        tm.update_topology(ip2, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
        tm.update_topology(ip3, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);
        tm.update_topology(ip4, locator::endpoint_dc_rack::default_location, std::nullopt, shard_count);

        tablet_map tmap(16);
        for (auto tid : tmap.tablet_ids()) {
            tmap.set_tablet(tid, tablet_info {
                tablet_replica_set {
                    tablet_replica {host1, tests::random::get_int<shard_id>(0, shard_count - 1)},
                    tablet_replica {host2, tests::random::get_int<shard_id>(0, shard_count - 1)},
                }
            });
        }
        tablet_metadata tmeta;
        tmeta.set_tablet_map(table1, std::move(tmap));
        tm.set_tablets(std::move(tmeta));
        return make_ready_future<>();
    }).get();

    rebalance_tablets(e.get_tablet_allocator().local(), stm);

    {
        load_sketch load(stm.get());
        load.populate().get();

        for (auto h : {host1, host2, host3, host4}) {
            testlog.debug("Checking host {}", h);
            BOOST_REQUIRE(load.get_avg_shard_load(h) == 4);
        }
    }
  }).get();
}

SEASTAR_THREAD_TEST_CASE(test_load_balancing_with_random_load) {
  do_with_cql_env_thread([] (auto& e) {
    const int n_hosts = 6;

    std::vector<host_id> hosts;
    for (int i = 0; i < n_hosts; ++i) {
        hosts.push_back(host_id(next_uuid()));
    }

    std::vector<endpoint_dc_rack> racks = {
        endpoint_dc_rack{ "dc1", "rack-1" },
        endpoint_dc_rack{ "dc1", "rack-2" }
    };

    for (int i = 0; i < 13; ++i) {
        std::unordered_map<sstring, std::vector<host_id>> hosts_by_rack;

        semaphore sem(1);
        shared_token_metadata stm([&sem]() noexcept { return get_units(sem, 1); }, locator::token_metadata::config {
                locator::topology::config {
                        .this_endpoint = inet_address("192.168.0.1"),
                        .local_dc_rack = racks[1]
                }
        });

        size_t total_tablet_count = 0;
        stm.mutate_token_metadata([&](auto& tm) {
            tablet_metadata tmeta;

            int i = 0;
            for (auto h : hosts) {
                auto ip = inet_address(format("192.168.0.{}", ++i));
                auto shard_count = 2;
                tm.update_host_id(h, ip);
                auto rack = racks[i % racks.size()];
                tm.update_topology(ip, rack, std::nullopt, shard_count);
                if (h != hosts[0]) {
                    // Leave the first host empty by making it invisible to allocation algorithm.
                    hosts_by_rack[rack.rack].push_back(h);
                }
            }

            size_t tablet_count_bits = 8;
            int rf = tests::random::get_int<shard_id>(2, 4);
            for (size_t log2_tablets = 0; log2_tablets < tablet_count_bits; ++log2_tablets) {
                if (tests::random::get_bool()) {
                    continue;
                }
                auto table = table_id(next_uuid());
                tablet_map tmap(1 << log2_tablets);
                for (auto tid : tmap.tablet_ids()) {
                    // Choose replicas randomly while loading racks evenly.
                    std::vector<host_id> replica_hosts;
                    for (int i = 0; i < rf; ++i) {
                        auto rack = racks[i % racks.size()];
                        auto& rack_hosts = hosts_by_rack[rack.rack];
                        while (true) {
                            auto candidate_host = rack_hosts[tests::random::get_int<shard_id>(0, rack_hosts.size() - 1)];
                            if (std::find(replica_hosts.begin(), replica_hosts.end(), candidate_host) == replica_hosts.end()) {
                                replica_hosts.push_back(candidate_host);
                                break;
                            }
                        }
                    }
                    tablet_replica_set replicas;
                    for (auto h : replica_hosts) {
                        auto shard_count = tm.get_topology().find_node(h)->get_shard_count();
                        auto shard = tests::random::get_int<shard_id>(0, shard_count - 1);
                        replicas.push_back(tablet_replica {h, shard});
                    }
                    tmap.set_tablet(tid, tablet_info {std::move(replicas)});
                }
                total_tablet_count += tmap.tablet_count();
                tmeta.set_tablet_map(table, std::move(tmap));
            }
            tm.set_tablets(std::move(tmeta));
            return make_ready_future<>();
        }).get();

        testlog.debug("tablet metadata: {}", stm.get()->tablets());
        testlog.info("Total tablet count: {}, hosts: {}", total_tablet_count, hosts.size());

        rebalance_tablets(e.get_tablet_allocator().local(), stm);

        {
            load_sketch load(stm.get());
            load.populate().get();

            min_max_tracker<unsigned> min_max_load;
            for (auto h: hosts) {
                auto l = load.get_avg_shard_load(h);
                testlog.info("Load on host {}: {}", h, l);
                min_max_load.update(l);
            }

            testlog.debug("tablet metadata: {}", stm.get()->tablets());
            testlog.debug("Min load: {}, max load: {}", min_max_load.min(), min_max_load.max());

//          FIXME: The algorithm cannot achieve balance in all cases yet, so we only check that it stops.
//          For example, if we have an overloaded node in one rack and target underloaded node in a different rack,
//          we won't be able to reduce the load gap by moving tablets between the two. We have to balance the overloaded
//          rack first, which is unconstrained.
//          Uncomment the following line when the algorithm is improved.
//          BOOST_REQUIRE(min_max_load.max() - min_max_load.min() <= 1);
        }
    }
  }).get();
}
