////////////////////////////////////////////////////////////////////////////
//
// Copyright 2019 Realm Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
////////////////////////////////////////////////////////////////////////////

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <util/test_file.hpp>
#include <util/test_utils.hpp>

#include <realm/db.hpp>
#include <realm/query_expression.hpp>

#include <realm/object-store/object_schema.hpp>
#include <realm/object-store/property.hpp>
#include <realm/object-store/results.hpp>
#include <realm/object-store/schema.hpp>
#include <realm/object-store/sectioned_results.hpp>
#include <realm/object-store/impl/realm_coordinator.hpp>

using namespace realm;

TEST_CASE("Benchmark results", "[benchmark][results]") {
    InMemoryTestFile config;
    config.schema = Schema{
        {"object",
         {
             {"value", PropertyType::Int},
             {"bool", PropertyType::Bool},
             {"data prop", PropertyType::Data},
             {"link", PropertyType::Object | PropertyType::Nullable, "object 2"},
             {"array", PropertyType::Object | PropertyType::Array, "object 2"},
         }},
        {"object 2",
         {
             {"value", PropertyType::Int},
             {"link", PropertyType::Object | PropertyType::Nullable, "object"},
         }},
    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto table2 = realm->read_group().get_table("class_object 2");
    Results r(realm, table);

    realm->begin_transaction();
    ObjKeys table_keys;
    ObjKeys table2_keys;
    table->create_objects(4, table_keys);
    table2->create_objects(4, table2_keys);
    ColKey col_link = table->get_column_key("link");
    ColKey col_value = table->get_column_key("value");
    ColKey col_link2 = table2->get_column_key("link");
    for (int i = 0; i < 4; ++i) {
        table->get_object(table_keys[i]).set_all((i + 2) % 4, bool(i % 2)).set(col_link, table2_keys[3 - i]);
        table2->get_object(table2_keys[i]).set_all((i + 1) % 4).set(col_link2, table_keys[i]);
    }
    realm->commit_transaction();
    /*
     | index | value | bool | link.value | link.link.value |
     |-------|-------|------|------------|-----------------|
     | 0     | 2     | 0    | 0          | 1               |
     | 1     | 3     | 1    | 3          | 0               |
     | 2     | 0     | 0    | 2          | 3               |
     | 3     | 1     | 1    | 1          | 2               |
     */

#define REQUIRE_ORDER(sort, ...)                                                                                     \
    do {                                                                                                             \
        ObjKeys expected({__VA_ARGS__});                                                                             \
        auto results = sort;                                                                                         \
        REQUIRE(results.size() == expected.size());                                                                  \
        for (size_t i = 0; i < expected.size(); ++i)                                                                 \
            REQUIRE(results.get(i).get_key() == expected[i]);                                                        \
    } while (0)

    SECTION("basics") {
        REQUIRE(r.filter(Query(table->where().less(col_value, 2))).size() == 2);
        BENCHMARK("basic filter") {
            return r.filter(Query(table->where().less(col_value, 2))).size();
        };

        REQUIRE_ORDER((r.sort({{"value", true}})), 2, 3, 0, 1);
        BENCHMARK("sort simple ints") {
            return r.sort({{"value", true}});
        };

        REQUIRE_ORDER((r.sort({{"bool", true}, {"value", true}})), 2, 0, 3, 1);
        BENCHMARK("sort over two properties") {
            return r.sort({{"bool", true}, {"value", true}});
        };

        REQUIRE_ORDER((r.sort({{"link.value", true}})), 0, 3, 2, 1);
        BENCHMARK("sort over link") {
            return r.sort({{"link.value", true}});
        };

        REQUIRE_ORDER((r.sort({{"link.link.value", true}})), 1, 0, 3, 2);
        BENCHMARK("sort over two links") {
            return r.sort({{"link.link.value", true}});
        };

        REQUIRE(r.distinct({{"value"}}).size() == 4);
        BENCHMARK("distinct ints") {
            return r.distinct({{"value"}});
        };

        REQUIRE(r.distinct({{"bool"}}).size() == 2);
        BENCHMARK("distinct bool") {
            return r.distinct({{"bool"}});
        };
    }

    SECTION("iteration") {
        const int additional_row_count = 10000;
        realm->begin_transaction();
        table->create_objects(additional_row_count, table_keys);
        for (int i = 0; i < additional_row_count; ++i)
            table->get_object(table_keys[i]).set_all((i + 2) % 4, bool(i % 2));
        realm->commit_transaction();

        BENCHMARK("Table forwards") {
            for (size_t i = 0, size = r.size(); i < size; ++i) {
                r.get<Obj>(i);
            }
        };

        BENCHMARK("Table reverse") {
            for (size_t i = 0, size = r.size(); i < size; ++i) {
                r.get<Obj>(size - i - 1);
            }
        };

        auto tv = r.snapshot();
        BENCHMARK("TableView forwards") {
            for (size_t i = 0, size = r.size(); i < size; ++i) {
                tv.get<Obj>(i);
            }
        };

        BENCHMARK("TableView reverse") {
            for (size_t i = 0, size = r.size(); i < size; ++i) {
                tv.get<Obj>(size - i - 1);
            }
        };
    }
}

TEST_CASE("Benchmark results notifier", "[benchmark][results]") {
    InMemoryTestFile config;

    SECTION("100 strongly connected tables") {
        std::vector<ObjectSchema> schema;
        for (int i = 0; i < 100; ++i) {
            ObjectSchema os;
            os.name = util::format("table %1", i);
            os.persisted_properties = {{"value", PropertyType::Int}};
            for (int j = 0; j < 100; ++j) {
                os.persisted_properties.push_back({util::format("column %1", j),
                                                   PropertyType::Object | PropertyType::Nullable,
                                                   util::format("table %1", j)});
            }
            schema.push_back(std::move(os));
        }
        config.schema = Schema{schema};
        auto realm = Realm::get_shared_realm(config);
        auto table_0 = realm->read_group().get_table("class_table 0");

        BENCHMARK("create notifier") {
            Results r(realm, table_0->where());
            r.evaluate_query_if_needed(true);
        };
    }

    config.automatic_change_notifications = false;
    static const int table_count = 6;
    static const int column_count = 50;
    static const int object_count = 50;

    auto make_schema = [](PropertyType link_type) {
        std::vector<ObjectSchema> schema;
        for (int i = 0; i < table_count; ++i) {
            ObjectSchema os;
            os.name = util::format("table %1", i);
            os.persisted_properties = {{"value", PropertyType::Int}};
            for (int j = 0; j < column_count; ++j) {
                os.persisted_properties.push_back(
                    {util::format("column %1", j), link_type, util::format("table %1", (i + 1) % table_count)});
            }
            schema.push_back(std::move(os));
        }
        return schema;
    };

    SECTION("chained tables using links") {
        std::vector<ObjectSchema> schema = make_schema(PropertyType::Object | PropertyType::Nullable);
        config.schema = Schema{schema};
        auto realm = Realm::get_shared_realm(config);

        auto& group = realm->read_group();
        std::vector<TableRef> tables;
        for (int i = 0; i < table_count; ++i) {
            tables.push_back(group.get_table(util::format("class_table %1", i)));
        }

        realm->begin_transaction();
        for (int i = 0; i < table_count; ++i) {
            for (int j = 0; j < object_count; ++j) {
                tables[i]->create_object();
            }
        }
        for (int i = 0; i < table_count; ++i) {
            auto target_table = tables[(i + 1) % table_count];
            for (int j = 0; j < object_count; ++j) {
                auto obj = tables[i]->get_object(j);
                for (int k = 0; k < column_count; ++k) {
                    obj.set(util::format("column %1", k), target_table->get_object((j + k) % object_count).get_key());
                }
            }
        }
        realm->commit_transaction();

        Results r(realm, tables[0]->where());
        auto token = r.add_notification_callback([](CollectionChangeSet) {});
        auto& coordinator = *_impl::RealmCoordinator::get_coordinator(config.path);
        coordinator.on_change();

        BENCHMARK("modify at depth 0", iteration) {
            realm->begin_transaction();
            for (int i = 0; i < object_count; ++i) {
                tables[0]->get_object(i).set_all(iteration);
            }
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 1", iteration) {
            realm->begin_transaction();
            tables[1]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 2", iteration) {
            realm->begin_transaction();
            tables[2]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 3", iteration) {
            realm->begin_transaction();
            tables[3]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };
    }

    SECTION("chained tables using lists") {
        std::vector<ObjectSchema> schema = make_schema(PropertyType::Object | PropertyType::Array);
        config.schema = Schema{schema};
        auto realm = Realm::get_shared_realm(config);

        auto& group = realm->read_group();
        std::vector<TableRef> tables;
        for (int i = 0; i < table_count; ++i) {
            tables.push_back(group.get_table(util::format("class_table %1", i)));
        }

        realm->begin_transaction();
        for (int i = 0; i < table_count; ++i) {
            for (int j = 0; j < object_count; ++j) {
                tables[i]->create_object();
            }
        }
        for (int i = 0; i < table_count; ++i) {
            auto target_table = tables[(i + 1) % table_count];
            for (int j = 0; j < object_count; ++j) {
                auto obj = tables[i]->get_object(j);
                for (int k = 0; k < column_count; ++k) {
                    obj.get_linklist(util::format("column %1", k))
                        .add(target_table->get_object((j + k) % object_count).get_key());
                }
            }
        }
        realm->commit_transaction();

        Results r(realm, tables[0]->where());
        auto token = r.add_notification_callback([](CollectionChangeSet) {});
        auto& coordinator = *_impl::RealmCoordinator::get_coordinator(config.path);
        coordinator.on_change();

        BENCHMARK("modify at depth 0", iteration) {
            realm->begin_transaction();
            for (int i = 0; i < object_count; ++i) {
                tables[0]->get_object(i).set_all(iteration);
            }
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 1", iteration) {
            realm->begin_transaction();
            tables[1]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 2", iteration) {
            realm->begin_transaction();
            tables[2]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };

        BENCHMARK("modify at depth 3", iteration) {
            realm->begin_transaction();
            tables[3]->get_object(0).set_all(iteration);
            realm->commit_transaction();
            coordinator.on_change();
        };
    }
}

TEST_CASE("aggregates", "[benchmark][aggregate]") {
    InMemoryTestFile config;
    config.schema = Schema{
        {"object",
         {
             {"value", PropertyType::Int},
         }},
        {"link",
         {
             {"list", PropertyType::Object | PropertyType::Array, "object"},
             {"set", PropertyType::Object | PropertyType::Set, "object"},
             {"dictionary", PropertyType::Object | PropertyType::Dictionary | PropertyType::Nullable, "object"},
         }},
        {"primitive",
         {
             {"list", PropertyType::Int | PropertyType::Array},
             {"set", PropertyType::Int | PropertyType::Set},
             {"dictionary", PropertyType::Int | PropertyType::Dictionary},
         }},
    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto link_table = realm->read_group().get_table("class_link");
    auto prim_table = realm->read_group().get_table("class_primitive");

    realm->begin_transaction();

    auto link_obj = link_table->create_object();
    auto obj_list = link_obj.get_linklist("list");
    auto obj_set = link_obj.get_linkset(link_obj.get_table()->get_column_key("set"));
    auto obj_dict = link_obj.get_dictionary("dictionary");

    auto prim_obj = prim_table->create_object();
    auto int_list = prim_obj.get_list<int64_t>("list");
    auto int_set = prim_obj.get_set<int64_t>("set");
    auto int_dict = prim_obj.get_dictionary("dictionary");

    const auto value_count = GENERATE(0, 100, 1'000'000);
    for (int i = 0; i < value_count; ++i) {
        auto key = table->create_object().set_all(int64_t(1)).get_key();
        obj_list.add(key);
        obj_set.insert(key);
        obj_dict.insert(std::to_string(i), key);
        int_list.add(1);
        int_set.insert(i);
        int_dict.insert(std::to_string(i), 1);
    }
    realm->commit_transaction();

    ColKey col = table->get_column_key("value");
    Results table_results(realm, table);
    Results query_results(realm, table->where());
    Results tv_results(realm, table->where());
    tv_results.evaluate_query_if_needed();

    BENCHMARK("table") {
        return table_results.sum(col);
    };
    BENCHMARK("query") {
        return query_results.sum(col);
    };
    BENCHMARK("tableview") {
        return tv_results.sum(col);
    };
    BENCHMARK("object list") {
        return List(realm, obj_list).sum(col);
    };
    BENCHMARK("object set") {
        return object_store::Set(realm, obj_set).sum(col);
    };
    BENCHMARK("object dictionary") {
        return object_store::Dictionary(realm, obj_dict).sum(col);
    };
    BENCHMARK("int list") {
        return List(realm, int_list).sum();
    };
    BENCHMARK("int set") {
        return object_store::Set(realm, int_set).sum();
    };
    BENCHMARK("int dictionary") {
        return object_store::Dictionary(realm, int_dict).sum();
    };
}

TEST_CASE("Benchmark sectioned results", "[benchmark][results]") {
    InMemoryTestFile config;
    config.automatic_change_notifications = false;
    config.schema = Schema{{"object", {{"value", PropertyType::Int}}}};

    auto realm = Realm::get_shared_realm(config);
    auto& coordinator = *_impl::RealmCoordinator::get_coordinator(config.path);
    auto table = realm->read_group().get_table("class_object");
    auto col = table->get_column_key("value");

    realm->begin_transaction();
    for (int64_t i = 0; i < 100'000; ++i) {
        table->create_object().set_all(i);
    }
    realm->commit_transaction();

    size_t section_count = GENERATE(1, 10, 1000, 10000);
    auto key_fn = [&](Mixed value, const std::shared_ptr<Realm>&) -> Mixed {
        return table->get_object(value.get_link().get_obj_key()).get<int64_t>(col) % int64_t(section_count);
    };

    BENCHMARK("create and get section count") {
        auto size = Results(realm, table).sectioned_results(key_fn).size();
        REQUIRE(size == section_count);
    };

    BENCHMARK_ADVANCED("iterate directly")(Catch::Benchmark::Chronometer meter)
    {
        auto results = Results(realm, table).sectioned_results(key_fn);
        static_cast<void>(results.size()); // evaluate sections
        meter.measure([&] {
            for (size_t i = 0, size = results.size(); i < size; ++i) {
                auto section = results[i];
                for (size_t j = 0, size = section.size(); j < size; ++j) {
                    static_cast<void>(section[j]);
                }
            }
        });
    };

    BENCHMARK_ADVANCED("iterate over a snapshot")(Catch::Benchmark::Chronometer meter)
    {
        auto results = Results(realm, table).sectioned_results(key_fn);
        static_cast<void>(results.size()); // evaluate sections
        meter.measure([&] {
            auto snapshot = results.snapshot();
            for (size_t i = 0, size = snapshot.size(); i < size; ++i) {
                auto section = snapshot[i];
                for (size_t j = 0, size = section.size(); j < size; ++j) {
                    static_cast<void>(section[j]);
                }
            }
        });
    };

    BENCHMARK_ADVANCED("change notification")(Catch::Benchmark::Chronometer meter)
    {
        auto results = Results(realm, table).sectioned_results(key_fn);
        auto token = results.add_notification_callback([](auto&&) {});
        coordinator.on_change();
        realm->notify();

        auto col = table->get_column_key("value");
        meter.measure([&] {
            realm->begin_transaction();
            for (auto& obj : *table) {
                obj.set(col, obj.get<int64_t>(col));
            }
            realm->commit_transaction();
            coordinator.on_change();
            realm->notify();
        });
    };

    BENCHMARK_ADVANCED("single section change notification")(Catch::Benchmark::Chronometer meter)
    {
        auto results = Results(realm, table).sectioned_results(key_fn);
        auto section = results[section_count > 5 ? 5 : 0];
        auto token = section.add_notification_callback([](auto&&) {});
        coordinator.on_change();
        realm->notify();

        auto col = table->get_column_key("value");
        meter.measure([&] {
            realm->begin_transaction();
            for (auto& obj : *table) {
                obj.set(col, obj.get<int64_t>(col));
            }
            realm->commit_transaction();
            coordinator.on_change();
            realm->notify();
        });
    };
}
