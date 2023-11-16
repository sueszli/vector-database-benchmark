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

#include <catch2/catch_all.hpp>

#include "util/event_loop.hpp"
#include "util/test_file.hpp"
#include "util/test_utils.hpp"

#include <realm/object-store/binding_context.hpp>
#include <realm/object-store/object_accessor.hpp>
#include <realm/object-store/object_schema.hpp>
#include <realm/object-store/object_store.hpp>
#include <realm/object-store/property.hpp>
#include <realm/object-store/results.hpp>
#include <realm/object-store/schema.hpp>
#include <realm/object-store/thread_safe_reference.hpp>

#include <realm/object-store/impl/object_accessor_impl.hpp>
#include <realm/object-store/impl/realm_coordinator.hpp>

#include <realm/db.hpp>
#include <realm/query_expression.hpp>

#if REALM_ENABLE_SYNC
#include <realm/object-store/sync/async_open_task.hpp>
#endif

#include <realm/util/scope_exit.hpp>

using namespace realm;
using util::any_cast;

TEST_CASE("Construct frozen Realm", "[frozen]") {
    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{
        {"object", {{"value", PropertyType::Int}}},
    };

    SECTION("Create frozen Realm directly") {
        auto realm = Realm::get_shared_realm(config);
        realm->read_group();
        auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());
        REQUIRE(frozen_realm->is_frozen());
        REQUIRE(realm->read_transaction_version() == *frozen_realm->current_transaction_version());
    }
}

TEST_CASE("Freeze Realm", "[frozen]") {
    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{
        {"object", {{"value", PropertyType::Int}}},
    };

    auto realm = Realm::get_shared_realm(config);
    realm->read_group();
    auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());

    SECTION("is_frozen") {
        REQUIRE(frozen_realm->is_frozen());
    }

    SECTION("refresh() returns false") {
        REQUIRE(!frozen_realm->refresh());
    }

    SECTION("wait_for_change() returns false") {
        REQUIRE(!frozen_realm->wait_for_change());
    }

    SECTION("auto_refresh") {
        REQUIRE(!frozen_realm->auto_refresh());
        REQUIRE_EXCEPTION(frozen_realm->set_auto_refresh(true), WrongTransactionState,
                          "Auto-refresh cannot be enabled for frozen Realms.");
        REQUIRE(!frozen_realm->auto_refresh());
    }

    SECTION("begin_transaction() throws") {
        REQUIRE_EXCEPTION(frozen_realm->begin_transaction(), WrongTransactionState,
                          "Can't perform transactions on a frozen Realm");
    }

    SECTION("can call methods on another thread") {
        JoiningThread thread([&] {
            // Smoke-test
            REQUIRE_NOTHROW(frozen_realm->write_copy());
            REQUIRE_NOTHROW(frozen_realm->read_transaction_version());
        });
    }

    SECTION("release all locks") {
        frozen_realm->close();
        realm->close();
        REQUIRE(DB::call_with_lock(config.path, [](auto) {}));
    }
}

TEST_CASE("Freeze Results", "[frozen]") {
    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{{"object",
                            {{"value", PropertyType::Int},
                             {"int_array", PropertyType::Array | PropertyType::Int},
                             {"int_dict", PropertyType::Dictionary | PropertyType::Int},
                             {"object_array", PropertyType::Array | PropertyType::Object, "linked to object"}}},
                           {"linked to object", {{"value", PropertyType::Int}}}

    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto linked_table = realm->read_group().get_table("class_linked to object");
    auto value_col = table->get_column_key("value");
    auto object_link_col = table->get_column_key("object_array");
    auto int_list_col = table->get_column_key("int_array");
    auto int_dict_col = table->get_column_key("int_dict");
    auto linked_object_value_col = linked_table->get_column_key("value");

    auto create_object = [&](int value, bool with_links = false) {
        Obj obj = table->create_object();
        obj.set(value_col, value);
        std::shared_ptr<LnkLst> object_link_view = obj.get_linklist_ptr(object_link_col);

        auto int_list = List(realm, obj, int_list_col);
        object_store::Dictionary int_dict(realm, obj, int_dict_col);

        for (int j = 0; j < 5; ++j) {
            if (with_links) {
                auto child_obj = linked_table->create_object();
                child_obj.set(linked_object_value_col, j + 10);
                object_link_view->add(child_obj.get_key());
            }
            int_list.add(static_cast<Int>(j + 42));
            std::string key = "Key" + util::to_string(j);
            int_dict.insert(key, value);
        }
        return obj;
    };

    auto write = [&](const std::function<void()>& fn) {
        realm->begin_transaction();
        fn();
        realm->commit_transaction();
    };

    write([&]() {
        for (int i = 0; i < 8; ++i)
            create_object(i + 2, true);
    });

    auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());

#define VERIFY_VALID_RESULTS(results, realm)                                                                         \
    REQUIRE(results.is_valid());                                                                                     \
    if (results.size() == 0) {                                                                                       \
        REQUIRE_EXCEPTION(results.get_any(0), OutOfBounds,                                                           \
                          "Requested index 0 calling get_any() on Results when empty");                              \
    }                                                                                                                \
    else {                                                                                                           \
        REQUIRE_FALSE(results.get_any(0).is_null());                                                                 \
    }                                                                                                                \
    REQUIRE(results.freeze(realm).is_valid());

#define VERIFY_INVALID_RESULTS(results, realm, exception, message)                                                   \
    REQUIRE_EXCEPTION(results.freeze(realm), exception, message);                                                    \
    REQUIRE_FALSE(results.is_valid());                                                                               \
    REQUIRE_EXCEPTION(results.size(), exception, message);                                                           \
    REQUIRE_EXCEPTION(results.get_any(0).is_null(), exception, message);

#define VERIFY_STALE_RESULTS(results, realm)                                                                         \
    VERIFY_INVALID_RESULTS(results, realm, StaleAccessor, "Access to invalidated Results objects")

    SECTION("is_frozen") {
        Results results(realm, table);
        Results frozen_results = results.freeze(frozen_realm);
        REQUIRE(!results.is_frozen());
        REQUIRE(frozen_results.is_frozen());
        JoiningThread thread([&] {
            // Check is_frozen across threads
            REQUIRE(!results.is_frozen());
            REQUIRE(frozen_results.is_frozen());
        });
    }

    SECTION("add_notification throws") {
        Results results(realm, table);
        Results frozen_results = results.freeze(frozen_realm);
        REQUIRE_EXCEPTION(frozen_results.add_notification_callback([&](CollectionChangeSet) {}),
                          WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
    }

    SECTION("Result constructor - Empty") {
        Results res = Results();
        REQUIRE(res.is_frozen()); // All Results are considered frozen
        Results frozen_res = res.freeze(frozen_realm);
        JoiningThread thread([&] {
            REQUIRE(frozen_res.is_frozen());
            REQUIRE(frozen_res.size() == 0);
            VERIFY_VALID_RESULTS(frozen_res, frozen_realm);
        });
    }

    SECTION("Result constructor - Table") {
        Results results = Results(frozen_realm, frozen_realm->read_group().get_table("class_object"));
        Results frozen_res = results.freeze(frozen_realm);
        JoiningThread thread([&] {
            auto obj = frozen_res.get(0);
            auto any = frozen_res.get_any(0);
            REQUIRE(obj.is_valid());
            REQUIRE(any.get_link() == obj.get_link());
            REQUIRE(Object(frozen_realm, obj).is_frozen());
            REQUIRE(frozen_res.get(0).get<int64_t>(value_col) == 2);
            REQUIRE(frozen_res.first()->get<int64_t>(value_col) == 2);

            VERIFY_VALID_RESULTS(results, frozen_realm);
            VERIFY_VALID_RESULTS(frozen_res, frozen_realm);
        });
    }

    SECTION("Result constructor - Primitive list") {
        const List list = List(frozen_realm, table->get_object(0), int_list_col);
        auto list_results = list.as_results();

        SECTION("unsorted") {
            Results frozen_res = list_results.freeze(frozen_realm);
            JoiningThread thread1([&] {
                REQUIRE(frozen_res.is_frozen());
                REQUIRE(frozen_res.size() == 5);
                REQUIRE(frozen_res.get<Int>(0) == 42);
            });
        }

        SECTION("sorted") {
            Results sorted_frozen_res = list.sort({{"self", false}}).freeze(frozen_realm);
            JoiningThread thread2([&] {
                REQUIRE(sorted_frozen_res.is_frozen());
                REQUIRE(sorted_frozen_res.size() == 5);
                REQUIRE(sorted_frozen_res.get<Int>(0) == 46);
            });
        }
    }

    SECTION("Result constructor - Dictionary") {
        const object_store::Dictionary dict(frozen_realm, table->get_object(0), int_dict_col);
        auto dict_results = dict.as_results();

        Results frozen_res = dict_results.freeze(frozen_realm);
        JoiningThread thread1([&] {
            REQUIRE(frozen_res.is_frozen());
            REQUIRE(frozen_res.size() == 5);
            REQUIRE(frozen_res.get<Int>(0) == 2);
        });

        /* FIXME causes ThreadSanitizer error in Catch2?
        write([&]() {
            table->remove_object(table->get_object(0).get_key());
        });
        VERIFY_STALE_RESULTS(dict_results, realm);
        */
    }

    SECTION("Result constructor - Query") {
        Query q = table->column<Int>(value_col) > 0;
        DescriptorOrdering ordering;
        ordering.append_sort(SortDescriptor({{value_col}}, {false}));
        Results query_results(realm, std::move(q), ordering);
        Results frozen_res = query_results.freeze(frozen_realm);
        JoiningThread thread([&] {
            auto obj = frozen_res.get(0);
            auto any = frozen_res.get_any(0);
            REQUIRE(obj.is_valid());
            REQUIRE(any.get_link() == obj.get_link());
            REQUIRE(Object(frozen_realm, obj).is_frozen());
            REQUIRE(frozen_res.get(0).get<Int>(value_col) == 9);
            REQUIRE(frozen_res.first()->get<Int>(value_col) == 9);
        });

        /* FIXME causes ThreadSanitizer error in Catch2?
        write([&] {
            realm->read_group().remove_table(table->get_name());
        });
        VERIFY_STALE_RESULTS(query_results, realm);
        */
    }

    SECTION("Result constructor - TableView") {
        Query q = table->column<Int>(value_col) > 2;
        DescriptorOrdering ordering;
        ordering.append_sort(SortDescriptor({{value_col}}, {false}));
        TableView tv = q.find_all();
        Results query_results(realm, tv, ordering);
        query_results.get(0);
        Results frozen_res = query_results.freeze(frozen_realm);
        JoiningThread thread([&] {
            auto obj = frozen_res.get(0);
            auto any = frozen_res.get_any(0);
            REQUIRE(any.get_link() == obj.get_link());
            REQUIRE(frozen_res.is_frozen());
            REQUIRE(obj.get<int64_t>(value_col) == 3);
            REQUIRE(frozen_res.first()->get<int64_t>(value_col) == 3);
        });
    }

    SECTION("Result constructor - LinkList") {
        Results results(realm, table);
        Obj obj = results.get(0);
        std::shared_ptr<LnkLst> link_list = obj.get_linklist_ptr(object_link_col);
        Results res = Results(realm, link_list);
        Results frozen_res = res.freeze(frozen_realm);
        JoiningThread thread([&] {
            REQUIRE(frozen_res.is_frozen());
            REQUIRE(frozen_res.size() == 5);
            auto any = frozen_res.get_any(0);
            Object o = Object(frozen_realm, any.get_link());
            REQUIRE(o.is_frozen());
            REQUIRE(o.get_column_value<Int>("value") == 10);
            REQUIRE(frozen_res.get(0).get<Int>(linked_object_value_col) == 10);
            REQUIRE(frozen_res.first()->get<Int>(linked_object_value_col) == 10);
        });
    }

    SECTION("release all locks") {
        frozen_realm->close();
        realm->close();
        REQUIRE(DB::call_with_lock(config.path, [](auto) {}));
    }

    SECTION("Results after source remove") {
        Results results;

        SECTION("Results on collection") {
            Obj obj;
            write([&]() {
                obj = create_object(42, true);
            });
            auto key = obj.get_key();

            SECTION("Dictionary") {
                results = Results(realm, obj.get_dictionary_ptr(int_dict_col));
                write([&]() {
                    table->remove_object(key);
                });
                // If Results is based on collection of primitives, the removal of
                // the collection should invalidate Results.
                VERIFY_STALE_RESULTS(results, realm);
            }

            SECTION("Links") {
                results = Results(realm, obj.get_linklist_ptr(object_link_col));
                auto snapshot = results.snapshot();
                // If Results is based on collection of objects, the removal of
                // the collection should not invalidate Results as the table still exists.
                write([&]() {
                    table->remove_object(key);
                    // Snapshot should not be affected by the removed collection
                    REQUIRE(snapshot.size() == 5);
                });
                REQUIRE(results.is_valid());
                REQUIRE(results.size() == 0);
                auto frozen = results.freeze(realm);
                REQUIRE(frozen.is_valid());
                REQUIRE(frozen.size() == 0);
            }
        }

        SECTION("Results on table") {
            results = Results(realm, table);
            write([&]() {
                realm->read_group().remove_table(table->get_key());
            });
            VERIFY_STALE_RESULTS(results, realm);
        }

        // FIXME? the test itself passes but crashes on teardown in notifier thread
        //       with realm::NoSuchTable on Query constructor through import_copy_of
        /* SECTION("Results on query") {
            results = Results(realm, table->column<Int>(value_col) > 0, DescriptorOrdering());
            do_remove = [&] {
                realm->read_group().remove_table(table->get_key());
            };
        } */
    }
}

TEST_CASE("Freeze List", "[frozen]") {

    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{{"object",
                            {{"value", PropertyType::Int},
                             {"int_array", PropertyType::Array | PropertyType::Int},
                             {"object_array", PropertyType::Array | PropertyType::Object, "linked to object"}}},
                           {"linked to object", {{"value", PropertyType::Int}}}

    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto linked_table = realm->read_group().get_table("class_linked to object");
    auto value_col = table->get_column_key("value");
    auto object_link_col = table->get_column_key("object_array");
    auto int_link_col = table->get_column_key("int_array");
    auto linked_object_value_col = linked_table->get_column_key("value");

    realm->begin_transaction();
    Obj obj = table->create_object();
    obj.set(value_col, 100);
    std::shared_ptr<LnkLst> object_link_view = obj.get_linklist_ptr(object_link_col);
    auto int_list = List(realm, obj, int_link_col);
    for (int j = 0; j < 5; ++j) {
        auto child_obj = linked_table->create_object();
        child_obj.set(linked_object_value_col, j + 10);
        object_link_view->add(child_obj.get_key());
        int_list.add(static_cast<Int>(j + 42));
    }
    realm->commit_transaction();

    Results results(realm, table);
    auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());

    std::shared_ptr<LnkLst> link_list = results.get(0).get_linklist_ptr(object_link_col);
    List frozen_link_list = List(realm, *link_list).freeze(frozen_realm);
    List frozen_primitive_list = List(realm, table->get_object(0), int_link_col).freeze(frozen_realm);

    SECTION("is_frozen") {
        REQUIRE(frozen_primitive_list.is_frozen());
        REQUIRE(frozen_link_list.is_frozen());
        JoiningThread thread([&] {
            REQUIRE(frozen_primitive_list.is_frozen());
            REQUIRE(frozen_link_list.is_frozen());
        });
    }

    SECTION("add_notification throws") {
        REQUIRE_EXCEPTION(frozen_link_list.add_notification_callback([&](CollectionChangeSet) {}),
                          WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
        REQUIRE_EXCEPTION(frozen_primitive_list.add_notification_callback([&](CollectionChangeSet) {}),
                          WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
    }

    SECTION("read across threads") {
        JoiningThread thread([&] {
            REQUIRE(frozen_primitive_list.size() == 5);
            REQUIRE(frozen_link_list.size() == 5);
            REQUIRE(frozen_primitive_list.get<Int>(0) == 42);
            REQUIRE(frozen_link_list.get(0).get<Int>(linked_object_value_col) == 10);
            REQUIRE(frozen_primitive_list.get<Int>(0) == 42);
            REQUIRE(frozen_link_list.get(0).get<Int>(linked_object_value_col) == 10);
        });
    }

    SECTION("release all locks") {
        frozen_realm->close();
        realm->close();
        REQUIRE(DB::call_with_lock(config.path, [](auto) {}));
    }
}

TEST_CASE("Reclaim Frozen", "[frozen]") {

#ifdef REALM_DEBUG
    constexpr int num_pending_transactions = 10;
    constexpr int num_iterations = 100;
    constexpr int num_objects = 5;
#else
    constexpr int num_pending_transactions = 100;
    constexpr int num_iterations = 10000;
    constexpr int num_objects = 20;
#endif
    constexpr int num_checks_pr_trans = 10;
    constexpr int num_trans_forgotten_rapidly = 5;
    struct Entry {
        SharedRealm realm;
        Object o;
        ObjKey link;
        int64_t linked_value;
        int64_t value;
        NotificationToken token;
    };
    std::vector<Entry> refs;
    refs.resize(num_pending_transactions);
    TestFile config;

    config.schema_version = 1;
    config.automatic_change_notifications = true;
    config.cache = false;
    config.schema = Schema{
        {"table", {{"value", PropertyType::Int}, {"link", PropertyType::Object | PropertyType::Nullable, "table"}}}};
    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_table");
    auto table_key = table->get_key();
    auto col = table->get_column_key("value");
    auto link_col = table->get_column_key("link");
    realm->begin_transaction();
    for (int j = 0; j < num_objects; ++j) {
        auto o = table->create_object(ObjKey(j));
        o.set(col, j);
        o.set(link_col, o.get_key());
    }
    realm->commit_transaction();
    int notifications = 0;
    for (int j = 0; j < num_iterations; ++j) {

        // pick a random earlier transaction
        int trans_number = (unsigned)random_int() % num_pending_transactions;
        auto& entry = refs[trans_number];

        // refresh chosen transaction so as to trigger notifications.
        if (entry.realm && !entry.realm->is_frozen()) {
            auto& r = entry.realm;
            REALM_ASSERT(r->is_in_read_transaction());
            auto before = r->current_transaction_version();
            r->refresh();
            auto after = r->current_transaction_version();
            REALM_ASSERT(before != after);
        }

        // set up and save a new realm for later refresh, replacing the old one
        // which we refreshed above
        int should_freeze = (random_int() % 5) != 0; // freeze 80%
        auto realm2 = should_freeze ? realm->freeze() : Realm::get_shared_realm(config);
        entry.realm = realm2;
        int key = (unsigned)random_int() % num_objects;
        auto table2 = realm2->read_group().get_table(table_key);
        auto& o = entry.o;
        o = Object(realm2, table2->get_object(key));
        entry.value = o.get_obj().get<Int>(col);
        entry.link = o.get_obj().get<ObjKey>(link_col);
        auto linked = table2->get_object(entry.link);
        entry.linked_value = linked.get<Int>(col);
        // add a dummy notification callback to later exercise the notification machinery
        if (!entry.realm->is_frozen()) {
            entry.token = o.add_notification_callback([&](CollectionChangeSet) {
                ++notifications;
            });
        }
        // create a number of new transactions.....
        for (int i = 0; i < num_trans_forgotten_rapidly; ++i) {
            realm->begin_transaction();
            auto key = ObjKey((unsigned)random_int() % num_objects);
            auto o = table->get_object(key);
            o.set(col, o.get<Int>(col) + j + 42);
            int link = (unsigned)random_int() % num_objects;
            o.set(link_col, ObjKey(link));
            realm->commit_transaction();
        }
        // verify a number of randomly selected saved transactions
        for (int k = 0; k < num_checks_pr_trans; ++k) {
            auto& entry = refs[(unsigned)random_int() % num_pending_transactions];
            if (entry.realm) {
                CHECK(entry.value == entry.o.get_obj().get<Int>(col));
                auto link = entry.o.get_obj().get<ObjKey>(link_col);
                CHECK(link == entry.link);
                auto table = entry.realm->read_group().get_table(table_key);
                auto linked_value = table->get_object(link).get<Int>(col);
                CHECK(entry.linked_value == linked_value);
            }
        }
    }
    // captured.reset();
    realm->begin_transaction();
    realm->commit_transaction();
    realm->begin_transaction();
    realm->commit_transaction();
    refs.clear();
    realm->begin_transaction();
    realm->commit_transaction();
    realm->begin_transaction();
    realm->commit_transaction();
}

TEST_CASE("Freeze Object", "[frozen]") {

    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{{"object",
                            {{"value", PropertyType::Int},
                             {"int_array", PropertyType::Array | PropertyType::Int},
                             {"object_array", PropertyType::Array | PropertyType::Object, "linked to object"}}},
                           {"linked to object", {{"value", PropertyType::Int}}}

    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto linked_table = realm->read_group().get_table("class_linked to object");
    auto value_col = table->get_column_key("value");
    auto object_link_col = table->get_column_key("object_array");
    auto int_link_col = table->get_column_key("int_array");
    auto linked_object_value_col = linked_table->get_column_key("value");

    realm->begin_transaction();
    Obj obj = table->create_object();
    obj.set(value_col, 100);
    std::shared_ptr<LnkLst> object_link_view = obj.get_linklist_ptr(object_link_col);
    auto int_list = List(realm, obj, int_link_col);
    for (int j = 0; j < 5; ++j) {
        auto child_obj = linked_table->create_object();
        child_obj.set(linked_object_value_col, j + 10);
        object_link_view->add(child_obj.get_key());
        int_list.add(static_cast<Int>(j + 42));
    }
    realm->commit_transaction();

    Results results(realm, table);
    auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());
    Object frozen_obj = Object(realm, table->get_object(0)).freeze(frozen_realm);
    CppContext ctx(frozen_realm);

    SECTION("is_frozen") {
        REQUIRE(frozen_obj.is_frozen());
    }

    SECTION("add_notification throws") {
        REQUIRE_EXCEPTION(frozen_obj.add_notification_callback([&](CollectionChangeSet) {}), WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
    }

    SECTION("read across threads") {
        JoiningThread thread([&] {
            REQUIRE(frozen_obj.is_valid());
            REQUIRE(util::any_cast<Int>(frozen_obj.get_property_value<std::any>(ctx, "value")) == 100);
            auto object_list = util::any_cast<List&&>(frozen_obj.get_property_value<std::any>(ctx, "object_array"));
            REQUIRE(object_list.is_frozen());
            REQUIRE(object_list.is_valid());
            REQUIRE(object_list.get(0).get<Int>(linked_object_value_col) == 10);
        });
    }

    SECTION("release all locks") {
        frozen_realm->close();
        realm->close();
        REQUIRE(DB::call_with_lock(config.path, [](auto) {}));
    }
}

TEST_CASE("Freeze dictionary", "[frozen]") {

    TestFile config;
    config.schema_version = 1;
    config.schema = Schema{
        {"object",
         {{"value", PropertyType::Int},
          {"integers", PropertyType::Dictionary | PropertyType::Int},
          {"links", PropertyType::Dictionary | PropertyType::Object | PropertyType::Nullable, "linked to object"}}},
        {"linked to object", {{"value", PropertyType::Int}}}

    };

    auto realm = Realm::get_shared_realm(config);
    auto table = realm->read_group().get_table("class_object");
    auto linked_table = realm->read_group().get_table("class_linked to object");
    auto value_col = table->get_column_key("value");
    auto object_col = table->get_column_key("links");
    auto int_col = table->get_column_key("integers");
    auto linked_object_value_col = linked_table->get_column_key("value");

    {
        realm->begin_transaction();
        Obj obj = table->create_object();
        obj.set(value_col, 100);
        auto object_dict = obj.get_dictionary(object_col);
        auto int_dict = obj.get_dictionary(int_col);
        const char* keys[5] = {"a", "b", "c", "d", "e"};
        for (int j = 0; j < 5; ++j) {
            auto child_obj = linked_table->create_object();
            child_obj.set(linked_object_value_col, j + 10);
            object_dict.insert(keys[j], child_obj.get_key());
            int_dict.insert(keys[j], static_cast<Int>(j + 42));
        }
        realm->commit_transaction();
    }

    Results results(realm, table);
    auto frozen_realm = Realm::get_frozen_realm(config, realm->read_transaction_version());

    auto obj_dict = results.get(0).get_dictionary(object_col);
    auto frozen_obj_dict = object_store::Dictionary(realm, obj_dict).freeze(frozen_realm);
    auto frozen_int_dict = object_store::Dictionary(realm, table->get_object(0), int_col).freeze(frozen_realm);

    SECTION("is_frozen") {
        REQUIRE(frozen_obj_dict.is_frozen());
        REQUIRE(frozen_int_dict.is_frozen());
        JoiningThread thread([&] {
            REQUIRE(frozen_int_dict.is_frozen());
            REQUIRE(frozen_obj_dict.is_frozen());
        });
    }

    SECTION("add_notification throws") {
        REQUIRE_EXCEPTION(frozen_obj_dict.add_notification_callback([&](CollectionChangeSet) {}),
                          WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
        REQUIRE_EXCEPTION(frozen_int_dict.add_notification_callback([&](CollectionChangeSet) {}),
                          WrongTransactionState,
                          "Notifications are not available on frozen collections since they do not change.");
    }

    SECTION("read across threads") {
        JoiningThread thread([&] {
            REQUIRE(frozen_int_dict.size() == 5);
            REQUIRE(frozen_obj_dict.size() == 5);
            REQUIRE(frozen_int_dict.get<Int>("a") == 42);
            REQUIRE(frozen_obj_dict.get_object("a").get<Int>(linked_object_value_col) == 10);
        });
    }

    SECTION("release all locks") {
        frozen_realm->close();
        realm->close();
        REQUIRE(DB::call_with_lock(config.path, [](auto) {}));
    }
}
