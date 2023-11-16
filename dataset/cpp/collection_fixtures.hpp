////////////////////////////////////////////////////////////////////////////
//
// Copyright 2021 Realm Inc.
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

#ifndef REALM_TEST_COLLECTION_FIXTURES_HPP
#define REALM_TEST_COLLECTION_FIXTURES_HPP

#include <catch2/catch_all.hpp>

#include <realm/object-store/list.hpp>
#include <realm/object-store/property.hpp>
#include <realm/object-store/results.hpp>

#include <realm/db.hpp>
#include <realm/util/any.hpp>
#include <realm/util/functional.hpp>

#include <string>
#include <type_traits>

namespace realm::collection_fixtures {

template <typename T>
constexpr bool always_false = false;

template <typename T>
inline T get(Mixed)
{
    static_assert(always_false<T>);
}

template <>
inline Mixed get(Mixed m)
{
    return m;
}

template <>
inline int64_t get(Mixed m)
{
    return m.get_int();
}
template <>
inline float get(Mixed m)
{
    return m.get_type() == type_Float ? m.get_float() : static_cast<float>(m.get_double());
}
template <>
inline double get(Mixed m)
{
    return m.get_double();
}
template <>
inline Timestamp get(Mixed m)
{
    return m.get_timestamp();
}
template <>
inline Decimal128 get(Mixed m)
{
    return m.get<Decimal128>();
}

template <PropertyType prop_type, typename T>
struct Base {
    using Type = T;
    using Wrapped = T;
    using Boxed = T;
    using AvgType = double;
    constexpr static bool is_optional = false;
    constexpr static bool can_sum = std::is_arithmetic<T>::value;
    constexpr static bool can_average = std::is_arithmetic<T>::value;
    constexpr static bool can_minmax = std::is_arithmetic<T>::value;
    constexpr static bool can_sort = true;
    constexpr static PropertyType property_type = prop_type;

    static std::any to_any(T value)
    {
        return value;
    }

    template <typename Fn>
    static auto unwrap(T value, Fn&& fn)
    {
        return fn(value);
    }

    static T min() = delete;
    static T max() = delete;
    static T sum() = delete;
    static AvgType average() = delete;
    static T empty_sum_value()
    {
        return T{};
    }
};

struct Int : Base<PropertyType::Int, int64_t> {
    constexpr static const char* name = "int";
    static std::vector<int64_t> values()
    {
        return {3, 1, 2};
    }
    constexpr static int64_t min()
    {
        return 1;
    }
    constexpr static int64_t max()
    {
        return 3;
    }
    constexpr static int64_t sum()
    {
        return 6;
    }
    constexpr static double average()
    {
        return 2.0;
    }
};

struct Bool : Base<PropertyType::Bool, bool> {
    constexpr static const char* name = "bool";
    constexpr static bool can_sum = false;
    constexpr static bool can_average = false;
    constexpr static bool can_minmax = false;
    static std::vector<bool> values()
    {
        return {true, false};
    }
};

struct Float : Base<PropertyType::Float, float> {
    constexpr static const char* name = "float";
    static std::vector<float> values()
    {
        return {3.3f, 1.1f, 2.2f};
    }
    constexpr static float min()
    {
        return 1.1f;
    }
    constexpr static float max()
    {
        return 3.3f;
    }
    static auto sum()
    {
        return Catch::Approx(6.6f);
    }
    static auto average()
    {
        return Catch::Approx(2.2f);
    }
};

struct Double : Base<PropertyType::Double, double> {
    constexpr static const char* name = "double";
    static std::vector<double> values()
    {
        return {3.3, 1.1, 2.2};
    }
    constexpr static double min()
    {
        return 1.1;
    }
    constexpr static double max()
    {
        return 3.3;
    }
    static auto sum()
    {
        return Catch::Approx(6.6);
    }
    static auto average()
    {
        return Catch::Approx(2.2);
    }
};

struct String : Base<PropertyType::String, StringData> {
    constexpr static const char* name = "string";
    using Boxed = std::string;
    static std::vector<StringData> values()
    {
        return {"c", "a", "b"};
    }
    static std::any to_any(StringData value)
    {
        return value ? std::string(value) : std::any();
    }
};

struct Binary : Base<PropertyType::Data, BinaryData> {
    constexpr static const char* name = "data";
    using Boxed = std::string;
    constexpr static bool can_sort = false;
    static std::any to_any(BinaryData value)
    {
        return value ? std::string(value) : std::any();
    }
    static std::vector<BinaryData> values()
    {
        return {BinaryData("c", 1), BinaryData("a", 1), BinaryData("b", 1)};
    }
};

struct Date : Base<PropertyType::Date, Timestamp> {
    constexpr static const char* name = "date";
    constexpr static bool can_minmax = true;
    static std::vector<Timestamp> values()
    {
        return {Timestamp(3, 3), Timestamp(1, 1), Timestamp(2, 2)};
    }
    static Timestamp min()
    {
        return Timestamp(1, 1);
    }
    static Timestamp max()
    {
        return Timestamp(3, 3);
    }
};

struct MixedVal : Base<PropertyType::Mixed, realm::Mixed> {
    constexpr static const char* name = "mixed";
    using AvgType = Decimal128;
    constexpr static bool is_optional = true;
    constexpr static bool can_sum = true;
    constexpr static bool can_average = true;
    constexpr static bool can_minmax = true;
    constexpr static PropertyType property_type = PropertyType::Mixed | PropertyType::Nullable;

    static std::vector<realm::Mixed> values()
    {
        return {
            Mixed{realm::UUID()},
            Mixed{},
            Mixed{realm::ObjectId()},

            // Mixed sorting considers all numerics to be the same time, so
            // ensure we have some interleaved values to test that
            Mixed{int64_t(1)},
            Mixed{int64_t(2)},
            Mixed{int64_t(3)},

            Mixed{double(1.2)},
            Mixed{double(2.2)},
            Mixed{double(3.2)},

            Mixed{float(1.1)},
            Mixed{float(2.1)},
            Mixed{float(3.1)},

            Mixed{Decimal128("1.3")},
            Mixed{Decimal128("2.3")},
            Mixed{Decimal128("3.3")},

            // Mixed sorting considers strings and binary to be the same time, so
            // ensure we have some interleaved values to test that
            Mixed{"a string"},
            Mixed{"b string"},
            Mixed{BinaryData("a binary", 8)},
            Mixed{BinaryData("b binary", 8)},
        };
    }
    static Mixed min()
    {
        return Mixed{int64_t(1)};
    }
    static Mixed max()
    {
        return Mixed{UUID()};
    }
    static Decimal128 sum()
    {
        return Decimal128{int64_t(1)} + Decimal128{int64_t(2)} + Decimal128{int64_t(3)} + Decimal128{double(1.2)} +
               Decimal128{double(2.2)} + Decimal128{double(3.2)} + Decimal128{float(1.1)} + Decimal128{float(2.1)} +
               Decimal128{float(3.1)} + Decimal128("1.3") + Decimal128("2.3") + Decimal128("3.3");
    }
    static Decimal128 average()
    {
        return (sum() / Decimal128(12));
    }
    static Mixed empty_sum_value()
    {
        return Mixed{0};
    }
};

struct OID : Base<PropertyType::ObjectId, ObjectId> {
    constexpr static const char* name = "object id";
    static std::vector<ObjectId> values()
    {
        return {ObjectId("bbbbbbbbbbbbbbbbbbbbbbbb"), ObjectId("aaaaaaaaaaaaaaaaaaaaaaaa")};
    }
};

struct UUID : Base<PropertyType::UUID, realm::UUID> {
    constexpr static const char* name = "uuid";
    static std::vector<realm::UUID> values()
    {
        return {realm::UUID("3b241101-e2bb-4255-8caf-4136c566a962"),
                realm::UUID("3b241101-a2b3-4255-8caf-4136c566a999")};
    }
};

struct Decimal : Base<PropertyType::Decimal, Decimal128> {
    constexpr static const char* name = "decimal128";
    using AvgType = Decimal128;
    constexpr static bool can_sum = true;
    constexpr static bool can_average = true;
    constexpr static bool can_minmax = true;

    static std::vector<Decimal128> values()
    {
        return {Decimal128("876.54e32"), Decimal128("123.45e6")};
    }
    static Decimal128 min()
    {
        return Decimal128("123.45e6");
    }
    static Decimal128 max()
    {
        return Decimal128("876.54e32");
    }
    static Decimal128 sum()
    {
        return Decimal128("123.45e6") + Decimal128("876.54e32");
    }
    static Decimal128 average()
    {
        return ((Decimal128("123.45e6") + Decimal128("876.54e32")) / Decimal128(2));
    }
};

template <typename BaseT>
struct BoxedOptional : BaseT {
    static const inline std::string name = std::string(BaseT::name) + "?";
    using Type = util::Optional<typename BaseT::Type>;
    using Boxed = Type;
    constexpr static bool is_optional = true;
    constexpr static PropertyType property_type = BaseT::property_type | PropertyType::Nullable;

    static std::vector<Type> values()
    {
        std::vector<Type> ret;
        for (auto v : BaseT::values())
            ret.push_back(Type(v));
        ret.push_back(util::none);
        return ret;
    }
    static auto unwrap(Type value)
    {
        return *value;
    }
    static std::any to_any(Type value)
    {
        return value ? std::any(*value) : std::any();
    }

    template <typename Fn>
    static auto unwrap(Type value, Fn&& fn)
    {
        return value ? fn(*value) : fn(null());
    }
};

template <typename BaseT>
struct UnboxedOptional : BaseT {
    static const inline std::string name = std::string(BaseT::name) + "?";
    constexpr static bool is_optional = true;
    constexpr static PropertyType property_type = BaseT::property_type | PropertyType::Nullable;

    static auto values() -> decltype(BaseT::values())
    {
        auto ret = BaseT::values();
        if constexpr (std::is_same_v<BaseT, collection_fixtures::Decimal>) {
            // The default Decimal128 ctr is 0, but we want a null value
            ret.push_back(Decimal128(realm::null()));
        }
        else {
            ret.push_back(typename BaseT::Type());
        }
        return ret;
    }
};

template <typename T>
std::vector<Obj> get_linked_objects(T collection)
{
    std::vector<Obj> links;
    auto group = collection.get_obj().get_table()->get_parent_group();
    REALM_ASSERT(group);

    for (size_t i = 0; i < collection.size(); ++i) {
        Mixed value = collection.get_any(i);
        if (value.is_type(type_TypedLink)) {
            ObjLink lnk = value.get_link();
            if (lnk && !lnk.is_unresolved()) {
                auto dst_table = group->get_table(lnk.get_table_key());
                REALM_ASSERT(dst_table);
                links.push_back(dst_table->get_object(lnk.get_obj_key()));
            }
        }
        else if (value.is_type(type_Link)) {
            ObjKey lnk = value.get<ObjKey>();
            if (lnk && !lnk.is_unresolved()) {
                auto dst_table = collection.get_obj().get_table()->get_opposite_table(collection.get_col_key());
                REALM_ASSERT(dst_table);
                links.push_back(dst_table->get_object(lnk));
            }
        }
        // null values and any other non-link value are ignored
    }
    return links;
}

struct LinkedCollectionBase {
    LinkedCollectionBase(const std::string& property_name, const std::string& dest_name)
        : m_prop_name(property_name)
        , m_dest_name(dest_name)
    {
    }

    void set_relation_updater(util::UniqueFunction<void()> updater)
    {
        m_relation_updater = std::move(updater);
    }

    ColKey get_link_col_key(TableRef source_table)
    {
        REALM_ASSERT(source_table);
        ColKey collection_col_key = source_table->get_column_key(m_prop_name);
        REALM_ASSERT(collection_col_key);
        return collection_col_key;
    }
    virtual void add_link(Obj from, ObjLink to) = 0;
    virtual bool remove_link(Obj from, ObjLink to) = 0;
    virtual void clear_collection(Obj obj) = 0;
    virtual std::vector<Obj> get_links(Obj obj) = 0;
    virtual void move(Obj, size_t, size_t) {}
    bool remove_linked_object(Obj obj, ObjLink to)
    {
        auto links = get_links(obj);
        for (auto& o : links) {
            if (o.get_key() == to.get_obj_key()) {
                o.remove();
                return true;
            }
        }
        return false;
    }
    virtual bool will_erase_removed_object_links()
    {
        return true; // only dictionaries are false
    }
    virtual void reset_test_state() {}

    std::string m_prop_name;
    std::string m_dest_name;
    util::UniqueFunction<void()> m_relation_updater;
};

struct ListOfObjects : public LinkedCollectionBase {
    ListOfObjects(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }

    Property property()
    {
        return {m_prop_name, PropertyType::Array | PropertyType::Object, m_dest_name};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        from.get_linklist(col).add(to.get_obj_key());
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        std::vector<Obj> links;
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_linklist(col);
        for (size_t i = 0; i < coll.size(); ++i) {
            links.push_back(coll.get_object(i));
        }
        return links;
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_linklist(col);
        size_t ndx = coll.find_first(to.get_obj_key());
        if (ndx != realm::not_found) {
            coll.remove(ndx);
            return true;
        }
        return false;
    }
    void move(Obj source, size_t from, size_t to) override
    {
        ColKey col = get_link_col_key(source.get_table());
        auto coll = source.get_linklist(col);
        coll.move(from, to);
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_linklist(col).size();
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        obj.get_linklist(col).clear();
    }
    size_t count_unresolved_links(Obj)
    {
        return 0;
    }
    constexpr static bool allows_storing_nulls = false;
};

struct ListOfMixedLinks : public LinkedCollectionBase {
    ListOfMixedLinks(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }

    Property property()
    {
        return {m_prop_name, PropertyType::Array | PropertyType::Mixed | PropertyType::Nullable};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        from.get_list<Mixed>(col).add(to);
        // When adding dynamic links through a mixed value, the relationship map needs to be dynamically updated.
        // In practice, this is triggered by the addition of backlink columns to any table.
        if (m_relation_updater) {
            m_relation_updater();
        }
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_list<Mixed>(col).size();
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_list<Mixed>(col);
        return get_linked_objects(coll);
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_list<Mixed>(col);
        size_t ndx = coll.find_first(Mixed{to});
        if (ndx != realm::not_found) {
            coll.remove(ndx);
            return true;
        }
        return false;
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        obj.get_list<Mixed>(col).clear();
    }
    void move(Obj obj, size_t from, size_t to) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        obj.get_list<Mixed>(col).move(from, to);
    }

    size_t count_unresolved_links(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        Lst<Mixed> list = obj.get_list<Mixed>(col);
        size_t num_unresolved = 0;
        for (auto value : list) {
            if (value.is_unresolved_link()) {
                ++num_unresolved;
            }
        }
        return num_unresolved;
    }
    constexpr static bool allows_storing_nulls = true;
};

struct SetOfObjects : public LinkedCollectionBase {
    SetOfObjects(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }

    Property property()
    {
        return {m_prop_name, PropertyType::Set | PropertyType::Object, m_dest_name};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        from.get_linkset(col).insert(to.get_obj_key());
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        std::vector<Obj> links;
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_linkset(col);
        for (size_t i = 0; i < coll.size(); ++i) {
            links.push_back(coll.get_object(i));
        }
        return links;
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_linkset(col);
        auto pair = coll.erase(to.get_obj_key());
        return pair.second;
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_linkset(col).size();
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        obj.get_linkset(col).clear();
    }
    size_t count_unresolved_links(Obj&)
    {
        return 0;
    }
    constexpr static bool allows_storing_nulls = false;
};

struct SetOfMixedLinks : public LinkedCollectionBase {
    SetOfMixedLinks(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }

    Property property()
    {
        return {m_prop_name, PropertyType::Set | PropertyType::Mixed | PropertyType::Nullable};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        from.get_set<Mixed>(col).insert(to);
        // When adding dynamic links through a mixed value, the relationship map needs to be dynamically updated.
        // In practice, this is triggered by the addition of backlink columns to any table.
        if (m_relation_updater) {
            m_relation_updater();
        }
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_set<Mixed>(col);
        return get_linked_objects(coll);
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_set<Mixed>(col);
        auto pair = coll.erase(Mixed{to});
        return pair.second;
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        obj.get_set<Mixed>(col).clear();
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_set<Mixed>(col).size();
    }
    size_t count_unresolved_links(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        Set<Mixed> set = obj.get_set<Mixed>(col);
        size_t num_unresolved = 0;
        for (auto value : set) {
            if (value.is_unresolved_link()) {
                ++num_unresolved;
            }
        }
        return num_unresolved;
    }
    constexpr static bool allows_storing_nulls = true;
};

struct DictionaryOfObjects : public LinkedCollectionBase {
    DictionaryOfObjects(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }
    Property property()
    {
        return {m_prop_name, PropertyType::Dictionary | PropertyType::Object | PropertyType::Nullable, m_dest_name};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey link_col = get_link_col_key(from.get_table());
        from.get_dictionary(link_col).insert(util::format("key_%1", key_counter++), to.get_obj_key());
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_dictionary(col);
        return get_linked_objects(coll);
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_dictionary(col).size();
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_dictionary(col);
        for (auto it = coll.begin(); it != coll.end(); ++it) {
            if ((*it).second == to) {
                coll.erase(it);
                return true;
            }
        }
        return false;
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        Dictionary dict = obj.get_dictionary(col);
        dict.clear();
    }
    size_t count_unresolved_links(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        Dictionary dict = obj.get_dictionary(col);
        size_t num_unresolved = 0;
        dict.for_all_values([&num_unresolved](Mixed val) {
            if (val.is_unresolved_link()) {
                ++num_unresolved;
            }
        });
        return num_unresolved;
    }
    bool will_erase_removed_object_links() override
    {
        return false;
    }
    void reset_test_state() override
    {
        key_counter = 0;
    }
    size_t key_counter = 0;
    constexpr static bool allows_storing_nulls = true;
};

struct DictionaryOfMixedLinks : public LinkedCollectionBase {
    DictionaryOfMixedLinks(const std::string& property_name, const std::string& dest_name)
        : LinkedCollectionBase(property_name, dest_name)
    {
    }
    Property property()
    {
        return {m_prop_name, PropertyType::Dictionary | PropertyType::Mixed | PropertyType::Nullable};
    }
    void add_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        from.get_dictionary(col).insert(util::format("key_%1", key_counter++), to);
        // When adding dynamic links through a mixed value, the relationship map needs to be dynamically updated.
        // In practice, this is triggered by the addition of backlink columns to any table.
        if (m_relation_updater) {
            m_relation_updater();
        }
    }
    std::vector<Obj> get_links(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        auto coll = obj.get_dictionary(col);
        return get_linked_objects(coll);
    }
    bool remove_link(Obj from, ObjLink to) override
    {
        ColKey col = get_link_col_key(from.get_table());
        auto coll = from.get_dictionary(col);
        for (auto it = coll.begin(); it != coll.end(); ++it) {
            if ((*it).second == to) {
                coll.erase(it);
                return true;
            }
        }
        return false;
    }
    size_t size_of_collection(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        return obj.get_dictionary(col).size();
    }
    void clear_collection(Obj obj) override
    {
        ColKey col = get_link_col_key(obj.get_table());
        Dictionary dict = obj.get_dictionary(col);
        dict.clear();
    }
    size_t count_unresolved_links(Obj obj)
    {
        ColKey col = get_link_col_key(obj.get_table());
        Dictionary dict = obj.get_dictionary(col);
        size_t num_unresolved = 0;
        dict.for_all_values([&num_unresolved](Mixed val) {
            if (val.is_unresolved_link()) {
                ++num_unresolved;
            }
        });
        return num_unresolved;
    }
    bool will_erase_removed_object_links() override
    {
        return false;
    }
    void reset_test_state() override
    {
        key_counter = 0;
    }
    size_t key_counter = 0;
    constexpr static bool allows_storing_nulls = true;
};


} // namespace realm::collection_fixtures

namespace realm {
template <typename T>
bool operator==(List const& list, std::vector<T> const& values)
{
    if (list.size() != values.size())
        return false;
    for (size_t i = 0; i < values.size(); ++i) {
        if (list.get<T>(i) != values[i])
            return false;
    }
    return true;
}

template <typename T>
bool operator==(Results const& results, std::vector<T> const& values)
{
    // FIXME: this is only necessary because Results::size() and ::get() are not const
    Results copy{results};
    if (copy.size() != values.size())
        return false;
    for (size_t i = 0; i < values.size(); ++i) {
        if (copy.get<T>(i) != values[i])
            return false;
    }
    return true;
}

} // namespace realm

#endif // REALM_TEST_COLLECTION_FIXTURES_HPP
