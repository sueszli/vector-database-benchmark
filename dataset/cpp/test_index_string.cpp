/*************************************************************************
 *
 * Copyright 2016 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#include "testsettings.hpp"
#ifdef TEST_INDEX_STRING

#include <realm.hpp>
#include <realm/index_string.hpp>
#include <realm/query_expression.hpp>
#include <realm/tokenizer.hpp>
#include <realm/util/to_string.hpp>
#include <set>
#include "test.hpp"
#include "util/misc.hpp"
#include "util/random.hpp"

using namespace realm;
using namespace util;
using namespace realm;
using namespace realm::util;
using namespace realm::test_util;
using unit_test::TestContext;

// Test independence and thread-safety
// -----------------------------------
//
// All tests must be thread safe and independent of each other. This
// is required because it allows for both shuffling of the execution
// order and for parallelized testing.
//
// In particular, avoid using std::rand() since it is not guaranteed
// to be thread safe. Instead use the API offered in
// `test/util/random.hpp`.
//
// All files created in tests must use the TEST_PATH macro (or one of
// its friends) to obtain a suitable file system path. See
// `test/util/test_path.hpp`.
//
//
// Debugging and the ONLY() macro
// ------------------------------
//
// A simple way of disabling all tests except one called `Foo`, is to
// replace TEST(Foo) with ONLY(Foo) and then recompile and rerun the
// test suite. Note that you can also use filtering by setting the
// environment varible `UNITTEST_FILTER`. See `README.md` for more on
// this.
//
// Another way to debug a particular test, is to copy that test into
// `experiments/testcase.cpp` and then run `sh build.sh
// check-testcase` (or one of its friends) from the command line.


namespace {

template <typename T>
class column {
public:
    class ColumnTestType {
    public:
        ColumnTestType(column* owner)
            : m_owner(owner)
        {
        }
        const StringIndex* create_search_index()
        {
            m_owner->m_table.add_search_index(m_owner->m_col_key);
            return m_owner->m_table.get_search_index(m_owner->m_col_key);
        }
        ObjKey key(size_t ndx) const
        {
            return m_keys[ndx];
        }
        size_t size() const
        {
            return m_keys.size();
        }
        void add(T value)
        {
            auto k = m_owner->m_table.create_object().set(m_owner->m_col_key, value).get_key();
            m_keys.push_back(k);
        }
        void add_null()
        {
            auto k = m_owner->m_table.create_object().set_null(m_owner->m_col_key).get_key();
            m_keys.push_back(k);
        }
        void set(size_t ndx, T value)
        {
            m_owner->m_table.get_object(m_keys[ndx]).set(m_owner->m_col_key, value);
        }
        void insert(size_t ndx, T value)
        {
            auto k = m_owner->m_table.create_object().set(m_owner->m_col_key, value).get_key();
            m_keys.insert(m_keys.begin() + ndx, k);
        }
        T get(size_t ndx)
        {
            return m_owner->m_table.get_object(m_keys[ndx]).template get<T>(m_owner->m_col_key);
        }
        T get(ObjKey obj_key)
        {
            return m_owner->m_table.get_object(obj_key).template get<T>(m_owner->m_col_key);
        }
        void erase(size_t ndx)
        {
            m_owner->m_table.remove_object(m_keys[ndx]);
            m_keys.erase(m_keys.begin() + ndx);
        }
        void clear()
        {
            m_owner->m_table.clear();
            m_keys.clear();
        }
        size_t find_first(T value) const
        {
            auto k = m_owner->m_table.find_first(m_owner->m_col_key, value);
            if (k == realm::null_key) {
                return realm::npos;
            }
            auto it = std::find(m_keys.begin(), m_keys.end(), k);
            return it - m_keys.begin();
        }
        size_t count(T value) const
        {
            return m_owner->m_table.count_string(m_owner->m_col_key, value);
        }
        void verify()
        {
            m_owner->m_table.verify();
        }

    private:
        column* m_owner;
        std::vector<ObjKey> m_keys;
    };

    column(bool nullable = false, bool enumerated = false)
        : m_column(this)
    {
        m_col_key = m_table.add_column(ColumnTypeTraits<T>::id, "values", nullable);
        if (enumerated) {
            m_table.enumerate_string_column(m_col_key);
        }
    }
    ColumnTestType& get_column()
    {
        return m_column;
    }

private:
    Table m_table;
    ColKey m_col_key;
    ColumnTestType m_column;
};

class string_column : public column<String> {
public:
    string_column()
        : column(false, false)
    {
    }
    static bool is_nullable()
    {
        return false;
    }
    static bool is_enumerated()
    {
        return false;
    }
};
class nullable_string_column : public column<String> {
public:
    nullable_string_column()
        : column(true, false)
    {
    }
    static bool is_nullable()
    {
        return true;
    }
    static bool is_enumerated()
    {
        return false;
    }
};
class enum_column : public column<String> {
public:
    enum_column()
        : column(false, true)
    {
    }
    static bool is_nullable()
    {
        return false;
    }
    static bool is_enumerated()
    {
        return true;
    }
};
class nullable_enum_column : public column<String> {
public:
    nullable_enum_column()
        : column(true, true)
    {
    }
    static bool is_nullable()
    {
        return true;
    }
    static bool is_enumerated()
    {
        return true;
    }
};

// disable to avoid warnings about not being used - enable when tests
// needed them are enabled again

// strings used by tests
const char s1[] = "John";
const char s2[] = "Brian";
const char s3[] = "Samantha";
const char s4[] = "Tom";
const char s5[] = "Johnathan";
const char s6[] = "Johnny";
const char s7[] = "Sam";

// integers used by integer index tests
std::vector<int64_t> ints = {0x1111,     0x11112222, 0x11113333, 0x1111333, 0x111122223333ull, 0x1111222233334ull,
                             0x22223333, 0x11112227, 0x11112227, 0x78923};

using nullable = std::true_type;
using non_nullable = std::false_type;

} // anonymous namespace

TEST(Tokenizer_Basic)
{
    auto tok = realm::Tokenizer::get_instance();

    tok->reset("to be or not to be");
    auto tokens = tok->get_all_tokens();
    CHECK_EQUAL(tokens.size(), 4);

    tok->reset("To be or not to be");
    realm::TokenInfoMap info = tok->get_token_info();
    CHECK_EQUAL(info.size(), 4);
    realm::TokenInfo& i(info["to"]);
    CHECK_EQUAL(i.positions.size(), 2);
    CHECK_EQUAL(i.positions[0], 0);
    CHECK_EQUAL(i.positions[1], 4);
    CHECK_EQUAL(i.ranges.size(), 2);
    CHECK_EQUAL(i.ranges[0].first, 0);
    CHECK_EQUAL(i.ranges[0].second, 2);
    CHECK_EQUAL(i.ranges[1].first, 13);
    CHECK_EQUAL(i.ranges[1].second, 15);

    tok->reset("Jeg gik mig over sø og land");
    info = tok->get_token_info();
    CHECK_EQUAL(info.size(), 7);
    realm::TokenInfo& j(info["sø"]);
    CHECK_EQUAL(j.ranges[0].first, 17);
    CHECK_EQUAL(j.ranges[0].second, 20);

    tok->reset("with-hyphen -term -other-term-plus");
    CHECK(tok->get_all_tokens() == std::set<std::string>({"with", "hyphen", "term", "other", "plus"}));
}

TEST(StringIndex_NonIndexable)
{
    // Create a column with string values
    Group group;
    TableRef table = group.add_table("table");
    TableRef target_table = group.add_table("target");
    table->add_column(*target_table, "link");
    table->add_column_list(*target_table, "linkList");
    table->add_column(type_Double, "double");
    table->add_column(type_Float, "float");
    table->add_column(type_Binary, "binary");

    for (auto col : table->get_column_keys()) {
        CHECK_LOGIC_ERROR(table->add_search_index(col), ErrorCodes::IllegalOperation);
    }
}

TEST_TYPES(StringIndex_BuildIndex, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value
    col.add(s5); // common prefix
    col.add(s6); // common prefix

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    const ObjKey r1 = ndx.find_first(s1);
    const ObjKey r2 = ndx.find_first(s2);
    const ObjKey r3 = ndx.find_first(s3);
    const ObjKey r4 = ndx.find_first(s4);
    const ObjKey r5 = ndx.find_first(s5);
    const ObjKey r6 = ndx.find_first(s6);

    CHECK_EQUAL(0, r1.value);
    CHECK_EQUAL(1, r2.value);
    CHECK_EQUAL(2, r3.value);
    CHECK_EQUAL(3, r4.value);
    CHECK_EQUAL(5, r5.value);
    CHECK_EQUAL(6, r6.value);
}

TEST_TYPES(StringIndex_DeleteAll, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value
    col.add(s5); // common prefix
    col.add(s6); // common prefix

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    // Delete all entries
    // (reverse order to avoid ref updates)
    col.erase(6);
    col.erase(5);
    col.erase(4);
    col.erase(3);
    col.erase(2);
    col.erase(1);
    col.erase(0);
    CHECK(ndx.is_empty());

    // Re-insert values
    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value
    col.add(s5); // common prefix
    col.add(s6); // common prefix

    // Delete all entries
    // (in order to force constant ref updating)
    col.erase(0);
    col.erase(0);
    col.erase(0);
    col.erase(0);
    col.erase(0);
    col.erase(0);
    col.erase(0);
    CHECK(ndx.is_empty());
}

TEST_TYPES(StringIndex_Delete, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    // Delete first item (in index)
    col.erase(1);

    CHECK_EQUAL(0, col.find_first(s1));
    CHECK_EQUAL(1, col.find_first(s3));
    CHECK_EQUAL(2, col.find_first(s4));
    CHECK_EQUAL(null_key, ndx.find_first(s2));

    // Delete last item (in index)
    col.erase(2);

    CHECK_EQUAL(0, col.find_first(s1));
    CHECK_EQUAL(1, col.find_first(s3));
    CHECK_EQUAL(not_found, col.find_first(s4));
    CHECK_EQUAL(not_found, col.find_first(s2));

    // Delete middle item (in index)
    col.erase(1);

    CHECK_EQUAL(0, col.find_first(s1));
    CHECK_EQUAL(not_found, col.find_first(s3));
    CHECK_EQUAL(not_found, col.find_first(s4));
    CHECK_EQUAL(not_found, col.find_first(s2));

    // Delete all items
    col.erase(0);
    col.erase(0);
    CHECK(ndx.is_empty());
}


TEST_TYPES(StringIndex_ClearEmpty, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    // Clear to remove all entries
    col.clear();
    CHECK(ndx.is_empty());
}

TEST_TYPES(StringIndex_Clear, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value
    col.add(s5); // common prefix
    col.add(s6); // common prefix

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    // Clear to remove all entries
    col.clear();
    CHECK(ndx.is_empty());

    // Re-insert values
    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value
    col.add(s5); // common prefix
    col.add(s6); // common prefix

    const ObjKey r1 = ndx.find_first(s1);
    const ObjKey r2 = ndx.find_first(s2);
    const ObjKey r3 = ndx.find_first(s3);
    const ObjKey r4 = ndx.find_first(s4);
    const ObjKey r5 = ndx.find_first(s5);
    const ObjKey r6 = ndx.find_first(s6);

    CHECK_EQUAL(col.key(0), r1);
    CHECK_EQUAL(col.key(1), r2);
    CHECK_EQUAL(col.key(2), r3);
    CHECK_EQUAL(col.key(3), r4);
    CHECK_EQUAL(col.key(5), r5);
    CHECK_EQUAL(col.key(6), r6);
}


TEST_TYPES(StringIndex_Set, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);
    col.add(s1); // duplicate value

    // Create a new index on column
    col.create_search_index();

    // Set top value
    col.set(0, s5);

    CHECK_EQUAL(0, col.find_first(s5));
    CHECK_EQUAL(1, col.find_first(s2));
    CHECK_EQUAL(2, col.find_first(s3));
    CHECK_EQUAL(3, col.find_first(s4));
    CHECK_EQUAL(4, col.find_first(s1));

    // Set bottom value
    col.set(4, s6);

    CHECK_EQUAL(not_found, col.find_first(s1));
    CHECK_EQUAL(0, col.find_first(s5));
    CHECK_EQUAL(1, col.find_first(s2));
    CHECK_EQUAL(2, col.find_first(s3));
    CHECK_EQUAL(3, col.find_first(s4));
    CHECK_EQUAL(4, col.find_first(s6));

    // Set middle value
    col.set(2, s7);

    CHECK_EQUAL(not_found, col.find_first(s3));
    CHECK_EQUAL(not_found, col.find_first(s1));
    CHECK_EQUAL(0, col.find_first(s5));
    CHECK_EQUAL(1, col.find_first(s2));
    CHECK_EQUAL(2, col.find_first(s7));
    CHECK_EQUAL(3, col.find_first(s4));
    CHECK_EQUAL(4, col.find_first(s6));
}

TEST_TYPES(StringIndex_Count, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s2);
    col.add(s3);
    col.add(s3);
    col.add(s3);
    col.add(s4);
    col.add(s4);
    col.add(s4);
    col.add(s4);

    // Create a new index on column
    col.create_search_index();

    // Counts
    const size_t c0 = col.count(s5);
    const size_t c1 = col.count(s1);
    const size_t c2 = col.count(s2);
    const size_t c3 = col.count(s3);
    const size_t c4 = col.count(s4);
    CHECK_EQUAL(0, c0);
    CHECK_EQUAL(1, c1);
    CHECK_EQUAL(2, c2);
    CHECK_EQUAL(3, c3);
    CHECK_EQUAL(4, c4);
}

TEST_TYPES(StringIndex_Distinct, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s2);
    col.add(s3);
    col.add(s3);
    col.add(s3);
    col.add(s4);
    col.add(s4);
    col.add(s4);
    col.add(s4);

    // Create a new index on column
    const StringIndex* ndx = col.create_search_index();
    CHECK(ndx->has_duplicate_values());
}

TEST_TYPES(StringIndex_FindAllNoCopy, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s2);
    col.add(s3);
    col.add(s3);
    col.add(s3);
    col.add(s4);
    col.add(s4);
    col.add(s4);
    col.add(s4);

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    InternalFindResult ref_2;
    FindRes res1 = ndx.find_all_no_copy(StringData("not there"), ref_2);
    CHECK_EQUAL(FindRes_not_found, res1);

    FindRes res2 = ndx.find_all_no_copy(s1, ref_2);
    CHECK_EQUAL(FindRes_single, res2);
    CHECK_EQUAL(0, ref_2.payload);

    FindRes res3 = ndx.find_all_no_copy(s4, ref_2);
    CHECK_EQUAL(FindRes_column, res3);
    BPlusTree<ObjKey> results(Allocator::get_default());
    results.init_from_ref(ref_type(ref_2.payload));

    CHECK_EQUAL(4, ref_2.end_ndx - ref_2.start_ndx);
    CHECK_EQUAL(4, results.size());
    CHECK_EQUAL(col.key(6), results.get(0));
    CHECK_EQUAL(col.key(7), results.get(1));
    CHECK_EQUAL(col.key(8), results.get(2));
    CHECK_EQUAL(col.key(9), results.get(3));
}

// If a column contains a specific value in multiple rows, then the index will store a list of these row numbers
// in form of a column. If you call find_all() on an index, it will return a *reference* to that column instead
// of copying it to you, as a performance optimization.
TEST(StringIndex_FindAllNoCopy2_Int)
{
    // Create a column with duplcate values
    column<Int> test_resources;
    auto col = test_resources.get_column();

    for (auto i : ints)
        col.add(i);

    // Create a new index on column
    col.create_search_index();
    const StringIndex& ndx = *col.create_search_index();
    InternalFindResult results;

    for (auto i : ints) {
        FindRes res = ndx.find_all_no_copy(i, results);

        size_t real = 0;
        for (auto j : ints) {
            if (i == j)
                real++;
        }

        if (real == 1) {
            CHECK_EQUAL(res, FindRes_single);
            CHECK_EQUAL(i, ints[size_t(results.payload)]);
        }
        else if (real > 1) {
            CHECK_EQUAL(FindRes_column, res);
            const IntegerColumn results_column(Allocator::get_default(), ref_type(results.payload));
            CHECK_EQUAL(real, results.end_ndx - results.start_ndx);
            CHECK_EQUAL(real, results_column.size());
            for (size_t y = 0; y < real; y++)
                CHECK_EQUAL(i, ints[size_t(results_column.get(y))]);
        }
    }
}

// If a column contains a specific value in multiple rows, then the index will store a list of these row numbers
// in form of a column. If you call find_all() on an index, it will return a *reference* to that column instead
// of copying it to you, as a performance optimization.
TEST(StringIndex_FindAllNoCopy2_IntNull)
{
    // Create a column with duplcate values
    column<Int> test_resources(true);
    auto col = test_resources.get_column();

    for (size_t t = 0; t < sizeof(ints) / sizeof(ints[0]); t++)
        col.add(ints[t]);
    col.add_null();

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();
    InternalFindResult results;

    for (size_t t = 0; t < sizeof(ints) / sizeof(ints[0]); t++) {
        FindRes res = ndx.find_all_no_copy(ints[t], results);

        size_t real = 0;
        for (size_t y = 0; y < sizeof(ints) / sizeof(ints[0]); y++) {
            if (ints[t] == ints[y])
                real++;
        }

        if (real == 1) {
            CHECK_EQUAL(res, FindRes_single);
            CHECK_EQUAL(ints[t], ints[size_t(results.payload)]);
        }
        else if (real > 1) {
            CHECK_EQUAL(FindRes_column, res);
            const IntegerColumn results2(Allocator::get_default(), ref_type(results.payload));
            CHECK_EQUAL(real, results.end_ndx - results.start_ndx);
            CHECK_EQUAL(real, results2.size());
            for (size_t y = 0; y < real; y++)
                CHECK_EQUAL(ints[t], ints[size_t(results2.get(y))]);
        }
    }

    FindRes res = ndx.find_all_no_copy(null{}, results);
    CHECK_EQUAL(FindRes_single, res);
    CHECK_EQUAL(results.payload, col.size() - 1);
}

TEST_TYPES(StringIndex_FindAllNoCopyCommonPrefixStrings, string_column, nullable_string_column, enum_column,
           nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();

    auto test_prefix_find = [&](std::string prefix) {
        std::string prefix_b = prefix + "b";
        std::string prefix_c = prefix + "c";
        std::string prefix_d = prefix + "d";
        std::string prefix_e = prefix + "e";
        StringData spb(prefix_b);
        StringData spc(prefix_c);
        StringData spd(prefix_d);
        StringData spe(prefix_e);

        size_t start_row = col.size();
        col.add(spb);
        col.add(spc);
        col.add(spc);
        col.add(spe);
        col.add(spe);
        col.add(spe);

        InternalFindResult results;
        FindRes res = ndx.find_all_no_copy(spb, results);
        CHECK_EQUAL(res, FindRes_single);
        CHECK_EQUAL(results.payload, start_row);

        res = ndx.find_all_no_copy(spc, results);
        CHECK_EQUAL(res, FindRes_column);
        CHECK_EQUAL(results.end_ndx - results.start_ndx, 2);
        const IntegerColumn results_c(Allocator::get_default(), ref_type(results.payload));
        CHECK_EQUAL(results_c.get(results.start_ndx), start_row + 1);
        CHECK_EQUAL(results_c.get(results.start_ndx + 1), start_row + 2);
        CHECK_EQUAL(col.get(size_t(results_c.get(results.start_ndx))), spc);
        CHECK_EQUAL(col.get(size_t(results_c.get(results.start_ndx + 1))), spc);

        res = ndx.find_all_no_copy(spd, results);
        CHECK_EQUAL(res, FindRes_not_found);

        res = ndx.find_all_no_copy(spe, results);
        CHECK_EQUAL(res, FindRes_column);
        CHECK_EQUAL(results.end_ndx - results.start_ndx, 3);
        const IntegerColumn results_e(Allocator::get_default(), ref_type(results.payload));
        CHECK_EQUAL(results_e.get(results.start_ndx), start_row + 3);
        CHECK_EQUAL(results_e.get(results.start_ndx + 1), start_row + 4);
        CHECK_EQUAL(results_e.get(results.start_ndx + 2), start_row + 5);
        CHECK_EQUAL(col.get(size_t(results_e.get(results.start_ndx))), spe);
        CHECK_EQUAL(col.get(size_t(results_e.get(results.start_ndx + 1))), spe);
        CHECK_EQUAL(col.get(size_t(results_e.get(results.start_ndx + 2))), spe);
    };

    std::string std_max(StringIndex::s_max_offset, 'a');
    std::string std_over_max = std_max + "a";
    std::string std_under_max(StringIndex::s_max_offset >> 1, 'a');

    test_prefix_find(std_max);
    test_prefix_find(std_over_max);
    test_prefix_find(std_under_max);
}

TEST(StringIndex_Count_Int)
{
    // Create a column with duplicate values
    column<Int> test_resources;
    auto col = test_resources.get_column();

    for (auto i : ints)
        col.add(i);

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    for (auto i : ints) {
        size_t count = ndx.count(i);

        size_t real = 0;
        for (auto j : ints) {
            if (i == j)
                real++;
        }

        CHECK_EQUAL(real, count);
    }
}


TEST(StringIndex_Distinct_Int)
{
    // Create a column with duplicate values
    column<Int> test_resources;
    auto col = test_resources.get_column();

    for (auto i : ints)
        col.add(i);

    // Create a new index on column
    auto ndx = col.create_search_index();
    CHECK(ndx->has_duplicate_values());
}


TEST(StringIndex_Set_Add_Erase_Insert_Int)
{
    column<Int> test_resources;
    auto col = test_resources.get_column();

    col.add(1);
    col.add(2);
    col.add(3);
    col.add(2);

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    ObjKey f = ndx.find_first(int64_t(2));
    CHECK_EQUAL(col.key(1), f);

    col.set(1, 5);

    f = ndx.find_first(int64_t(2));
    CHECK_EQUAL(col.key(3), f);

    col.erase(1);

    f = ndx.find_first(int64_t(2));
    CHECK_EQUAL(col.key(2), f);

    col.insert(1, 5);
    CHECK_EQUAL(col.get(1), 5);

    f = ndx.find_first(int64_t(2));
    CHECK_EQUAL(col.key(3), f);

    col.add(7);
    CHECK_EQUAL(col.get(4), 7);
    col.set(4, 10);
    CHECK_EQUAL(col.get(4), 10);

    f = ndx.find_first(int64_t(10));
    CHECK_EQUAL(col.key(col.size() - 1), f);

    col.add(9);
    f = ndx.find_first(int64_t(9));
    CHECK_EQUAL(col.key(col.size() - 1), f);

    col.clear();
    f = ndx.find_first(int64_t(2));
    CHECK_EQUAL(null_key, f);
}

TEST(StringIndex_FuzzyTest_Int)
{
    column<Int> test_resources;
    auto col = test_resources.get_column();
    Random random(random_int<unsigned long>());
    const size_t n = static_cast<size_t>(1.2 * REALM_MAX_BPNODE_SIZE);

    col.create_search_index();

    for (size_t t = 0; t < n; ++t) {
        col.add(random.draw_int_max(0xffffffffffffffff));
    }

    for (size_t t = 0; t < n; ++t) {
        int64_t r;
        if (random.draw_bool())
            r = col.get(t);
        else
            r = random.draw_int_max(0xffffffffffffffff);

        size_t m = col.find_first(r);
        for (size_t t_2 = 0; t_2 < n; ++t_2) {
            if (col.get(t_2) == r) {
                CHECK_EQUAL(t_2, m);
                break;
            }
        }
    }
}

namespace {

// Generate string where the bit pattern in bits is converted to NUL bytes. E.g. (length=2):
// bits=0 -> "\0\0", bits=1 -> "\x\0", bits=2 -> "\0\x", bits=3 -> "\x\x", where x is a random byte
StringData create_string_with_nuls(const size_t bits, const size_t length, char* tmp, Random& random)
{
    for (size_t i = 0; i < length; ++i) {
        bool insert_nul_at_pos = (bits & (size_t(1) << i)) == 0;
        if (insert_nul_at_pos) {
            tmp[i] = '\0';
        }
        else {
            // Avoid stray \0 chars, since we are already testing all combinations.
            // All casts are necessary to preserve the bitpattern.
            tmp[i] = static_cast<char>(static_cast<unsigned char>(random.draw_int<unsigned int>(1, UCHAR_MAX)));
        }
    }
    return StringData(tmp, length);
}

} // anonymous namespace


// Test for generated strings of length 1..16 with all combinations of embedded NUL bytes
TEST_TYPES_IF(StringIndex_EmbeddedZeroesCombinations, TEST_DURATION > 1, string_column, nullable_string_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();

    constexpr unsigned int seed = 42;
    const size_t MAX_LENGTH = 16; // Test medium
    char tmp[MAX_LENGTH];         // this is a bit of a hack, that relies on the string being copied in column.add()

    for (size_t length = 1; length <= MAX_LENGTH; ++length) {

        {
            Random random(seed);
            const size_t combinations = size_t(1) << length;
            for (size_t i = 0; i < combinations; ++i) {
                StringData str = create_string_with_nuls(i, length, tmp, random);
                col.add(str);
            }
        }

        // check index up to this length
        size_t expected_index = 0;
        for (size_t l = 1; l <= length; ++l) {
            Random random(seed);
            const size_t combinations = size_t(1) << l;
            for (size_t i = 0; i < combinations; ++i) {
                StringData needle = create_string_with_nuls(i, l, tmp, random);
                CHECK_EQUAL(ndx.find_first(needle), col.key(expected_index));
                CHECK(strncmp(col.get(expected_index).data(), needle.data(), l) == 0);
                CHECK_EQUAL(col.get(expected_index).size(), needle.size());
                expected_index++;
            }
        }
    }
}

// Tests for a bug with strings containing zeroes
TEST_TYPES(StringIndex_EmbeddedZeroes, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col2 = test_resources.get_column();
    const StringIndex& ndx2 = *col2.create_search_index();

    // FIXME: re-enable once embedded nuls work
    col2.add(StringData("\0", 1));
    col2.add(StringData("\1", 1));
    col2.add(StringData("\0\0", 2));
    col2.add(StringData("\0\1", 2));
    col2.add(StringData("\1\0", 2));

    CHECK_EQUAL(ndx2.find_first(StringData("\0", 1)), col2.key(0));
    CHECK_EQUAL(ndx2.find_first(StringData("\1", 1)), col2.key(1));
    CHECK_EQUAL(ndx2.find_first(StringData("\2", 1)), null_key);
    CHECK_EQUAL(ndx2.find_first(StringData("\0\0", 2)), col2.key(2));
    CHECK_EQUAL(ndx2.find_first(StringData("\0\1", 2)), col2.key(3));
    CHECK_EQUAL(ndx2.find_first(StringData("\1\0", 2)), col2.key(4));
    CHECK_EQUAL(ndx2.find_first(StringData("\1\0\0", 3)), null_key);

    // Integer index (uses String index internally)
    int64_t v = 1ULL << 41;
    column<Int> test_resources_1;
    auto col = test_resources_1.get_column();
    const StringIndex& ndx = *col.create_search_index();
    col.add(1ULL << 40);
    auto f = ndx.find_first(v);
    CHECK_EQUAL(f, null_key);
}

TEST_TYPES(StringIndex_Null, nullable_string_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add("");
    col.add(realm::null());

    const StringIndex& ndx = *col.create_search_index();

    auto r1 = ndx.find_first(realm::null());
    CHECK_EQUAL(r1, col.key(1));
}


TEST_TYPES(StringIndex_Zero_Crash, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    bool nullable = TEST_TYPE::is_nullable();

    // StringIndex could crash if strings ended with one or more 0-bytes
    Table table;
    auto col = table.add_column(type_String, "strings", nullable);

    auto k0 = table.create_object().set(col, StringData("")).get_key();
    auto k1 = table.create_object().set(col, StringData("\0", 1)).get_key();
    auto k2 = table.create_object().set(col, StringData("\0\0", 2)).get_key();
    table.add_search_index(col);

    if (TEST_TYPE::is_enumerated())
        table.enumerate_string_column(col);

    ObjKey t;

    t = table.find_first_string(col, StringData(""));
    CHECK_EQUAL(k0, t);

    t = table.find_first_string(col, StringData("\0", 1));
    CHECK_EQUAL(k1, t);

    t = table.find_first_string(col, StringData("\0\0", 2));
    CHECK_EQUAL(k2, t);
}

TEST_TYPES(StringIndex_Zero_Crash2, std::true_type, std::false_type)
{
    Random random(random_int<unsigned long>());

    constexpr bool add_common_prefix = TEST_TYPE::value;

    for (size_t iter = 0; iter < 10 + TEST_DURATION * 100; iter++) {
        // StringIndex could crash if strings ended with one or more 0-bytes
        Table table;
        auto col = table.add_column(type_String, "string", true);

        table.add_search_index(col);

        for (size_t i = 0; i < 100 + TEST_DURATION * 1000; i++) {
            unsigned char action = static_cast<unsigned char>(random.draw_int_max<unsigned int>(100));
            if (action == 0) {
                table.remove_search_index(col);
                table.add_search_index(col);
            }
            else if (action > 48 && table.size() < 10) {
                // Generate string with equal probability of being empty, null, short, medium and long, and with
                // their contents having equal proability of being either random or a duplicate of a previous
                // string. When it's random, each char must have equal probability of being 0 or non-0e
                static std::string buf =
                    "This string is around 90 bytes long, which falls in the long-string type of Realm strings";

                std::string copy = buf;

                static std::string buf2 =
                    "                                                                                         ";
                std::string copy2 = buf2;
                StringData sd;

                size_t len = random.draw_int_max<size_t>(3);
                if (len == 0)
                    len = 0;
                else if (len == 1)
                    len = 7;
                else if (len == 2)
                    len = 27;
                else
                    len = random.draw_int_max<size_t>(90);

                copy = copy.substr(0, len);
                if (add_common_prefix) {
                    std::string prefix(StringIndex::s_max_offset, 'a');
                    copy = prefix + copy;
                }

                if (random.draw_int_max<int>(1) == 0) {
                    // duplicate string
                    sd = StringData(copy);
                }
                else {
                    // random string
                    for (size_t t = 0; t < len; t++) {
                        if (random.draw_int_max<int>(100) > 20)
                            copy2[t] = 0; // zero byte
                        else
                            copy2[t] = static_cast<char>(random.draw_int<int>()); // random byte
                    }
                    // no generated string can equal "null" (our vector magic value for null) because
                    // len == 4 is not possible
                    copy2 = copy2.substr(0, len);
                    if (add_common_prefix) {
                        std::string prefix(StringIndex::s_max_offset, 'a');
                        copy2 = prefix + copy2;
                    }
                    sd = StringData(copy2);
                }

                bool done = false;
                do {
                    int64_t key_val = random.draw_int_max<int64_t>(10000);
                    try {
                        table.create_object(ObjKey(key_val)).set(col, sd);
                        done = true;
                    }
                    catch (...) {
                    }
                } while (!done);
                table.verify();
            }
            else if (table.size() > 0) {
                // delete
                size_t row = random.draw_int_max<size_t>(table.size() - 1);
                Obj obj = table.get_object(row);
                obj.remove();
            }

            action = static_cast<unsigned char>(random.draw_int_max<unsigned int>(100));
            if (table.size() > 0) {
                // Search for value that exists
                size_t row = random.draw_int_max<size_t>(table.size() - 1);
                Obj obj = table.get_object(row);
                StringData sd = obj.get<String>(col);
                ObjKey t = table.find_first_string(col, sd);
                StringData sd2 = table.get_object(t).get<String>(col);
                CHECK_EQUAL(sd, sd2);
            }
        }
    }
}

TEST(StringIndex_Integer_Increasing)
{
    const size_t rows = 2000 + 1000000 * TEST_DURATION;

    // StringIndex could crash if strings ended with one or more 0-bytes
    Table table;
    auto col = table.add_column(type_Int, "int");
    table.add_search_index(col);

    std::multiset<int64_t> reference;

    for (size_t row = 0; row < rows; row++) {
        int64_t r = fastrand((TEST_DURATION == 0) ? 2000 : 0x100000);
        table.create_object().set(col, r);
        reference.insert(r);
    }

    for (auto obj : table) {
        int64_t v = obj.get<Int>(col);
        size_t c = table.count_int(col, v);
        size_t ref_count = reference.count(v);
        CHECK_EQUAL(c, ref_count);
    }
}

TEST_TYPES(StringIndex_Duplicate_Values, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    col.add(s1);
    col.add(s2);
    col.add(s3);
    col.add(s4);

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    CHECK(!ndx.has_duplicate_values());

    col.add(s1); // duplicate value

    CHECK(ndx.has_duplicate_values());

    // remove and test again.
    col.erase(4);
    CHECK(!ndx.has_duplicate_values());
    col.add(s1);
    CHECK(ndx.has_duplicate_values());
    col.erase(0);
    CHECK(!ndx.has_duplicate_values());
    col.clear();

    // check emptied set
    CHECK(ndx.is_empty());
    CHECK(!ndx.has_duplicate_values());

    const size_t num_rows = 100;

    for (size_t i = 0; i < num_rows; ++i) {
        std::string to_insert(util::to_string(i));
        col.add(to_insert);
    }
    CHECK(!ndx.has_duplicate_values());

    std::string a_string = "a";
    for (size_t i = 0; i < num_rows; ++i) {
        col.add(a_string);
        a_string += "a";
    }
    std::string str_num_rows(util::to_string(num_rows));
    CHECK(!ndx.has_duplicate_values());
    col.add(a_string);
    col.add(a_string);
    CHECK(ndx.has_duplicate_values());
    col.erase(col.size() - 1);
    CHECK(!ndx.has_duplicate_values());

    // Insert into the middle unique value of num_rows
    col.insert(num_rows / 2, str_num_rows);

    CHECK(!ndx.has_duplicate_values());

    // Set the next element to be num_rows too
    col.set(num_rows / 2 + 1, str_num_rows);

    CHECK(ndx.has_duplicate_values());

    col.clear();
    CHECK(!ndx.has_duplicate_values());
    CHECK(col.size() == 0);
}

TEST_TYPES(StringIndex_MaxBytes, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    std::string std_max(StringIndex::s_max_offset, 'a');
    std::string std_over_max(std_max + "a");
    std::string std_under_max(StringIndex::s_max_offset >> 1, 'a');
    StringData max(std_max);
    StringData over_max(std_over_max);
    StringData under_max(std_under_max);

    const StringIndex& ndx = *col.create_search_index();

    CHECK_EQUAL(col.size(), 0);

    auto duplicate_check = [&](size_t num_dups, StringData s) {
        CHECK(col.size() == 0);
        for (size_t i = 0; i < num_dups; ++i) {
            col.add(s);
        }
        CHECK_EQUAL(col.size(), num_dups);
        CHECK(ndx.has_duplicate_values() == (num_dups > 1));
        CHECK_EQUAL(col.get(0), s);
        CHECK_EQUAL(col.count(s), num_dups);
        CHECK_EQUAL(col.find_first(s), 0);
        col.clear();
    };

    std::vector<size_t> num_duplicates_list = {
        1, 10, REALM_MAX_BPNODE_SIZE - 1, REALM_MAX_BPNODE_SIZE, REALM_MAX_BPNODE_SIZE + 1,
    };
    for (auto& dups : num_duplicates_list) {
        duplicate_check(dups, under_max);
        duplicate_check(dups, max);
        duplicate_check(dups, over_max);
    }
}


// There is a corner case where two very long strings are
// inserted into the string index which are identical except
// for the characters at the end (they have an identical very
// long prefix). This was causing a stack overflow because of
// the recursive nature of the insert function.
TEST_TYPES(StringIndex_InsertLongPrefix, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();

    col.add("test_index_string1");
    col.add("test_index_string2");

    CHECK_EQUAL(col.find_first("test_index_string1"), 0);
    CHECK_EQUAL(col.find_first("test_index_string2"), 1);

    std::string std_base(107, 'a');
    std::string std_base_b = std_base + "b";
    std::string std_base_c = std_base + "c";
    StringData base_b(std_base_b);
    StringData base_c(std_base_c);
    col.add(base_b);
    col.add(base_c);

    CHECK_EQUAL(col.find_first(base_b), 2);
    CHECK_EQUAL(col.find_first(base_c), 3);

    // To trigger the bug, the length must be more than 10000.
    // Array::destroy_deep() will stack overflow at around recursion depths of
    // lengths > 90000 on mac and less on android devices.
    std::string std_base2(100000, 'a');
    std::string std_base2_b = std_base2 + "b";
    std::string std_base2_c = std_base2 + "c";
    StringData base2(std_base2);
    StringData base2_b(std_base2_b);
    StringData base2_c(std_base2_c);
    col.add(base2_b);
    col.add(base2_c);

    CHECK_EQUAL(col.find_first(base2_b), 4);
    CHECK_EQUAL(col.find_first(base2_c), 5);

    col.add(base2);
    CHECK(!ndx.has_duplicate_values());
    ndx.verify();
    col.add(base2_b); // adds a duplicate in the middle of the list

    CHECK(ndx.has_duplicate_values());
    std::vector<ObjKey> find_all_result;
    CHECK_EQUAL(col.find_first(base2_b), 4);
    ndx.find_all(find_all_result, base2_b);
    CHECK_EQUAL(find_all_result.size(), 2);
    CHECK_EQUAL(find_all_result[0], col.key(4));
    CHECK_EQUAL(find_all_result[1], col.key(7));
    find_all_result.clear();
    CHECK_EQUAL(ndx.count(base2_b), 2);
    col.verify();

    col.erase(7);
    CHECK_EQUAL(col.find_first(base2_b), 4);
    CHECK_EQUAL(ndx.count(base2_b), 1);
    ndx.find_all(find_all_result, base2_b);
    CHECK_EQUAL(find_all_result.size(), 1);
    CHECK_EQUAL(find_all_result[0], col.key(4));
    find_all_result.clear();
    col.verify();

    col.set(6, base2_b);
    CHECK_EQUAL(ndx.count(base2_b), 2);
    CHECK_EQUAL(col.find_first(base2_b), 4);
    ndx.find_all(find_all_result, base2_b);
    CHECK_EQUAL(find_all_result.size(), 2);
    CHECK_EQUAL(find_all_result[0], col.key(4));
    CHECK_EQUAL(find_all_result[1], col.key(6));
    col.verify();

    col.clear(); // calls recursive function Array::destroy_deep()
}

TEST_TYPES(StringIndex_InsertLongPrefixAndQuery, string_column, nullable_string_column, enum_column,
           nullable_enum_column)
{
    constexpr int half_node_size = REALM_MAX_BPNODE_SIZE / 2;
    bool nullable_column = TEST_TYPE::is_nullable();
    Group g;
    auto t = g.add_table("StringsOnly");
    auto col = t->add_column(type_String, "first", nullable_column);
    t->add_search_index(col);

    std::string base(StringIndex::s_max_offset, 'a');
    std::string str_a = base + "aaaaa";
    std::string str_a0 = base + "aaaa0";
    std::string str_ax = base + "aaaax";
    std::string str_b = base + "bbbbb";
    std::string str_c = base + "ccccc";
    std::string str_c0 = base + "cccc0";
    std::string str_cx = base + "ccccx";

    for (int i = 0; i < half_node_size * 3; i++) {
        t->create_object().set(col, str_a);
        t->create_object().set(col, str_b);
        t->create_object().set(col, str_c);
    }
    t->create_object().set(col, str_ax);
    t->create_object().set(col, str_ax);
    t->create_object().set(col, str_a0);
    /*
    {
        std::ofstream o("index.dot");
        index->to_dot(o, "");
    }
    */
    if (TEST_TYPE::is_enumerated())
        t->enumerate_string_column(col);

    auto ndx_a = t->where().equal(col, StringData(str_a)).find();
    auto cnt = t->count_string(col, StringData(str_a));
    auto tw_a = t->where().equal(col, StringData(str_a)).find_all();
    CHECK_EQUAL(ndx_a, ObjKey(0));
    CHECK_EQUAL(cnt, half_node_size * 3);
    CHECK_EQUAL(tw_a.size(), half_node_size * 3);
    ndx_a = t->where().equal(col, StringData(str_c0)).find();
    CHECK_EQUAL(ndx_a, null_key);
    ndx_a = t->where().equal(col, StringData(str_cx)).find();
    CHECK_EQUAL(ndx_a, null_key);
    // Find string that is 'less' than strings in the table, but with identical last key
    tw_a = t->where().equal(col, StringData(str_c0)).find_all();
    CHECK_EQUAL(tw_a.size(), 0);
    // Find string that is 'greater' than strings in the table, but with identical last key
    tw_a = t->where().equal(col, StringData(str_cx)).find_all();
    CHECK_EQUAL(tw_a.size(), 0);

    // Same as above, but just for 'count' method
    cnt = t->count_string(col, StringData(str_c0));
    CHECK_EQUAL(cnt, 0);
    cnt = t->count_string(col, StringData(str_cx));
    CHECK_EQUAL(cnt, 0);
}


TEST(StringIndex_Fuzzy)
{
    constexpr size_t chunkcount = 50;
    constexpr size_t rowcount = 100 + 1000 * TEST_DURATION;

    for (size_t main_rounds = 0; main_rounds < 2 + 10 * TEST_DURATION; main_rounds++) {

        Group g;

        auto t = g.add_table("StringsOnly");
        auto col0 = t->add_column(type_String, "first");
        auto col1 = t->add_column(type_String, "second");

        t->add_search_index(col0);

        std::string strings[chunkcount];

        for (size_t j = 0; j < chunkcount; j++) {
            size_t len = fastrand() % REALM_MAX_BPNODE_SIZE;

            for (size_t i = 0; i < len; i++)
                strings[j] += char(fastrand());
        }

        for (size_t rows = 0; rows < rowcount; rows++) {
            // Strings consisting of 2 concatenated strings are very interesting
            size_t chunks;
            if (fastrand() % 2 == 0)
                chunks = fastrand() % 4;
            else
                chunks = 2;

            std::string str;

            for (size_t c = 0; c < chunks; c++) {
                str += strings[fastrand() % chunkcount];
            }

            t->create_object().set_all(str, str);
        }

        for (size_t rounds = 0; rounds < 1 + 10 * TEST_DURATION; rounds++) {
            for (auto obj : *t) {

                TableView tv0 = (t->column<String>(col0) == obj.get<String>(col0)).find_all();
                TableView tv1 = (t->column<String>(col1) == obj.get<String>(col1)).find_all();

                CHECK_EQUAL(tv0.size(), tv1.size());

                for (size_t v = 0; v < tv0.size(); v++) {
                    CHECK_EQUAL(tv0.get_key(v), tv1.get_key(v));
                }
            }


            for (size_t r = 0; r < 5 + 1000 * TEST_DURATION; r++) {
                size_t chunks;
                if (fastrand() % 2 == 0)
                    chunks = fastrand() % 4;
                else
                    chunks = 2;

                std::string str;

                for (size_t c = 0; c < chunks; c++) {
                    str += strings[fastrand() % chunkcount];
                }

                TableView tv0 = (t->column<String>(col0) == str).find_all();
                TableView tv1 = (t->column<String>(col1) == str).find_all();

                CHECK_EQUAL(tv0.size(), tv1.size());

                for (size_t v = 0; v < tv0.size(); v++) {
                    CHECK_EQUAL(tv0.get_key(v), tv1.get_key(v));
                }
            }
            if (t->size() > 10)
                t->get_object(0).remove();

            size_t r1 = fastrand() % t->size();
            size_t r2 = fastrand() % t->size();

            std::string str = t->get_object(r2).get<String>(col0);
            Obj obj = t->get_object(r1);
            obj.set<String>(col0, StringData(str));
            obj.set<String>(col1, StringData(str));
        }
    }
}

namespace {

// results returned by the index should be in ascending row order
// this requirement is assumed by the query system which runs find_gte
// and this will return wrong results unless the results are ordered
void check_result_order(const std::vector<ObjKey>& results, TestContext& test_context)
{
    const size_t num_results = results.size();
    for (size_t i = 1; i < num_results; ++i) {
        CHECK(results[i - 1] < results[i]);
    }
}

} // end anonymous namespace


TEST_TYPES(StringIndex_Insensitive, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    const char* strings[] = {
        "john", "John", "jOhn", "JOhn", "joHn", "JoHn", "jOHn", "JOHn", "johN", "JohN", "jOhN", "JOhN", "joHN", "JoHN", "jOHN", "JOHN", "john" /* yes, an extra to test the "bucket" case as well */,
        "hans", "Hansapark", "george", "billion dollar startup",
        "abcde", "abcdE", "Abcde", "AbcdE",
        "common", "common"
    };

    for (const char* string : strings) {
        col.add(string);
    }

    // Generate 255 strings with 1..255 'a' chars
    for (int i = 1; i < 256; ++i) {
        col.add(std::string(i, 'a').c_str());
    }

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    std::vector<ObjKey> results;
    {
        // case sensitive
        ndx.find_all(results, strings[0]);
        CHECK_EQUAL(2, results.size());
        CHECK_EQUAL(col.get(results[0]), strings[0]);
        CHECK_EQUAL(col.get(results[1]), strings[0]);
        check_result_order(results, test_context);
        results.clear();
    }

    {
        constexpr bool case_insensitive = true;
        const char* needle = "john";
        auto upper_needle = case_map(needle, true);
        ndx.find_all(results, needle, case_insensitive);
        CHECK_EQUAL(17, results.size());
        for (size_t i = 0; i < results.size(); ++i) {
            auto upper_result = case_map(col.get(results[i]), true);
            CHECK_EQUAL(upper_result, upper_needle);

        }
        check_result_order(results, test_context);
        results.clear();
    }


    {
        struct TestData {
            const bool case_insensitive;
            const char* const needle;
            const size_t result_size;
        };

        TestData td[] = {
            {true, "Hans", 1},
            {true, "Geor", 0},
            {true, "George", 1},
            {true, "geoRge", 1},
            {true, "Billion Dollar Startup", 1},
            {true, "ABCDE", 4},
            {true, "commON", 2},
        };

        for (const TestData& t : td) {
            ndx.find_all(results, t.needle, t.case_insensitive);
            CHECK_EQUAL(t.result_size, results.size());
            check_result_order(results, test_context);
            results.clear();
        }
    }

    // Test generated 'a'-strings
    for (int i = 1; i < 256; ++i) {
        const std::string str = std::string(i, 'A');
        ndx.find_all(results, str.c_str(), false);
        CHECK_EQUAL(0, results.size());
        ndx.find_all(results, str.c_str(), true);
        CHECK_EQUAL(1, results.size());
        results.clear();
    }
}


/* Disabled until we have better support for case mapping unicode characters

TEST_TYPES(StringIndex_Insensitive_Unicode, non_nullable, nullable)
{
    constexpr bool nullable = TEST_TYPE::value;

    // Create a column with string values
    ref_type ref = StringColumn::create(Allocator::get_default());
    StringColumn col(Allocator::get_default(), ref, nullable);

    const char* strings[] = {
        "æøå", "ÆØÅ",
    };

    for (const char* string : strings) {
        col.add(string);
    }

    // Create a new index on column
    const StringIndex& ndx = *col.create_search_index();

    ref_type results_ref = IntegerColumn::create(Allocator::get_default());
    IntegerColumn results(Allocator::get_default(), results_ref);

    {
        struct TestData {
            const bool case_insensitive;
            const char* const needle;
            const size_t result_size;
        };

        TestData td[] = {
            {false, "æøå", 1},
            {false, "ÆØÅ", 1},
            {true, "æøå", 2},
            {true, "Æøå", 2},
            {true, "æØå", 2},
            {true, "ÆØå", 2},
            {true, "æøÅ", 2},
            {true, "ÆøÅ", 2},
            {true, "æØÅ", 2},
            {true, "ÆØÅ", 2},
        };

        for (const TestData& t : td) {
            ndx.find_all(results, t.needle, t.case_insensitive);
            CHECK_EQUAL(t.result_size, results.size());
            results.clear();
        }
    }

    // Clean up
    results.destroy();
    col.destroy();
}

*/


TEST_TYPES(StringIndex_45, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();
    std::string a4 = std::string(4, 'a');
    std::string A5 = std::string(5, 'A');

    col.add(a4);
    col.add(a4);

    std::vector<ObjKey> res;

    ndx.find_all(res, A5.c_str(), true);
    CHECK_EQUAL(res.size(), 0);
}


namespace {

std::string create_random_a_string(size_t max_len) {
    std::string s;
    size_t len = size_t(fastrand(max_len));
    for (size_t p = 0; p < len; p++) {
        s += fastrand(1) == 0 ? 'a' : 'A';
    }
    return s;
}

}


// Excluded when run with valgrind because it takes a long time
TEST_TYPES_IF(StringIndex_Insensitive_Fuzz, TEST_DURATION > 1, string_column, nullable_string_column, enum_column,
              nullable_enum_column)
{
    const size_t max_str_len = 9;
    const size_t iters = 3;

    for (size_t iter = 0; iter < iters; iter++) {
        TEST_TYPE test_resources;
        typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

        size_t rows = size_t(fastrand(2 * REALM_MAX_BPNODE_SIZE - 1));

        // Add 'rows' number of rows in the column
        for (size_t t = 0; t < rows; t++) {
            std::string str = create_random_a_string(max_str_len);
            col.add(str);
        }

        const StringIndex& ndx = *col.create_search_index();

        for (size_t t = 0; t < 1000; t++) {
            std::string needle = create_random_a_string(max_str_len);

            std::vector<ObjKey> res;

            ndx.find_all(res, needle.c_str(), true);
            check_result_order(res, test_context);

            // Check that all items in 'res' point at a match in 'col'
            auto needle_upper = case_map(needle, true);
            for (size_t res_ndx = 0; res_ndx < res.size(); res_ndx++) {
                auto res_upper = case_map(col.get(res[res_ndx]), true);
                CHECK_EQUAL(res_upper, needle_upper);
            }

            // Check that all matches in 'col' exist in 'res'
            for (size_t col_ndx = 0; col_ndx < col.size(); col_ndx++) {
                auto str_upper = case_map(col.get(col_ndx), true);
                if (str_upper == needle_upper) {
                    CHECK(std::find(res.begin(), res.end(), col.key(col_ndx)) != res.end());
                }
            }
        }
    }
}

// Exercise the StringIndex case insensitive search for strings with very long, common prefixes
// to cover the special case code paths where different strings are stored in a list.
TEST_TYPES(StringIndex_Insensitive_VeryLongStrings, string_column, nullable_string_column, enum_column,
           nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();

    std::string long1 = std::string(StringIndex::s_max_offset + 10, 'a');
    std::string long2 = long1 + "b";
    std::string long3 = long1 + "c";

    // Add the strings in a "random" order
    col.add(long1);
    col.add(long2);
    col.add(long2);
    col.add(long1);
    col.add(long3);
    col.add(long2);
    col.add(long1);
    col.add(long1);

    std::vector<ObjKey> results;

    ndx.find_all(results, long1.c_str(), true);
    CHECK_EQUAL(results.size(), 4);
    check_result_order(results, test_context);
    results.clear();
    ndx.find_all(results, long2.c_str(), true);
    CHECK_EQUAL(results.size(), 3);
    results.clear();
    ndx.find_all(results, long3.c_str(), true);
    CHECK_EQUAL(results.size(), 1);
    results.clear();
}


// Bug with case insensitive search on numbers that gives duplicate results
TEST_TYPES(StringIndex_Insensitive_Numbers, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();
    const StringIndex& ndx = *col.create_search_index();

    constexpr const char* number_string_16 = "1111111111111111";
    constexpr const char* number_string_17 = "11111111111111111";

    col.add(number_string_16);
    col.add(number_string_17);

    std::vector<ObjKey> results;

    ndx.find_all(results, number_string_16, true);
    CHECK_EQUAL(results.size(), 1);
}


TEST_TYPES(StringIndex_Rover, string_column, nullable_string_column, enum_column, nullable_enum_column)
{
    TEST_TYPE test_resources;
    typename TEST_TYPE::ColumnTestType& col = test_resources.get_column();

    const StringIndex& ndx = *col.create_search_index();

    col.add("ROVER");
    col.add("Rover");

    std::vector<ObjKey> results;

    ndx.find_all(results, "rover", true);
    CHECK_EQUAL(results.size(), 2);
    check_result_order(results, test_context);
}

TEST(StringIndex_QuerySingleObject)
{
    Group g;
    auto table = g.add_table_with_primary_key("class_StringClass", type_String, "name", true);
    table->create_object_with_primary_key("Foo");

    auto q = table->where().equal(table->get_column_key("name"), "Foo", true);
    CHECK_EQUAL(q.count(), 1);
    q = table->where().equal(table->get_column_key("name"), "Bar", true);
    CHECK_EQUAL(q.count(), 0);
}

TEST(StringIndex_MixedNonEmptyTable)
{
    Group g;
    auto table = g.add_table("foo");
    auto col = table->add_column(type_Mixed, "any");
    table->create_object().set(col, Mixed("abcdefgh"));
    table->add_search_index(col);
}

TEST(StringIndex_MixedEqualBitPattern)
{
    Group g;
    auto table = g.add_table("foo");
    auto col = table->add_column(type_Mixed, "any");
    table->add_search_index(col);

    Mixed val1(int64_t(0x6867666564636261));
    table->create_object().set(col, Mixed("abcdefgh"));
    // From single value to list
    table->create_object().set(col, val1);

    auto tv = table->where().equal(col, val1).find_all();
    CHECK_EQUAL(tv.size(), 1);
    CHECK_EQUAL(tv.get_object(0).get_any(col), val1);

    table->clear();
    table->create_object().set(col, Mixed("abcdefgh"));
    table->create_object().set(col, Mixed("abcdefgh"));
    // Insert in existing list
    table->create_object().set(col, val1);

    tv = table->where().equal(col, val1).find_all();
    CHECK_EQUAL(tv.size(), 1);
    CHECK_EQUAL(tv.get_object(0).get_any(col), val1);
    tv = table->where().equal(col, Mixed("abcdefgh")).find_all();
    CHECK_EQUAL(tv.size(), 2);

    // Add another one into existing list
    table->create_object().set(col, val1);
    tv = table->where().equal(col, val1).find_all();
    CHECK_EQUAL(tv.size(), 2);
    CHECK_EQUAL(tv.get_object(0).get_any(col), val1);
    CHECK_EQUAL(tv.get_object(1).get_any(col), val1);
}

TEST(Unicode_Casemap)
{
    std::string inp = "±ÀÁÂÃÄÅÆÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝß×÷";
    auto out = case_map(inp, false);
    if (CHECK(out)) {
        CHECK_EQUAL(*out, "±àáâãäåæèéêëìíîïñòóôõöøùúûüýß×÷");
    }
    out = case_map(*out, true);
    if (CHECK(out)) {
        CHECK_EQUAL(*out, inp);
    }

    inp = "A very old house 🏠 is on 🔥, we have to save the 🦄";
    out = case_map(inp, true);
    if (CHECK(out)) {
        CHECK_EQUAL(*out, "A VERY OLD HOUSE 🏠 IS ON 🔥, WE HAVE TO SAVE THE 🦄");
    }

    StringData trailing_garbage(inp.data(), 19); // String terminated inside icon
    out = case_map(trailing_garbage, true);
    CHECK_NOT(out);

    inp = "rødgrød med fløde";
    out = case_map(inp, true);
    if (CHECK(out)) {
        CHECK_EQUAL(*out, "RØDGRØD MED FLØDE");
    }
    out = case_map(out, false);
    if (CHECK(out)) {
        CHECK_EQUAL(*out, inp);
    }
}
#endif // TEST_INDEX_STRING
