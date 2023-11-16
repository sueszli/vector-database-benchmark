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

#include <realm/query_engine.hpp>

#include <realm/query_expression.hpp>
#include <realm/index_string.hpp>
#include <realm/db.hpp>
#include <realm/utilities.hpp>

namespace realm {

ParentNode::ParentNode(const ParentNode& from)
    : m_child(from.m_child ? from.m_child->clone() : nullptr)
    , m_condition_column_key(from.m_condition_column_key)
    , m_dD(from.m_dD)
    , m_dT(from.m_dT)
    , m_probes(from.m_probes)
    , m_matches(from.m_matches)
    , m_table(from.m_table)
{
}


size_t ParentNode::find_first(size_t start, size_t end)
{
    size_t sz = m_children.size();
    size_t current_cond = 0;
    size_t nb_cond_to_test = sz;

    while (REALM_LIKELY(start < end)) {
        size_t m = m_children[current_cond]->find_first_local(start, end);

        if (m != start) {
            // Pointer advanced - we will have to check all other conditions
            nb_cond_to_test = sz;
            start = m;
        }

        nb_cond_to_test--;

        // Optimized for one condition where this will be true first time
        if (REALM_LIKELY(nb_cond_to_test == 0))
            return m;

        current_cond++;

        if (current_cond == sz)
            current_cond = 0;
    }
    return not_found;
}

template <class T>
inline bool Obj::evaluate(T func) const
{
    REALM_ASSERT(is_valid());
    Cluster cluster(0, get_alloc(), m_table->m_clusters);
    cluster.init(m_mem);
    cluster.set_offset(m_key.value - cluster.get_key_value(m_row_ndx));
    return func(&cluster, m_row_ndx);
}

bool ParentNode::match(const Obj& obj)
{
    return obj.evaluate([this](const Cluster* cluster, size_t row) {
        set_cluster(cluster);
        size_t m = find_first(row, row + 1);
        return m != npos;
    });
}

size_t ParentNode::aggregate_local(QueryStateBase* st, size_t start, size_t end, size_t local_limit,
                                   ArrayPayload* source_column)
{
    // aggregate called on non-integer column type. Speed of this function is not as critical as speed of the
    // integer version, because find_first_local() is relatively slower here (because it's non-integers).
    //
    // Todo: Two speedups are possible. Simple: Initially test if there are no sub criterias and run
    // find_first_local()
    // in a tight loop if so (instead of testing if there are sub criterias after each match). Harder: Specialize
    // data type array to make array call match() directly on each match, like for integers.

    m_state = st;
    m_state->set_payload_column(source_column);
    size_t local_matches = 0;

    if (m_children.size() == 1) {
        return find_all_local(start, end);
    }

    size_t r = start - 1;
    for (;;) {
        if (local_matches == local_limit) {
            m_dD = double(r - start) / (local_matches + 1.1);
            return r + 1;
        }

        // Find first match in this condition node
        auto pos = r + 1;
        r = find_first_local(pos, end);
        if (r == not_found) {
            m_dD = double(pos - start) / (local_matches + 1.1);
            return end;
        }

        local_matches++;

        // Find first match in remaining condition nodes
        size_t m = r;

        for (size_t c = 1; c < m_children.size(); c++) {
            m = m_children[c]->find_first_local(r, r + 1);
            if (m != r) {
                break;
            }
        }

        // If index of first match in this node equals index of first match in all remaining nodes, we have a final
        // match
        if (m == r) {
            bool cont = st->match(r);
            if (!cont) {
                return static_cast<size_t>(-1);
            }
        }
    }
}

size_t ParentNode::find_all_local(size_t start, size_t end)
{
    while (start < end) {
        start = find_first_local(start, end);
        if (start != not_found) {
            bool cont = m_state->match(start);
            if (!cont) {
                return static_cast<size_t>(-1);
            }
            start++;
        }
    }
    return end;
}

void MixedNode<Equal>::init(bool will_query_ranges)
{
    MixedNodeBase::init(will_query_ranges);

    REALM_ASSERT(bool(m_index_evaluator) ==
                 (m_table.unchecked_ptr()->search_index_type(m_condition_column_key) == IndexType::General));
    if (m_index_evaluator) {
        auto index = ParentNode::m_table->get_search_index(ParentNode::m_condition_column_key);
        m_index_evaluator->init(index, m_value);
        m_dT = 0.0;
    }
    else {
        m_dT = 10.0;
    }
}

size_t MixedNode<Equal>::find_first_local(size_t start, size_t end)
{
    REALM_ASSERT(m_table);

    if (m_index_evaluator) {
        return m_index_evaluator->do_search_index(m_cluster, start, end);
    }
    else {
        return m_leaf->find_first(m_value, start, end);
    }

    return not_found;
}

void MixedNode<EqualIns>::init(bool will_query_ranges)
{
    MixedNodeBase::init(will_query_ranges);

    StringData val_as_string;
    if (m_value.is_type(type_String)) {
        val_as_string = m_value.get<StringData>();
    }
    else if (m_value.is_type(type_Binary)) {
        BinaryData bin = m_value.get<BinaryData>();
        val_as_string = StringData(bin.data(), bin.size());
    }
    REALM_ASSERT(bool(m_index_evaluator) ==
                 (m_table.unchecked_ptr()->search_index_type(m_condition_column_key) == IndexType::General));
    if (m_index_evaluator) {
        auto index = ParentNode::m_table->get_search_index(ParentNode::m_condition_column_key);
        if (!val_as_string.is_null()) {
            m_index_matches.clear();
            constexpr bool case_insensitive = true;
            index->find_all(m_index_matches, val_as_string, case_insensitive);
            m_index_evaluator->init(&m_index_matches);
        }
        else {
            // search for non string can use exact match
            m_index_evaluator->init(index, m_value);
        }
        m_dT = 0.0;
    }
    else {
        m_dT = 10.0;
        if (!val_as_string.is_null()) {
            auto upper = case_map(val_as_string, true);
            auto lower = case_map(val_as_string, false);
            if (!upper || !lower) {
                throw query_parser::InvalidQueryError(util::format("Malformed UTF-8: %1", val_as_string));
            }
            else {
                m_ucase = std::move(*upper);
                m_lcase = std::move(*lower);
            }
        }
    }
}

size_t MixedNode<EqualIns>::find_first_local(size_t start, size_t end)
{
    REALM_ASSERT(m_table);

    EqualIns cond;
    if (m_value.is_type(type_String)) {
        for (size_t i = start; i < end; i++) {
            QueryValue val(m_leaf->get(i));
            StringData val_as_str;
            if (val.is_type(type_String)) {
                val_as_str = val.get<StringData>();
            }
            else if (val.is_type(type_Binary)) {
                val_as_str = StringData(val.get<BinaryData>().data(), val.get<BinaryData>().size());
            }
            if (!val_as_str.is_null() &&
                cond(m_value.get<StringData>(), m_ucase.c_str(), m_lcase.c_str(), val_as_str))
                return i;
        }
    }
    else {
        for (size_t i = start; i < end; i++) {
            QueryValue val(m_leaf->get(i));
            if (cond(val, m_value))
                return i;
        }
    }

    return not_found;
}

void StringNodeEqualBase::init(bool will_query_ranges)
{
    StringNodeBase::init(will_query_ranges);

    const bool uses_index = has_search_index();
    if (m_is_string_enum) {
        m_dT = 1.0;
    }
    else if (uses_index) {
        m_dT = 0.0;
    }
    else {
        m_dT = 10.0;
    }

    if (uses_index) {
        _search_index_init();
    }
}

size_t StringNodeEqualBase::find_first_local(size_t start, size_t end)
{
    REALM_ASSERT(m_table);

    if (m_index_evaluator) {
        return m_index_evaluator->do_search_index(m_cluster, start, end);
    }

    return _find_first_local(start, end);
}


void IndexEvaluator::init(StringIndex* index, Mixed value)
{
    REALM_ASSERT(index);
    m_matching_keys = nullptr;
    FindRes fr;
    InternalFindResult res;

    m_last_start_key = ObjKey();
    m_results_start = 0;
    fr = index->find_all_no_copy(value, res);

    m_index_matches.reset();
    switch (fr) {
        case FindRes_single:
            m_actual_key = ObjKey(res.payload);
            m_results_end = 1;
            break;
        case FindRes_column:
            m_index_matches.reset(new IntegerColumn(index->get_alloc(), ref_type(res.payload))); // Throws
            m_results_start = res.start_ndx;
            m_results_end = res.end_ndx;
            m_actual_key = ObjKey(m_index_matches->get(m_results_start));
            break;
        case FindRes_not_found:
            m_results_end = 0;
            break;
    }
    m_results_ndx = m_results_start;
}

void IndexEvaluator::init(std::vector<ObjKey>* storage)
{
    REALM_ASSERT(storage);
    m_matching_keys = storage;
    m_actual_key = ObjKey();
    m_last_start_key = ObjKey();
    m_results_start = 0;
    m_results_end = m_matching_keys->size();
    m_results_ndx = 0;
    if (m_results_start != m_results_end) {
        m_actual_key = m_matching_keys->at(0);
    }
}

size_t IndexEvaluator::do_search_index(const Cluster* cluster, size_t start, size_t end)
{
    if (start >= end) {
        return not_found;
    }

    ObjKey first_key = cluster->get_real_key(start);
    if (first_key < m_last_start_key) {
        // We are not advancing through the clusters. We basically don't know where we are,
        // so just start over from the beginning.
        m_results_ndx = m_results_start;
        m_actual_key = (m_results_start != m_results_end) ? get_internal(m_results_start) : ObjKey();
    }
    m_last_start_key = first_key;

    // Check if we can expect to find more keys
    if (m_results_ndx < m_results_end) {
        // Check if we should advance to next key to search for
        while (first_key > m_actual_key) {
            m_results_ndx++;
            if (m_results_ndx == m_results_end) {
                return not_found;
            }
            m_actual_key = get_internal(m_results_ndx);
        }

        // If actual_key is bigger than last key, it is not in this leaf
        ObjKey last_key = cluster->get_real_key(end - 1);
        if (m_actual_key > last_key)
            return not_found;

        // Now actual_key must be found in leaf keys
        return cluster->lower_bound_key(ObjKey(m_actual_key.value - cluster->get_offset()));
    }
    return not_found;
}

void StringNode<Equal>::_search_index_init()
{
    REALM_ASSERT(bool(m_index_evaluator));
    auto index = ParentNode::m_table.unchecked_ptr()->get_search_index(ParentNode::m_condition_column_key);
    m_index_evaluator->init(index, StringData(StringNodeBase::m_value));
}

bool StringNode<Equal>::do_consume_condition(ParentNode& node)
{
    // Don't use the search index if present since we're in a scenario where
    // it'd be slower
    m_index_evaluator.reset();

    auto& other = static_cast<StringNode<Equal>&>(node);
    REALM_ASSERT(m_condition_column_key == other.m_condition_column_key);
    REALM_ASSERT(other.m_needles.empty());
    if (m_needles.empty()) {
        m_needles.insert(m_value ? StringData(*m_value) : StringData());
    }
    if (auto& str = other.m_value) {
        m_needle_storage.push_back(std::make_unique<char[]>(str->size()));
        std::copy(str->data(), str->data() + str->size(), m_needle_storage.back().get());
        m_needles.insert(StringData(m_needle_storage.back().get(), str->size()));
    }
    else {
        m_needles.emplace();
    }
    return true;
}

size_t StringNode<Equal>::_find_first_local(size_t start, size_t end)
{
    if (m_needles.empty()) {
        return m_leaf->find_first(m_value, start, end);
    }
    else {
        if (end == npos)
            end = m_leaf->size();
        REALM_ASSERT_3(start, <=, end);
        return find_first_haystack<20>(*m_leaf, m_needles, start, end);
    }
}

std::string StringNode<Equal>::describe(util::serializer::SerialisationState& state) const
{
    if (m_needles.empty()) {
        return StringNodeEqualBase::describe(state);
    }

    std::string list_contents;
    bool is_first = true;
    for (auto it : m_needles) {
        StringData sd(it.data(), it.size());
        list_contents += util::format("%1%2", is_first ? "" : ", ", util::serializer::print_value(sd));
        is_first = false;
    }
    std::string col_descr = state.describe_column(ParentNode::m_table, m_condition_column_key);
    std::string desc = util::format("%1 IN {%2}", col_descr, list_contents);
    return desc;
}


void StringNode<EqualIns>::_search_index_init()
{
    auto index = ParentNode::m_table->get_search_index(ParentNode::m_condition_column_key);
    m_index_matches.clear();
    constexpr bool case_insensitive = true;
    index->find_all(m_index_matches, StringData(StringNodeBase::m_value), case_insensitive);
    m_index_evaluator->init(&m_index_matches);
}

size_t StringNode<EqualIns>::_find_first_local(size_t start, size_t end)
{
    EqualIns cond;
    for (size_t s = start; s < end; ++s) {
        StringData t = get_string(s);

        if (cond(StringData(m_value), m_ucase.c_str(), m_lcase.c_str(), t))
            return s;
    }

    return not_found;
}

StringNodeFulltext::StringNodeFulltext(StringData v, ColKey column, std::unique_ptr<LinkMap> lm)
    : StringNodeEqualBase(v, column)
    , m_link_map(std::move(lm))
{
    if (!m_link_map)
        m_link_map = std::make_unique<LinkMap>();
}

void StringNodeFulltext::table_changed()
{
    StringNodeEqualBase::table_changed();
    m_link_map->set_base_table(m_table);
}

StringNodeFulltext::StringNodeFulltext(const StringNodeFulltext& other)
    : StringNodeEqualBase(other)
{
    m_link_map = std::make_unique<LinkMap>(*other.m_link_map);
}

void StringNodeFulltext::_search_index_init()
{
    auto index = m_link_map->get_target_table()->get_search_index(ParentNode::m_condition_column_key);
    REALM_ASSERT(index && index->is_fulltext_index());
    m_index_matches.clear();
    index->find_all_fulltext(m_index_matches, StringData(StringNodeBase::m_value));

    // If links exists, use backlinks to find the original objects
    if (m_link_map->links_exist()) {
        std::set<ObjKey> tmp;
        for (auto k : m_index_matches) {
            auto ndxs = m_link_map->get_origin_ndxs(k);
            tmp.insert(ndxs.begin(), ndxs.end());
        }
        m_index_matches.assign(tmp.begin(), tmp.end());
    }

    m_index_evaluator = IndexEvaluator{};
    m_index_evaluator->init(&m_index_matches);
}

std::unique_ptr<ArrayPayload> TwoColumnsNodeBase::update_cached_leaf_pointers_for_column(Allocator& alloc,
                                                                                         const ColKey& col_key)
{
    switch (col_key.get_type()) {
        case col_type_Int:
            if (col_key.is_nullable()) {
                return std::make_unique<ArrayIntNull>(alloc);
            }
            return std::make_unique<ArrayInteger>(alloc);
        case col_type_Bool:
            return std::make_unique<ArrayBoolNull>(alloc);
        case col_type_String:
            return std::make_unique<ArrayString>(alloc);
        case col_type_Binary:
            return std::make_unique<ArrayBinary>(alloc);
        case col_type_Mixed:
            return std::make_unique<ArrayMixed>(alloc);
        case col_type_Timestamp:
            return std::make_unique<ArrayTimestamp>(alloc);
        case col_type_Float:
            return std::make_unique<ArrayFloatNull>(alloc);
        case col_type_Double:
            return std::make_unique<ArrayDoubleNull>(alloc);
        case col_type_Decimal:
            return std::make_unique<ArrayDecimal128>(alloc);
        case col_type_Link:
            return std::make_unique<ArrayKey>(alloc);
        case col_type_ObjectId:
            return std::make_unique<ArrayObjectIdNull>(alloc);
        case col_type_UUID:
            return std::make_unique<ArrayUUIDNull>(alloc);
        case col_type_TypedLink:
        case col_type_BackLink:
        case col_type_LinkList:
            break;
    };
    REALM_UNREACHABLE();
    return {};
}

size_t size_of_list_from_ref(ref_type ref, Allocator& alloc, ColumnType col_type, bool is_nullable)
{
    switch (col_type) {
        case col_type_Int: {
            if (is_nullable) {
                BPlusTree<util::Optional<Int>> list(alloc);
                list.init_from_ref(ref);
                return list.size();
            }
            else {
                BPlusTree<Int> list(alloc);
                list.init_from_ref(ref);
                return list.size();
            }
        }
        case col_type_Bool: {
            BPlusTree<Bool> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_String: {
            BPlusTree<String> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Binary: {
            BPlusTree<Binary> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Timestamp: {
            BPlusTree<Timestamp> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Float: {
            BPlusTree<Float> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Double: {
            BPlusTree<Double> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Decimal: {
            BPlusTree<Decimal128> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_ObjectId: {
            BPlusTree<ObjectId> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_UUID: {
            BPlusTree<UUID> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Mixed: {
            BPlusTree<Mixed> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_LinkList: {
            BPlusTree<ObjKey> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_TypedLink: {
            BPlusTree<ObjLink> list(alloc);
            list.init_from_ref(ref);
            return list.size();
        }
        case col_type_Link:
        case col_type_BackLink:
            break;
    }
    REALM_TERMINATE("Unsupported column type.");
}

size_t NotNode::find_first_local(size_t start, size_t end)
{
    if (start <= m_known_range_start && end >= m_known_range_end) {
        return find_first_covers_known(start, end);
    }
    else if (start >= m_known_range_start && end <= m_known_range_end) {
        return find_first_covered_by_known(start, end);
    }
    else if (start < m_known_range_start && end >= m_known_range_start) {
        return find_first_overlap_lower(start, end);
    }
    else if (start <= m_known_range_end && end > m_known_range_end) {
        return find_first_overlap_upper(start, end);
    }
    else { // start > m_known_range_end || end < m_known_range_start
        return find_first_no_overlap(start, end);
    }
}

bool NotNode::evaluate_at(size_t rowndx)
{
    return m_condition->find_first(rowndx, rowndx + 1) == not_found;
}

void NotNode::update_known(size_t start, size_t end, size_t first)
{
    m_known_range_start = start;
    m_known_range_end = end;
    m_first_in_known_range = first;
}

size_t NotNode::find_first_loop(size_t start, size_t end)
{
    for (size_t i = start; i < end; ++i) {
        if (evaluate_at(i)) {
            return i;
        }
    }
    return not_found;
}

size_t NotNode::find_first_covers_known(size_t start, size_t end)
{
    // CASE: start-end covers the known range
    // [    ######    ]
    REALM_ASSERT_DEBUG(start <= m_known_range_start && end >= m_known_range_end);
    size_t result = find_first_loop(start, m_known_range_start);
    if (result != not_found) {
        update_known(start, m_known_range_end, result);
    }
    else {
        if (m_first_in_known_range != not_found) {
            update_known(start, m_known_range_end, m_first_in_known_range);
            result = m_first_in_known_range;
        }
        else {
            result = find_first_loop(m_known_range_end, end);
            update_known(start, end, result);
        }
    }
    return result;
}

size_t NotNode::find_first_covered_by_known(size_t start, size_t end)
{
    REALM_ASSERT_DEBUG(start >= m_known_range_start && end <= m_known_range_end);
    // CASE: the known range covers start-end
    // ###[#####]###
    if (m_first_in_known_range != not_found) {
        if (m_first_in_known_range > end) {
            return not_found;
        }
        else if (m_first_in_known_range >= start) {
            return m_first_in_known_range;
        }
    }
    // The first known match is before start, so we can't use the results to improve
    // heuristics.
    return find_first_loop(start, end);
}

size_t NotNode::find_first_overlap_lower(size_t start, size_t end)
{
    REALM_ASSERT_DEBUG(start < m_known_range_start && end >= m_known_range_start && end <= m_known_range_end);
    static_cast<void>(end);
    // CASE: partial overlap, lower end
    // [   ###]#####
    size_t result;
    result = find_first_loop(start, m_known_range_start);
    if (result == not_found) {
        result = m_first_in_known_range;
    }
    update_known(start, m_known_range_end, result);
    return result < end ? result : not_found;
}

size_t NotNode::find_first_overlap_upper(size_t start, size_t end)
{
    REALM_ASSERT_DEBUG(start <= m_known_range_end && start >= m_known_range_start && end > m_known_range_end);
    // CASE: partial overlap, upper end
    // ####[###    ]
    size_t result;
    if (m_first_in_known_range != not_found) {
        if (m_first_in_known_range >= start) {
            result = m_first_in_known_range;
            update_known(m_known_range_start, end, result);
        }
        else {
            result = find_first_loop(start, end);
            update_known(m_known_range_start, end, m_first_in_known_range);
        }
    }
    else {
        result = find_first_loop(m_known_range_end, end);
        update_known(m_known_range_start, end, result);
    }
    return result;
}

size_t NotNode::find_first_no_overlap(size_t start, size_t end)
{
    REALM_ASSERT_DEBUG((start < m_known_range_start && end < m_known_range_start) ||
                       (start > m_known_range_end && end > m_known_range_end));
    // CASE: no overlap
    // ### [    ]   or    [    ] ####
    // if input is a larger range, discard and replace with results.
    size_t result = find_first_loop(start, end);
    if (end - start > m_known_range_end - m_known_range_start) {
        update_known(start, end, result);
    }
    return result;
}

ExpressionNode::ExpressionNode(std::unique_ptr<Expression> expression)
    : m_expression(std::move(expression))
{
    m_dT = 50.0;
}

void ExpressionNode::table_changed()
{
    m_expression->set_base_table(m_table);
}

void ExpressionNode::cluster_changed()
{
    m_expression->set_cluster(m_cluster);
}

void ExpressionNode::init(bool will_query_ranges)
{
    ParentNode::init(will_query_ranges);
    m_dT = m_expression->init();
}

std::string ExpressionNode::describe(util::serializer::SerialisationState& state) const
{
    if (m_expression) {
        return m_expression->description(state);
    }
    else {
        return "empty expression";
    }
}

void ExpressionNode::collect_dependencies(std::vector<TableKey>& tables) const
{
    m_expression->collect_dependencies(tables);
}

size_t ExpressionNode::find_first_local(size_t start, size_t end)
{
    return m_expression->find_first(start, end);
}

std::unique_ptr<ParentNode> ExpressionNode::clone() const
{
    return std::unique_ptr<ParentNode>(new ExpressionNode(*this));
}

ExpressionNode::ExpressionNode(const ExpressionNode& from)
    : ParentNode(from)
    , m_expression(from.m_expression->clone())
{
}

template <>
size_t LinksToNode<Equal>::find_first_local(size_t start, size_t end)
{
    if (m_condition_column_key.is_collection()) {
        Allocator& alloc = m_table.unchecked_ptr()->get_alloc();
        if (m_condition_column_key.is_dictionary()) {
            auto target_table_key = m_table->get_opposite_table(m_condition_column_key)->get_key();
            Array top(alloc);
            for (size_t i = start; i < end; i++) {
                if (ref_type ref = get_ref(i)) {
                    top.init_from_ref(ref);
                    BPlusTree<Mixed> values(alloc);
                    values.set_parent(&top, 1);
                    values.init_from_parent();
                    for (auto& key : m_target_keys) {
                        ObjLink link(target_table_key, key);
                        if (values.find_first(link) != not_found)
                            return i;
                    }
                }
            }
        }
        else {
            // LinkLists never contain null
            if (!m_target_keys[0] && m_target_keys.size() == 1 && start != end)
                return not_found;

            BPlusTree<ObjKey> links(alloc);
            for (size_t i = start; i < end; i++) {
                if (ref_type ref = get_ref(i)) {
                    links.init_from_ref(ref);
                    for (auto& key : m_target_keys) {
                        if (key) {
                            if (links.find_first(key) != not_found)
                                return i;
                        }
                    }
                }
            }
        }
    }
    else if (m_list) {
        for (auto& key : m_target_keys) {
            auto pos = m_list->find_first(key, start, end);
            if (pos != realm::npos) {
                return pos;
            }
        }
    }

    return not_found;
}

template <>
size_t LinksToNode<NotEqual>::find_first_local(size_t start, size_t end)
{
    // NotEqual only makes sense for a single value
    REALM_ASSERT(m_target_keys.size() == 1);
    ObjKey key = m_target_keys[0];

    if (m_condition_column_key.is_collection()) {
        Allocator& alloc = m_table.unchecked_ptr()->get_alloc();
        if (m_condition_column_key.is_dictionary()) {
            auto target_table_key = m_table->get_opposite_table(m_condition_column_key)->get_key();
            Array top(alloc);
            for (size_t i = start; i < end; i++) {
                if (ref_type ref = get_ref(i)) {
                    top.init_from_ref(ref);
                    BPlusTree<Mixed> values(alloc);
                    values.set_parent(&top, 1);
                    values.init_from_parent();

                    ObjLink link(target_table_key, key);
                    bool found = false;
                    values.for_all([&](const Mixed& val) {
                        if (val != link) {
                            found = true;
                        }
                        return !found;
                    });
                    if (found)
                        return i;
                }
            }
        }
        else {
            BPlusTree<ObjKey> links(alloc);
            for (size_t i = start; i < end; i++) {
                if (ref_type ref = get_ref(i)) {
                    links.init_from_ref(ref);
                    auto sz = links.size();
                    for (size_t j = 0; j < sz; j++) {
                        if (links.get(j) != key) {
                            return i;
                        }
                    }
                }
            }
        }
    }
    else if (m_list) {
        for (size_t i = start; i < end; i++) {
            if (m_list->get(i) != key) {
                return i;
            }
        }
    }

    return not_found;
}

} // namespace realm
