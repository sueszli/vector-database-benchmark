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

#include <realm/query_expression.hpp>
#include <realm/group.hpp>
#include <realm/dictionary.hpp>

namespace realm {

void LinkMap::set_base_table(ConstTableRef table)
{
    if (table == get_base_table())
        return;

    m_tables.clear();
    m_tables.push_back(table);
    m_link_types.clear();
    m_only_unary_links = true;

    for (size_t i = 0; i < m_link_column_keys.size(); i++) {
        ColKey link_column_key = m_link_column_keys[i];
        // Link column can be either LinkList or single Link
        ColumnType type = link_column_key.get_type();
        REALM_ASSERT(Table::is_link_type(type) || type == col_type_BackLink);
        if (type == col_type_LinkList || type == col_type_BackLink ||
            (type == col_type_Link && link_column_key.is_collection())) {
            m_only_unary_links = false;
        }

        m_link_types.push_back(type);
        REALM_ASSERT(table->valid_column(link_column_key));
        table = table.unchecked_ptr()->get_opposite_table(link_column_key);
        m_tables.push_back(table);
    }
}

void LinkMap::collect_dependencies(std::vector<TableKey>& tables) const
{
    for (auto& t : m_tables) {
        TableKey k = t->get_key();
        if (find(tables.begin(), tables.end(), k) == tables.end()) {
            tables.push_back(k);
        }
    }
}

std::string LinkMap::description(util::serializer::SerialisationState& state) const
{
    std::string s;
    for (size_t i = 0; i < m_link_column_keys.size(); ++i) {
        if (i < m_tables.size() && m_tables[i]) {
            s += state.get_column_name(m_tables[i], m_link_column_keys[i]);
            if (i != m_link_column_keys.size() - 1) {
                s += util::serializer::value_separator;
            }
        }
    }
    return s;
}

bool LinkMap::map_links(size_t column, ObjKey key, LinkMapFunction lm) const
{
    if (!key || key.is_unresolved())
        return true;
    if (column == m_link_column_keys.size()) {
        return lm(key);
    }

    ColumnType type = m_link_types[column];
    ColKey column_key = m_link_column_keys[column];
    const Obj obj = m_tables[column]->get_object(key);
    if (column_key.is_collection()) {
        auto coll = obj.get_linkcollection_ptr(column_key);
        size_t sz = coll->size();
        for (size_t t = 0; t < sz; t++) {
            if (!map_links(column + 1, coll->get_key(t), lm))
                return false;
        }
    }
    else if (type == col_type_Link) {
        return map_links(column + 1, obj.get<ObjKey>(column_key), lm);
    }
    else if (type == col_type_BackLink) {
        auto backlinks = obj.get_all_backlinks(column_key);
        for (auto k : backlinks) {
            if (!map_links(column + 1, k, lm))
                return false;
        }
    }
    else {
        REALM_TERMINATE("Invalid column type in LinkMap::map_links()");
    }
    return true;
}

void LinkMap::map_links(size_t column, size_t row, LinkMapFunction lm) const
{
    ColumnType type = m_link_types[column];
    ColKey column_key = m_link_column_keys[column];
    if (type == col_type_Link && !column_key.is_set()) {
        if (column_key.is_dictionary()) {
            auto& leaf = mpark::get<ArrayInteger>(m_leaf);
            if (leaf.get(row)) {
                Allocator& alloc = get_base_table()->get_alloc();
                Array top(alloc);
                top.set_parent(const_cast<ArrayInteger*>(&leaf), row);
                top.init_from_parent();
                BPlusTree<Mixed> values(alloc);
                values.set_parent(&top, 1);
                values.init_from_parent();

                // Iterate through values and insert all link values
                values.for_all([&](Mixed m) {
                    if (m.is_type(type_TypedLink)) {
                        auto link = m.get_link();
                        REALM_ASSERT(link.get_table_key() == this->m_tables[column + 1]->get_key());
                        if (!map_links(column + 1, link.get_obj_key(), lm))
                            return false;
                    }
                    return true;
                });
            }
        }
        else {
            REALM_ASSERT(!column_key.is_collection());
            map_links(column + 1, mpark::get<ArrayKey>(m_leaf).get(row), lm);
        }
    }
    else if (type == col_type_LinkList || (type == col_type_Link && column_key.is_set())) {
        ref_type ref;
        if (auto list = mpark::get_if<ArrayList>(&m_leaf)) {
            ref = list->get(row);
        }
        else {
            ref = mpark::get<ArrayKey>(m_leaf).get_as_ref(row);
        }

        if (ref) {
            BPlusTree<ObjKey> links(get_base_table()->get_alloc());
            links.init_from_ref(ref);
            size_t sz = links.size();
            for (size_t t = 0; t < sz; t++) {
                if (!map_links(column + 1, links.get(t), lm))
                    break;
            }
        }
    }
    else if (type == col_type_BackLink) {
        auto& back_links = mpark::get<ArrayBacklink>(m_leaf);
        size_t sz = back_links.get_backlink_count(row);
        for (size_t t = 0; t < sz; t++) {
            ObjKey k = back_links.get_backlink(row, t);
            if (!map_links(column + 1, k, lm))
                break;
        }
    }
    else {
        REALM_ASSERT(false);
    }
}

std::vector<ObjKey> LinkMap::get_origin_ndxs(ObjKey key, size_t column) const
{
    if (column == m_link_types.size()) {
        return {key};
    }
    std::vector<ObjKey> keys = get_origin_ndxs(key, column + 1);
    std::vector<ObjKey> ret;
    auto origin_col = m_link_column_keys[column];
    auto origin = m_tables[column];
    auto link_type = m_link_types[column];
    if (link_type == col_type_BackLink) {
        auto link_table = origin->get_opposite_table(origin_col);
        ColKey link_col_key = origin->get_opposite_column(origin_col);

        for (auto k : keys) {
            const Obj o = link_table.unchecked_ptr()->get_object(k);
            if (link_col_key.is_collection()) {
                auto coll = o.get_linkcollection_ptr(link_col_key);
                auto sz = coll->size();
                for (size_t i = 0; i < sz; i++) {
                    if (ObjKey x = coll->get_key(i))
                        ret.push_back(x);
                }
            }
            else if (link_col_key.get_type() == col_type_Link) {
                ret.push_back(o.get<ObjKey>(link_col_key));
            }
        }
    }
    else {
        auto target = m_tables[column + 1];
        for (auto k : keys) {
            const Obj o = target->get_object(k);
            auto cnt = o.get_backlink_count(*origin, origin_col);
            for (size_t i = 0; i < cnt; i++) {
                ret.push_back(o.get_backlink(*origin, origin_col, i));
            }
        }
    }
    return ret;
}

ColumnDictionaryKey Columns<Dictionary>::key(const Mixed& key_value)
{
    if (m_key_type != type_Mixed && key_value.get_type() != m_key_type) {
        throw InvalidArgument(ErrorCodes::TypeMismatch, util::format("Key not a %1", m_key_type));
    }

    return ColumnDictionaryKey(key_value, *this);
}

ColumnDictionaryKeys Columns<Dictionary>::keys()
{
    return ColumnDictionaryKeys(*this);
}

void ColumnDictionaryKey::init_key(Mixed key_value)
{
    REALM_ASSERT(!key_value.is_null());

    m_key = key_value;
    m_key.use_buffer(m_buffer);
}

void ColumnDictionaryKeys::set_cluster(const Cluster* cluster)
{
    if (m_link_map.has_links()) {
        m_link_map.set_cluster(cluster);
    }
    else {
        m_leaf.emplace(m_link_map.get_base_table()->get_alloc());
        cluster->init_leaf(m_column_key, &*m_leaf);
    }
}


void ColumnDictionaryKeys::evaluate(size_t index, ValueBase& destination)
{
    if (m_link_map.has_links()) {
        REALM_ASSERT(!m_leaf);
        std::vector<ObjKey> links = m_link_map.get_links(index);
        auto sz = links.size();

        // Here we don't really know how many values to expect
        std::vector<Mixed> values;
        for (size_t t = 0; t < sz; t++) {
            const Obj obj = m_link_map.get_target_table()->get_object(links[t]);
            auto dict = obj.get_dictionary(m_column_key);
            // Insert all values
            dict.for_all_keys<StringData>([&values](const Mixed& value) {
                values.emplace_back(value);
            });
        }

        // Copy values over
        destination.init(true, values.size());
        destination.set(values.begin(), values.end());
    }
    else {
        // Not a link column
        Allocator& alloc = get_base_table()->get_alloc();

        REALM_ASSERT(m_leaf);
        if (m_leaf->get(index)) {
            Array top(alloc);
            top.set_parent(&*m_leaf, index);
            top.init_from_parent();
            BPlusTree<StringData> keys(alloc);
            keys.set_parent(&top, 0);
            keys.init_from_parent();

            destination.init(true, keys.size());
            size_t n = 0;
            // Iterate through BPlusTree and insert all keys
            keys.for_all([&](StringData str) {
                destination.set(n, str);
                n++;
            });
        }
    }
}

void ColumnDictionaryKey::evaluate(size_t index, ValueBase& destination)
{
    if (links_exist()) {
        REALM_ASSERT(!m_leaf);
        std::vector<ObjKey> links = m_link_map.get_links(index);
        auto sz = links.size();

        destination.init_for_links(m_link_map.only_unary_links(), sz);
        for (size_t t = 0; t < sz; t++) {
            const Obj obj = m_link_map.get_target_table()->get_object(links[t]);
            auto dict = obj.get_dictionary(m_column_key);
            Mixed val;
            if (auto opt_val = dict.try_get(m_key)) {
                val = *opt_val;
                if (m_prop_list.size()) {
                    if (val.is_type(type_TypedLink)) {
                        auto obj = get_base_table()->get_parent_group()->get_object(val.get<ObjLink>());
                        val = obj.get_any(m_prop_list.begin(), m_prop_list.end());
                    }
                    else {
                        val = {};
                    }
                }
            }
            destination.set(t, val);
        }
    }
    else {
        // Not a link column
        Allocator& alloc = get_base_table()->get_alloc();

        REALM_ASSERT(m_leaf);
        if (m_leaf->get(index)) {
            Array top(alloc);
            top.set_parent(&*m_leaf, index);
            top.init_from_parent();
            BPlusTree<StringData> keys(alloc);
            keys.set_parent(&top, 0);
            keys.init_from_parent();

            Mixed val;
            size_t ndx = keys.find_first(m_key.get_string());
            if (ndx != realm::npos) {
                BPlusTree<Mixed> values(alloc);
                values.set_parent(&top, 1);
                values.init_from_parent();
                val = values.get(ndx);
                if (m_prop_list.size()) {
                    if (val.is_type(type_TypedLink)) {
                        auto obj = get_base_table()->get_parent_group()->get_object(val.get<ObjLink>());
                        val = obj.get_any(m_prop_list.begin(), m_prop_list.end());
                    }
                    else {
                        val = {};
                    }
                }
            }
            destination.set(0, val);
        }
    }
}

class DictionarySize : public Columns<Dictionary> {
public:
    DictionarySize(const Columns<Dictionary>& other)
        : Columns<Dictionary>(other)
    {
    }
    void evaluate(size_t index, ValueBase& destination) override
    {
        Allocator& alloc = this->m_link_map.get_target_table()->get_alloc();
        Value<int64_t> list_refs;
        this->get_lists(index, list_refs, 1);
        destination.init(list_refs.m_from_list, list_refs.size());
        for (size_t i = 0; i < list_refs.size(); i++) {
            ref_type ref = to_ref(list_refs[i].get_int());
            size_t s = _impl::get_collection_size_from_ref(ref, alloc);
            destination.set(i, int64_t(s));
        }
    }

    std::unique_ptr<Subexpr> clone() const override
    {
        return std::unique_ptr<Subexpr>(new DictionarySize(*this));
    }
};

SizeOperator<int64_t> Columns<Dictionary>::size()
{
    std::unique_ptr<Subexpr> ptr(new DictionarySize(*this));
    return SizeOperator<int64_t>(std::move(ptr));
}

void Columns<Dictionary>::evaluate(size_t index, ValueBase& destination)
{
    if (links_exist()) {
        REALM_ASSERT(!m_leaf);
        std::vector<ObjKey> links = m_link_map.get_links(index);
        auto sz = links.size();

        // Here we don't really know how many values to expect
        std::vector<Mixed> values;
        for (size_t t = 0; t < sz; t++) {
            const Obj obj = m_link_map.get_target_table()->get_object(links[t]);
            auto dict = obj.get_dictionary(m_column_key);
            // Insert all values
            dict.for_all_values([&values](const Mixed& value) {
                values.emplace_back(value);
            });
        }

        // Copy values over
        destination.init(true, values.size());
        destination.set(values.begin(), values.end());
    }
    else {
        // Not a link column
        Allocator& alloc = get_base_table()->get_alloc();

        REALM_ASSERT(m_leaf);
        if (m_leaf->get(index)) {
            Array top(alloc);
            top.set_parent(&*m_leaf, index);
            top.init_from_parent();
            BPlusTree<Mixed> values(alloc);
            values.set_parent(&top, 1);
            values.init_from_parent();

            destination.init(true, values.size());
            size_t n = 0;
            // Iterate through BPlusTreee and insert all values
            values.for_all([&](Mixed val) {
                destination.set(n, val);
                n++;
            });
        }
    }
}


void Columns<Link>::evaluate(size_t index, ValueBase& destination)
{
    // Destination must be of Key type. It only makes sense to
    // compare keys with keys
    std::vector<ObjKey> links = m_link_map.get_links(index);

    if (m_link_map.only_unary_links()) {
        ObjKey key;
        if (!links.empty()) {
            key = links[0];
        }
        destination.init(false, 1);
        destination.set(0, key);
    }
    else {
        destination.init(true, links.size());
        destination.set(links.begin(), links.end());
    }
}

void ColumnListBase::set_cluster(const Cluster* cluster)
{
    if (m_link_map.has_links()) {
        m_link_map.set_cluster(cluster);
    }
    else {
        m_leaf.emplace(m_link_map.get_base_table()->get_alloc());
        cluster->init_leaf(m_column_key, &*m_leaf);
    }
}

void ColumnListBase::get_lists(size_t index, Value<int64_t>& destination, size_t nb_elements)
{
    if (m_link_map.has_links()) {
        std::vector<ObjKey> links = m_link_map.get_links(index);
        auto sz = links.size();

        if (m_link_map.only_unary_links()) {
            int64_t val = 0;
            if (sz == 1) {
                const Obj obj = m_link_map.get_target_table()->get_object(links[0]);
                val = obj._get<int64_t>(m_column_key.get_index());
            }
            destination.init(false, 1);
            destination.set(0, val);
        }
        else {
            destination.init(true, sz);
            for (size_t t = 0; t < sz; t++) {
                const Obj obj = m_link_map.get_target_table()->get_object(links[t]);
                int64_t val = obj._get<int64_t>(m_column_key.get_index());
                destination.set(t, val);
            }
        }
    }
    else {
        size_t rows = std::min(m_leaf->size() - index, nb_elements);

        destination.init(false, rows);

        for (size_t t = 0; t < rows; t++) {
            destination.set(t, m_leaf->get(index + t));
        }
    }
}

void LinkCount::evaluate(size_t index, ValueBase& destination)
{
    if (m_column_key) {
        REALM_ASSERT(m_link_map.has_links());
        std::vector<ObjKey> links = m_link_map.get_links(index);
        auto sz = links.size();

        if (sz == 0) {
            destination.init(true, 0);
        }
        else {
            destination.init(true, sz);
            Allocator& alloc = m_link_map.get_target_table()->get_alloc();
            for (size_t i = 0; i < sz; i++) {
                const Obj obj = m_link_map.get_target_table()->get_object(links[i]);
                auto val = obj._get<int64_t>(m_column_key.get_index());
                size_t s;
                if (m_column_key.get_type() == col_type_Link && !m_column_key.is_collection()) {
                    // It is a single link column
                    s = (val == 0) ? 0 : 1;
                }
                else if (val & 1) {
                    // It is a backlink column with just one value
                    s = 1;
                }
                else {
                    // This is some kind of collection or backlink column
                    s = _impl::get_collection_size_from_ref(to_ref(val), alloc);
                }
                destination.set(i, int64_t(s));
            }
        }
    }
    else {
        destination = Value<int64_t>(m_link_map.count_links(index));
    }
}

std::string LinkCount::description(util::serializer::SerialisationState& state) const
{
    return state.describe_columns(m_link_map, m_column_key) + util::serializer::value_separator + "@count";
}

Query Subexpr2<StringData>::equal(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, Equal, EqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::equal(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<Equal, EqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<StringData>::not_equal(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, NotEqual, NotEqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::not_equal(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<NotEqual, NotEqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<StringData>::begins_with(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, BeginsWith, BeginsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::begins_with(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<BeginsWith, BeginsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<StringData>::ends_with(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, EndsWith, EndsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::ends_with(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<EndsWith, EndsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<StringData>::contains(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, Contains, ContainsIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::contains(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<Contains, ContainsIns>(*this, col, case_sensitive);
}

Query Subexpr2<StringData>::like(StringData sd, bool case_sensitive)
{
    return string_compare<StringData, Like, LikeIns>(*this, sd, case_sensitive);
}

Query Subexpr2<StringData>::like(const Subexpr2<StringData>& col, bool case_sensitive)
{
    return string_compare<Like, LikeIns>(*this, col, case_sensitive);
}

Query Columns<StringData>::fulltext(StringData text) const
{
    const LinkMap& link_map = get_link_map();
    return link_map.get_base_table()->where().fulltext(column_key(), text, link_map);
}


// BinaryData

Query Subexpr2<BinaryData>::equal(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, Equal, EqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::equal(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<Equal, EqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<BinaryData>::not_equal(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, NotEqual, NotEqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::not_equal(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<NotEqual, NotEqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<BinaryData>::begins_with(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, BeginsWith, BeginsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::begins_with(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<BeginsWith, BeginsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<BinaryData>::ends_with(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, EndsWith, EndsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::ends_with(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<EndsWith, EndsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<BinaryData>::contains(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, Contains, ContainsIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::contains(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<Contains, ContainsIns>(*this, col, case_sensitive);
}

Query Subexpr2<BinaryData>::like(BinaryData sd, bool case_sensitive)
{
    return binary_compare<BinaryData, Like, LikeIns>(*this, sd, case_sensitive);
}

Query Subexpr2<BinaryData>::like(const Subexpr2<BinaryData>& col, bool case_sensitive)
{
    return binary_compare<Like, LikeIns>(*this, col, case_sensitive);
}

// Mixed

Query Subexpr2<Mixed>::equal(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, Equal, EqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::equal(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<Equal, EqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<Mixed>::not_equal(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, NotEqual, NotEqualIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::not_equal(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<NotEqual, NotEqualIns>(*this, col, case_sensitive);
}

Query Subexpr2<Mixed>::begins_with(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, BeginsWith, BeginsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::begins_with(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<BeginsWith, BeginsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<Mixed>::ends_with(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, EndsWith, EndsWithIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::ends_with(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<EndsWith, EndsWithIns>(*this, col, case_sensitive);
}

Query Subexpr2<Mixed>::contains(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, Contains, ContainsIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::contains(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<Contains, ContainsIns>(*this, col, case_sensitive);
}

Query Subexpr2<Mixed>::like(Mixed sd, bool case_sensitive)
{
    return mixed_compare<Mixed, Like, LikeIns>(*this, sd, case_sensitive);
}

Query Subexpr2<Mixed>::like(const Subexpr2<Mixed>& col, bool case_sensitive)
{
    return mixed_compare<Like, LikeIns>(*this, col, case_sensitive);
}

} // namespace realm
