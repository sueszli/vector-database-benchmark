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

#include <realm/sort_descriptor.hpp>
#include <realm/table.hpp>
#include <realm/table_view.hpp>
#include <realm/db.hpp>
#include <realm/util/assert.hpp>
#include <realm/list.hpp>
#include <realm/dictionary.hpp>

using namespace realm;

ConstTableRef ExtendedColumnKey::get_target_table(const Table* table) const
{
    return (m_colkey.get_type() == col_type_Link) ? table->get_link_target(m_colkey) : ConstTableRef{};
}

std::string ExtendedColumnKey::get_description(const Table* table) const
{
    std::string description = table->get_column_name(m_colkey);
    if (!m_index.is_null()) {
        description += util::format("[%1]", util::serializer::print_value(m_index));
    }
    return description;
}

bool ExtendedColumnKey::is_collection() const
{
    return m_colkey.is_collection() && m_index.is_null();
}

ObjKey ExtendedColumnKey::get_link_target(const Obj& obj) const
{
    if (m_index.is_null()) {
        return obj.get<ObjKey>(m_colkey);
    }
    else if (m_colkey.is_dictionary()) {
        const auto dictionary = obj.get_dictionary(m_colkey);
        auto val = dictionary.try_get(m_index);
        if (val && val->is_type(type_TypedLink)) {
            return val->get<ObjKey>();
        }
    }
    return {};
}

Mixed ExtendedColumnKey::get_value(const Obj& obj) const
{
    if (m_index.is_null()) {
        return obj.get_any(m_colkey);
    }
    else if (m_colkey.is_dictionary()) {
        const auto dictionary = obj.get_dictionary(m_colkey);
        auto val = dictionary.try_get(m_index);
        if (val) {
            return *val;
        }
    }
    return {};
}

LinkPathPart::LinkPathPart(ColKey col_key, ConstTableRef source)
    : column_key(col_key)
    , from(source->get_key())
{
}


ColumnsDescriptor::ColumnsDescriptor(std::vector<std::vector<ExtendedColumnKey>> column_keys)
    : m_column_keys(std::move(column_keys))
{
}

std::unique_ptr<BaseDescriptor> DistinctDescriptor::clone() const
{
    return std::unique_ptr<DistinctDescriptor>(new DistinctDescriptor(*this));
}

void ColumnsDescriptor::collect_dependencies(const Table* table, std::vector<TableKey>& table_keys) const
{
    for (auto& columns : m_column_keys) {
        auto sz = columns.size();
        // If size is 0 or 1 there is no link chain and hence no additional tables to check
        if (sz > 1) {
            const Table* t = table;
            for (size_t i = 0; i < sz - 1; i++) {
                const auto& col = columns[i];
                ConstTableRef target_table = col.get_target_table(t);
                if (!target_table)
                    return;
                table_keys.push_back(target_table->get_key());
                t = target_table.unchecked_ptr();
            }
        }
    }
}

std::string DistinctDescriptor::get_description(ConstTableRef attached_table) const
{
    std::string description = "DISTINCT(";
    for (size_t i = 0; i < m_column_keys.size(); ++i) {
        const size_t chain_size = m_column_keys[i].size();
        ConstTableRef cur_link_table = attached_table;
        for (size_t j = 0; j < chain_size; ++j) {
            const auto& col_key = m_column_keys[i][j];
            description += col_key.get_description(cur_link_table.unchecked_ptr());
            if (j < chain_size - 1) {
                description += ".";
                cur_link_table = col_key.get_target_table(cur_link_table.unchecked_ptr());
            }
        }
        if (i < m_column_keys.size() - 1) {
            description += ", ";
        }
    }
    description += ")";
    return description;
}

std::string SortDescriptor::get_description(ConstTableRef attached_table) const
{
    std::string description = "SORT(";
    for (size_t i = 0; i < m_column_keys.size(); ++i) {
        const size_t chain_size = m_column_keys[i].size();
        ConstTableRef cur_link_table = attached_table;
        for (size_t j = 0; j < chain_size; ++j) {
            const auto& col_key = m_column_keys[i][j];
            description += col_key.get_description(cur_link_table.unchecked_ptr());
            if (j < chain_size - 1) {
                description += ".";
                cur_link_table = col_key.get_target_table(cur_link_table.unchecked_ptr());
            }
        }
        description += " ";
        if (i < m_ascending.size()) {
            if (m_ascending[i]) {
                description += "ASC";
            }
            else {
                description += "DESC";
            }
        }
        if (i < m_column_keys.size() - 1) {
            description += ", ";
        }
    }
    description += ")";
    return description;
}

SortDescriptor::SortDescriptor(std::vector<std::vector<ExtendedColumnKey>> column_keys, std::vector<bool> ascending)
    : ColumnsDescriptor(std::move(column_keys))
    , m_ascending(std::move(ascending))
{
    REALM_ASSERT_EX(m_ascending.empty() || m_ascending.size() == m_column_keys.size(), m_ascending.size(),
                    m_column_keys.size());
    if (m_ascending.empty())
        m_ascending.resize(m_column_keys.size(), true);
}

std::unique_ptr<BaseDescriptor> SortDescriptor::clone() const
{
    return std::unique_ptr<ColumnsDescriptor>(new SortDescriptor(*this));
}

void SortDescriptor::merge(SortDescriptor&& other, MergeMode mode)
{
    if (mode == MergeMode::replace) {
        m_column_keys = std::move(other.m_column_keys);
        m_ascending = std::move(other.m_ascending);
        return;
    }

    m_column_keys.insert(mode == MergeMode::prepend ? m_column_keys.begin() : m_column_keys.end(),
                         other.m_column_keys.begin(), other.m_column_keys.end());
    // Do not use a move iterator on a vector of bools!
    // It will form a reference to a temporary and return incorrect results.
    m_ascending.insert(mode == MergeMode::prepend ? m_ascending.begin() : m_ascending.end(),
                       other.m_ascending.begin(), other.m_ascending.end());
}

BaseDescriptor::Sorter::Sorter(std::vector<std::vector<ExtendedColumnKey>> const& column_lists,
                               std::vector<bool> const& ascending, Table const& root_table, const IndexPairs& indexes)
{
    REALM_ASSERT(!column_lists.empty());
    REALM_ASSERT_EX(column_lists.size() == ascending.size(), column_lists.size(), ascending.size());
    size_t translated_size = std::max_element(indexes.begin(), indexes.end())->index_in_view + 1;

    m_columns.reserve(column_lists.size());
    for (size_t i = 0; i < column_lists.size(); ++i) {
        auto& columns = column_lists[i];
        auto sz = columns.size();
        REALM_ASSERT_EX(!columns.empty(), i);

        if (columns.empty()) {
            throw InvalidArgument(ErrorCodes::InvalidSortDescriptor, "Missing property");
        }
        if (columns.empty() || columns.back().is_collection()) {
            throw InvalidArgument(ErrorCodes::InvalidSortDescriptor, "Cannot sort on a collection property");
        }

        if (sz == 1) { // no link chain
            m_columns.emplace_back(&root_table, columns[0], ascending[i]);
            continue;
        }

        std::vector<const Table*> tables = {&root_table};
        tables.resize(sz);
        for (size_t j = 0; j + 1 < sz; ++j) {
            ColKey col = columns[j].get_col_key();
            if (!tables[j]->valid_column(col)) {
                throw InvalidArgument(ErrorCodes::InvalidSortDescriptor, "Invalid property");
            }
            if (col.get_type() != col_type_Link) {
                // Only last column in link chain is allowed to be non-link
                throw InvalidArgument(ErrorCodes::InvalidSortDescriptor, "All but last property must be a link");
            }
            tables[j + 1] = tables[j]->get_link_target(col).unchecked_ptr();
        }

        m_columns.emplace_back(tables.back(), columns.back(), ascending[i]);

        auto& translated_keys = m_columns.back().translated_keys;
        translated_keys.resize(translated_size);

        for (const auto& index : indexes) {
            size_t index_in_view = index.index_in_view;
            ObjKey translated_key = index.key_for_object;
            for (size_t j = 0; j + 1 < sz; ++j) {
                const Obj obj = tables[j]->get_object(translated_key);
                // type was checked when creating the ColumnsDescriptor
                translated_key = columns[j].get_link_target(obj);
                if (!translated_key || translated_key.is_unresolved()) {
                    translated_key = null_key; // normalize unresolve to null
                    break;
                }
            }
            translated_keys[index_in_view] = translated_key;
        }
    }
    m_cache.resize(column_lists.size() - 1);
}

BaseDescriptor::Sorter DistinctDescriptor::sorter(Table const& table, const IndexPairs& indexes) const
{
    REALM_ASSERT(!m_column_keys.empty());
    std::vector<bool> ascending(m_column_keys.size(), true);
    return Sorter(m_column_keys, ascending, table, indexes);
}

void DistinctDescriptor::execute(IndexPairs& v, const Sorter& predicate, const BaseDescriptor* next) const
{
    using IP = ColumnsDescriptor::IndexPair;
    // Remove all rows which have a null link along the way to the distinct columns
    if (predicate.has_links()) {
        auto nulls = std::remove_if(v.begin(), v.end(), [&](const IP& index) {
            return predicate.any_is_null(index);
        });
        v.erase(nulls, v.end());
    }

    // Sort by the columns to distinct on
    std::sort(v.begin(), v.end(), std::ref(predicate));

    // Move duplicates to the back - "not less than" is "equal" since they're sorted
    auto duplicates = std::unique(v.begin(), v.end(), [&](const IP& a, const IP& b) {
        return !predicate(a, b, false);
    });
    // Erase the duplicates
    v.erase(duplicates, v.end());
    bool will_be_sorted_next = next && next->get_type() == DescriptorType::Sort;
    if (!will_be_sorted_next) {
        // Restore the original order, this is either the original
        // tableview order or the order of the previous sort
        std::sort(v.begin(), v.end(), [](const IP& a, const IP& b) {
            return a.index_in_view < b.index_in_view;
        });
    }
}

BaseDescriptor::Sorter SortDescriptor::sorter(Table const& table, const IndexPairs& indexes) const
{
    REALM_ASSERT(!m_column_keys.empty());
    return Sorter(m_column_keys, m_ascending, table, indexes);
}

void SortDescriptor::execute(IndexPairs& v, const Sorter& predicate, const BaseDescriptor* next) const
{
    size_t limit = size_t(-1);
    if (next && next->get_type() == DescriptorType::Limit) {
        limit = static_cast<const LimitDescriptor*>(next)->get_limit();
    }
    // Measurements shows that if limit is smaller than size / 16, then
    // it is quicker to make a sorted insert into a smaller vector
    if (limit < (v.size() >> 4)) {
        IndexPairs buffer;
        buffer.reserve(limit + 1);
        for (auto& elem : v) {
            auto it = std::lower_bound(buffer.begin(), buffer.end(), elem, std::ref(predicate));
            buffer.insert(it, elem);
            if (buffer.size() > limit) {
                buffer.pop_back();
            }
        }
        v.m_removed_by_limit += v.size() - limit;
        v.erase(v.begin() + limit, v.end());
        std::move(buffer.begin(), buffer.end(), v.begin());
    }
    else {
        std::sort(v.begin(), v.end(), std::ref(predicate));
    }

    // not doing this on the last step is an optimisation
    if (next) {
        const size_t v_size = v.size();
        // Distinct must choose the winning unique elements by sorted
        // order not by the previous tableview order, the lowest
        // "index_in_view" wins.
        for (size_t i = 0; i < v_size; ++i) {
            v[i].index_in_view = i;
        }
    }
}

std::string LimitDescriptor::get_description(ConstTableRef) const
{
    return "LIMIT(" + util::serializer::print_value(m_limit) + ")";
}

std::unique_ptr<BaseDescriptor> LimitDescriptor::clone() const
{
    return std::unique_ptr<BaseDescriptor>(new LimitDescriptor(*this));
}

void LimitDescriptor::execute(const Table&, KeyValues& key_values, const BaseDescriptor*) const
{
    if (key_values.size() > m_limit) {
        key_values.erase(key_values.begin() + m_limit, key_values.end());
    }
}

std::string FilterDescriptor::get_description(ConstTableRef) const
{
    throw SerializationError("Serialization of FilterDescriptor is not supported");
    return "";
}

std::unique_ptr<BaseDescriptor> FilterDescriptor::clone() const
{
    return std::unique_ptr<BaseDescriptor>(new FilterDescriptor(*this));
}

void FilterDescriptor::execute(const Table& table, KeyValues& key_values, const BaseDescriptor*) const
{
    KeyValues filtered;
    filtered.create();
    auto sz = key_values.size();
    for (size_t i = 0; i < sz; i++) {
        auto key = key_values.get(i);
        Obj obj = table.try_get_object(key);
        if (obj && m_predicate(obj)) {
            filtered.add(key);
        }
    }
    key_values = std::move(filtered);
}

// This function must conform to 'is less' predicate - that is:
// return true if i is strictly smaller than j
bool BaseDescriptor::Sorter::operator()(IndexPair i, IndexPair j, bool total_ordering) const
{
    // Sorting can be specified by multiple columns, so that if two entries in the first column are
    // identical, then the rows are ordered according to the second column, and so forth. For the
    // first column, all the payload of the View is cached in IndexPair::cached_value.
    for (size_t t = 0; t < m_columns.size(); t++) {
        ObjKey key_i = i.key_for_object;
        ObjKey key_j = j.key_for_object;

        if (!m_columns[t].translated_keys.empty()) {
            key_i = m_columns[t].translated_keys[i.index_in_view];
            key_j = m_columns[t].translated_keys[j.index_in_view];

            bool null_i = !key_i;
            bool null_j = !key_j;

            if (null_i && null_j) {
                continue;
            }
            if (null_i || null_j) {
                // Sort null links at the end if m_ascending[t], else at beginning.
                return m_columns[t].ascending != null_i;
            }
        }

        int c;

        if (t == 0) {
            c = i.cached_value.compare(j.cached_value);
        }
        else {
            if (m_cache[t - 1].empty()) {
                m_cache[t - 1].resize(256);
            }
            ObjCache& cache_i = m_cache[t - 1][key_i.value & 0xFF];
            ObjCache& cache_j = m_cache[t - 1][key_j.value & 0xFF];

            if (cache_i.key != key_i) {
                const auto& obj = m_columns[t].table->get_object(key_i);
                const auto& col_key = m_columns[t].col_key;

                cache_i.value = col_key.get_value(obj);
                cache_i.key = key_i;
            }
            Mixed val_i = cache_i.value;

            if (cache_j.key != key_j) {
                const auto& obj = m_columns[t].table->get_object(key_j);
                const auto& col_key = m_columns[t].col_key;

                cache_j.value = col_key.get_value(obj);
                cache_j.key = key_j;
            }

            c = val_i.compare(cache_j.value);
        }
        // if c is negative i comes before j
        if (c) {
            return m_columns[t].ascending ? c < 0 : c > 0;
        }
    }
    // make sort stable by using original index as final comparison
    return total_ordering ? i.index_in_view < j.index_in_view : 0;
}

void BaseDescriptor::Sorter::cache_first_column(IndexPairs& v)
{
    if (m_columns.empty())
        return;

    auto& col = m_columns[0];
    const auto& ck = col.col_key;
    for (size_t i = 0; i < v.size(); i++) {
        IndexPair& index = v[i];
        ObjKey key = index.key_for_object;

        if (!col.translated_keys.empty()) {
            key = col.translated_keys[v[i].index_in_view];
            if (!key) {
                index.cached_value = Mixed();
                continue;
            }
        }

        const auto obj = col.table->get_object(key);
        index.cached_value = ck.get_value(obj);
    }
}

DescriptorOrdering::DescriptorOrdering(const DescriptorOrdering& other)
    : AtomicRefCountBase()
{
    for (const auto& d : other.m_descriptors) {
        m_descriptors.emplace_back(d->clone());
    }
}

DescriptorOrdering& DescriptorOrdering::operator=(const DescriptorOrdering& rhs)
{
    if (&rhs != this) {
        m_descriptors.clear();
        for (const auto& d : rhs.m_descriptors) {
            m_descriptors.emplace_back(d->clone());
        }
    }
    return *this;
}

void DescriptorOrdering::append_sort(SortDescriptor sort, SortDescriptor::MergeMode mode)
{
    if (!sort.is_valid()) {
        return;
    }
    if (!m_descriptors.empty()) {
        if (SortDescriptor* previous_sort = dynamic_cast<SortDescriptor*>(m_descriptors.back().get())) {
            previous_sort->merge(std::move(sort), mode);
            return;
        }
    }
    m_descriptors.emplace_back(new SortDescriptor(std::move(sort)));
}

void DescriptorOrdering::append_distinct(DistinctDescriptor distinct)
{
    if (distinct.is_valid()) {
        m_descriptors.emplace_back(new DistinctDescriptor(std::move(distinct)));
    }
}

void DescriptorOrdering::append_limit(LimitDescriptor limit)
{
    if (limit.is_valid()) {
        m_descriptors.emplace_back(new LimitDescriptor(std::move(limit)));
    }
}

void DescriptorOrdering::append_filter(FilterDescriptor filter)
{
    if (filter.is_valid()) {
        m_descriptors.emplace_back(new FilterDescriptor(std::move(filter)));
    }
}

void DescriptorOrdering::append(const DescriptorOrdering& other)
{
    for (const auto& d : other.m_descriptors) {
        m_descriptors.emplace_back(d->clone());
    }
}

void DescriptorOrdering::append(DescriptorOrdering&& other)
{
    std::move(other.m_descriptors.begin(), other.m_descriptors.end(), std::back_inserter(m_descriptors));
    other.m_descriptors.clear();
}

DescriptorType DescriptorOrdering::get_type(size_t index) const
{
    REALM_ASSERT(index < m_descriptors.size());
    return m_descriptors[index]->get_type();
}

const BaseDescriptor* DescriptorOrdering::operator[](size_t ndx) const
{
    if (ndx >= m_descriptors.size()) {
        throw OutOfBounds("DescriptorOrdering[]", ndx, m_descriptors.size());
    }
    return m_descriptors[ndx].get();
}

bool DescriptorOrdering::will_apply_sort() const
{
    return std::any_of(m_descriptors.begin(), m_descriptors.end(), [](const std::unique_ptr<BaseDescriptor>& desc) {
        REALM_ASSERT(desc->is_valid());
        return desc->get_type() == DescriptorType::Sort;
    });
}

bool DescriptorOrdering::will_apply_distinct() const
{
    return std::any_of(m_descriptors.begin(), m_descriptors.end(), [](const std::unique_ptr<BaseDescriptor>& desc) {
        REALM_ASSERT(desc->is_valid());
        return desc->get_type() == DescriptorType::Distinct;
    });
}

bool DescriptorOrdering::will_apply_limit() const
{
    return std::any_of(m_descriptors.begin(), m_descriptors.end(), [](const std::unique_ptr<BaseDescriptor>& desc) {
        REALM_ASSERT(desc->is_valid());
        return desc->get_type() == DescriptorType::Limit;
    });
}

bool DescriptorOrdering::will_apply_filter() const
{
    return std::any_of(m_descriptors.begin(), m_descriptors.end(), [](const std::unique_ptr<BaseDescriptor>& desc) {
        REALM_ASSERT(desc->is_valid());
        return desc->get_type() == DescriptorType::Filter;
    });
}

realm::util::Optional<size_t> DescriptorOrdering::get_min_limit() const
{
    realm::util::Optional<size_t> min_limit;
    for (auto it = m_descriptors.begin(); it != m_descriptors.end(); it++) {
        if ((*it)->get_type() == DescriptorType::Limit) {
            const LimitDescriptor* limit = static_cast<const LimitDescriptor*>(it->get());
            REALM_ASSERT(limit);
            min_limit = bool(min_limit) ? std::min(*min_limit, limit->get_limit()) : limit->get_limit();
        }
    }
    return min_limit;
}

util::Optional<size_t> DescriptorOrdering::remove_all_limits()
{
    size_t min_limit = size_t(-1);
    for (auto it = m_descriptors.begin(); it != m_descriptors.end();) {
        if ((*it)->get_type() == DescriptorType::Limit) {
            const LimitDescriptor* limit = static_cast<const LimitDescriptor*>(it->get());
            if (limit->get_limit() < min_limit) {
                min_limit = limit->get_limit();
            }
            it = m_descriptors.erase(it);
        }
        else {
            ++it;
        }
    }
    return min_limit == size_t(-1) ? util::none : util::some<size_t>(min_limit);
}

bool DescriptorOrdering::will_limit_to_zero() const
{
    return std::any_of(m_descriptors.begin(), m_descriptors.end(), [](const std::unique_ptr<BaseDescriptor>& desc) {
        REALM_ASSERT(desc.get()->is_valid());
        return (desc->get_type() == DescriptorType::Limit &&
                static_cast<LimitDescriptor*>(desc.get())->get_limit() == 0);
    });
}

std::string DescriptorOrdering::get_description(ConstTableRef target_table) const
{
    std::string description = "";
    for (auto it = m_descriptors.begin(); it != m_descriptors.end(); ++it) {
        REALM_ASSERT_DEBUG(bool(*it));
        description += (*it)->get_description(target_table);
        if (it != m_descriptors.end() - 1) {
            description += " ";
        }
    }
    return description;
}

void DescriptorOrdering::collect_dependencies(const Table* table)
{
    m_dependencies.clear();
    for (auto& descr : m_descriptors) {
        descr->collect_dependencies(table, m_dependencies);
    }
}

void DescriptorOrdering::get_versions(const Group* group, TableVersions& versions) const
{
    for (auto table_key : m_dependencies) {
        REALM_ASSERT_DEBUG(group);
        versions.emplace_back(table_key, group->get_table(table_key)->get_content_version());
    }
}
