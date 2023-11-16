////////////////////////////////////////////////////////////////////////////
//
// Copyright 2016 Realm Inc.
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

#include <realm/object-store/impl/list_notifier.hpp>

#include <realm/object-store/list.hpp>

#include <realm/transaction.hpp>

using namespace realm;
using namespace realm::_impl;

ListNotifier::ListNotifier(std::shared_ptr<Realm> realm, CollectionBase const& list, PropertyType type)
    : CollectionNotifier(std::move(realm))
    , m_type(type)
    , m_prev_size(list.size())
{
    attach(list);
}

void ListNotifier::release_data() noexcept
{
    m_list = {};
    CollectionNotifier::release_data();
}

void ListNotifier::reattach()
{
    attach(*m_list);
}

void ListNotifier::attach(CollectionBase const& src)
{
    auto& tr = transaction();
    try {
        auto obj = tr.get_table(src.get_table()->get_key())->get_object(src.get_owner_key());
        m_list = obj.get_collection_ptr(src.get_col_key());
    }
    catch (const KeyNotFound&) {
        m_list = nullptr;
    }
}

bool ListNotifier::do_add_required_change_info(TransactionChangeInfo& info)
{
    if (!m_list || !m_list->is_attached())
        return false; // origin row was deleted after the notification was added

    info.collections.push_back(
        {m_list->get_table()->get_key(), m_list->get_owner_key(), m_list->get_col_key(), &m_change});

    m_info = &info;

    // When adding or removing a callback, the related tables can change due to the way we calculate related tables
    // when key path filters are set, hence we need to recalculate every time the callbacks are changed.
    // We only need to do this for lists that link to other lists. Lists of primitives cannot have related tables.
    util::CheckedLockGuard lock(m_callback_mutex);
    if (m_did_modify_callbacks && m_type == PropertyType::Object) {
        update_related_tables(*m_list->get_table());
    }

    return true;
}

void ListNotifier::run()
{
    if (!m_list || !m_list->is_attached()) {
        // List was deleted, so report all of the rows being removed if this is
        // the first run after that
        if (m_prev_size) {
            m_change.deletions.set(m_prev_size);
            m_prev_size = 0;
        }
        else {
            m_change = {};
        }
        report_collection_root_is_deleted();
        return;
    }

    m_prev_size = m_list->size();

    if (m_info && m_type == PropertyType::Object) {
        auto object_did_change = get_modification_checker(*m_info, m_list->get_target_table());
        for (size_t i = 0; i < m_prev_size; ++i) {
            if (m_change.modifications.contains(i))
                continue;
            auto m = m_list->get_any(i);
            if (!m.is_null() && object_did_change(m.get<ObjKey>()))
                m_change.modifications.add(i);
        }

        for (auto const& move : m_change.moves) {
            if (m_change.modifications.contains(move.to))
                continue;
            if (object_did_change(m_list->get_any(move.to).get<ObjKey>()))
                m_change.modifications.add(move.to);
        }
    }
}
