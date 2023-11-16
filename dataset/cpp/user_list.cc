//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/peer/user_list.h"

#include "base/strings/string_util.h"

namespace base {

//--------------------------------------------------------------------------------------------------
UserList::UserList() = default;

//--------------------------------------------------------------------------------------------------
UserList::UserList(const std::vector<User>& list, const ByteArray& seed_key)
    : seed_key_(seed_key),
      list_(list)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
UserList::~UserList() = default;

//--------------------------------------------------------------------------------------------------
// static
std::unique_ptr<UserList> UserList::createEmpty()
{
    return std::unique_ptr<UserList>(new UserList());
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<UserList> UserList::duplicate() const
{
    return std::unique_ptr<UserList>(new UserList(list_, seed_key_));
}

//--------------------------------------------------------------------------------------------------
void UserList::add(const User& user)
{
    if (user.isValid())
        list_.emplace_back(user);
}

//--------------------------------------------------------------------------------------------------
void UserList::merge(const UserList& user_list)
{
    for (const auto& user : user_list.list_)
        add(user);
}

//--------------------------------------------------------------------------------------------------
User UserList::find(std::u16string_view username) const
{
    const User* user = &User::kInvalidUser;

    for (const auto& item : list_)
    {
        if (compareCaseInsensitive(username, item.name) == 0)
            user = &item;
    }

    return *user;
}

//--------------------------------------------------------------------------------------------------
void UserList::setSeedKey(const ByteArray& seed_key)
{
    seed_key_ = seed_key;
}

} // namespace base
