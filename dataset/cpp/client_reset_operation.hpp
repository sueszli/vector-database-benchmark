/*************************************************************************
 *
 * Copyright 2021 Realm Inc.
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

#ifndef REALM_NOINST_CLIENT_RESET_OPERATION_HPP
#define REALM_NOINST_CLIENT_RESET_OPERATION_HPP

#include <realm/db.hpp>
#include <realm/util/functional.hpp>
#include <realm/util/function_ref.hpp>
#include <realm/util/logger.hpp>
#include <realm/sync/config.hpp>
#include <realm/sync/protocol.hpp>

namespace realm::sync {
class SubscriptionStore;
}

namespace realm::_impl::client_reset {
using CallbackBeforeType = util::UniqueFunction<VersionID()>;
using CallbackAfterType = util::UniqueFunction<void(VersionID, bool)>;

std::string get_fresh_path_for(const std::string& realm_path);
bool is_fresh_path(const std::string& realm_path);

bool perform_client_reset(util::Logger& logger, DB& target_db, DB& fresh_db, ClientResyncMode mode,
                          CallbackBeforeType notify_before, CallbackAfterType notify_after,
                          sync::SaltedFileIdent new_file_ident, sync::SubscriptionStore*,
                          util::FunctionRef<void(int64_t)> on_flx_version, bool recovery_is_allowed);

} // namespace realm::_impl::client_reset

#endif // REALM_NOINST_CLIENT_RESET_OPERATION_HPP
