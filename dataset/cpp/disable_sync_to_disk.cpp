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

#include <atomic>

#include <realm/disable_sync_to_disk.hpp>

using namespace realm;

namespace {

std::atomic<bool> g_disable_sync_to_disk(false);

} // anonymous namespace

// LCOV_EXCL_START We will sync to disc during coverage run
void realm::disable_sync_to_disk(bool disable)
{
    g_disable_sync_to_disk = disable;
}
// LCOV_EXCL_STOP

bool realm::get_disable_sync_to_disk() noexcept
{
    return g_disable_sync_to_disk;
}
