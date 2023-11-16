////////////////////////////////////////////////////////////////////////////
//
// Copyright 2023 Realm Inc.
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

#if defined(__EMSCRIPTEN_PTHREADS__) || defined(__EMSCRIPTEN_WASM_WORKERS__)
#error "This ExternalCommitHelper implementation is not compatible with multi-threaded WebAssembly."
#endif

#include <realm/object-store/impl/external_commit_helper.hpp>
#include <realm/object-store/impl/realm_coordinator.hpp>

using namespace realm;
using namespace realm::_impl;

ExternalCommitHelper::ExternalCommitHelper(RealmCoordinator& parent, const RealmConfig&)
    : m_parent(parent)
{
}

void ExternalCommitHelper::notify_others()
{
    m_parent.on_change();
}
