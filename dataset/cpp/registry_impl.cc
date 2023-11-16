// Copyright 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////////////

#include "tink/internal/registry_impl.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tink/input_stream.h"
#include "tink/internal/keyset_wrapper_store.h"
#include "tink/key_manager.h"
#include "tink/monitoring/monitoring.h"
#include "tink/util/errors.h"
#include "tink/util/status.h"
#include "tink/util/statusor.h"
#include "proto/tink.pb.h"

namespace crypto {
namespace tink {
namespace internal {

using ::crypto::tink::MonitoringClientFactory;
using ::google::crypto::tink::KeyData;
using ::google::crypto::tink::KeyTemplate;

util::StatusOr<const KeyTypeInfoStore::Info*> RegistryImpl::get_key_type_info(
    absl::string_view type_url) const {
  absl::MutexLock lock(&maps_mutex_);
  return key_type_info_store_.Get(type_url);
}

util::StatusOr<std::unique_ptr<KeyData>> RegistryImpl::NewKeyData(
    const KeyTemplate& key_template) const {
  util::StatusOr<const internal::KeyTypeInfoStore::Info*> info =
      get_key_type_info(key_template.type_url());
  if (!info.ok()) {
    return info.status();
  }
  if (!(*info)->new_key_allowed()) {
    return crypto::tink::util::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("KeyManager for type ", key_template.type_url(),
                     " does not allow for creation of new keys."));
  }
  return (*info)->key_factory().NewKeyData(key_template.value());
}

util::StatusOr<std::unique_ptr<KeyData>> RegistryImpl::GetPublicKeyData(
    absl::string_view type_url,
    absl::string_view serialized_private_key) const {
  util::StatusOr<const internal::KeyTypeInfoStore::Info*> info =
      get_key_type_info(type_url);
  if (!info.ok()) {
    return info.status();
  }
  auto factory =
      dynamic_cast<const PrivateKeyFactory*>(&(*info)->key_factory());
  if (factory == nullptr) {
    return ToStatusF(absl::StatusCode::kInvalidArgument,
                     "KeyManager for type '%s' does not have "
                     "a PrivateKeyFactory.",
                     type_url);
  }
  auto result = factory->GetPublicKeyData(serialized_private_key);
  return result;
}

util::StatusOr<KeyData> RegistryImpl::DeriveKey(const KeyTemplate& key_template,
                                                InputStream* randomness) const {
  util::StatusOr<const internal::KeyTypeInfoStore::Info*> info =
      get_key_type_info(key_template.type_url());
  if (!info.ok()) {
    return info.status();
  }
  if (!(*info)->key_deriver()) {
    return crypto::tink::util::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Manager for type '", key_template.type_url(),
                     "' cannot derive keys."));
  }
  return (*info)->key_deriver()(key_template.value(), randomness);
}

util::Status RegistryImpl::RegisterMonitoringClientFactory(
    std::unique_ptr<MonitoringClientFactory> factory) {
  absl::MutexLock lock(&monitoring_factory_mutex_);
  if (monitoring_factory_ != nullptr) {
    return util::Status(absl::StatusCode::kAlreadyExists,
                        "A monitoring factory is already registered");
  }
  monitoring_factory_ = std::move(factory);
  return util::OkStatus();
}

void RegistryImpl::Reset() {
  {
    absl::MutexLock lock(&maps_mutex_);
    key_type_info_store_ = KeyTypeInfoStore();
    keyset_wrapper_store_ = KeysetWrapperStore();
  }
  {
    absl::MutexLock lock(&monitoring_factory_mutex_);
    monitoring_factory_.reset();
  }
}

}  // namespace internal
}  // namespace tink
}  // namespace crypto
