// Copyright 2021 Google LLC
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
#include "tink/jwt/internal/jwt_rsa_ssa_pss_verify_key_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"

namespace crypto {
namespace tink {
namespace jwt_internal {

using crypto::tink::util::Status;
using crypto::tink::util::StatusOr;
using google::crypto::tink::JwtRsaSsaPssAlgorithm;
using google::crypto::tink::JwtRsaSsaPssPublicKey;

StatusOr<std::unique_ptr<JwtPublicKeyVerifyInternal>>
JwtRsaSsaPssVerifyKeyManager::PublicKeyVerifyFactory::Create(
    const JwtRsaSsaPssPublicKey& jwt_rsa_ssa_pss_public_key) const {
  StatusOr<std::string> name =
      AlgorithmName(jwt_rsa_ssa_pss_public_key.algorithm());
  if (!name.ok()) {
    return name.status();
  }
  StatusOr<std::unique_ptr<PublicKeyVerify>> verify =
      raw_key_manager_.GetPrimitive<PublicKeyVerify>(
          jwt_rsa_ssa_pss_public_key);
  if (!verify.ok()) {
    return verify.status();
  }
  absl::optional<absl::string_view> custom_kid = absl::nullopt;
  if (jwt_rsa_ssa_pss_public_key.has_custom_kid()) {
    custom_kid = jwt_rsa_ssa_pss_public_key.custom_kid().value();
  }
  std::unique_ptr<JwtPublicKeyVerifyInternal> jwt_public_key_verify =
      absl::make_unique<jwt_internal::JwtPublicKeyVerifyImpl>(
          *std::move(verify), *name, custom_kid);
  return std::move(jwt_public_key_verify);
}

uint32_t JwtRsaSsaPssVerifyKeyManager::get_version() const {
  return raw_key_manager_.get_version();
}

google::crypto::tink::KeyData::KeyMaterialType
JwtRsaSsaPssVerifyKeyManager::key_material_type() const {
  return raw_key_manager_.key_material_type();
}

const std::string& JwtRsaSsaPssVerifyKeyManager::get_key_type() const {
  return raw_key_manager_.get_key_type();
}

Status JwtRsaSsaPssVerifyKeyManager::ValidateKey(
    const JwtRsaSsaPssPublicKey& key) const {
  return raw_key_manager_.ValidateKey(key);
}

StatusOr<std::string> JwtRsaSsaPssVerifyKeyManager::AlgorithmName(
    const JwtRsaSsaPssAlgorithm& algorithm) {
  switch (algorithm) {
    case JwtRsaSsaPssAlgorithm::PS256:
      return std::string("PS256");
    case JwtRsaSsaPssAlgorithm::PS384:
      return std::string("PS384");
    case JwtRsaSsaPssAlgorithm::PS512:
      return std::string("PS512");
    default:
      return Status(absl::StatusCode::kInvalidArgument,
                    "Unsupported RSA SSA PKCS1 Algorithm");
  }
}

}  // namespace jwt_internal
}  // namespace tink
}  // namespace crypto
