// Copyright 2019 Google LLC
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

#include "tink/cc/cc_jwt_config.h"

#include "tink/jwt/internal/raw_jwt_ecdsa_sign_key_manager.h"
#include "tink/jwt/internal/raw_jwt_ecdsa_verify_key_manager.h"
#include "tink/jwt/internal/raw_jwt_hmac_key_manager.h"
#include "tink/jwt/internal/raw_jwt_rsa_ssa_pkcs1_sign_key_manager.h"
#include "tink/jwt/internal/raw_jwt_rsa_ssa_pkcs1_verify_key_manager.h"
#include "tink/jwt/internal/raw_jwt_rsa_ssa_pss_sign_key_manager.h"
#include "tink/jwt/internal/raw_jwt_rsa_ssa_pss_verify_key_manager.h"

namespace crypto {
namespace tink {

util::Status CcJwtConfigRegister() {
  util::Status status = Registry::RegisterKeyTypeManager(
      absl::make_unique<jwt_internal::RawJwtHmacKeyManager>(), true);
  if (!status.ok()) {
    return status;
  }
  status = Registry::RegisterAsymmetricKeyManagers(
      absl::make_unique<jwt_internal::RawJwtEcdsaSignKeyManager>(),
      absl::make_unique<jwt_internal::RawJwtEcdsaVerifyKeyManager>(), true);
  if (!status.ok()) {
    return status;
  }
  status = Registry::RegisterAsymmetricKeyManagers(
      absl::make_unique<RawJwtRsaSsaPkcs1SignKeyManager>(),
      absl::make_unique<RawJwtRsaSsaPkcs1VerifyKeyManager>(), true);
  if (!status.ok()) {
    return status;
  }
  return Registry::RegisterAsymmetricKeyManagers(
      absl::make_unique<RawJwtRsaSsaPssSignKeyManager>(),
      absl::make_unique<RawJwtRsaSsaPssVerifyKeyManager>(), true);
}

}  // namespace tink
}  // namespace crypto
