// Copyright 2023 Google LLC
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
////////////////////////////////////////////////////////////////////////////////

#include "tink/config/v0.h"

#include "absl/log/check.h"
#include "tink/config/internal/aead_v0.h"
#include "tink/configuration.h"
#include "tink/daead/aes_siv_key_manager.h"
#include "tink/daead/deterministic_aead_wrapper.h"
#include "tink/hybrid/ecies_aead_hkdf_private_key_manager.h"
#include "tink/hybrid/ecies_aead_hkdf_public_key_manager.h"
#include "tink/hybrid/hybrid_decrypt_wrapper.h"
#include "tink/hybrid/hybrid_encrypt_wrapper.h"
#include "tink/hybrid/internal/hpke_private_key_manager.h"
#include "tink/hybrid/internal/hpke_public_key_manager.h"
#include "tink/internal/configuration_impl.h"
#include "tink/mac/aes_cmac_key_manager.h"
#include "tink/mac/hmac_key_manager.h"
#include "tink/mac/internal/chunked_mac_wrapper.h"
#include "tink/mac/mac_wrapper.h"
#include "tink/prf/aes_cmac_prf_key_manager.h"
#include "tink/prf/hkdf_prf_key_manager.h"
#include "tink/prf/hmac_prf_key_manager.h"
#include "tink/prf/prf_set_wrapper.h"
#include "tink/signature/ecdsa_verify_key_manager.h"
#include "tink/signature/ed25519_sign_key_manager.h"
#include "tink/signature/ed25519_verify_key_manager.h"
#include "tink/signature/public_key_sign_wrapper.h"
#include "tink/signature/public_key_verify_wrapper.h"
#include "tink/signature/rsa_ssa_pkcs1_sign_key_manager.h"
#include "tink/signature/rsa_ssa_pkcs1_verify_key_manager.h"
#include "tink/signature/rsa_ssa_pss_sign_key_manager.h"
#include "tink/signature/rsa_ssa_pss_verify_key_manager.h"
#include "tink/streamingaead/aes_ctr_hmac_streaming_key_manager.h"
#include "tink/streamingaead/aes_gcm_hkdf_streaming_key_manager.h"
#include "tink/streamingaead/streaming_aead_wrapper.h"
#include "tink/signature/ecdsa_sign_key_manager.h"

namespace crypto {
namespace tink {
namespace {

util::Status AddMac(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<MacWrapper>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<internal::ChunkedMacWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  status = internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<HmacKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  return internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<AesCmacKeyManager>(), config);
}

util::Status AddDeterministicAead(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<DeterministicAeadWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  return internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<AesSivKeyManager>(), config);
}

util::Status AddStreamingAead(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<StreamingAeadWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  status = internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<AesGcmHkdfStreamingKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  return internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<AesCtrHmacStreamingKeyManager>(), config);
}

util::Status AddHybrid(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<HybridEncryptWrapper>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<HybridDecryptWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  status = internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<EciesAeadHkdfPrivateKeyManager>(),
      absl::make_unique<EciesAeadHkdfPublicKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  return internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<internal::HpkePrivateKeyManager>(),
      absl::make_unique<internal::HpkePublicKeyManager>(), config);
}

util::Status AddPrf(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<PrfSetWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  status = internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<HmacPrfKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<HkdfPrfKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  return internal::ConfigurationImpl::AddKeyTypeManager(
      absl::make_unique<AesCmacPrfKeyManager>(), config);
}

util::Status AddSignature(Configuration& config) {
  util::Status status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<PublicKeySignWrapper>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddPrimitiveWrapper(
      absl::make_unique<PublicKeyVerifyWrapper>(), config);
  if (!status.ok()) {
    return status;
  }

  status = internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<EcdsaSignKeyManager>(),
      absl::make_unique<EcdsaVerifyKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<RsaSsaPssSignKeyManager>(),
      absl::make_unique<RsaSsaPssVerifyKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  status = internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<RsaSsaPkcs1SignKeyManager>(),
      absl::make_unique<RsaSsaPkcs1VerifyKeyManager>(), config);
  if (!status.ok()) {
    return status;
  }
  return internal::ConfigurationImpl::AddAsymmetricKeyManagers(
      absl::make_unique<Ed25519SignKeyManager>(),
      absl::make_unique<Ed25519VerifyKeyManager>(), config);
}

}  // namespace

const Configuration& ConfigV0() {
  static const Configuration* instance = [] {
    static Configuration* config = new Configuration();
    CHECK_OK(AddMac(*config));
    CHECK_OK(internal::AddAeadV0(*config));
    CHECK_OK(AddDeterministicAead(*config));
    CHECK_OK(AddStreamingAead(*config));
    CHECK_OK(AddHybrid(*config));
    CHECK_OK(AddPrf(*config));
    CHECK_OK(AddSignature(*config));
    return config;
  }();
  return *instance;
}

}  // namespace tink
}  // namespace crypto
