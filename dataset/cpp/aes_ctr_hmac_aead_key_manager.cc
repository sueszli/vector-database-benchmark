// Copyright 2017 Google Inc.
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

#include "tink/aead/aes_ctr_hmac_aead_key_manager.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tink/aead.h"
#include "tink/mac.h"
#include "tink/mac/hmac_key_manager.h"
#include "tink/subtle/aes_ctr_boringssl.h"
#include "tink/subtle/encrypt_then_authenticate.h"
#include "tink/subtle/ind_cpa_cipher.h"
#include "tink/subtle/random.h"
#include "tink/util/enums.h"
#include "tink/util/input_stream_util.h"
#include "tink/util/secret_data.h"
#include "tink/util/status.h"
#include "tink/util/statusor.h"
#include "tink/util/validation.h"
#include "proto/aes_ctr.pb.h"
#include "proto/aes_ctr_hmac_aead.pb.h"
#include "proto/common.pb.h"
#include "proto/hmac.pb.h"

namespace crypto {
namespace tink {

namespace {
constexpr int kMinKeySizeInBytes = 16;
constexpr int kMinIvSizeInBytes = 12;
constexpr int kMinTagSizeInBytes = 10;
}

using ::crypto::tink::util::Enums;
using ::crypto::tink::util::Status;
using ::crypto::tink::util::StatusOr;
using ::google::crypto::tink::AesCtrHmacAeadKey;
using ::google::crypto::tink::AesCtrHmacAeadKeyFormat;
using ::google::crypto::tink::AesCtrKey;
using ::google::crypto::tink::HashType;
using ::google::crypto::tink::HmacKey;

StatusOr<AesCtrHmacAeadKey> AesCtrHmacAeadKeyManager::CreateKey(
    const AesCtrHmacAeadKeyFormat& aes_ctr_hmac_aead_key_format) const {
  AesCtrHmacAeadKey aes_ctr_hmac_aead_key;
  aes_ctr_hmac_aead_key.set_version(get_version());

  // Generate AesCtrKey.
  auto aes_ctr_key = aes_ctr_hmac_aead_key.mutable_aes_ctr_key();
  aes_ctr_key->set_version(get_version());
  *(aes_ctr_key->mutable_params()) =
      aes_ctr_hmac_aead_key_format.aes_ctr_key_format().params();
  aes_ctr_key->set_key_value(subtle::Random::GetRandomBytes(
      aes_ctr_hmac_aead_key_format.aes_ctr_key_format().key_size()));

  // Generate HmacKey.
  auto hmac_key_or = HmacKeyManager().CreateKey(
      aes_ctr_hmac_aead_key_format.hmac_key_format());
  if (!hmac_key_or.status().ok()) {
    return hmac_key_or.status();
  }
  *aes_ctr_hmac_aead_key.mutable_hmac_key() = hmac_key_or.value();

  return aes_ctr_hmac_aead_key;
}

StatusOr<std::unique_ptr<Aead>> AesCtrHmacAeadKeyManager::AeadFactory::Create(
    const AesCtrHmacAeadKey& key) const {
  auto aes_ctr_result = subtle::AesCtrBoringSsl::New(
      util::SecretDataFromStringView(key.aes_ctr_key().key_value()),
      key.aes_ctr_key().params().iv_size());
  if (!aes_ctr_result.ok()) return aes_ctr_result.status();

  auto hmac_result = HmacKeyManager().GetPrimitive<Mac>(key.hmac_key());
  if (!hmac_result.ok()) return hmac_result.status();

  auto cipher_res = subtle::EncryptThenAuthenticate::New(
      std::move(aes_ctr_result.value()), std::move(hmac_result.value()),
      key.hmac_key().params().tag_size());
  if (!cipher_res.ok()) {
    return cipher_res.status();
  }
  return std::move(cipher_res.value());
}

Status AesCtrHmacAeadKeyManager::ValidateKey(
    const AesCtrHmacAeadKey& key) const {
  Status status = ValidateVersion(key.version(), get_version());
  if (!status.ok()) return status;

  status = ValidateVersion(key.aes_ctr_key().version(), get_version());
  if (!status.ok()) return status;

  // Validate AesCtrKey.
  auto aes_ctr_key = key.aes_ctr_key();
  uint32_t aes_key_size = aes_ctr_key.key_value().size();
  status = ValidateAesKeySize(aes_key_size);
  if (!status.ok()) {
    return status;
  }
  if (aes_ctr_key.params().iv_size() < kMinIvSizeInBytes ||
      aes_ctr_key.params().iv_size() > 16) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "Invalid AesCtrHmacAeadKey: IV size out of range.");
  }
  return HmacKeyManager().ValidateKey(key.hmac_key());
}

Status AesCtrHmacAeadKeyManager::ValidateKeyFormat(
    const AesCtrHmacAeadKeyFormat& key_format) const {
  // Validate AesCtrKeyFormat.
  auto aes_ctr_key_format = key_format.aes_ctr_key_format();
  auto status = ValidateAesKeySize(aes_ctr_key_format.key_size());
  if (!status.ok()) {
    return status;
  }
  if (aes_ctr_key_format.params().iv_size() < kMinIvSizeInBytes ||
      aes_ctr_key_format.params().iv_size() > 16) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        "Invalid AesCtrHmacAeadKeyFormat: IV size out of range.");
  }

  // Validate HmacKeyFormat.
  auto hmac_key_format = key_format.hmac_key_format();
  if (hmac_key_format.key_size() < kMinKeySizeInBytes) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        "Invalid AesCtrHmacAeadKeyFormat: HMAC key_size is too small.");
  }
  auto params = hmac_key_format.params();
  if (params.tag_size() < kMinTagSizeInBytes) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrCat("Invalid HmacParams: tag_size ",
                                     params.tag_size(), " is too small."));
  }
  std::map<HashType, uint32_t> max_tag_size = {{HashType::SHA1, 20},
                                               {HashType::SHA224, 28},
                                               {HashType::SHA256, 32},
                                               {HashType::SHA384, 48},
                                               {HashType::SHA512, 64}};
  if (max_tag_size.find(params.hash()) == max_tag_size.end()) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Invalid HmacParams: HashType '",
                     Enums::HashName(params.hash()), "' not supported."));
  } else {
    if (params.tag_size() > max_tag_size[params.hash()]) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Invalid HmacParams: tag_size ", params.tag_size(),
                       " is too big for HashType '",
                       Enums::HashName(params.hash()), "'."));
    }
  }

  return HmacKeyManager().ValidateKeyFormat(key_format.hmac_key_format());
}

// To ensure the resulting key can provide key commitment, the AES-CTR key must
// be derived first, then the HMAC key. This avoids situation where it's
// possible to brute force raw key material so that the 32th byte of the
// keystream is a 0 Give party A a key with this raw key material, saying that
// the size of the HMAC key is 32 bytes and the size of the AES key is 16 bytes.
// Give party B a key with this raw key material, saying that the size of the
// HMAC key is 31 bytes and the size of the AES key is 16 bytes. Since HMAC will
// pad the key with zeroes, this leads to both parties using the same HMAC key,
// but a different AES key (offset by 1 byte)
StatusOr<AesCtrHmacAeadKey> AesCtrHmacAeadKeyManager::DeriveKey(
    const AesCtrHmacAeadKeyFormat& key_format,
    InputStream* input_stream) const {
  Status status = ValidateKeyFormat(key_format);
  if (!status.ok()) {
    return status;
  }
  StatusOr<std::string> aes_ctr_randomness = ReadBytesFromStream(
      key_format.aes_ctr_key_format().key_size(), input_stream);
  if (!aes_ctr_randomness.ok()) {
    if (absl::IsOutOfRange(aes_ctr_randomness.status())) {
      return crypto::tink::util::Status(
          absl::StatusCode::kInvalidArgument,
          "Could not get enough pseudorandomness from input stream");
    }
    return aes_ctr_randomness.status();
  }
  StatusOr<HmacKey> hmac_key =
      HmacKeyManager().DeriveKey(key_format.hmac_key_format(), input_stream);
  if (!hmac_key.ok()) {
    return hmac_key.status();
  }

  google::crypto::tink::AesCtrHmacAeadKey key;
  key.set_version(get_version());
  *key.mutable_hmac_key() = hmac_key.value();

  AesCtrKey* aes_ctr_key = key.mutable_aes_ctr_key();
  aes_ctr_key->set_version(get_version());
  aes_ctr_key->set_key_value(aes_ctr_randomness.value());
  *aes_ctr_key->mutable_params() = key_format.aes_ctr_key_format().params();

  status = ValidateKey(key);
  if (!status.ok()) {
    return status;
  }
  return key;
}

}  // namespace tink
}  // namespace crypto
