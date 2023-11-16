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

#include "tink/subtle/ecies_hkdf_sender_kem_boringssl.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "openssl/bn.h"
#include "openssl/evp.h"
#include "tink/internal/ec_util.h"
#include "tink/internal/ssl_unique_ptr.h"
#include "tink/subtle/common_enums.h"
#include "tink/subtle/hkdf.h"
#include "tink/util/secret_data.h"

namespace crypto {
namespace tink {
namespace subtle {

// static
util::StatusOr<std::unique_ptr<const EciesHkdfSenderKemBoringSsl>>
EciesHkdfSenderKemBoringSsl::New(subtle::EllipticCurveType curve,
                                 const std::string& pubx,
                                 const std::string& puby) {
  switch (curve) {
    case EllipticCurveType::NIST_P256:
    case EllipticCurveType::NIST_P384:
    case EllipticCurveType::NIST_P521:
      return EciesHkdfNistPCurveSendKemBoringSsl::New(curve, pubx, puby);
    case EllipticCurveType::CURVE25519:
      return EciesHkdfX25519SendKemBoringSsl::New(curve, pubx, puby);
    default:
      return util::Status(absl::StatusCode::kUnimplemented,
                          "Unsupported elliptic curve");
  }
}

EciesHkdfNistPCurveSendKemBoringSsl::EciesHkdfNistPCurveSendKemBoringSsl(
    subtle::EllipticCurveType curve, const std::string& pubx,
    const std::string& puby, internal::SslUniquePtr<EC_POINT> peer_pub_key)
    : curve_(curve),
      pubx_(pubx),
      puby_(puby),
      peer_pub_key_(std::move(peer_pub_key)) {}

// static
util::StatusOr<std::unique_ptr<const EciesHkdfSenderKemBoringSsl>>
EciesHkdfNistPCurveSendKemBoringSsl::New(subtle::EllipticCurveType curve,
                                         const std::string& pubx,
                                         const std::string& puby) {
  auto status =
      internal::CheckFipsCompatibility<EciesHkdfNistPCurveSendKemBoringSsl>();
  if (!status.ok()) return status;

  auto status_or_ec_point = internal::GetEcPoint(curve, pubx, puby);
  if (!status_or_ec_point.ok()) return status_or_ec_point.status();
  std::unique_ptr<const EciesHkdfSenderKemBoringSsl> sender_kem(
      new EciesHkdfNistPCurveSendKemBoringSsl(
          curve, pubx, puby, std::move(status_or_ec_point.value())));
  return std::move(sender_kem);
}

util::StatusOr<std::unique_ptr<const EciesHkdfSenderKemBoringSsl::KemKey>>
EciesHkdfNistPCurveSendKemBoringSsl::GenerateKey(
    subtle::HashType hash, absl::string_view hkdf_salt,
    absl::string_view hkdf_info, uint32_t key_size_in_bytes,
    subtle::EcPointFormat point_format) const {
  if (peer_pub_key_.get() == nullptr) {
    return util::Status(absl::StatusCode::kInternal,
                        "peer_pub_key_ wasn't initialized");
  }

  auto status_or_ec_group = internal::EcGroupFromCurveType(curve_);
  if (!status_or_ec_group.ok()) {
    return status_or_ec_group.status();
  }
  internal::SslUniquePtr<EC_GROUP> group =
      std::move(status_or_ec_group.value());
  internal::SslUniquePtr<EC_KEY> ephemeral_key(EC_KEY_new());
  if (1 != EC_KEY_set_group(ephemeral_key.get(), group.get())) {
    return util::Status(absl::StatusCode::kInternal, "EC_KEY_set_group failed");
  }
  if (1 != EC_KEY_generate_key(ephemeral_key.get())) {
    return util::Status(absl::StatusCode::kInternal,
                        "EC_KEY_generate_key failed");
  }
  const BIGNUM* ephemeral_priv = EC_KEY_get0_private_key(ephemeral_key.get());
  const EC_POINT* ephemeral_pub = EC_KEY_get0_public_key(ephemeral_key.get());
  auto status_or_string_kem =
      internal::EcPointEncode(curve_, point_format, ephemeral_pub);
  if (!status_or_string_kem.ok()) {
    return status_or_string_kem.status();
  }
  std::string kem_bytes = status_or_string_kem.value();
  auto status_or_string_shared_secret = internal::ComputeEcdhSharedSecret(
      curve_, ephemeral_priv, peer_pub_key_.get());
  if (!status_or_string_shared_secret.ok()) {
    return status_or_string_shared_secret.status();
  }
  util::SecretData shared_secret = status_or_string_shared_secret.value();
  auto symmetric_key_or = Hkdf::ComputeEciesHkdfSymmetricKey(
      hash, kem_bytes, shared_secret, hkdf_salt, hkdf_info, key_size_in_bytes);
  if (!symmetric_key_or.ok()) {
    return symmetric_key_or.status();
  }
  util::SecretData symmetric_key = symmetric_key_or.value();
  return absl::make_unique<const KemKey>(std::move(kem_bytes),
                                         std::move(symmetric_key));
}

EciesHkdfX25519SendKemBoringSsl::EciesHkdfX25519SendKemBoringSsl(
    internal::SslUniquePtr<EVP_PKEY> peer_public_key)
    : peer_public_key_(std::move(peer_public_key)) {}

// static
util::StatusOr<std::unique_ptr<const EciesHkdfSenderKemBoringSsl>>
EciesHkdfX25519SendKemBoringSsl::New(subtle::EllipticCurveType curve,
                                     const std::string& pubx,
                                     const std::string& puby) {
  auto status =
      internal::CheckFipsCompatibility<EciesHkdfX25519SendKemBoringSsl>();
  if (!status.ok()) return status;

  if (curve != CURVE25519) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "curve is not CURVE25519");
  }
  if (pubx.size() != internal::X25519KeyPubKeySize()) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "pubx has unexpected length");
  }
  if (!puby.empty()) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "puby is not empty");
  }

  internal::SslUniquePtr<EVP_PKEY> peer_public_key(EVP_PKEY_new_raw_public_key(
      /*type=*/EVP_PKEY_X25519, /*unused=*/nullptr,
      /*in=*/reinterpret_cast<const uint8_t*>(pubx.data()),
      /*len=*/internal::Ed25519KeyPubKeySize()));
  if (peer_public_key == nullptr) {
    return util::Status(absl::StatusCode::kInternal,
                        "EVP_PKEY_new_raw_public_key failed");
  }
  std::unique_ptr<const EciesHkdfSenderKemBoringSsl> sender_kem(
      new EciesHkdfX25519SendKemBoringSsl(std::move(peer_public_key)));
  return std::move(sender_kem);
}

util::StatusOr<std::unique_ptr<const EciesHkdfSenderKemBoringSsl::KemKey>>
EciesHkdfX25519SendKemBoringSsl::GenerateKey(
    subtle::HashType hash, absl::string_view hkdf_salt,
    absl::string_view hkdf_info, uint32_t key_size_in_bytes,
    subtle::EcPointFormat point_format) const {
  if (point_format != EcPointFormat::COMPRESSED) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        "X25519 only supports compressed elliptic curve points");
  }

  // Generate an ephemeral key pair; the public key is the KEM key to use.
  util::StatusOr<std::unique_ptr<internal::X25519Key>> ephemeral_key =
      internal::NewX25519Key();

  internal::SslUniquePtr<EVP_PKEY> ssl_priv_key(EVP_PKEY_new_raw_private_key(
      /*type=*/EVP_PKEY_X25519, /*unused=*/nullptr,
      /*in=*/(*ephemeral_key)->private_key,
      /*len=*/internal::Ed25519KeyPrivKeySize()));
  if (ssl_priv_key == nullptr) {
    return util::Status(absl::StatusCode::kInternal,
                        "EVP_PKEY_new_raw_private_key failed");
  }

  util::StatusOr<util::SecretData> shared_secret =
      internal::ComputeX25519SharedSecret(ssl_priv_key.get(),
                                          peer_public_key_.get());

  auto public_key = absl::string_view(
      reinterpret_cast<const char*>((*ephemeral_key)->public_value),
      internal::X25519KeyPubKeySize());

  util::StatusOr<util::SecretData> symmetric_key_or =
      Hkdf::ComputeEciesHkdfSymmetricKey(hash, public_key, *shared_secret,
                                         hkdf_salt, hkdf_info,
                                         key_size_in_bytes);
  if (!symmetric_key_or.ok()) {
    return symmetric_key_or.status();
  }
  util::SecretData symmetric_key = *symmetric_key_or;
  return absl::make_unique<const KemKey>(std::string(public_key),
                                         symmetric_key);
}

}  // namespace subtle
}  // namespace tink
}  // namespace crypto
