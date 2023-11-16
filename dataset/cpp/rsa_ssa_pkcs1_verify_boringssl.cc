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

#include "tink/subtle/rsa_ssa_pkcs1_verify_boringssl.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "openssl/bn.h"
#include "openssl/evp.h"
#include "openssl/rsa.h"
#include "tink/internal/md_util.h"
#include "tink/internal/rsa_util.h"
#include "tink/internal/ssl_unique_ptr.h"
#include "tink/internal/util.h"
#include "tink/subtle/common_enums.h"
#include "tink/util/errors.h"
#include "tink/util/statusor.h"

namespace crypto {
namespace tink {
namespace subtle {

util::StatusOr<std::unique_ptr<RsaSsaPkcs1VerifyBoringSsl>>
RsaSsaPkcs1VerifyBoringSsl::New(const internal::RsaPublicKey& pub_key,
                                const internal::RsaSsaPkcs1Params& params) {
  util::Status status =
      internal::CheckFipsCompatibility<RsaSsaPkcs1VerifyBoringSsl>();
  if (!status.ok()) {
    return status;
  }

  // Check if the hash type is safe to use.
  util::Status is_safe = internal::IsHashTypeSafeForSignature(params.hash_type);
  if (!is_safe.ok()) {
    return is_safe;
  }

  util::StatusOr<const EVP_MD*> sig_hash =
      internal::EvpHashFromHashType(params.hash_type);
  if (!sig_hash.ok()) {
    return sig_hash.status();
  }

  // The RSA modulus and exponent are checked as part of the conversion to
  // internal::SslUniquePtr<RSA>.
  util::StatusOr<internal::SslUniquePtr<RSA>> rsa =
      internal::RsaPublicKeyToRsa(pub_key);
  if (!rsa.ok()) {
    return rsa.status();
  }

  std::unique_ptr<RsaSsaPkcs1VerifyBoringSsl> verify(
      new RsaSsaPkcs1VerifyBoringSsl(*std::move(rsa), *sig_hash));
  return std::move(verify);
}

util::Status RsaSsaPkcs1VerifyBoringSsl::Verify(absl::string_view signature,
                                                absl::string_view data) const {
  // BoringSSL expects a non-null pointer for data,
  // regardless of whether the size is 0.
  data = internal::EnsureStringNonNull(data);

  util::StatusOr<std::string> digest = internal::ComputeHash(data, *sig_hash_);
  if (!digest.ok()) {
    return digest.status();
  }

  if (RSA_verify(EVP_MD_type(sig_hash_),
                 /*digest=*/reinterpret_cast<const uint8_t*>(digest->data()),
                 /*digest_len=*/digest->size(),
                 /*sig=*/reinterpret_cast<const uint8_t*>(signature.data()),
                 /*sig_len=*/signature.length(),
                 /*rsa=*/rsa_.get()) != 1) {
    // Signature is invalid.
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "Signature is not valid.");
  }

  return util::OkStatus();
}

}  // namespace subtle
}  // namespace tink
}  // namespace crypto
