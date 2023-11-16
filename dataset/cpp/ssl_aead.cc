// Copyright 2021 Google LLC.
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
#include "tink/aead/internal/ssl_aead.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "openssl/crypto.h"
#include "openssl/evp.h"
#include "tink/aead/internal/aead_util.h"
#include "tink/internal/err_util.h"
#include "tink/internal/ssl_unique_ptr.h"
#include "tink/internal/util.h"
#include "tink/util/secret_data.h"
#include "tink/util/status.h"
#include "tink/util/statusor.h"

namespace crypto {
namespace tink {
namespace internal {

ABSL_CONST_INIT const int kXchacha20Poly1305TagSizeInBytes = 16;
ABSL_CONST_INIT const int kAesGcmTagSizeInBytes = 16;
ABSL_CONST_INIT const int kAesGcmSivTagSizeInBytes = 16;

namespace {

// Encrypts/Decrypts `data` and writes the result into `out`. The direction
// (encrypt/decrypt) is given by `context`. `out` is assumed to be large enough
// to hold the encrypted/decrypted content.
util::StatusOr<int64_t> UpdateCipher(EVP_CIPHER_CTX *context,
                                     absl::string_view data,
                                     absl::Span<char> out) {
  // We encrypt/decrypt in chunks of at most MAX int.
  const int64_t kMaxChunkSize = std::numeric_limits<int>::max();
  // Keep track of the bytes written to out.
  int64_t total_written_bytes = 0;
  // In practical cases data.size() is assumed to fit into a int64_t.
  int64_t left_to_update = data.size();
  while (left_to_update > 0) {
    const int chunk_size = std::min(kMaxChunkSize, left_to_update);
    auto *buffer_ptr =
        reinterpret_cast<uint8_t *>(out.data() + total_written_bytes);
    absl::string_view data_chunk = data.substr(total_written_bytes, chunk_size);
    int written_bytes = 0;
    if (EVP_CipherUpdate(context, buffer_ptr, &written_bytes,
                         reinterpret_cast<const uint8_t *>(data_chunk.data()),
                         data_chunk.size()) <= 0) {
      const bool is_encrypting = EVP_CIPHER_CTX_encrypting(context) == 1;
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat(is_encrypting ? "Encryption" : "Decryption", " failed"));
    }
    left_to_update -= written_bytes;
    total_written_bytes += written_bytes;
  }
  return total_written_bytes;
}

class OpenSslOneShotAeadImpl : public SslOneShotAead {
 public:
  explicit OpenSslOneShotAeadImpl(const util::SecretData &key,
                                  const EVP_CIPHER *cipher, size_t tag_size)
      : key_(key), cipher_(cipher), tag_size_(tag_size) {}

  util::StatusOr<int64_t> Encrypt(absl::string_view plaintext,
                                  absl::string_view associated_data,
                                  absl::string_view iv,
                                  absl::Span<char> out) const override {
    absl::string_view plaintext_data = internal::EnsureStringNonNull(plaintext);
    absl::string_view ad = internal::EnsureStringNonNull(associated_data);

    const int64_t min_out_buff_size = CiphertextSize(plaintext.size());
    if (out.size() < min_out_buff_size) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Encryption buffer too small; expected at least ",
                       min_out_buff_size, " bytes, got ", out.size()));
    }

    if (BuffersOverlap(plaintext, absl::string_view(out.data(), out.size()))) {
      return util::Status(absl::StatusCode::kInvalidArgument,
                          "Plaintext and output buffer must not overlap");
    }

    if (associated_data.size() > std::numeric_limits<int>::max()) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Associated data too large; expected at most ",
                       std::numeric_limits<int>::max(), " got ",
                       associated_data.size()));
    }

    util::StatusOr<internal::SslUniquePtr<EVP_CIPHER_CTX>> context =
        GetContext(iv, /*encryption=*/true);
    if (!context.ok()) {
      return context.status();
    }

    // Set the associated data.
    int len = 0;
    if (EVP_EncryptUpdate(context->get(), /*out=*/nullptr, &len,
                          reinterpret_cast<const uint8_t *>(ad.data()),
                          ad.size()) <= 0) {
      return util::Status(absl::StatusCode::kInternal,
                          "Failed to set associated data");
    }

    util::StatusOr<int64_t> raw_ciphertext_bytes =
        UpdateCipher(context->get(), plaintext_data, out);
    if (!raw_ciphertext_bytes.ok()) {
      return raw_ciphertext_bytes.status();
    }

    if (EVP_EncryptFinal_ex(context->get(), /*out=*/nullptr, &len) <= 0) {
      return util::Status(absl::StatusCode::kInternal, "Finalization failed");
    }

    // Write the tag after the ciphertext.
    absl::Span<char> tag = out.subspan(*raw_ciphertext_bytes, tag_size_);
    if (EVP_CIPHER_CTX_ctrl(context->get(), EVP_CTRL_AEAD_GET_TAG, tag_size_,
                            reinterpret_cast<uint8_t *>(tag.data())) <= 0) {
      return util::Status(absl::StatusCode::kInternal, "Failed to get the tag");
    }
    return *raw_ciphertext_bytes + tag_size_;
  }

  util::StatusOr<int64_t> Decrypt(absl::string_view ciphertext,
                                  absl::string_view associated_data,
                                  absl::string_view iv,
                                  absl::Span<char> out) const override {
    absl::string_view ad = internal::EnsureStringNonNull(associated_data);

    if (ciphertext.size() < tag_size_) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Ciphertext buffer too small; expected at least ",
                       tag_size_, " got ", ciphertext.size()));
    }

    const int64_t min_out_buff_size = PlaintextSize(ciphertext.size());
    if (out.size() < min_out_buff_size) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Output buffer too small; expected at least ",
                       min_out_buff_size, " got ", out.size()));
    }

    if (BuffersOverlap(ciphertext, absl::string_view(out.data(), out.size()))) {
      return util::Status(absl::StatusCode::kInvalidArgument,
                          "Ciphertext and output buffer must not overlap");
    }

    if (associated_data.size() > std::numeric_limits<int>::max()) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Associated data too large; expected at most ",
                       std::numeric_limits<int>::max(), " got ",
                       associated_data.size()));
    }

    util::StatusOr<internal::SslUniquePtr<EVP_CIPHER_CTX>> context =
        GetContext(iv, /*encryption=*/false);
    if (!context.ok()) {
      return context.status();
    }

    int len = 0;
    // Add the associated data.
    if (EVP_DecryptUpdate(context->get(), /*out=*/nullptr, &len,
                          reinterpret_cast<const uint8_t *>(ad.data()),
                          ad.size()) <= 0) {
      return util::Status(absl::StatusCode::kInternal,
                          "Failed to set associated_data");
    }

    const int64_t raw_ciphertext_size = ciphertext.size() - tag_size_;
    // "Unpack" the ciphertext.
    absl::string_view raw_ciphertext =
        ciphertext.substr(0, raw_ciphertext_size);
    // This copy is needed since EVP_CIPHER_CTX_ctrl requires a non-const
    // pointer even if the EVP_CTRL_AEAD_SET_TAG operation doesn't modify the
    // content of the buffer.
    auto tag = std::string(ciphertext.substr(raw_ciphertext_size, tag_size_));

    // Set the tag.
    if (EVP_CIPHER_CTX_ctrl(context->get(), EVP_CTRL_AEAD_SET_TAG, tag_size_,
                            reinterpret_cast<uint8_t *>(&tag[0])) <= 0) {
      return util::Status(absl::StatusCode::kInternal,
                          "Could not set authentication tag");
    }

    // If out.empty() accessing the 0th element would result in an out of
    // bound violation. This makes sure we pass a pointer to at least one byte
    // when calling into OpenSSL.
    char buffer_if_size_is_zero = '\0';
    auto out_buffer = absl::Span<char>(&buffer_if_size_is_zero, /*length=*/1);
    if (!out.empty()) {
      out_buffer = out.subspan(0, min_out_buff_size - tag_size_);
    }

    // Zero the plaintext buffer in case decryption fails before returning an
    // error.
    auto output_eraser =
        absl::MakeCleanup([out] { OPENSSL_cleanse(out.data(), out.size()); });

    util::StatusOr<int64_t> written_bytes =
        UpdateCipher(context->get(), raw_ciphertext, out_buffer);
    if (!written_bytes.ok()) {
      return written_bytes.status();
    }

    if (!EVP_DecryptFinal_ex(context->get(), /*out=*/nullptr, &len)) {
      return util::Status(absl::StatusCode::kInternal, "Authentication failed");
    }

    // Decryption executed correctly, cancel cleanup on the output buffer.
    std::move(output_eraser).Cancel();
    return *written_bytes;
  }

  int64_t CiphertextSize(int64_t plaintext_length) const override {
    return plaintext_length + tag_size_;
  }

  int64_t PlaintextSize(int64_t ciphertext_length) const override {
    if (ciphertext_length < tag_size_) {
      return 0;
    }
    return ciphertext_length - tag_size_;
  }

 private:
  // Returns a new EVP_CIPHER_CTX for encryption (`ecryption` == true) or
  // decryption (`encryption` == false).
  util::StatusOr<internal::SslUniquePtr<EVP_CIPHER_CTX>> GetContext(
      absl::string_view iv, bool encryption) const {
    internal::SslUniquePtr<EVP_CIPHER_CTX> context(EVP_CIPHER_CTX_new());
    if (context == nullptr) {
      return util::Status(absl::StatusCode::kInternal,
                          "EVP_CIPHER_CTX_new failed");
    }
    const int encryption_flag = encryption ? 1 : 0;
    if (EVP_CipherInit_ex(context.get(), cipher_, /*impl=*/nullptr,
                          /*key=*/nullptr, /*iv=*/nullptr,
                          encryption_flag) <= 0) {
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat("Failed initializializing context for ",
                       encryption ? "encryption" : "decryption"));
    }
    // Set the size for IV first, then set the IV bytes.
    if (EVP_CIPHER_CTX_ctrl(context.get(), EVP_CTRL_AEAD_SET_IVLEN, iv.size(),
                            /*ptr=*/nullptr) <= 0) {
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat("Failed stting size of the IV to ", iv.size()));
    }
    if (EVP_CipherInit_ex(context.get(), /*cipher=*/nullptr, /*impl=*/nullptr,
                          reinterpret_cast<const uint8_t *>(key_.data()),
                          reinterpret_cast<const uint8_t *>(iv.data()),
                          encryption_flag) <= 0) {
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat("Failed to set key of size ", key_.size(),
                       "and IV of size ", iv.size()));
    }

    return std::move(context);
  }

  const util::SecretData key_;
  const EVP_CIPHER *cipher_;
  const size_t tag_size_;
};

#ifdef OPENSSL_IS_BORINGSSL

// Implementation of the one-shot AEAD cypter. This is purposely internal to
// an anonymous namespace to disallow direct use of this class other than
// through the Create* functions below.
class BoringSslOneShotAeadImpl : public SslOneShotAead {
 public:
  explicit BoringSslOneShotAeadImpl(
      internal::SslUniquePtr<EVP_AEAD_CTX> context, size_t tag_size)
      : context_(std::move(context)), tag_size_(tag_size) {}

  util::StatusOr<int64_t> Encrypt(absl::string_view plaintext,
                                  absl::string_view associated_data,
                                  absl::string_view iv,
                                  absl::Span<char> out) const override {
    // BoringSSL expects a non-null pointer for associated_data,
    // regardless of whether the size is 0.
    plaintext = internal::EnsureStringNonNull(plaintext);
    associated_data = internal::EnsureStringNonNull(associated_data);
    iv = internal::EnsureStringNonNull(iv);

    if (BuffersOverlap(plaintext, absl::string_view(out.data(), out.size()))) {
      return util::Status(absl::StatusCode::kInvalidArgument,
                          "Plaintext and output buffer must not overlap");
    }

    const int64_t min_out_buff_size = CiphertextSize(plaintext.size());
    if (out.size() < min_out_buff_size) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Output buffer too small; expected at least ",
                       min_out_buff_size, " got ", out.size()));
    }
    size_t out_len = 0;
    if (!EVP_AEAD_CTX_seal(
            context_.get(), reinterpret_cast<uint8_t *>(&out[0]), &out_len,
            out.size(), reinterpret_cast<const uint8_t *>(iv.data()), iv.size(),
            reinterpret_cast<const uint8_t *>(plaintext.data()),
            plaintext.size(),
            /*ad=*/reinterpret_cast<const uint8_t *>(associated_data.data()),
            /*ad_len=*/associated_data.size())) {
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat("Encryption failed: ", internal::GetSslErrors()));
    }

    return out_len;
  }

  util::StatusOr<int64_t> Decrypt(absl::string_view ciphertext,
                                  absl::string_view associated_data,
                                  absl::string_view iv,
                                  absl::Span<char> out) const override {
    ciphertext = internal::EnsureStringNonNull(ciphertext);
    associated_data = internal::EnsureStringNonNull(associated_data);
    iv = internal::EnsureStringNonNull(iv);

    if (BuffersOverlap(ciphertext, absl::string_view(out.data(), out.size()))) {
      return util::Status(absl::StatusCode::kInvalidArgument,
                          "Ciphertext and output buffer must not overlap");
    }

    if (ciphertext.size() < tag_size_) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Ciphertext buffer too small; expected at least ",
                       tag_size_, " got ", ciphertext.size()));
    }

    const int64_t min_out_buff_size = PlaintextSize(ciphertext.size());
    if (out.size() < min_out_buff_size) {
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Output buffer too small; expected at least ",
                       min_out_buff_size, " got ", out.size()));
    }

    // If out.empty() accessing the 0th element would result in an out of
    // bound violation. This makes sure we pass a pointer to at least one byte
    // when calling into OpenSSL.
    uint8_t buffer_if_size_is_zero;
    uint8_t *buffer_ptr = &buffer_if_size_is_zero;
    if (!out.empty()) {
      buffer_ptr = reinterpret_cast<uint8_t *>(&out[0]);
    }

    size_t out_len = 0;
    if (!EVP_AEAD_CTX_open(
            context_.get(), buffer_ptr, &out_len, out.size(),
            reinterpret_cast<const uint8_t *>(iv.data()), iv.size(),
            reinterpret_cast<const uint8_t *>(ciphertext.data()),
            ciphertext.size(),
            /*ad=*/reinterpret_cast<const uint8_t *>(associated_data.data()),
            /*ad_len=*/associated_data.size())) {
      return util::Status(
          absl::StatusCode::kInternal,
          absl::StrCat("Authentication failed: ", internal::GetSslErrors()));
    }

    return out_len;
  }

  int64_t CiphertextSize(int64_t plaintext_length) const override {
    return plaintext_length + tag_size_;
  }

  int64_t PlaintextSize(int64_t ciphertext_length) const override {
    if (ciphertext_length < tag_size_) {
      return 0;
    }
    return ciphertext_length - tag_size_;
  }

 private:
  const internal::SslUniquePtr<EVP_AEAD_CTX> context_;
  const size_t tag_size_;
};

#endif

}  // namespace

util::StatusOr<std::unique_ptr<SslOneShotAead>> CreateAesGcmOneShotCrypter(
    const util::SecretData &key) {
#ifdef OPENSSL_IS_BORINGSSL
  util::StatusOr<const EVP_AEAD *> aead_cipher =
      GetAesGcmAeadForKeySize(key.size());
  if (!aead_cipher.ok()) {
    return aead_cipher.status();
  }

  internal::SslUniquePtr<EVP_AEAD_CTX> context(EVP_AEAD_CTX_new(
      *aead_cipher, key.data(), key.size(), kAesGcmTagSizeInBytes));
  if (context == nullptr) {
    return util::Status(
        absl::StatusCode::kInternal,
        absl::StrCat("EVP_AEAD_CTX_new failed: ", internal::GetSslErrors()));
  }
  return {absl::make_unique<BoringSslOneShotAeadImpl>(std::move(context),
                                                      kAesGcmTagSizeInBytes)};
#else
  util::StatusOr<const EVP_CIPHER *> aead_cipher =
      GetAesGcmCipherForKeySize(key.size());
  if (!aead_cipher.ok()) {
    return aead_cipher.status();
  }

  return absl::make_unique<OpenSslOneShotAeadImpl>(key, *aead_cipher,
                                                   kAesGcmTagSizeInBytes);
#endif
}

util::StatusOr<std::unique_ptr<SslOneShotAead>> CreateAesGcmSivOneShotCrypter(
    const util::SecretData &key) {
#ifdef OPENSSL_IS_BORINGSSL
  util::StatusOr<const EVP_AEAD *> aead_cipher =
      GetAesGcmSivAeadCipherForKeySize(key.size());
  if (!aead_cipher.ok()) {
    return aead_cipher.status();
  }
  internal::SslUniquePtr<EVP_AEAD_CTX> context(EVP_AEAD_CTX_new(
      *aead_cipher, key.data(), key.size(), kAesGcmTagSizeInBytes));
  if (context == nullptr) {
    return util::Status(absl::StatusCode::kInternal,
                        absl::StrCat("EVP_AEAD_CTX_new initialization Failed: ",
                                     internal::GetSslErrors()));
  }
  return {absl::make_unique<BoringSslOneShotAeadImpl>(
      std::move(context), kAesGcmSivTagSizeInBytes)};
#else
  return util::Status(absl::StatusCode::kUnimplemented,
                      "AES-GCM-SIV is unimplemented for OpenSSL");
#endif
}

util::StatusOr<std::unique_ptr<SslOneShotAead>>
CreateXchacha20Poly1305OneShotCrypter(const util::SecretData &key) {
#ifdef OPENSSL_IS_BORINGSSL
  if (key.size() != 32) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Invalid key size; valid values are {32} bytes, got ",
                     key.size()));
  }

  internal::SslUniquePtr<EVP_AEAD_CTX> context(
      EVP_AEAD_CTX_new(EVP_aead_xchacha20_poly1305(), key.data(), key.size(),
                       kAesGcmTagSizeInBytes));
  if (context == nullptr) {
    return util::Status(absl::StatusCode::kInternal,
                        absl::StrCat("EVP_AEAD_CTX_new initialization Failed: ",
                                     internal::GetSslErrors()));
  }
  return {absl::make_unique<BoringSslOneShotAeadImpl>(
      std::move(context), kXchacha20Poly1305TagSizeInBytes)};
#else
  return util::Status(absl::StatusCode::kUnimplemented,
                      "Xchacha20-Poly1305 is unimplemented for OpenSSL");
#endif
}

}  // namespace internal
}  // namespace tink
}  // namespace crypto
