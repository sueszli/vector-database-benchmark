// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
////////////////////////////////////////////////////////////////////////////////

#include "tink/subtle/aes_eax_boringssl.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "openssl/err.h"
#include "tink/config/tink_fips.h"
#include "tink/subtle/wycheproof_util.h"
#include "tink/util/secret_data.h"
#include "tink/util/status.h"
#include "tink/util/statusor.h"
#include "tink/util/test_matchers.h"
#include "tink/util/test_util.h"

namespace crypto {
namespace tink {
namespace subtle {
namespace {

using ::crypto::tink::test::StatusIs;

TEST(AesEaxBoringSslTest, TestBasic) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  size_t nonce_size = 12;
  auto res = AesEaxBoringSsl::New(key, nonce_size);
  EXPECT_TRUE(res.ok()) << res.status();
  auto cipher = std::move(res.value());
  std::string message = "Some data to encrypt.";
  std::string associated_data = "Some data to authenticate.";
  auto ct = cipher->Encrypt(message, associated_data);
  EXPECT_TRUE(ct.ok()) << ct.status();
  EXPECT_EQ(ct.value().size(), message.size() + nonce_size + 16);
  auto pt = cipher->Decrypt(ct.value(), associated_data);
  EXPECT_TRUE(pt.ok()) << pt.status();
  EXPECT_EQ(pt.value(), message);
}

TEST(AesEaxBoringSslTest, TestMessageSize) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  size_t nonce_size = 12;
  auto res = AesEaxBoringSsl::New(key, nonce_size);
  EXPECT_TRUE(res.ok()) << res.status();
  auto cipher = std::move(res.value());
  for (size_t size = 0; size < 260; size++) {
    std::string message(size, 'x');
    std::string associated_data = "";
    auto ct = cipher->Encrypt(message, associated_data);
    EXPECT_TRUE(ct.ok()) << ct.status();
    EXPECT_EQ(ct.value().size(), message.size() + nonce_size + 16);
    auto pt = cipher->Decrypt(ct.value(), associated_data);
    EXPECT_TRUE(pt.ok()) << pt.status();
    EXPECT_EQ(pt.value(), message);
  }
}

TEST(AesEaxBoringSslTest, TestAssociatedDataSize) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  size_t nonce_size = 12;
  auto res = AesEaxBoringSsl::New(key, nonce_size);
  EXPECT_TRUE(res.ok()) << res.status();
  auto cipher = std::move(res.value());
  for (size_t size = 0; size < 260; size++) {
    std::string message("Some message");
    std::string associated_data(size, 'x');
    auto ct = cipher->Encrypt(message, associated_data);
    EXPECT_TRUE(ct.ok()) << ct.status();
    EXPECT_EQ(ct.value().size(), message.size() + nonce_size + 16);
    auto pt = cipher->Decrypt(ct.value(), associated_data);
    EXPECT_TRUE(pt.ok()) << pt.status();
    EXPECT_EQ(pt.value(), message);
  }
}

TEST(AesEaxBoringSslTest, TestLongNonce) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  size_t nonce_size = 16;
  auto res = AesEaxBoringSsl::New(key, nonce_size);
  EXPECT_TRUE(res.ok()) << res.status();
  auto cipher = std::move(res.value());
  std::string message = "Some data to encrypt.";
  std::string associated_data = "Some associated data.";
  auto ct = cipher->Encrypt(message, associated_data);
  EXPECT_TRUE(ct.ok()) << ct.status();
  EXPECT_EQ(ct.value().size(), message.size() + nonce_size + 16);
  auto pt = cipher->Decrypt(ct.value(), associated_data);
  EXPECT_TRUE(pt.ok()) << pt.status();
  EXPECT_EQ(pt.value(), message);
}

TEST(AesEaxBoringSslTest, TestModification) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  size_t nonce_size = 12;
  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  auto cipher = std::move(AesEaxBoringSsl::New(key, nonce_size).value());
  std::string message = "Some data to encrypt.";
  std::string associated_data = "Some data to authenticate.";
  std::string ct = cipher->Encrypt(message, associated_data).value();
  EXPECT_TRUE(cipher->Decrypt(ct, associated_data).ok());
  // Modify the ciphertext
  for (size_t i = 0; i < ct.size() * 8; i++) {
    std::string modified_ct = ct;
    modified_ct[i / 8] ^= 1 << (i % 8);
    EXPECT_FALSE(cipher->Decrypt(modified_ct, associated_data).ok()) << i;
  }
  // Modify the associated data
  for (size_t i = 0; i < associated_data.size() * 8; i++) {
    std::string modified_associated_data = associated_data;
    modified_associated_data[i / 8] ^= 1 << (i % 8);
    auto decrypted = cipher->Decrypt(ct, modified_associated_data);
    EXPECT_FALSE(decrypted.ok()) << i << " pt:" << decrypted.value();
  }
  // Truncate the ciphertext
  for (size_t i = 0; i < ct.size(); i++) {
    std::string truncated_ct(ct, 0, i);
    EXPECT_FALSE(cipher->Decrypt(truncated_ct, associated_data).ok()) << i;
  }
}

TEST(AesEaxBoringSslTest, TestInvalidKeySizes) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  size_t nonce_size = 12;
  for (int keysize = 0; keysize < 65; keysize++) {
    if (keysize == 16 || keysize == 32) {
      continue;
    }
    util::SecretData key(keysize, 'x');
    auto cipher = AesEaxBoringSsl::New(key, nonce_size);
    EXPECT_FALSE(cipher.ok());
  }
}

TEST(AesEaxBoringSslTest, TestEmpty) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  size_t nonce_size = 12;
  util::SecretData key = util::SecretDataFromStringView(
      test::HexDecodeOrDie("bedcfb5a011ebc84600fcb296c15af0d"));
  std::string nonce(test::HexDecodeOrDie("438a547a94ea88dce46c6c85"));
  // Expected tag is an empty string with an empty tag is encrypted with
  // the nonce above;
  std::string tag(test::HexDecodeOrDie("9607977cd7556b1dfedf0c73a35a5197"));
  std::string ciphertext = nonce + tag;
  auto res = AesEaxBoringSsl::New(key, nonce_size);
  EXPECT_TRUE(res.ok()) << res.status();
  auto cipher = std::move(res.value());

  // Test decryption of the arguments above.
  std::string empty_string("");
  absl::string_view empty_string_view("");
  absl::string_view null_string_view;

  auto pt = cipher->Decrypt(ciphertext, empty_string);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  pt = cipher->Decrypt(ciphertext, empty_string_view);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  pt = cipher->Decrypt(ciphertext, null_string_view);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  // Test encryption.
  auto ct = cipher->Encrypt(empty_string, empty_string);
  EXPECT_TRUE(ct.ok());
  pt = cipher->Decrypt(ct.value(), empty_string);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  ct = cipher->Encrypt(empty_string_view, empty_string_view);
  EXPECT_TRUE(ct.ok());
  pt = cipher->Decrypt(ct.value(), empty_string);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  ct = cipher->Encrypt(empty_string_view, empty_string_view);
  EXPECT_TRUE(ct.ok());
  pt = cipher->Decrypt(ct.value(), empty_string);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());

  ct = cipher->Encrypt(null_string_view, null_string_view);
  EXPECT_TRUE(ct.ok());
  pt = cipher->Decrypt(ct.value(), empty_string);
  EXPECT_TRUE(pt.ok());
  EXPECT_EQ(0, pt.value().size());
}

static std::string GetError() {
  auto err = ERR_peek_last_error();
  // Sometimes there is no error message on the stack.
  if (err == 0) {
    return "";
  }
  std::string lib(ERR_lib_error_string(err));
  std::string func(ERR_func_error_string(err));
  std::string reason(ERR_reason_error_string(err));
  return lib + ":" + func + ":" + reason;
}

// Test with test vectors from project Wycheproof.
// AesEaxBoringSsl does not allow to pass in IVs. Therefore this test
// can only test decryption.
// Currently AesEaxBoringSsl is restricted to encryption with 12 byte
// IVs and 16 byte tags. Therefore it is necessary to skip tests with
// other parameter sizes.
bool WycheproofTest(const rapidjson::Document &root) {
  int errors = 0;
  for (const rapidjson::Value& test_group : root["testGroups"].GetArray()) {
    const size_t iv_size = test_group["ivSize"].GetInt();
    const size_t key_size = test_group["keySize"].GetInt();
    const size_t tag_size = test_group["tagSize"].GetInt();
    if (key_size != 128 && key_size != 256) {
      // Not supported
      continue;
    }
    if (iv_size != 128 && iv_size != 96) {
      // Not supported
      continue;
    }
    if (tag_size != 128) {
      // Not supported
      continue;
    }
    for (const rapidjson::Value& test : test_group["tests"].GetArray()) {
      std::string comment = test["comment"].GetString();
      util::SecretData key =
          util::SecretDataFromStringView(WycheproofUtil::GetBytes(test["key"]));
      std::string iv = WycheproofUtil::GetBytes(test["iv"]);
      std::string msg = WycheproofUtil::GetBytes(test["msg"]);
      std::string ct = WycheproofUtil::GetBytes(test["ct"]);
      std::string associated_data = WycheproofUtil::GetBytes(test["aad"]);
      std::string tag = WycheproofUtil::GetBytes(test["tag"]);
      std::string id = absl::StrCat(test["tcId"].GetInt());
      std::string expected = test["result"].GetString();
      auto cipher = std::move(AesEaxBoringSsl::New(key, iv_size / 8).value());
      auto result = cipher->Decrypt(iv + ct + tag, associated_data);
      bool success = result.ok();
      if (success) {
        std::string decrypted = result.value();
        if (expected == "invalid") {
          ADD_FAILURE() << "decrypted invalid ciphertext:" << id;
          errors++;
        } else if (msg != decrypted) {
          ADD_FAILURE() << "Incorrect decryption:" << id;
          errors++;
        }
      } else {
        if (expected == "valid" || expected == "acceptable") {
          ADD_FAILURE()
              << "Could not decrypt test with tcId:" << id
              << " iv_size:" << iv_size
              << " tag_size:" << tag_size
              << " key_size:" << key_size
              << " error:" << GetError();
          errors++;
        }
      }
    }
  }
  return errors == 0;
}

TEST(AesEaxBoringSslTest, TestVectors) {
  if (IsFipsModeEnabled()) {
    GTEST_SKIP() << "Not supported in FIPS-only mode";
  }

  std::unique_ptr<rapidjson::Document> root =
      WycheproofUtil::ReadTestVectors("aes_eax_test.json");
  ASSERT_TRUE(WycheproofTest(*root));
}

TEST(AesEaxBoringSslTest, TestFipsOnly) {
  if (!IsFipsModeEnabled()) {
    GTEST_SKIP() << "Only supported in FIPS-only mode";
  }

  util::SecretData key128 = util::SecretDataFromStringView(
      test::HexDecodeOrDie("000102030405060708090a0b0c0d0e0f"));
  util::SecretData key256 = util::SecretDataFromStringView(test::HexDecodeOrDie(
      "000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f"));

  EXPECT_THAT(subtle::AesEaxBoringSsl::New(key128, 16).status(),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(subtle::AesEaxBoringSsl::New(key256, 16).status(),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace subtle
}  // namespace tink
}  // namespace crypto

