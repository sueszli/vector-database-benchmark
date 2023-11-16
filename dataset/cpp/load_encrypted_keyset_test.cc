// Copyright 2022 Google LLC
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

#include "walkthrough/load_encrypted_keyset.h"

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tink/aead.h"
#include "tink/aead/aead_config.h"
#include "walkthrough/load_cleartext_keyset.h"
#include "walkthrough/test_util.h"
#include "tink/kms_clients.h"
#include "tink/registry.h"
#include "tink/util/statusor.h"
#include "tink/util/test_matchers.h"

namespace tink_walkthrough {
namespace {

constexpr absl::string_view kSerializedMasterKeyKeyset = R"json({
  "key": [
    {
      "keyData": {
        "keyMaterialType": "SYMMETRIC",
        "typeUrl": "type.googleapis.com/google.crypto.tink.AesGcmKey",
        "value": "GiBWyUfGgYk3RTRhj/LIUzSudIWlyjCftCOypTr0jCNSLg=="
      },
      "keyId": 294406504,
      "outputPrefixType": "TINK",
      "status": "ENABLED"
    }
  ],
  "primaryKeyId": 294406504
})json";

constexpr absl::string_view kSerializedKeysetToEncrypt = R"json({
  "key": [
    {
      "keyData": {
        "keyMaterialType": "SYMMETRIC",
        "typeUrl": "type.googleapis.com/google.crypto.tink.AesGcmKey",
        "value": "GhD+9l0RANZjzZEZ8PDp7LRW"
      },
      "keyId": 1931667682,
      "outputPrefixType": "TINK",
      "status": "ENABLED"
    }
  ],
  "primaryKeyId": 1931667682
})json";

// Encryption of kSerializedKeysetToEncrypt using kSerializedMasterKeyKeyset.
constexpr absl::string_view kEncryptedKeyset = R"json({
  "encryptedKeyset": "ARGMSWi6YHyZ/Oqxl00XSq631a0q2UPmf+rCvCIAggSZrwCmxFF797MpY0dqgaXu1fz2eQ8zFNhlyTXv9kwg1kY6COpyhY/68zNBUkyKX4CharLYfpg1LgRl+6rMzIQa0XDHh7ZDmp1CevzecZIKnG83uDRHxxSv3h8c/Kc="
})json";

constexpr absl::string_view kFakeKmsKeyUri = "fake://some_key";

using ::crypto::tink::Aead;
using ::crypto::tink::KeysetHandle;
using ::crypto::tink::Registry;
using ::crypto::tink::test::IsOk;
using ::crypto::tink::test::IsOkAndHolds;
using ::crypto::tink::test::StatusIs;
using ::crypto::tink::util::Status;
using ::crypto::tink::util::StatusOr;
using ::testing::Environment;

// Test environment used to register KMS clients only once for the whole test.
class LoadKeysetTestEnvironment : public Environment {
 public:
  ~LoadKeysetTestEnvironment() override = default;

  // Register FakeKmsClient and AlwaysFailingFakeKmsClient.
  void SetUp() override {
    auto fake_kms =
        absl::make_unique<FakeKmsClient>(kSerializedMasterKeyKeyset);
    ASSERT_THAT(crypto::tink::KmsClients::Add(std::move(fake_kms)), IsOk());
    auto failing_kms = absl::make_unique<AlwaysFailingFakeKmsClient>();
    ASSERT_THAT(crypto::tink::KmsClients::Add(std::move(failing_kms)), IsOk());
  }
};

// Unused.
Environment *const test_env =
    testing::AddGlobalTestEnvironment(new LoadKeysetTestEnvironment());

class LoadKeysetTest : public ::testing::Test {
 public:
  void TearDown() override { Registry::Reset(); }
};

TEST_F(LoadKeysetTest, LoadKeysetFailsWhenNoKmsRegistered) {
  StatusOr<std::unique_ptr<KeysetHandle>> expected_keyset =
      LoadKeyset(kEncryptedKeyset, /*master_key_uri=*/"other_kms://some_key");
  EXPECT_THAT(expected_keyset.status(), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoadKeysetTest, LoadKeysetFailsWhenKmsClientFails) {
  StatusOr<std::unique_ptr<KeysetHandle>> expected_keyset =
      LoadKeyset(kEncryptedKeyset, /*master_key_uri=*/"failing://some_key");
  EXPECT_THAT(expected_keyset.status(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LoadKeysetTest, LoadKeysetFailsWhenAeadNotRegistered) {
  StatusOr<std::unique_ptr<KeysetHandle>> expected_keyset =
      LoadKeyset(kEncryptedKeyset, kFakeKmsKeyUri);
  EXPECT_THAT(expected_keyset.status(), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LoadKeysetTest, LoadKeysetFailsWhenInvalidKeyset) {
  ASSERT_THAT(crypto::tink::AeadConfig::Register(), IsOk());
  StatusOr<std::unique_ptr<KeysetHandle>> expected_keyset =
      LoadKeyset("invalid", kFakeKmsKeyUri);
  EXPECT_THAT(expected_keyset.status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  Registry::Reset();
}

TEST_F(LoadKeysetTest, LoadKeysetSucceeds) {
  ASSERT_THAT(crypto::tink::AeadConfig::Register(), IsOk());
  StatusOr<std::unique_ptr<KeysetHandle>> handle =
      LoadKeyset(kEncryptedKeyset, kFakeKmsKeyUri);
  ASSERT_THAT(handle, IsOk());
  StatusOr<std::unique_ptr<Aead>> aead =
      (*handle)->GetPrimitive<crypto::tink::Aead>(
          crypto::tink::ConfigGlobalRegistry());
  ASSERT_THAT(aead, IsOk());

  StatusOr<std::unique_ptr<KeysetHandle>> expected_keyset =
      LoadKeyset(kSerializedKeysetToEncrypt);
  ASSERT_THAT(expected_keyset, IsOk());
  StatusOr<std::unique_ptr<Aead>> expected_aead =
      (*expected_keyset)
          ->GetPrimitive<crypto::tink::Aead>(
              crypto::tink::ConfigGlobalRegistry());
  ASSERT_THAT(expected_aead, IsOk());

  std::string associated_data = "Some associated data";
  std::string plaintext = "Some plaintext";

  StatusOr<std::string> ciphertext =
      (*aead)->Encrypt(plaintext, associated_data);
  ASSERT_THAT(ciphertext, IsOk());
  EXPECT_THAT((*expected_aead)->Decrypt(*ciphertext, associated_data),
              IsOkAndHolds(plaintext));
}

}  // namespace
}  // namespace tink_walkthrough
