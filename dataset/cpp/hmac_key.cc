// Copyright 2023 Google LLC
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

#include "tink/mac/hmac_key.h"

#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tink/mac/hmac_parameters.h"
#include "tink/partial_key_access_token.h"
#include "tink/restricted_data.h"
#include "tink/subtle/subtle_util.h"
#include "tink/util/status.h"
#include "tink/util/statusor.h"

namespace crypto {
namespace tink {

util::StatusOr<HmacKey> HmacKey::Create(const HmacParameters& parameters,
                                        const RestrictedData& key_bytes,
                                        absl::optional<int> id_requirement,
                                        PartialKeyAccessToken token) {
  if (parameters.KeySizeInBytes() != key_bytes.size()) {
    return util::Status(absl::StatusCode::kInvalidArgument,
                        "Key size does not match HMAC parameters");
  }
  if (parameters.HasIdRequirement() && !id_requirement.has_value()) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        "Cannot create key without ID requirement with parameters with ID "
        "requirement");
  }
  if (!parameters.HasIdRequirement() && id_requirement.has_value()) {
    return util::Status(
        absl::StatusCode::kInvalidArgument,
        "Cannot create key with ID requirement with parameters without ID "
        "requirement");
  }
  util::StatusOr<std::string> output_prefix =
      ComputeOutputPrefix(parameters, id_requirement);
  if (!output_prefix.ok()) {
    return output_prefix.status();
  }
  return HmacKey(parameters, key_bytes, id_requirement,
                 *std::move(output_prefix));
}

util::StatusOr<std::string> HmacKey::ComputeOutputPrefix(
    const HmacParameters& parameters, absl::optional<int> id_requirement) {
  switch (parameters.GetVariant()) {
    case HmacParameters::Variant::kNoPrefix:
      return std::string("");  // Empty prefix.
    case HmacParameters::Variant::kLegacy:
      ABSL_FALLTHROUGH_INTENDED;
    case HmacParameters::Variant::kCrunchy:
      if (!id_requirement.has_value()) {
        return util::Status(
            absl::StatusCode::kInvalidArgument,
            "id requirement must have value with kCrunchy or kLegacy");
      }
      return absl::StrCat(absl::HexStringToBytes("00"),
                          subtle::BigEndian32(*id_requirement));
    case HmacParameters::Variant::kTink:
      if (!id_requirement.has_value()) {
        return util::Status(absl::StatusCode::kInvalidArgument,
                            "id requirement must have value with kTink");
      }
      return absl::StrCat(absl::HexStringToBytes("01"),
                          subtle::BigEndian32(*id_requirement));
    default:
      return util::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Invalid variant: ", parameters.GetVariant()));
  }
}

bool HmacKey::operator==(const Key& other) const {
  const HmacKey* that = dynamic_cast<const HmacKey*>(&other);
  if (that == nullptr) {
    return false;
  }
  if (GetParameters() != that->GetParameters()) {
    return false;
  }
  if (id_requirement_ != that->id_requirement_) {
    return false;
  }
  return key_bytes_ == that->key_bytes_;
}

}  // namespace tink
}  // namespace crypto
