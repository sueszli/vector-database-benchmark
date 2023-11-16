/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yggdrasil_decision_forests/utils/protobuf.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

TEST(Protobuf, ParseTextProto) {
  const auto message = ParseTextProto<proto::IntegerDistributionInt64>(R"(
    sum: 5
  )")
                           .value();
  EXPECT_EQ(message.sum(), 5);
}

TEST(Protobuf, ParseBinaryProto) {
  proto::IntegerDistributionInt64 m1;
  m1.set_sum(5);
  const auto m2 =
      ParseBinaryProto<proto::IntegerDistributionInt64>(m1.SerializeAsString())
          .value();
  EXPECT_EQ(m2.sum(), 5);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
