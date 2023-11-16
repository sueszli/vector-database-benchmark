// Copyright 2018 Google Inc.
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

#include "tink/version.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tink/internal/util.h"

namespace crypto {
namespace tink {
namespace {

using ::testing::AnyOf;
using ::testing::MatchesRegex;

TEST(VersionTest, VersionHasCorrectFormat) {
  // The regex represents Semantic Versioning syntax (www.semver.org),
  // i.e. three dot-separated numbers, with an optional suffix
  // that starts with a hyphen, to cover alpha/beta releases and
  // release candiates, for example:
  //   1.2.3
  //   1.2.3-beta
  //   1.2.3-RC1
  if (crypto::tink::internal::IsWindows()) {
    // Using the syntax in
    // https://github.com/google/googletest/blob/main/docs/advanced.md#regular-expression-syntax.
    EXPECT_THAT(Version::kTinkVersion,
                AnyOf(MatchesRegex(R"regex(\d+\.\d+\.\d+)regex"),
                      MatchesRegex(R"regex(\d+\.\d+\.\d+-\w+)regex")));
  } else {
    std::string version_regex = "[0-9]+[.][0-9]+[.][0-9]+(-[A-Za-z0-9]+)?";
    EXPECT_THAT(Version::kTinkVersion, testing::MatchesRegex(version_regex));
  }
}

}  // namespace
}  // namespace tink
}  // namespace crypto
