// Copyright 2019 Google LLC
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

#include "tink/streamingaead/shared_random_access_stream.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tink/internal/test_random_access_stream.h"
#include "tink/random_access_stream.h"
#include "tink/subtle/random.h"

namespace crypto {
namespace tink {
namespace streamingaead {
namespace {

TEST(SharedRandomAccessStreamTest, ReadingStreams) {
  for (auto stream_size : {0, 10, 100, 1000, 10000, 1000000}) {
    SCOPED_TRACE(absl::StrCat("stream_size = ", stream_size));
    std::string stream_content = subtle::Random::GetRandomBytes(stream_size);
    auto ra_stream =
        absl::make_unique<internal::TestRandomAccessStream>(stream_content);
    SharedRandomAccessStream shared_stream(ra_stream.get());
    std::string stream_contents;
    auto status = internal::ReadAllFromRandomAccessStream(
        &shared_stream, stream_contents, /*chunk_size=*/1 + (stream_size / 10));
    EXPECT_EQ(absl::StatusCode::kOutOfRange, status.code());
    EXPECT_EQ("EOF", status.message());
    EXPECT_EQ(stream_content, stream_contents);
    EXPECT_EQ(stream_size, shared_stream.size().value());
  }
}


}  // namespace
}  // namespace streamingaead
}  // namespace tink
}  // namespace crypto
