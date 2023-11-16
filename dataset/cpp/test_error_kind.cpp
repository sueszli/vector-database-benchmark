// Copyright (c) 2023 by Apex.AI Inc. All rights reserved.
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
// SPDX-License-Identifier: Apache-2.0

#include "test.hpp"
#include <gtest/gtest.h>

#include "iceoryx_hoofs/error_reporting/custom/error_kind.hpp"
#include "iceoryx_hoofs/error_reporting/error_kind.hpp"

namespace
{

using namespace ::testing;
using namespace iox::er;

// check the type traits of the error kinds

// tautologies are always true ...
TEST(ErrorKind_test, fatalErrorsAreFatal)
{
    ::testing::Test::RecordProperty("TEST_ID", "2524192f-a29c-45bc-b950-ae29deb8b3ae");

    FatalKind sut;
    EXPECT_TRUE(IsFatal<FatalKind>::value);
    EXPECT_TRUE(isFatal(sut));
}

TEST(ErrorKind_test, preconditionViolationsAreFatal)
{
    ::testing::Test::RecordProperty("TEST_ID", "21b79757-e46b-44fe-854a-7579b7f2243b");

    PreconditionViolationKind sut;
    EXPECT_TRUE(IsFatal<PreconditionViolationKind>::value);
    EXPECT_TRUE(isFatal(sut));
    EXPECT_TRUE(isFatal(PRECONDITION_VIOLATION));
}

TEST(ErrorKind_test, assumptionViolationsAreFatal)
{
    ::testing::Test::RecordProperty("TEST_ID", "b502e70e-157d-45a0-9654-61ada213531d");

    AssumptionViolationKind sut;
    EXPECT_TRUE(IsFatal<AssumptionViolationKind>::value);
    EXPECT_TRUE(isFatal(sut));
    EXPECT_TRUE(isFatal(ASSUMPTION_VIOLATION));
}

TEST(ErrorKind_test, runtimeErrorsAreNotFatal)
{
    ::testing::Test::RecordProperty("TEST_ID", "22c69c24-5082-4e81-8b3f-306e624731a5");

    RuntimeErrorKind sut;
    EXPECT_FALSE(IsFatal<RuntimeErrorKind>::value);
    EXPECT_FALSE(isFatal(sut));
}

} // namespace
