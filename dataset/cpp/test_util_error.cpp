/*************************************************************************
 *
 * Copyright 2016 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#include "testsettings.hpp"
#ifdef TEST_UTIL_ERROR

#include <realm/util/basic_system_errors.hpp>

#include "test.hpp"

using namespace realm;
using namespace realm::util;

// Test independence and thread-safety
// -----------------------------------
//
// All tests must be thread safe and independent of each other. This
// is required because it allows for both shuffling of the execution
// order and for parallelized testing.
//
// In particular, avoid using std::rand() since it is not guaranteed
// to be thread safe. Instead use the API offered in
// `test/util/random.hpp`.
//
// All files created in tests must use the TEST_PATH macro (or one of
// its friends) to obtain a suitable file system path. See
// `test/util/test_path.hpp`.
//
//
// Debugging and the ONLY() macro
// ------------------------------
//
// A simple way of disabling all tests except one called `Foo`, is to
// replace TEST(Foo) with ONLY(Foo) and then recompile and rerun the
// test suite. Note that you can also use filtering by setting the
// environment varible `UNITTEST_FILTER`. See `README.md` for more on
// this.
//
// Another way to debug a particular test, is to copy that test into
// `experiments/testcase.cpp` and then run `sh build.sh
// check-testcase` (or one of its friends) from the command line.

TEST(BasicSystemErrors_Category)
{
    std::error_code err = make_error_code(error::operation_aborted);
    CHECK_EQUAL(err.category().name(), "realm.basic_system");
}


TEST(BasicSystemErrors_Messages)
{
#if defined(__linux__) && !defined(__GLIBC__)
    // Linux and not glibc implies Musl, which has its own message
    const std::string error_message("No error information");
#else
    const std::string error_message("Unknown error");
#endif

    {
        std::error_code err = make_error_code(error::address_family_not_supported);

        CHECK_GREATER(err.message().length(), 0);
#ifndef _WIN32 // Older versions of the Windows CRT return "Unknown error" for this error instead of an actual message
        CHECK_NOT_EQUAL(err.message(), error_message);
#endif
    }
    {
        std::error_code err = make_error_code(error::invalid_argument);
        CHECK_GREATER(err.message().length(), 0);
        CHECK_NOT_EQUAL(err.message(), error_message);
    }
    {
        std::error_code err = make_error_code(error::no_memory);
        CHECK_GREATER(err.message().length(), 0);
        CHECK_NOT_EQUAL(err.message(), error_message);
    }
    {
        std::error_code err = make_error_code(error::operation_aborted);
        CHECK_GREATER(err.message().length(), 0);
#ifndef _WIN32 // Older versions of the Windows CRT return "Unknown error" for this error instead of an actual message
        CHECK_NOT_EQUAL(err.message(), error_message);
#endif
    }

#if !REALM_HAVE_CLANG_FEATURE(undefined_behavior_sanitizer)
    // Ensure that if we pass an unknown error code, we get some error reporting
    // This may potentially pass on some operating system. If this test starts
    // failing, simply change the magic number below.
    {
        std::error_code err = make_error_code(static_cast<error::basic_system_errors>(64532));
        CHECK_GREATER(err.message().length(), 0);
        // Check that `err.message()` begins with `error_message_prefix`
        // On Linux and Apple systems, the returned string is of format:
        // "Unknown error: <errcode>"
        CHECK(err.message().compare(0, error_message.length(), error_message) == 0);
    }
#endif // !REALM_HAVE_CLANG_FEATURE(undefined_behavior_sanitizer)
}

#endif // TEST_BASIC_SYSTEM_ERRORS
