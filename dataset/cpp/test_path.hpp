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

#ifndef REALM_TEST_UTIL_TEST_PATH_HPP
#define REALM_TEST_UTIL_TEST_PATH_HPP

#include <string>
#include <memory>

#include <realm/string_data.hpp>
#include <realm/util/features.h>

#define TEST_PATH_HELPER(class_name, var_name, suffix)                                                               \
    class_name var_name(realm::test_util::get_test_path(test_context.get_test_name(), "." #var_name "." suffix))

#if REALM_PLATFORM_APPLE
// Apple doesn't support file names with "�"
#define TEST_PATH(var_name) TEST_PATH_HELPER(realm::test_util::TestPathGuard, var_name, "test");
#else
#define TEST_PATH(var_name) TEST_PATH_HELPER(realm::test_util::TestPathGuard, var_name, "tempor�re");
#endif

#define TEST_DIR(var_name) TEST_PATH_HELPER(realm::test_util::TestDirGuard, var_name, "test-dir");

#define GROUP_TEST_PATH(var_name) TEST_PATH_HELPER(realm::test_util::TestPathGuard, var_name, "realm");

#define SHARED_GROUP_TEST_PATH(var_name) TEST_PATH_HELPER(realm::test_util::DBTestPathGuard, var_name, "realm");

namespace realm {

class DB;

namespace test_util {

/// Disable automatic removal of test files.
///
/// This function is **not** thread-safe. If you call it, be sure to call it
/// prior to any execution of the TEST_PATH or TEST_DIR family of macros.
void keep_test_files();


/// This function is thread-safe as long as there are no concurrent invocations
/// of set_test_path_prefix().
std::string get_test_path_prefix();

/// This function is thread-safe as long as there are no concurrent invocations
/// of set_test_path_prefix().
std::string get_test_path(const std::string& path, const std::string& suffix);

/// Initialize the test path prefix, resource path, and working directory. This function is not thread-safe and should
/// be called exactly once on startup.
bool initialize_test_path(int argc, const char* argv[]);

/// Check if get_test_path_prefix() will give a path located on an exFAT
/// filesystem, which does not support all of the features a typical unix
/// filesystem does.
bool test_dir_is_exfat();


/// This function is thread-safe as long as there are no concurrent invocations
/// of set_test_resource_path().
std::string get_test_resource_path();

/// This function is thread-safe as long as there are no concurrent invocations
/// of initialize_test_path
std::string get_test_exe_name();

// This is an adapter class which replaces dragging in the whole test framework
// by implementing the `get_test_name()` method from the TestContext class.
// It allows use of TestPathGuard and friends outside of a unit test:
// RealmPathInfo test_context { path };
struct RealmPathInfo {
    std::string m_path;
    std::string get_test_name() const
    {
        return m_path;
    }
};


/// Constructor and destructor removes file if it exists.
class TestPathGuard {
public:
    TestPathGuard(const std::string& path);
    ~TestPathGuard() noexcept;
    operator std::string() const
    {
        return m_path;
    }
    operator StringData() const
    {
        return m_path;
    }
    const char* c_str() const noexcept
    {
        return m_path.c_str();
    }
    TestPathGuard(const TestPathGuard&) = delete;
    TestPathGuard& operator=(const TestPathGuard&) = delete;
    TestPathGuard(TestPathGuard&&) noexcept;
    TestPathGuard& operator=(TestPathGuard&&) noexcept;

protected:
    std::string m_path;
    bool m_do_remove;
};

/// The constructor creates the directory if it does not already exist, then
/// removes any files already in it. The destructor removes files in the
/// directory, then removes the directory.
class TestDirGuard {
public:
    TestDirGuard(const std::string& path, bool init_clean = true);
    ~TestDirGuard() noexcept;
    operator std::string() const
    {
        return m_path;
    }
    operator StringData() const
    {
        return m_path;
    }
    const char* c_str() const
    {
        return m_path.c_str();
    }

    bool do_remove = true;

    void clean_dir()
    {
        clean_dir(m_path);
    }

private:
    std::string m_path;
    void clean_dir(const std::string& path);
};

class DBTestPathGuard : public TestPathGuard {
public:
    DBTestPathGuard(const std::string& path);
    std::string get_lock_path() const
    {
        return m_path + ".lock"; // ".management/access_control";
    }
    ~DBTestPathGuard() noexcept;
    DBTestPathGuard(DBTestPathGuard&&) = default;
    DBTestPathGuard& operator=(DBTestPathGuard&&) = default;

private:
    void cleanup() const noexcept;
};

class TestDirNameGenerator {
public:
    TestDirNameGenerator(std::string path);

    std::string next();

private:
    std::string m_path;
    std::size_t m_counter = 0;
};

std::shared_ptr<DB> get_test_db(const std::string& path, const char* crypt_key = nullptr);

} // namespace test_util
} // namespace realm

#endif // REALM_TEST_UTIL_TEST_PATH_HPP
