#include "testsettings.hpp"
#ifdef TEST_UTIL_FILE

#include <realm/util/file.hpp>
#include <filesystem>

#include "test.hpp"
#include <cstdio>

#ifndef _WIN32
#include <unistd.h>
#endif

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

// FIXME: Methods on File are not yet implemented on Windows

TEST(Utils_File_dir)
{
#ifndef _WIN32
    if (getuid() == 0) {
        std::cout << "Utils_File_dir test skipped because you are running it as root\n\n";
        return;
    }
#endif

    std::string dir_name = File::resolve("tempdir", test_util::get_test_path_prefix());

    // Create directory
    bool dir_exists = File::is_dir(dir_name);
    CHECK_NOT(dir_exists);

    make_dir(dir_name);
    try {
        make_dir(dir_name);
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.code(), ErrorCodes::FileAlreadyExists);
        CHECK_EQUAL(e.get_path(), dir_name);
        dir_exists = File::is_dir(dir_name);
    }
    CHECK(dir_exists);

#ifndef _WIN32
    bool perm_denied = false;
    try {
        make_dir("/foobar");
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.get_path(), "/foobar");
        perm_denied = true;
    }
    CHECK(perm_denied);

    perm_denied = false;
    try {
        remove_dir("/usr");
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.get_path(), "/usr");
        perm_denied = true;
    }
    CHECK(perm_denied);
#endif

    // Remove directory
    remove_dir(dir_name);
    try {
        remove_dir(dir_name);
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.code(), ErrorCodes::FileNotFound);
        CHECK_EQUAL(e.get_path(), dir_name);
        dir_exists = false;
    }
    CHECK_NOT(dir_exists);

    // try_remove_dir missing directory
    dir_exists = try_remove_dir(dir_name);
    CHECK_NOT(dir_exists);

    // try_remove_dir existing directory
    make_dir(dir_name);
    dir_exists = try_remove_dir(dir_name);
    CHECK(dir_exists);
}

TEST(Utils_File_dir_unicode)
{
    using std::filesystem::u8path;

    constexpr char all_the_unicode[] = u8"фоо-бар Λορεμ ლორემ 植物 החלל جمعت søren";
    std::string dir_name = File::resolve(all_the_unicode, test_util::get_test_path_prefix());

    // Create directory
    bool dir_exists = File::is_dir(dir_name);
    CHECK_NOT(dir_exists);

    make_dir(dir_name);
    try {
        make_dir(dir_name);
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.code(), ErrorCodes::FileAlreadyExists);
        CHECK_EQUAL(e.get_path(), dir_name);
        dir_exists = File::is_dir(dir_name);
    }
    CHECK(dir_exists);

    // Double-check using filesystem facilities with enforce UTF-8 handling
    CHECK(std::filesystem::exists(u8path(test_util::get_test_path_prefix()) / u8path(all_the_unicode)));

    // Create file
    File f(File::resolve("test.realm", dir_name), File::mode_Write);
    f.close();
    File::remove(f.get_path());

    // Remove directory
    remove_dir(dir_name);
    try {
        remove_dir(dir_name);
    }
    catch (const FileAccessError& e) {
        CHECK_EQUAL(e.code(), ErrorCodes::FileNotFound);
        CHECK_EQUAL(e.get_path(), dir_name);
        dir_exists = false;
    }
    CHECK_NOT(dir_exists);

    // try_remove_dir missing directory
    dir_exists = try_remove_dir(dir_name);
    CHECK_NOT(dir_exists);

    // try_remove_dir existing directory
    make_dir(dir_name);
    dir_exists = try_remove_dir(dir_name);
    CHECK(dir_exists);
}

TEST(Utils_File_resolve)
{
    std::string res;
    res = File::resolve("", "");

#if REALM_HAVE_STD_FILESYSTEM
    // The std::filesystem-based implementation canonicalizes the resolved path in terms of '.' and '..'
    // This doesn't affect the actual behavior of the file APIs since 'some/dir/.' == 'some/dir'
    // but it does produce a different string value.
    CHECK_EQUAL(res, "");
#else
    CHECK_EQUAL(res, ".");
#endif

#ifdef _WIN32
    res = File::resolve("C:\\foo\\bar", "dir");
    CHECK_EQUAL(res, "C:\\foo\\bar");

    res = File::resolve("foo\\bar", "");
    CHECK_EQUAL(res, "foo\\bar");

    res = File::resolve("file", "dir");
    CHECK_EQUAL(res, "dir\\file");

    res = File::resolve("file\\", "dir");
    CHECK_EQUAL(res, "dir\\file\\");
#else
    res = File::resolve("/foo/bar", "dir");
    CHECK_EQUAL(res, "/foo/bar");

    res = File::resolve("foo/bar", "");
    CHECK_EQUAL(res, "foo/bar");

    res = File::resolve("file", "dir");
    CHECK_EQUAL(res, "dir/file");

    res = File::resolve("file/", "dir");
    CHECK_EQUAL(res, "dir/file/");
#endif

    /* Function does not work as specified - but not used
    res = File::resolve("../baz", "/foo/bar");
    CHECK_EQUAL(res, "/foo/baz");
    */
}

TEST(Utils_File_TryRemoveDirRecursive)
{
    TEST_DIR(dir_0);
    bool did_exist = false;

    std::string dir_1  = File::resolve("dir_1",  dir_0);
    make_dir(dir_1);
    did_exist = try_remove_dir_recursive(dir_1);
    CHECK(did_exist);

    std::string dir_2  = File::resolve("dir_2",  dir_0);
    did_exist = try_remove_dir_recursive(dir_2);
    CHECK(!did_exist);

    std::string dir_3  = File::resolve("dir_3",  dir_0);
    make_dir(dir_3);
    std::string file_1 = File::resolve("file_1", dir_3);
    File(file_1, File::mode_Write);
    did_exist = try_remove_dir_recursive(dir_3);
    CHECK(did_exist);

    // Try to remove dir_3 again;
    did_exist = try_remove_dir_recursive(dir_3);
    CHECK(!did_exist);
}

TEST(Utils_File_ForEach)
{
    TEST_DIR(dir_0);
    auto touch = [](const std::string& path) {
        File(path, File::mode_Write);
    };
    std::string dir_1  = File::resolve("dir_1",  dir_0);
    make_dir(dir_1);
    std::string file_1 = File::resolve("file_1", dir_0);
    touch(file_1);
    std::string dir_2  = File::resolve("dir_2",  dir_0);
    make_dir(dir_2);
    std::string file_2 = File::resolve("file_2", dir_0);
    touch(file_2);
    std::string dir_3  = File::resolve("dir_3",  dir_1);
    make_dir(dir_3);
    std::string file_3 = File::resolve("file_3", dir_1);
    touch(file_3);
    std::string dir_4  = File::resolve("dir_4",  dir_2);
    make_dir(dir_4);
    std::string file_4 = File::resolve("file_4", dir_2);
    touch(file_4);
    std::string file_5 = File::resolve("file_5", dir_3);
    touch(file_5);
    std::string file_6 = File::resolve("file_6", dir_4);
    touch(file_6);
    std::vector<std::pair<std::string, std::string>> files;
    auto handler = [&](const std::string& file, const std::string& dir) {
        files.emplace_back(dir, file);
        return true;
    };
    File::for_each(dir_0, handler);
    std::sort(files.begin(), files.end());
    std::string dir_1_3 = File::resolve("dir_3", "dir_1");
    std::string dir_2_4 = File::resolve("dir_4", "dir_2");
    if (CHECK_EQUAL(6, files.size())) {
        CHECK_EQUAL("",       files[0].first);
        CHECK_EQUAL("file_1", files[0].second);
        CHECK_EQUAL("",       files[1].first);
        CHECK_EQUAL("file_2", files[1].second);
        CHECK_EQUAL("dir_1",  files[2].first);
        CHECK_EQUAL("file_3", files[2].second);
        CHECK_EQUAL(dir_1_3,  files[3].first);
        CHECK_EQUAL("file_5", files[3].second);
        CHECK_EQUAL("dir_2",  files[4].first);
        CHECK_EQUAL("file_4", files[4].second);
        CHECK_EQUAL(dir_2_4,  files[5].first);
        CHECK_EQUAL("file_6", files[5].second);
    }
}

TEST(Utils_File_Lock)
{
    TEST_DIR(dir);
    util::try_make_dir(dir);
    auto file = File::resolve("test", dir);
    File f1(file, File::mode_Write);
    File f2(file);
    CHECK(f1.try_rw_lock_exclusive());
    CHECK_NOT(f2.try_rw_lock_shared());
    f1.rw_unlock();
    CHECK(f1.try_rw_lock_shared());
    CHECK_NOT(f2.try_rw_lock_exclusive());
}

TEST(Utils_File_SystemErrorMessage)
{
    std::error_code err = std::make_error_code(std::errc::too_many_files_open);
    std::string_view message = "my message";
#ifdef _WIN32
    const char* expected = "my message: too many files open (%1)";
#elif defined(__linux__) && !defined(__GLIBC__)
    // Linux and not glibc implies Musl, which has its own message
    const char* expected = "my message: No file descriptors available (%1)";
#else
    const char* expected = "my message: Too many open files (%1)";
#endif
    CHECK_THROW_CONTAINING_MESSAGE(throw SystemError(err, message), message);
    CHECK_THROW_CONTAINING_MESSAGE(throw SystemError(err.value(), message), util::format(expected, err.value()));
}

#endif
