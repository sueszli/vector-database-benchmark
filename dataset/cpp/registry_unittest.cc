//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/win/registry.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace base::win {

namespace {

class RegistryTest : public testing::Test
{
protected:
#if defined(_WIN64)
    static const REGSAM kNativeViewMask = KEY_WOW64_64KEY;
    static const REGSAM kRedirectedViewMask = KEY_WOW64_32KEY;
#else
    static const REGSAM kNativeViewMask = KEY_WOW64_32KEY;
    static const REGSAM kRedirectedViewMask = KEY_WOW64_64KEY;
#endif  //  _WIN64

    RegistryTest() = default;

    void SetUp() override
    {
        // Create a temporary key.
        RegistryKey key(HKEY_CURRENT_USER, L"", KEY_ALL_ACCESS);
        key.deleteKey(kRootKey);
        ASSERT_NE(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_READ));
        ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, kRootKey, KEY_READ));
        foo_software_key_ = L"Software\\";
        foo_software_key_ += kRootKey;
        foo_software_key_ += L"\\Foo";
    }

    void TearDown() override
    {
        // Clean up the temporary key.
        RegistryKey key(HKEY_CURRENT_USER, L"", KEY_SET_VALUE);
        ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(kRootKey));
        ASSERT_NE(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_READ));
    }

    static bool isRedirectorPresent()
    {
#if defined(_WIN64)
        return true;
#else
        typedef BOOL(WINAPI* IsWow64ProcessFunc)(HANDLE, PBOOL);

        IsWow64ProcessFunc is_wow64_process = reinterpret_cast<IsWow64ProcessFunc>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process"));
        if (!is_wow64_process)
            return false;

        BOOL is_wow64 = FALSE;
        if (!(*is_wow64_process)(GetCurrentProcess(), &is_wow64))
            return false;

        return !!is_wow64;
#endif
    }

    const wchar_t* const kRootKey = L"Base_Registry_Unittest";
    std::wstring foo_software_key_;

private:
    DISALLOW_COPY_AND_ASSIGN(RegistryTest);
};

// static
const REGSAM RegistryTest::kNativeViewMask;
const REGSAM RegistryTest::kRedirectedViewMask;

TEST_F(RegistryTest, ValueTest)
{
    RegistryKey key;

    std::wstring foo_key(kRootKey);
    foo_key += L"\\Foo";
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));

    {
        ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(),
                                          KEY_READ | KEY_SET_VALUE));
        ASSERT_TRUE(key.isValid());

        const wchar_t kStringValueName[] = L"StringValue";
        const wchar_t kDWORDValueName[] = L"DWORDValue";
        const wchar_t kInt64ValueName[] = L"Int64Value";
        const wchar_t kStringData[] = L"string data";
        const DWORD kDWORDData = 0xdeadbabe;
        const int64_t kInt64Data = 0xdeadbabedeadbabeLL;

        // Test value creation
        ASSERT_EQ(ERROR_SUCCESS, key.writeValue(kStringValueName, kStringData));
        ASSERT_EQ(ERROR_SUCCESS, key.writeValue(kDWORDValueName, kDWORDData));
        ASSERT_EQ(ERROR_SUCCESS, key.writeValue(kInt64ValueName, &kInt64Data,
                                                sizeof(kInt64Data), REG_QWORD));
        EXPECT_EQ(3U, key.valueCount());
        EXPECT_TRUE(key.hasValue(kStringValueName));
        EXPECT_TRUE(key.hasValue(kDWORDValueName));
        EXPECT_TRUE(key.hasValue(kInt64ValueName));

        // Test Read
        std::wstring string_value;
        DWORD dword_value = 0;
        int64_t int64_value = 0;
        ASSERT_EQ(ERROR_SUCCESS, key.readValue(kStringValueName, &string_value));
        ASSERT_EQ(ERROR_SUCCESS, key.readValueDW(kDWORDValueName, &dword_value));
        ASSERT_EQ(ERROR_SUCCESS, key.readInt64(kInt64ValueName, &int64_value));
        EXPECT_STREQ(kStringData, string_value.c_str());
        EXPECT_EQ(kDWORDData, dword_value);
        EXPECT_EQ(kInt64Data, int64_value);

        // Make sure out args are not touched if ReadValue fails
        const wchar_t* kNonExistent = L"NonExistent";
        ASSERT_NE(ERROR_SUCCESS, key.readValue(kNonExistent, &string_value));
        ASSERT_NE(ERROR_SUCCESS, key.readValueDW(kNonExistent, &dword_value));
        ASSERT_NE(ERROR_SUCCESS, key.readInt64(kNonExistent, &int64_value));
        EXPECT_STREQ(kStringData, string_value.c_str());
        EXPECT_EQ(kDWORDData, dword_value);
        EXPECT_EQ(kInt64Data, int64_value);

        // Test delete
        ASSERT_EQ(ERROR_SUCCESS, key.deleteValue(kStringValueName));
        ASSERT_EQ(ERROR_SUCCESS, key.deleteValue(kDWORDValueName));
        ASSERT_EQ(ERROR_SUCCESS, key.deleteValue(kInt64ValueName));
        EXPECT_EQ(0U, key.valueCount());
        EXPECT_FALSE(key.hasValue(kStringValueName));
        EXPECT_FALSE(key.hasValue(kDWORDValueName));
        EXPECT_FALSE(key.hasValue(kInt64ValueName));
    }
}

TEST_F(RegistryTest, BigValueIteratorTest)
{
    RegistryKey key;
    std::wstring foo_key(kRootKey);
    foo_key += L"\\Foo";
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(),
                                      KEY_READ | KEY_SET_VALUE));
    ASSERT_TRUE(key.isValid());

    // Create a test value that is larger than MAX_PATH.
    std::wstring data(MAX_PATH * 2, L'a');

    ASSERT_EQ(ERROR_SUCCESS, key.writeValue(data.c_str(), data.c_str()));

    RegistryValueIterator iterator(HKEY_CURRENT_USER, foo_key.c_str());
    ASSERT_TRUE(iterator.valid());
    EXPECT_STREQ(data.c_str(), iterator.name());
    EXPECT_STREQ(data.c_str(), iterator.value());
    // ValueSize() is in bytes, including NUL.
    EXPECT_EQ((MAX_PATH * 2 + 1) * sizeof(wchar_t), iterator.valueSize());
    ++iterator;
    EXPECT_FALSE(iterator.valid());
}

TEST_F(RegistryTest, TruncatedCharTest)
{
    RegistryKey key;
    std::wstring foo_key(kRootKey);
    foo_key += L"\\Foo";
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(),
                                      KEY_READ | KEY_SET_VALUE));
    ASSERT_TRUE(key.isValid());

    const wchar_t kName[] = L"name";
    // kData size is not a multiple of sizeof(wchar_t).
    const uint8_t kData[] = { 1, 2, 3, 4, 5 };
    EXPECT_EQ(5u, std::size(kData));
    ASSERT_EQ(ERROR_SUCCESS,
        key.writeValue(kName, kData, static_cast<DWORD>(std::size(kData)), REG_BINARY));

    RegistryValueIterator iterator(HKEY_CURRENT_USER, foo_key.c_str());
    ASSERT_TRUE(iterator.valid());
    EXPECT_STREQ(kName, iterator.name());
    // ValueSize() is in bytes.
    ASSERT_EQ(std::size(kData), iterator.valueSize());
    // Value() is NUL terminated.
    int end = (iterator.valueSize() + sizeof(wchar_t) - 1) / sizeof(wchar_t);
    EXPECT_NE(L'\0', iterator.value()[end - 1]);
    EXPECT_EQ(L'\0', iterator.value()[end]);
    EXPECT_EQ(0, std::memcmp(kData, iterator.value(), std::size(kData)));
    ++iterator;
    EXPECT_FALSE(iterator.valid());
}

TEST_F(RegistryTest, RecursiveDelete)
{
    RegistryKey key;
    // Create kRootKey->Foo
    //                  \->Bar (TestValue)
    //                     \->Foo (TestValue)
    //                        \->Bar
    //                           \->Foo
    //                  \->Moo
    //                  \->Foo
    // and delete kRootKey->Foo
    std::wstring foo_key(kRootKey);
    foo_key += L"\\Foo";
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Bar", KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.writeValue(L"TestValue", L"TestData"));
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Moo", KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.create(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Foo", KEY_WRITE));
    foo_key += L"\\Bar";
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    foo_key += L"\\Foo";
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Foo", KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.writeValue(L"TestValue", L"TestData"));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));

    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_WRITE));
    ASSERT_NE(ERROR_SUCCESS, key.deleteKey(L"Bar"));
    ASSERT_NE(ERROR_SUCCESS, key.deleteEmptyKey(L"Foo"));
    ASSERT_NE(ERROR_SUCCESS, key.deleteEmptyKey(L"Foo\\Bar\\Foo"));
    ASSERT_NE(ERROR_SUCCESS, key.deleteEmptyKey(L"Foo\\Bar"));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteEmptyKey(L"Foo\\Foo"));

    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Bar", KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"Foo", KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(L""));
    ASSERT_NE(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));

    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(L"Foo"));
    ASSERT_NE(ERROR_SUCCESS, key.deleteKey(L"Foo"));
    ASSERT_NE(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));
}

// This test requires running as an Administrator as it tests redirected
// registry writes to HKLM\Software
// http://msdn.microsoft.com/en-us/library/windows/desktop/aa384253.aspx
// TODO(wfh): flaky test on Vista.  See http://crbug.com/377917
TEST_F(RegistryTest, DISABLED_Wow64RedirectedFromNative)
{
    if (!isRedirectorPresent())
        return;

    RegistryKey key;

    // Test redirected key access from non-redirected.
    ASSERT_EQ(ERROR_SUCCESS,
              key.create(HKEY_LOCAL_MACHINE,
                         foo_software_key_.c_str(),
                         KEY_WRITE | kRedirectedViewMask));
    ASSERT_NE(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, foo_software_key_.c_str(), KEY_READ));
    ASSERT_NE(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, foo_software_key_.c_str(), KEY_READ | kNativeViewMask));

    // Open the non-redirected view of the parent and try to delete the test key.
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_SET_VALUE));
    ASSERT_NE(ERROR_SUCCESS, key.deleteKey(kRootKey));
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_SET_VALUE | kNativeViewMask));
    ASSERT_NE(ERROR_SUCCESS, key.deleteKey(kRootKey));

    // Open the redirected view and delete the key created above.
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_SET_VALUE | kRedirectedViewMask));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(kRootKey));
}

// Test for the issue found in http://crbug.com/384587 where OpenKey would call
// Close() and reset wow64_access_ flag to 0 and cause a NOTREACHED to hit on a
// subsequent OpenKey call.
TEST_F(RegistryTest, SameWowFlags)
{
    RegistryKey key;

    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_READ | KEY_WOW64_64KEY));
    ASSERT_EQ(ERROR_SUCCESS, key.openKey(L"Microsoft", KEY_READ | KEY_WOW64_64KEY));
    ASSERT_EQ(ERROR_SUCCESS, key.openKey(L"Windows", KEY_READ | KEY_WOW64_64KEY));
}

// TODO(wfh): flaky test on Vista.  See http://crbug.com/377917
TEST_F(RegistryTest, DISABLED_Wow64NativeFromRedirected)
{
    if (!isRedirectorPresent())
        return;

    RegistryKey key;

    // Test non-redirected key access from redirected.
    ASSERT_EQ(ERROR_SUCCESS,
              key.create(HKEY_LOCAL_MACHINE,
                         foo_software_key_.c_str(),
                         KEY_WRITE | kNativeViewMask));
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, foo_software_key_.c_str(), KEY_READ));
    ASSERT_NE(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE,
                       foo_software_key_.c_str(),
                       KEY_READ | kRedirectedViewMask));

    // Open the redirected view of the parent and try to delete the test key
    // from the non-redirected view.
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_SET_VALUE | kRedirectedViewMask));
    ASSERT_NE(ERROR_SUCCESS, key.deleteKey(kRootKey));

    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_LOCAL_MACHINE, L"Software", KEY_SET_VALUE | kNativeViewMask));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(kRootKey));
}

TEST_F(RegistryTest, OpenSubKey)
{
    RegistryKey key;
    ASSERT_EQ(ERROR_SUCCESS,
              key.open(HKEY_CURRENT_USER, kRootKey, KEY_READ | KEY_CREATE_SUB_KEY));

    ASSERT_NE(ERROR_SUCCESS, key.openKey(L"foo", KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.createKey(L"foo", KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.openKey(L"foo", KEY_READ));

    std::wstring foo_key(kRootKey);
    foo_key += L"\\Foo";

    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, foo_key.c_str(), KEY_READ));
    ASSERT_EQ(ERROR_SUCCESS, key.open(HKEY_CURRENT_USER, kRootKey, KEY_WRITE));
    ASSERT_EQ(ERROR_SUCCESS, key.deleteKey(L"foo"));
}

} // namespace

} // namespace base::win
