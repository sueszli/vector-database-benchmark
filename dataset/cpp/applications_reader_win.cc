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

#include "base/applications_reader.h"

#include "base/logging.h"
#include "base/strings/string_printf.h"
#include "base/strings/unicode.h"
#include "base/win/registry.h"

namespace base {

namespace {

const wchar_t kUninstallKeyPath[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall";
const wchar_t kDisplayName[] = L"DisplayName";
const wchar_t kDisplayVersion[] = L"DisplayVersion";
const wchar_t kPublisher[] = L"Publisher";
const wchar_t kInstallDate[] = L"InstallDate";
const wchar_t kInstallLocation[] = L"InstallLocation";
const wchar_t kSystemComponent[] = L"SystemComponent";
const wchar_t kParentKeyName[] = L"ParentKeyName";

//--------------------------------------------------------------------------------------------------
bool addApplication(
    proto::system_info::Applications* message, const wchar_t* key_name, REGSAM access)
{
    std::wstring key_path = stringPrintf(L"%s\\%s", kUninstallKeyPath, key_name);

    win::RegistryKey key;

    LONG status = key.open(HKEY_LOCAL_MACHINE, key_path.c_str(), access | KEY_READ);
    if (status != ERROR_SUCCESS)
    {
        LOG(LS_ERROR) << "Unable to open registry key: "
                      << SystemError(static_cast<DWORD>(status)).toString();
        return false;
    }

    DWORD system_component = 0;

    status = key.readValueDW(kSystemComponent, &system_component);
    if (status == ERROR_SUCCESS && system_component == 1)
        return false;

    std::wstring value;

    status = key.readValue(kParentKeyName, &value);
    if (status == ERROR_SUCCESS)
        return false;

    status = key.readValue(kDisplayName, &value);
    if (status != ERROR_SUCCESS)
        return false;

    proto::system_info::Applications::Application* item = message->add_application();

    item->set_name(utf8FromWide(value));

    status = key.readValue(kDisplayVersion, &value);
    if (status == ERROR_SUCCESS)
        item->set_version(utf8FromWide(value));

    status = key.readValue(kPublisher, &value);
    if (status == ERROR_SUCCESS)
        item->set_publisher(utf8FromWide(value));

    status = key.readValue(kInstallDate, &value);
    if (status == ERROR_SUCCESS)
        item->set_install_date(utf8FromWide(value));

    status = key.readValue(kInstallLocation, &value);
    if (status == ERROR_SUCCESS)
        item->set_install_location(utf8FromWide(value));

    return true;
}

} // namespace

//--------------------------------------------------------------------------------------------------
void readApplicationsInformation(proto::system_info::Applications* applications)
{
    win::RegistryKeyIterator machine_key_iterator(HKEY_LOCAL_MACHINE, kUninstallKeyPath);

    while (machine_key_iterator.valid())
    {
        addApplication(applications, machine_key_iterator.name(), 0);
        ++machine_key_iterator;
    }

    win::RegistryKeyIterator user_key_iterator(HKEY_CURRENT_USER, kUninstallKeyPath);

    while (user_key_iterator.valid())
    {
        addApplication(applications, user_key_iterator.name(), 0);
        ++user_key_iterator;
    }

#if (ARCH_CPU_X86 == 1)

    BOOL is_wow64;

    // If the x86 application is running in a x64 system.
    if (IsWow64Process(GetCurrentProcess(), &is_wow64) && is_wow64)
    {
        win::RegistryKeyIterator machine64_key_iterator(HKEY_LOCAL_MACHINE,
                                                        kUninstallKeyPath,
                                                        KEY_WOW64_64KEY);

        while (machine64_key_iterator.valid())
        {
            addApplication(applications, machine64_key_iterator.name(), KEY_WOW64_64KEY);
            ++machine64_key_iterator;
        }

        win::RegistryKeyIterator user64_key_iterator(HKEY_CURRENT_USER,
                                                     kUninstallKeyPath,
                                                     KEY_WOW64_64KEY);

        while (user64_key_iterator.valid())
        {
            addApplication(applications, user64_key_iterator.name(), KEY_WOW64_64KEY);
            ++user64_key_iterator;
        }
    }

#elif (ARCH_CPU_X86_64 == 1)

    win::RegistryKeyIterator machine32_key_iterator(HKEY_LOCAL_MACHINE,
                                                    kUninstallKeyPath,
                                                    KEY_WOW64_32KEY);

    while (machine32_key_iterator.valid())
    {
        addApplication(applications, machine32_key_iterator.name(), KEY_WOW64_32KEY);
        ++machine32_key_iterator;
    }

    win::RegistryKeyIterator user32_key_iterator(HKEY_CURRENT_USER,
                                                 kUninstallKeyPath,
                                                 KEY_WOW64_32KEY);

    while (user32_key_iterator.valid())
    {
        addApplication(applications, user32_key_iterator.name(), KEY_WOW64_32KEY);
        ++user32_key_iterator;
    }

#else
#error Unknown Architecture
#endif
}

} // namespace base
