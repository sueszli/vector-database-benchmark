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

#include "base/win/monitor_enumerator.h"

#include "base/logging.h"
#include "base/strings/string_printf.h"
#include "base/strings/unicode.h"
#include "base/win/registry.h"

#include <devguid.h>

namespace base::win {

//--------------------------------------------------------------------------------------------------
MonitorEnumerator::MonitorEnumerator()
    : DeviceEnumerator(&GUID_DEVCLASS_MONITOR, DIGCF_PROFILE | DIGCF_PRESENT)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<Edid> MonitorEnumerator::edid() const
{
    std::wstring key_path =
        stringPrintf(L"SYSTEM\\CurrentControlSet\\Enum\\%s\\Device Parameters",
                     wideFromUtf8(deviceID()).c_str());

    RegistryKey key;
    LONG status = key.open(HKEY_LOCAL_MACHINE, key_path.c_str(), KEY_READ);
    if (status != ERROR_SUCCESS)
    {
        DLOG(LS_ERROR) << "Unable to open registry key: "
                       << SystemError(static_cast<DWORD>(status)).toString();
        return nullptr;
    }

    DWORD type;
    DWORD size = 128;
    std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(size);

    status = key.readValue(L"EDID", data.get(), &size, &type);
    if (status != ERROR_SUCCESS)
    {
        if (status == ERROR_MORE_DATA)
        {
            data = std::make_unique<uint8_t[]>(size);
            status = key.readValue(L"EDID", data.get(), &size, &type);
        }

        if (status != ERROR_SUCCESS)
        {
            DLOG(LS_ERROR) << "Unable to read EDID data from registry: "
                           << SystemError(static_cast<DWORD>(status)).toString();
            return nullptr;
        }
    }

    if (type != REG_BINARY)
    {
        DLOG(LS_ERROR) << "Unexpected data type: " << type;
        return nullptr;
    }

    return Edid::create(std::move(data), size);
}

} // namespace base::win
