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

#include "base/smbios_reader.h"

#include "base/logging.h"

#include <Windows.h>

namespace base {

//--------------------------------------------------------------------------------------------------
std::string readSmbiosDump()
{
    UINT buffer_size = GetSystemFirmwareTable('RSMB', 'PCAF', nullptr, 0);
    if (!buffer_size)
    {
        PLOG(LS_ERROR) << "GetSystemFirmwareTable failed";
        return std::string();
    }

    std::string buffer;
    buffer.resize(buffer_size);

    if (!GetSystemFirmwareTable('RSMB', 'PCAF', buffer.data(), buffer_size))
    {
        PLOG(LS_ERROR) << "GetSystemFirmwareTable failed";
        return std::string();
    }

    return buffer;
}

} // namespace base
