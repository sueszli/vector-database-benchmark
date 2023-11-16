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

#include "base/win/window_station.h"

#include "base/logging.h"

namespace base {

//--------------------------------------------------------------------------------------------------
WindowStation::WindowStation() = default;

//--------------------------------------------------------------------------------------------------
WindowStation::WindowStation(HWINSTA winsta, bool own)
    : winsta_(winsta),
      own_(own)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
WindowStation::WindowStation(WindowStation&& other) noexcept
{
    winsta_ = other.winsta_;
    own_ = other.own_;

    other.winsta_ = nullptr;
    other.own_ = false;
}

//--------------------------------------------------------------------------------------------------
WindowStation::~WindowStation()
{
    close();
}

//--------------------------------------------------------------------------------------------------
// static
WindowStation WindowStation::open(const wchar_t* name)
{
    HWINSTA winsta = OpenWindowStationW(name, TRUE, GENERIC_ALL);
    if (!winsta)
    {
        PLOG(LS_ERROR) << "OpenWindowStationW failed";
        return WindowStation();
    }

    return WindowStation(winsta, true);
}

//--------------------------------------------------------------------------------------------------
// static
WindowStation WindowStation::forCurrentProcess()
{
    HWINSTA winsta = GetProcessWindowStation();
    if (!winsta)
    {
        PLOG(LS_ERROR) << "OpenWindowStationW failed";
        return WindowStation();
    }

    return WindowStation(winsta, false);
}

//--------------------------------------------------------------------------------------------------
// static
std::vector<std::wstring> WindowStation::windowStationList()
{
    std::vector<std::wstring> list;

    if (!EnumWindowStationsW(enumWindowStationProc, reinterpret_cast<LPARAM>(&list)))
    {
        PLOG(LS_ERROR) << "EnumWindowStationsW failed";
        return {};
    }

    return list;
}

//--------------------------------------------------------------------------------------------------
bool WindowStation::isValid() const
{
    return winsta_ != nullptr;
}

//--------------------------------------------------------------------------------------------------
HWINSTA WindowStation::get() const
{
    return winsta_;
}

//--------------------------------------------------------------------------------------------------
bool WindowStation::setProcessWindowStation()
{
    if (!SetProcessWindowStation(winsta_))
    {
        PLOG(LS_ERROR) << "SetProcessWindowStation failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
std::wstring WindowStation::name()
{
    if (!winsta_)
        return std::wstring();

    wchar_t buffer[128] = { 0 };

    if (!GetUserObjectInformationW(winsta_, UOI_NAME, buffer, sizeof(buffer), nullptr))
    {
        PLOG(LS_ERROR) << "GetUserObjectInformationW failed";
        return std::wstring();
    }

    return buffer;
}

//--------------------------------------------------------------------------------------------------
void WindowStation::close()
{
    if (winsta_ && own_)
    {
        if (!CloseWindowStation(winsta_))
        {
            PLOG(LS_ERROR) << "CloseWindowStation failed";
        }
    }

    winsta_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
WindowStation& WindowStation::operator=(WindowStation&& other) noexcept
{
    close();

    winsta_ = other.winsta_;
    own_ = other.own_;

    other.winsta_ = nullptr;
    other.own_ = false;

    return *this;
}

//--------------------------------------------------------------------------------------------------
// static
BOOL WindowStation::enumWindowStationProc(LPWSTR window_station, LPARAM lparam)
{
    std::vector<std::wstring>* list = reinterpret_cast<std::vector<std::wstring>*>(lparam);
    if (!list)
    {
        LOG(LS_ERROR) << "Invalid window station list pointer";
        return FALSE;
    }

    if (!window_station)
    {
        LOG(LS_ERROR) << "Invalid window station name";;
        return FALSE;
    }

    list->emplace_back(window_station);
    return TRUE;
}


} // namespace base
