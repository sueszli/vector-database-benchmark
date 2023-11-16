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

#include "base/net/route_enumerator.h"

#include <Windows.h>
#include <iphlpapi.h>
#include <WS2tcpip.h>
#include <WinSock2.h>

namespace base {

namespace {

//--------------------------------------------------------------------------------------------------
std::string ipToString(DWORD ip)
{
    char buffer[46 + 1];

    if (!inet_ntop(AF_INET, &ip, buffer, _countof(buffer)))
        return std::string();

    return buffer;
}

} // namespace

//--------------------------------------------------------------------------------------------------
RouteEnumerator::RouteEnumerator()
{
    ULONG buffer_size = sizeof(MIB_IPFORWARDTABLE);

    std::unique_ptr<uint8_t[]> forward_table_buffer =
        std::make_unique<uint8_t[]>(buffer_size);

    PMIB_IPFORWARDTABLE forward_table =
        reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer.get());

    DWORD error_code = GetIpForwardTable(forward_table, &buffer_size, 0);
    if (error_code != NO_ERROR)
    {
        if (error_code == ERROR_INSUFFICIENT_BUFFER)
        {
            forward_table_buffer = std::make_unique<uint8_t[]>(buffer_size);
            forward_table = reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer.get());

            error_code = GetIpForwardTable(forward_table, &buffer_size, 0);
        }
    }

    if (error_code != NO_ERROR)
        return;

    forward_table_buffer_ = std::move(forward_table_buffer);
    num_entries_ = forward_table->dwNumEntries;
}

//--------------------------------------------------------------------------------------------------
RouteEnumerator::~RouteEnumerator() = default;

//--------------------------------------------------------------------------------------------------
bool RouteEnumerator::isAtEnd() const
{
    return pos_ >= num_entries_;
}

//--------------------------------------------------------------------------------------------------
void RouteEnumerator::advance()
{
    ++pos_;
}

//--------------------------------------------------------------------------------------------------
std::string RouteEnumerator::destonation() const
{
    PMIB_IPFORWARDTABLE forward_table =
        reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer_.get());
    if (!forward_table)
        return std::string();

    return ipToString(forward_table->table[pos_].dwForwardDest);
}

//--------------------------------------------------------------------------------------------------
std::string RouteEnumerator::mask() const
{
    PMIB_IPFORWARDTABLE forward_table =
        reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer_.get());
    if (!forward_table)
        return std::string();

    return ipToString(forward_table->table[pos_].dwForwardMask);
}

//--------------------------------------------------------------------------------------------------
std::string RouteEnumerator::gateway() const
{
    PMIB_IPFORWARDTABLE forward_table =
        reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer_.get());
    if (!forward_table)
        return std::string();

    return ipToString(forward_table->table[pos_].dwForwardNextHop);
}

//--------------------------------------------------------------------------------------------------
uint32_t RouteEnumerator::metric() const
{
    PMIB_IPFORWARDTABLE forward_table =
        reinterpret_cast<PMIB_IPFORWARDTABLE>(forward_table_buffer_.get());
    if (!forward_table)
        return 0;

    return forward_table->table[pos_].dwForwardMetric1;
}

} // namespace base
