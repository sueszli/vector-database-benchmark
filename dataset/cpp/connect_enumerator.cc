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

#include "base/net/connect_enumerator.h"

#include "base/endian_util.h"
#include "base/logging.h"
#include "base/strings/string_printf.h"
#include "base/strings/unicode.h"

#include <iphlpapi.h>
#include <TlHelp32.h>
#include <WinSock2.h>

namespace base {

namespace {

//--------------------------------------------------------------------------------------------------
std::unique_ptr<uint8_t[]> initializeTcpTable()
{
    ULONG table_buffer_size = sizeof(MIB_TCPTABLE);

    std::unique_ptr<uint8_t[]> table_buffer = std::make_unique<uint8_t[]>(table_buffer_size);

    DWORD ret = GetExtendedTcpTable(reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer.get()),
                                    &table_buffer_size, TRUE, AF_INET, TCP_TABLE_OWNER_PID_ALL, 0);

    if (ret == ERROR_INSUFFICIENT_BUFFER)
    {
        table_buffer = std::make_unique<uint8_t[]>(table_buffer_size);

        ret = GetExtendedTcpTable(reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer.get()),
                                  &table_buffer_size, TRUE, AF_INET, TCP_TABLE_OWNER_PID_ALL, 0);
    }

    if (ret != NO_ERROR)
        return nullptr;

    return table_buffer;
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<uint8_t[]> initializeUdpTable()
{
    ULONG table_buffer_size = sizeof(MIB_UDPTABLE);

    std::unique_ptr<uint8_t[]> table_buffer = std::make_unique<uint8_t[]>(table_buffer_size);

    DWORD ret = GetExtendedUdpTable(reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer.get()),
                                    &table_buffer_size, TRUE, AF_INET, UDP_TABLE_OWNER_PID, 0);

    if (ret == ERROR_INSUFFICIENT_BUFFER)
    {
        table_buffer = std::make_unique<uint8_t[]>(table_buffer_size);

        ret = GetExtendedUdpTable(reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer.get()),
                                  &table_buffer_size, TRUE, AF_INET, UDP_TABLE_OWNER_PID, 0);
    }

    if (ret != NO_ERROR)
        return nullptr;

    return table_buffer;
}

//--------------------------------------------------------------------------------------------------
std::string processNameByPid(HANDLE process_snapshot, DWORD process_id)
{
    PROCESSENTRY32W process_entry;
    process_entry.dwSize = sizeof(process_entry);

    if (Process32FirstW(process_snapshot, &process_entry))
    {
        do
        {
            if (process_entry.th32ProcessID == process_id)
                return utf8FromWide(process_entry.szExeFile);
        } while (Process32NextW(process_snapshot, &process_entry));
    }

    return std::string();
}

//--------------------------------------------------------------------------------------------------
std::string addressToString(uint32_t address)
{
    address = base::EndianUtil::byteSwap(address);

    return stringPrintf("%u.%u.%u.%u",
                        (address >> 24) & 0xFF,
                        (address >> 16) & 0xFF,
                        (address >> 8)  & 0xFF,
                        (address)       & 0xFF);
}

//--------------------------------------------------------------------------------------------------
std::string stateToString(DWORD state)
{
    switch (state)
    {
        case 0:  return "UNKNOWN";
        case 1:  return "CLOSED";
        case 2:  return "LISTENING";
        case 3:  return "SYN_SENT";
        case 4:  return "SYN_RCVD";
        case 5:  return "ESTABLISHED";
        case 6:  return "FIN_WAIT1";
        case 7:  return "FIN_WAIT2";
        case 8:  return "CLOSE_WAIT";
        case 9:  return "CLOSING";
        case 10: return "LAST_ACK";
        case 11: return "TIME_WAIT";
        case 12: return "DELETE_TCB";
        default: return "UNKNOWN";
    }
}

} // namespace

//--------------------------------------------------------------------------------------------------
ConnectEnumerator::ConnectEnumerator(Mode mode)
    : mode_(mode),
      snapshot_(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0))
{
    if (mode_ == Mode::TCP)
    {
        table_buffer_ = initializeTcpTable();

        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (tcp_table)
            num_entries_ = tcp_table->dwNumEntries;
    }
    else
    {
        DCHECK_EQ(mode_, Mode::UDP);
        table_buffer_ = initializeUdpTable();

        PMIB_UDPTABLE_OWNER_PID udp_table =
            reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer_.get());
        if (udp_table)
            num_entries_ = udp_table->dwNumEntries;
    }
}

//--------------------------------------------------------------------------------------------------
ConnectEnumerator::~ConnectEnumerator() = default;

//--------------------------------------------------------------------------------------------------
bool ConnectEnumerator::isAtEnd() const
{
    return pos_ >= num_entries_;
}

//--------------------------------------------------------------------------------------------------
void ConnectEnumerator::advance()
{
    ++pos_;
}

//--------------------------------------------------------------------------------------------------
std::string ConnectEnumerator::protocol() const
{
    return (mode_ == Mode::TCP) ? "TCP" : "UDP";
}

//--------------------------------------------------------------------------------------------------
std::string ConnectEnumerator::processName() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return std::string();

        return processNameByPid(snapshot_.get(), tcp_table->table[pos_].dwOwningPid);
    }
    else
    {
        PMIB_UDPTABLE_OWNER_PID udp_table =
            reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer_.get());
        if (!udp_table)
            return std::string();

        return processNameByPid(snapshot_.get(), udp_table->table[pos_].dwOwningPid);
    }
}

//--------------------------------------------------------------------------------------------------
std::string ConnectEnumerator::localAddress() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return std::string();

        return addressToString(tcp_table->table[pos_].dwLocalAddr);
    }
    else
    {
        PMIB_UDPTABLE_OWNER_PID udp_table =
            reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer_.get());
        if (!udp_table)
            return std::string();

        return addressToString(udp_table->table[pos_].dwLocalAddr);
    }
}

//--------------------------------------------------------------------------------------------------
std::string ConnectEnumerator::remoteAddress() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return std::string();

        return addressToString(tcp_table->table[pos_].dwRemoteAddr);
    }
    else
    {
        return std::string();
    }
}

//--------------------------------------------------------------------------------------------------
uint16_t ConnectEnumerator::localPort() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return 0;

        return static_cast<uint16_t>(tcp_table->table[pos_].dwLocalPort);
    }
    else
    {
        PMIB_UDPTABLE_OWNER_PID udp_table =
            reinterpret_cast<PMIB_UDPTABLE_OWNER_PID>(table_buffer_.get());
        if (!udp_table)
            return 0;

        return static_cast<uint16_t>(udp_table->table[pos_].dwLocalPort);
    }
}

//--------------------------------------------------------------------------------------------------
uint16_t ConnectEnumerator::remotePort() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return 0;

        return static_cast<uint16_t>(tcp_table->table[pos_].dwRemotePort);
    }
    else
    {
        return 0;
    }
}

//--------------------------------------------------------------------------------------------------
std::string ConnectEnumerator::state() const
{
    if (mode_ == Mode::TCP)
    {
        PMIB_TCPTABLE_OWNER_PID tcp_table =
            reinterpret_cast<PMIB_TCPTABLE_OWNER_PID>(table_buffer_.get());
        if (!tcp_table)
            return std::string();

        return stateToString(tcp_table->table[pos_].dwState);
    }
    else
    {
        return std::string();
    }
}

} // namespace base
