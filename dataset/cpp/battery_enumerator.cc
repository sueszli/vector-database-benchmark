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

#include "base/win/battery_enumerator.h"

#include "base/logging.h"
#include "base/strings/string_printf.h"
#include "base/strings/unicode.h"

#include <batclass.h>
#include <winioctl.h>
#include <devguid.h>

namespace base::win {

namespace {

//--------------------------------------------------------------------------------------------------
bool batteryInformation(Device& battery,
                        ULONG tag,
                        BATTERY_QUERY_INFORMATION_LEVEL level,
                        LPVOID buffer,
                        ULONG buffer_size)
{
    BATTERY_QUERY_INFORMATION battery_info;

    memset(&battery_info, 0, sizeof(battery_info));
    battery_info.BatteryTag = tag;
    battery_info.InformationLevel = level;

    ULONG bytes_returned;

    return battery.ioControl(IOCTL_BATTERY_QUERY_INFORMATION,
                             &battery_info, sizeof(battery_info),
                             buffer, buffer_size,
                             &bytes_returned);
}

//--------------------------------------------------------------------------------------------------
bool batteryStatus(Device& battery, ULONG tag, BATTERY_STATUS* status)
{
    BATTERY_WAIT_STATUS status_request;
    memset(&status_request, 0, sizeof(status_request));
    status_request.BatteryTag = tag;

    BATTERY_STATUS status_reply;
    memset(&status_reply, 0, sizeof(status_reply));

    DWORD bytes_returned = 0;

    if (!battery.ioControl(IOCTL_BATTERY_QUERY_STATUS,
                           &status_request, sizeof(status_request),
                           &status_reply, sizeof(status_reply),
                           &bytes_returned))
    {
        return false;
    }

    *status = status_reply;
    return true;
}

} // namespace

//--------------------------------------------------------------------------------------------------
BatteryEnumerator::BatteryEnumerator()
{
    const DWORD flags = DIGCF_PROFILE | DIGCF_PRESENT | DIGCF_DEVICEINTERFACE;
    device_info_.reset(SetupDiGetClassDevsW(&GUID_DEVCLASS_BATTERY, nullptr, nullptr, flags));
    if (!device_info_.isValid())
    {
        PLOG(LS_ERROR) << "SetupDiGetClassDevsW failed";
        return;
    }
}

//--------------------------------------------------------------------------------------------------
BatteryEnumerator::~BatteryEnumerator() = default;

//--------------------------------------------------------------------------------------------------
bool BatteryEnumerator::isAtEnd() const
{
    SP_DEVICE_INTERFACE_DATA device_iface_data;

    memset(&device_iface_data, 0, sizeof(device_iface_data));
    device_iface_data.cbSize = sizeof(device_iface_data);

    if (!SetupDiEnumDeviceInterfaces(device_info_.get(),
                                     nullptr,
                                     &GUID_DEVCLASS_BATTERY,
                                     device_index_,
                                     &device_iface_data))
    {
        DWORD error_code = GetLastError();

        if (error_code != ERROR_NO_MORE_ITEMS)
        {
            DLOG(LS_ERROR) << "SetupDiEnumDeviceInfo failed: "
                           << SystemError(error_code).toString();
        }

        return true;
    }

    DWORD required_size = 0;
    if (SetupDiGetDeviceInterfaceDetailW(device_info_.get(),
                                         &device_iface_data,
                                         nullptr,
                                         0,
                                         &required_size,
                                         nullptr) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        PLOG(LS_ERROR) << "Unexpected return value";
        return true;
    }

    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(required_size);
    PSP_DEVICE_INTERFACE_DETAIL_DATA_W detail_data =
        reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA_W>(buffer.get());

    detail_data->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);

    if (!SetupDiGetDeviceInterfaceDetailW(device_info_.get(),
                                          &device_iface_data,
                                          detail_data,
                                          required_size,
                                          &required_size,
                                          nullptr))
    {
        PLOG(LS_ERROR) << "SetupDiGetDeviceInterfaceDetailW failed";
        return true;
    }

    if (!battery_.open(detail_data->DevicePath))
        return true;

    ULONG bytes_returned;
    ULONG input_buffer = 0;
    ULONG output_buffer = 0;

    if (!battery_.ioControl(IOCTL_BATTERY_QUERY_TAG,
                            &input_buffer, sizeof(input_buffer),
                            &output_buffer, sizeof(output_buffer),
                            &bytes_returned))
    {
        return true;
    }

    battery_tag_ = output_buffer;
    return false;
}

//--------------------------------------------------------------------------------------------------
void BatteryEnumerator::advance()
{
    ++device_index_;
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::deviceName() const
{
    wchar_t buffer[256] = { 0 };

    if (!batteryInformation(battery_, battery_tag_, BatteryDeviceName, buffer, sizeof(buffer)))
        return std::string();

    return utf8FromWide(buffer);
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::manufacturer() const
{
    wchar_t buffer[256] = { 0 };

    if (!batteryInformation(battery_, battery_tag_, BatteryManufactureName, buffer, sizeof(buffer)))
        return std::string();

    return utf8FromWide(buffer);
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::manufactureDate() const
{
    BATTERY_MANUFACTURE_DATE date;

    memset(&date, 0, sizeof(date));

    if (!batteryInformation(battery_, battery_tag_, BatteryManufactureDate, &date, sizeof(date)))
        return std::string();

    return stringPrintf("%u-%u-%u", date.Day, date.Month, date.Year);
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::uniqueId() const
{
    wchar_t buffer[256] = { 0 };

    if (!batteryInformation(battery_, battery_tag_, BatteryUniqueID, buffer, sizeof(buffer)))
        return std::string();

    return utf8FromWide(buffer);
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::serialNumber() const
{
    wchar_t buffer[256] = { 0 };

    if (!batteryInformation(battery_, battery_tag_, BatterySerialNumber, buffer, sizeof(buffer)))
        return std::string();

    return utf8FromWide(buffer);
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::temperature() const
{
    wchar_t buffer[256] = { 0 };

    if (!batteryInformation(battery_, battery_tag_, BatteryTemperature, buffer, sizeof(buffer)))
        return std::string();

    return utf8FromWide(buffer);
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::designCapacity() const
{
    BATTERY_INFORMATION battery_info;

    memset(&battery_info, 0, sizeof(battery_info));

    if (!batteryInformation(battery_, battery_tag_, BatteryInformation,
                            &battery_info, sizeof(battery_info)))
        return 0;

    return battery_info.DesignedCapacity;
}

//--------------------------------------------------------------------------------------------------
std::string BatteryEnumerator::type() const
{
    BATTERY_INFORMATION battery_info;
    memset(&battery_info, 0, sizeof(battery_info));

    if (batteryInformation(battery_, battery_tag_, BatteryInformation,
                           &battery_info, sizeof(battery_info)))
    {
        if (memcmp(&battery_info.Chemistry[0], "PbAc", 4) == 0)
            return "Lead Acid";

        if (memcmp(&battery_info.Chemistry[0], "LION", 4) == 0 ||
            memcmp(&battery_info.Chemistry[0], "Li-I", 4) == 0)
            return "Lithium Ion";

        if (memcmp(&battery_info.Chemistry[0], "NiCd", 4) == 0)
            return "Nickel Cadmium";

        if (memcmp(&battery_info.Chemistry[0], "NiMH", 4) == 0)
            return "Nickel Metal Hydride";

        if (memcmp(&battery_info.Chemistry[0], "NiZn", 4) == 0)
            return "Nickel Zinc";

        if (memcmp(&battery_info.Chemistry[0], "RAM", 3) == 0)
            return "Rechargeable Alkaline-Manganese";
    }

    return std::string();
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::fullChargedCapacity() const
{
    BATTERY_INFORMATION battery_info;
    memset(&battery_info, 0, sizeof(battery_info));

    if (!batteryInformation(battery_, battery_tag_, BatteryInformation,
                            &battery_info, sizeof(battery_info)))
        return 0;

    return battery_info.FullChargedCapacity;
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::depreciation() const
{
    BATTERY_INFORMATION battery_info;
    memset(&battery_info, 0, sizeof(battery_info));

    if (!batteryInformation(battery_, battery_tag_, BatteryInformation,
                            &battery_info, sizeof(battery_info)) ||
        battery_info.DesignedCapacity == 0)
    {
        return 0;
    }

    const int percent = 100 - (static_cast<int>(battery_info.FullChargedCapacity) * 100) /
        static_cast<int>(battery_info.DesignedCapacity);

    return (percent > 0) ? static_cast<uint32_t>(percent) : 0;
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::currentCapacity() const
{
    BATTERY_STATUS battery_status;
    if (!batteryStatus(battery_, battery_tag_, &battery_status))
        return 0;

    return battery_status.Capacity;
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::voltage() const
{
    BATTERY_STATUS battery_status;
    if (!batteryStatus(battery_, battery_tag_, &battery_status))
        return 0;

    return battery_status.Voltage;
}

//--------------------------------------------------------------------------------------------------
uint32_t BatteryEnumerator::state() const
{
    BATTERY_STATUS battery_status;
    if (!batteryStatus(battery_, battery_tag_, &battery_status))
        return 0;

    uint32_t result = 0;

    if (battery_status.PowerState & BATTERY_CHARGING)
        result |= CHARGING;

    if (battery_status.PowerState & BATTERY_CRITICAL)
        result |= CRITICAL;

    if (battery_status.PowerState & BATTERY_DISCHARGING)
        result |= DISCHARGING;

    if (battery_status.PowerState & BATTERY_POWER_ON_LINE)
        result |= POWER_ONLINE;

    return result;
}

} // namespace base::win
