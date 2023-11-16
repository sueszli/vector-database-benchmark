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

#include "base/win/service_controller.h"

#include "base/logging.h"
#include "base/strings/string_util.h"

#include <memory>

namespace base::win {

//--------------------------------------------------------------------------------------------------
ServiceController::ServiceController() = default;

//--------------------------------------------------------------------------------------------------
ServiceController::ServiceController(SC_HANDLE sc_manager, SC_HANDLE service)
    : sc_manager_(sc_manager),
      service_(service)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
ServiceController::ServiceController(ServiceController&& other) noexcept
    : sc_manager_(std::move(other.sc_manager_)),
      service_(std::move(other.service_))
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
ServiceController& ServiceController::operator=(ServiceController&& other) noexcept
{
    sc_manager_ = std::move(other.sc_manager_);
    service_ = std::move(other.service_);
    return *this;
}

//--------------------------------------------------------------------------------------------------
ServiceController::~ServiceController() = default;

//--------------------------------------------------------------------------------------------------
// static
ServiceController ServiceController::open(std::u16string_view name)
{
    win::ScopedScHandle sc_manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS));
    if (!sc_manager.isValid())
    {
        PLOG(LS_ERROR) << "OpenSCManagerW failed";
        return ServiceController();
    }

    win::ScopedScHandle service(OpenServiceW(sc_manager, asWide(name), SERVICE_ALL_ACCESS));
    if (!service.isValid())
    {
        PLOG(LS_ERROR) << "OpenServiceW failed";
        return ServiceController();
    }

    return ServiceController(sc_manager.release(), service.release());
}

//--------------------------------------------------------------------------------------------------
// static
ServiceController ServiceController::install(std::u16string_view name,
                                             std::u16string_view display_name,
                                             const std::filesystem::path& file_path)
{
    win::ScopedScHandle sc_manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS));
    if (!sc_manager.isValid())
    {
        PLOG(LS_ERROR) << "OpenSCManagerW failed";
        return ServiceController();
    }

    win::ScopedScHandle service(CreateServiceW(sc_manager,
                                               asWide(name),
                                               asWide(display_name),
                                               SERVICE_ALL_ACCESS,
                                               SERVICE_WIN32_OWN_PROCESS,
                                               SERVICE_AUTO_START,
                                               SERVICE_ERROR_NORMAL,
                                               file_path.c_str(),
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr));
    if (!service.isValid())
    {
        PLOG(LS_ERROR) << "CreateServiceW failed";
        return ServiceController();
    }

    SC_ACTION action;
    action.Type = SC_ACTION_RESTART;
    action.Delay = 60000; // 60 seconds

    SERVICE_FAILURE_ACTIONS actions;
    actions.dwResetPeriod = 0;
    actions.lpRebootMsg   = nullptr;
    actions.lpCommand     = nullptr;
    actions.cActions      = 1;
    actions.lpsaActions   = &action;

    if (!ChangeServiceConfig2W(service, SERVICE_CONFIG_FAILURE_ACTIONS, &actions))
    {
        PLOG(LS_ERROR) << "ChangeServiceConfig2W failed";
        return ServiceController();
    }

    return ServiceController(sc_manager.release(), service.release());
}

//--------------------------------------------------------------------------------------------------
// static
bool ServiceController::remove(std::u16string_view name)
{
    win::ScopedScHandle sc_manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS));
    if (!sc_manager.isValid())
    {
        PLOG(LS_ERROR) << "OpenSCManagerW failed";
        return false;
    }

    win::ScopedScHandle service(OpenServiceW(sc_manager, asWide(name), SERVICE_ALL_ACCESS));
    if (!service.isValid())
    {
        PLOG(LS_ERROR) << "OpenServiceW failed";
        return false;
    }

    if (!DeleteService(service))
    {
        PLOG(LS_ERROR) << "DeleteService failed";
        return false;
    }

    service.reset();
    sc_manager.reset();

    static const int kMaxAttempts = 15;
    static const int kAttemptInterval = 100; // ms

    for (int i = 0; i < kMaxAttempts; ++i)
    {
        if (!isInstalled(name))
            return true;

        Sleep(kAttemptInterval);
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
// static
bool ServiceController::isInstalled(std::u16string_view name)
{
    win::ScopedScHandle sc_manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT));
    if (!sc_manager.isValid())
    {
        PLOG(LS_ERROR) << "OpenSCManagerW failed";
        return false;
    }

    win::ScopedScHandle service(OpenServiceW(sc_manager, asWide(name), SERVICE_QUERY_CONFIG));
    if (!service.isValid())
    {
        if (GetLastError() != ERROR_SERVICE_DOES_NOT_EXIST)
        {
            PLOG(LS_ERROR) << "OpenServiceW failed";
        }

        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
// static
bool ServiceController::isRunning(std::u16string_view name)
{
    win::ScopedScHandle sc_manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT));
    if (!sc_manager.isValid())
    {
        PLOG(LS_ERROR) << "OpenSCManagerW failed";
        return false;
    }

    win::ScopedScHandle service(OpenServiceW(sc_manager, asWide(name), SERVICE_QUERY_STATUS));
    if (!service.isValid())
    {
        PLOG(LS_ERROR) << "OpenServiceW failed";
        return false;
    }

    SERVICE_STATUS status;

    if (!QueryServiceStatus(service, &status))
    {
        PLOG(LS_ERROR) << "QueryServiceStatus failed";
        return false;
    }

    return status.dwCurrentState != SERVICE_STOPPED;
}

//--------------------------------------------------------------------------------------------------
void ServiceController::close()
{
    service_.reset();
    sc_manager_.reset();
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::setDescription(std::u16string_view description)
{
    SERVICE_DESCRIPTIONW service_description;
    service_description.lpDescription = const_cast<LPWSTR>(asWide(description));

    // Set the service description.
    if (!ChangeServiceConfig2W(service_, SERVICE_CONFIG_DESCRIPTION, &service_description))
    {
        PLOG(LS_ERROR) << "ChangeServiceConfig2W failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
std::u16string ServiceController::description() const
{
    DWORD bytes_needed = 0;

    if (QueryServiceConfig2W(service_, SERVICE_CONFIG_DESCRIPTION, nullptr, 0, &bytes_needed) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        LOG(LS_FATAL) << "QueryServiceConfig2W: unexpected result";
        return std::u16string();
    }

    if (!bytes_needed)
        return std::u16string();

    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(bytes_needed);

    if (!QueryServiceConfig2W(service_, SERVICE_CONFIG_DESCRIPTION, buffer.get(), bytes_needed,
                             &bytes_needed))
    {
        PLOG(LS_ERROR) << "QueryServiceConfig2W failed";
        return std::u16string();
    }

    SERVICE_DESCRIPTION* service_description =
        reinterpret_cast<SERVICE_DESCRIPTION*>(buffer.get());
    if (!service_description->lpDescription)
        return std::u16string();

    return asUtf16(service_description->lpDescription);
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::setDependencies(const std::vector<std::u16string>& dependencies)
{
    std::string buffer;

    for (auto it = dependencies.cbegin(); it != dependencies.cend(); ++it)
    {
        const std::u16string& str = *it;

        buffer.append(reinterpret_cast<const char*>(str.c_str()),
                      (str.length() + 1) * sizeof(wchar_t));
    }

    buffer.push_back(static_cast<char>(0));
    buffer.push_back(static_cast<char>(0));

    if (!ChangeServiceConfigW(service_,
                              SERVICE_NO_CHANGE,
                              SERVICE_NO_CHANGE,
                              SERVICE_NO_CHANGE,
                              nullptr, nullptr, nullptr,
                              reinterpret_cast<const wchar_t*>(buffer.data()),
                              nullptr, nullptr, nullptr))
    {
        PLOG(LS_ERROR) << "ChangeServiceConfigW failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
std::vector<std::u16string> ServiceController::dependencies() const
{
    DWORD bytes_needed = 0;

    if (QueryServiceConfigW(service_, nullptr, 0, &bytes_needed) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        LOG(LS_FATAL) << "QueryServiceConfigW: unexpected result";
        return std::vector<std::u16string>();
    }

    if (!bytes_needed)
        return std::vector<std::u16string>();

    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(bytes_needed);
    QUERY_SERVICE_CONFIGW* service_config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(buffer.get());

    if (!QueryServiceConfigW(service_, service_config, bytes_needed, &bytes_needed))
    {
        PLOG(LS_ERROR) << "QueryServiceConfigW failed";
        return std::vector<std::u16string>();
    }

    if (!service_config->lpDependencies)
        return std::vector<std::u16string>();

    std::vector<std::u16string> list;
    size_t len = 0;

    for (;;)
    {
        std::u16string str = asUtf16(service_config->lpDependencies + len);

        len += str.length() + 1;

        if (str.empty())
            break;

        list.emplace_back(std::move(str));
    }

    return list;
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::setAccount(std::u16string_view username, std::u16string_view password)
{
    if (!ChangeServiceConfigW(service_,
                              SERVICE_NO_CHANGE,
                              SERVICE_NO_CHANGE,
                              SERVICE_NO_CHANGE,
                              nullptr, nullptr, nullptr, nullptr,
                              asWide(username), asWide(password),
                              nullptr))
    {
        PLOG(LS_ERROR) << "ChangeServiceConfigW failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
std::filesystem::path ServiceController::filePath() const
{
    DWORD bytes_needed = 0;

    if (QueryServiceConfigW(service_, nullptr, 0, &bytes_needed) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        LOG(LS_FATAL) << "QueryServiceConfigW: unexpected result";
        return std::u16string();
    }

    if (!bytes_needed)
        return std::u16string();

    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(bytes_needed);
    QUERY_SERVICE_CONFIGW* service_config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(buffer.get());

    if (!QueryServiceConfigW(service_, service_config, bytes_needed, &bytes_needed))
    {
        PLOG(LS_ERROR) << "QueryServiceConfigW failed";
        return std::u16string();
    }

    if (!service_config->lpBinaryPathName)
        return std::u16string();

    return service_config->lpBinaryPathName;
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::isValid() const
{
    return sc_manager_.isValid() && service_.isValid();
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::isRunning() const
{
    SERVICE_STATUS status;

    if (!QueryServiceStatus(service_, &status))
    {
        PLOG(LS_ERROR) << "QueryServiceStatus failed";
        return false;
    }

    return status.dwCurrentState != SERVICE_STOPPED;
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::start()
{
    if (!StartServiceW(service_, 0, nullptr))
    {
        PLOG(LS_ERROR) << "StartServiceW failed";
        return false;
    }

    SERVICE_STATUS status;
    if (QueryServiceStatus(service_, &status))
    {
        bool is_started = status.dwCurrentState == SERVICE_RUNNING;
        int number_of_attempts = 0;

        while (!is_started && number_of_attempts < 15)
        {
            Sleep(250);

            if (!QueryServiceStatus(service_, &status))
                break;

            is_started = status.dwCurrentState == SERVICE_RUNNING;
            ++number_of_attempts;
        }
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool ServiceController::stop()
{
    SERVICE_STATUS status;

    if (!ControlService(service_, SERVICE_CONTROL_STOP, &status))
    {
        PLOG(LS_ERROR) << "ControlService failed";
        return false;
    }

    bool is_stopped = status.dwCurrentState == SERVICE_STOPPED;
    int number_of_attempts = 0;

    while (!is_stopped && number_of_attempts < 15)
    {
        Sleep(250);

        if (!QueryServiceStatus(service_, &status))
            break;

        is_stopped = status.dwCurrentState == SERVICE_STOPPED;
        ++number_of_attempts;
    }

    return is_stopped;
}

} // namespace base::win
