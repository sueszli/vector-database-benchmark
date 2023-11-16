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

#include "host/desktop_agent_main.h"

#include "build/build_config.h"
#include "base/command_line.h"
#include "base/environment.h"
#include "base/scoped_logging.h"
#include "base/sys_info.h"
#include "base/message_loop/message_loop.h"
#include "build/version.h"
#include "host/desktop_session_agent.h"

#if defined(OS_WIN)
#include "base/win/mini_dump_writer.h"
#include "base/win/session_info.h"
#include "base/win/window_station.h"

#include <dxgi.h>
#include <d3d11.h>
#include <comdef.h>
#include <wrl/client.h>
#endif // defined(OS_WIN)

//--------------------------------------------------------------------------------------------------
void desktopAgentMain(int argc, const char* const* argv)
{
#if defined(OS_WIN)
    base::installFailureHandler(L"aspia_desktop_agent");
#endif // defined(OS_WIN)

    base::LoggingSettings logging_settings;
    logging_settings.min_log_level = base::LOG_LS_INFO;

    base::ScopedLogging scoped_logging(logging_settings);

    base::CommandLine::init(argc, argv);
    base::CommandLine* command_line = base::CommandLine::forCurrentProcess();

    LOG(LS_INFO) << "Version: " << ASPIA_VERSION_STRING << " (arch: " << ARCH_CPU_STRING << ")";
#if defined(GIT_CURRENT_BRANCH) && defined(GIT_COMMIT_HASH)
    LOG(LS_INFO) << "Git branch: " << GIT_CURRENT_BRANCH;
    LOG(LS_INFO) << "Git commit: " << GIT_COMMIT_HASH;
#endif
    LOG(LS_INFO) << "Command line: " << command_line->commandLineString();
    LOG(LS_INFO) << "OS: " << base::SysInfo::operatingSystemName()
                 << " (version: " << base::SysInfo::operatingSystemVersion()
                 <<  " arch: " << base::SysInfo::operatingSystemArchitecture() << ")";
    LOG(LS_INFO) << "CPU: " << base::SysInfo::processorName()
                 << " (vendor: " << base::SysInfo::processorVendor()
                 << " packages: " << base::SysInfo::processorPackages()
                 << " cores: " << base::SysInfo::processorCores()
                 << " threads: " << base::SysInfo::processorThreads() << ")";

#if defined(OS_WIN)
    MEMORYSTATUSEX memory_status;
    memset(&memory_status, 0, sizeof(memory_status));
    memory_status.dwLength = sizeof(memory_status);

    if (GlobalMemoryStatusEx(&memory_status))
    {
        static const uint32_t kMB = 1024 * 1024;

        LOG(LS_INFO) << "Total physical memory: " << (memory_status.ullTotalPhys / kMB)
                     << "MB (free: " << (memory_status.ullAvailPhys / kMB) << "MB)";
        LOG(LS_INFO) << "Total page file: " << (memory_status.ullTotalPageFile / kMB)
                     << "MB (free: " << (memory_status.ullAvailPageFile / kMB) << "MB)";
        LOG(LS_INFO) << "Total virtual memory: " << (memory_status.ullTotalVirtual / kMB)
                     << "MB (free: " << (memory_status.ullAvailVirtual / kMB) << "MB)";
    }
    else
    {
        PLOG(LS_ERROR) << "GlobalMemoryStatusEx failed";
    }

    LOG(LS_INFO) << "Video adapters";
    LOG(LS_INFO) << "#####################################################";
    Microsoft::WRL::ComPtr<IDXGIFactory> factory;
    _com_error error = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                         reinterpret_cast<void**>(factory.GetAddressOf()));
    if (error.Error() != S_OK || !factory)
    {
        LOG(LS_INFO) << "CreateDXGIFactory failed: " << error.ErrorMessage();
    }
    else
    {
        UINT adapter_index = 0;

        while (true)
        {
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            error = factory->EnumAdapters(adapter_index, adapter.GetAddressOf());
            if (error.Error() != S_OK)
                break;

            DXGI_ADAPTER_DESC adapter_desc;
            memset(&adapter_desc, 0, sizeof(adapter_desc));

            if (SUCCEEDED(adapter->GetDesc(&adapter_desc)))
            {
                static const uint32_t kMB = 1024 * 1024;

                LOG(LS_INFO) << adapter_desc.Description << " ("
                             << "video memory: " << (adapter_desc.DedicatedVideoMemory / kMB) << "MB"
                             << ", system memory: " << (adapter_desc.DedicatedSystemMemory / kMB) << "MB"
                             << ", shared memory: " << (adapter_desc.SharedSystemMemory / kMB) << "MB"
                             << ")";
            }

            ++adapter_index;
        }
    }
    LOG(LS_INFO) << "#####################################################";

    DWORD session_id = 0;
    if (!ProcessIdToSessionId(GetCurrentProcessId(), &session_id))
    {
        PLOG(LS_ERROR) << "ProcessIdToSessionId failed";
    }
    else
    {
        base::win::SessionInfo session_info(session_id);
        if (!session_info.isValid())
        {
            LOG(LS_ERROR) << "Unable to get session info";
        }
        else
        {
            LOG(LS_INFO) << "Process session ID: " << session_id;
            LOG(LS_INFO) << "Running in user session: '" << session_info.userName() << "'";
            LOG(LS_INFO) << "Session connect state: "
                << base::win::SessionInfo::connectStateToString(session_info.connectState());
            LOG(LS_INFO) << "WinStation name: '" << session_info.winStationName() << "'";
            LOG(LS_INFO) << "Domain name: '" << session_info.domain() << "'";
            LOG(LS_INFO) << "User Locked: " << session_info.isUserLocked();
        }
    }

    wchar_t username[64] = { 0 };
    DWORD username_size = sizeof(username) / sizeof(username[0]);
    if (!GetUserNameW(username, &username_size))
    {
        PLOG(LS_ERROR) << "GetUserNameW failed";
    }

    LOG(LS_INFO) << "Running as user: '" << username << "'";
    LOG(LS_INFO) << "Active console session ID: " << WTSGetActiveConsoleSessionId();
    LOG(LS_INFO) << "Computer name: '" << base::SysInfo::computerName() << "'";
    LOG(LS_INFO) << "Process WindowStation: " << base::WindowStation::forCurrentProcess().name();

    LOG(LS_INFO) << "WindowStation list";
    LOG(LS_INFO) << "#####################################################";
    for (const auto& window_station_name : base::WindowStation::windowStationList())
    {
        std::wstring desktops;

        base::WindowStation window_station = base::WindowStation::open(window_station_name.data());
        if (window_station.isValid())
        {
            std::vector<std::wstring> list = base::Desktop::desktopList(window_station.get());

            for (size_t i = 0; i < list.size(); ++i)
            {
                desktops += list[i];
                if ((i + 1) != list.size())
                    desktops += L", ";
            }
        }

        LOG(LS_INFO) << window_station_name << " (desktops: " << desktops << ")";
    }
    LOG(LS_INFO) << "#####################################################";

    LOG(LS_INFO) << "Environment variables";
    LOG(LS_INFO) << "#####################################################";
    for (const auto& variable : base::Environment::list())
    {
        LOG(LS_INFO) << variable.first << ": " << variable.second;
    }
    LOG(LS_INFO) << "#####################################################";
#endif // defined(OS_WIN)

    if (command_line->hasSwitch(u"channel_id"))
    {
        std::unique_ptr<base::MessageLoop> message_loop =
            std::make_unique<base::MessageLoop>(base::MessageLoop::Type::ASIO);

        std::shared_ptr<host::DesktopSessionAgent> desktop_agent =
            std::make_shared<host::DesktopSessionAgent>(message_loop->taskRunner());

        desktop_agent->start(command_line->switchValue(u"channel_id"));
        message_loop->run();
    }
    else
    {
        LOG(LS_ERROR) << "Parameter channel_id is not specified";
    }
}
