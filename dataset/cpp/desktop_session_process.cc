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

#include "host/desktop_session_process.h"

#include "base/command_line.h"
#include "base/logging.h"
#include "base/stl_util.h"
#include "base/files/base_paths.h"
#include "base/strings/string_printf.h"
#include "base/strings/string_util.h"
#include "base/strings/string_split.h"
#include "base/strings/unicode.h"

#if defined(OS_WIN)
#include "base/win/scoped_impersonator.h"

#include <UserEnv.h>
#endif // defined(OS_WIN)

#if defined(OS_LINUX)
#include <signal.h>
#include <spawn.h>
#endif // defined(OS_LINUX)

namespace host {

namespace {

#if defined(OS_LINUX)
const char16_t kDesktopAgentFile[] = u"aspia_desktop_agent";
#endif

#if defined(OS_WIN)
// Name of the default session desktop.
const char16_t kDefaultDesktopName[] = u"winsta0\\default";
const char16_t kDesktopAgentFile[] = u"aspia_desktop_agent.exe";

//--------------------------------------------------------------------------------------------------
bool copyProcessToken(DWORD desired_access, base::win::ScopedHandle* token_out)
{
    base::win::ScopedHandle process_token;

    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_DUPLICATE | desired_access,
                          process_token.recieve()))
    {
        PLOG(LS_ERROR) << "OpenProcessToken failed";
        return false;
    }

    if (!DuplicateTokenEx(process_token,
                          desired_access,
                          nullptr,
                          SecurityImpersonation,
                          TokenPrimary,
                          token_out->recieve()))
    {
        PLOG(LS_ERROR) << "DuplicateTokenEx failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool createPrivilegedToken(base::win::ScopedHandle* token_out)
{
    base::win::ScopedHandle privileged_token;
    const DWORD desired_access = TOKEN_ADJUST_PRIVILEGES | TOKEN_IMPERSONATE |
        TOKEN_DUPLICATE | TOKEN_QUERY;

    if (!copyProcessToken(desired_access, &privileged_token))
    {
        LOG(LS_ERROR) << "copyProcessToken failed";
        return false;
    }

    // Get the LUID for the SE_TCB_NAME privilege.
    TOKEN_PRIVILEGES state;
    state.PrivilegeCount = 1;
    state.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!LookupPrivilegeValueW(nullptr, SE_TCB_NAME, &state.Privileges[0].Luid))
    {
        PLOG(LS_ERROR) << "LookupPrivilegeValueW failed";
        return false;
    }

    // Enable the SE_TCB_NAME privilege.
    if (!AdjustTokenPrivileges(privileged_token, FALSE, &state, 0, nullptr, nullptr))
    {
        PLOG(LS_ERROR) << "AdjustTokenPrivileges failed";
        return false;
    }

    token_out->reset(privileged_token.release());
    return true;
}

//--------------------------------------------------------------------------------------------------
// Creates a copy of the current process token for the given |session_id| so
// it can be used to launch a process in that session.
bool createSessionToken(DWORD session_id, base::win::ScopedHandle* token_out)
{
    base::win::ScopedHandle session_token;
    const DWORD desired_access = TOKEN_ADJUST_DEFAULT | TOKEN_ADJUST_SESSIONID |
        TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY;

    if (!copyProcessToken(desired_access, &session_token))
    {
        LOG(LS_ERROR) << "copyProcessToken failed";
        return false;
    }

    base::win::ScopedHandle privileged_token;

    if (!createPrivilegedToken(&privileged_token))
    {
        LOG(LS_ERROR) << "createPrivilegedToken failed";
        return false;
    }

    base::win::ScopedImpersonator impersonator;

    if (!impersonator.loggedOnUser(privileged_token))
    {
        LOG(LS_ERROR) << "Failed to impersonate thread";
        return false;
    }

    // Change the session ID of the token.
    if (!SetTokenInformation(session_token, TokenSessionId, &session_id, sizeof(session_id)))
    {
        PLOG(LS_ERROR) << "SetTokenInformation failed";
        return false;
    }

    DWORD ui_access = 1;
    if (!SetTokenInformation(session_token, TokenUIAccess, &ui_access, sizeof(ui_access)))
    {
        PLOG(LS_ERROR) << "SetTokenInformation failed";
        return false;
    }

    token_out->reset(session_token.release());
    return true;
}

//--------------------------------------------------------------------------------------------------
bool startProcessWithToken(HANDLE token,
                           const base::CommandLine& command_line,
                           base::win::ScopedHandle* process,
                           base::win::ScopedHandle* thread)
{
    STARTUPINFOW startup_info;
    memset(&startup_info, 0, sizeof(startup_info));

    startup_info.cb = sizeof(startup_info);
    startup_info.lpDesktop = const_cast<wchar_t*>(
        reinterpret_cast<const wchar_t*>(kDefaultDesktopName));

    void* environment = nullptr;

    if (!CreateEnvironmentBlock(&environment, token, FALSE))
    {
        PLOG(LS_ERROR) << "CreateEnvironmentBlock failed";
        return false;
    }

    PROCESS_INFORMATION process_info;
    memset(&process_info, 0, sizeof(process_info));

    if (!CreateProcessAsUserW(token,
                              nullptr,
                              const_cast<wchar_t*>(reinterpret_cast<const wchar_t*>(
                                  command_line.commandLineString().c_str())),
                              nullptr,
                              nullptr,
                              FALSE,
                              CREATE_UNICODE_ENVIRONMENT | HIGH_PRIORITY_CLASS,
                              environment,
                              nullptr,
                              &startup_info,
                              &process_info))
    {
        PLOG(LS_ERROR) << "CreateProcessAsUserW failed";
        if (!DestroyEnvironmentBlock(environment))
        {
            PLOG(LS_ERROR) << "DestroyEnvironmentBlock failed";
        }
        return false;
    }

    thread->reset(process_info.hThread);
    process->reset(process_info.hProcess);

    if (!DestroyEnvironmentBlock(environment))
    {
        PLOG(LS_ERROR) << "DestroyEnvironmentBlock failed";
    }

    return true;
}
#endif // defined(OS_WIN)

} // namespace

//--------------------------------------------------------------------------------------------------
#if defined(OS_WIN)
DesktopSessionProcess::DesktopSessionProcess(
    base::win::ScopedHandle&& process, base::win::ScopedHandle&& thread)
    : process_(std::move(process)),
      thread_(std::move(thread))
{
    LOG(LS_INFO) << "Ctor";
}
#elif defined(OS_LINUX)
DesktopSessionProcess::DesktopSessionProcess(pid_t pid)
    : pid_(pid)
{
    LOG(LS_INFO) << "Ctor";
}
#else
DesktopSessionProcess::DesktopSessionProcess()
{
    LOG(LS_INFO) << "Ctor";
}
#endif

//--------------------------------------------------------------------------------------------------
DesktopSessionProcess::~DesktopSessionProcess()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
// static
std::unique_ptr<DesktopSessionProcess> DesktopSessionProcess::create(
    base::SessionId session_id, std::u16string_view channel_id)
{
    if (session_id == base::kInvalidSessionId)
    {
        LOG(LS_ERROR) << "An attempt was detected to start a process in a INVALID session";
        return nullptr;
    }

#if defined(OS_WIN)
    if (session_id == base::kServiceSessionId)
    {
        LOG(LS_ERROR) << "An attempt was detected to start a process in a SERVICES session";
        return nullptr;
    }

    base::CommandLine command_line(filePath());
    command_line.appendSwitch(u"channel_id", channel_id);

    base::win::ScopedHandle session_token;
    if (!createSessionToken(session_id, &session_token))
    {
        LOG(LS_ERROR) << "createSessionToken failed";
        return nullptr;
    }

    base::win::ScopedHandle process_handle;
    base::win::ScopedHandle thread_handle;

    if (!startProcessWithToken(session_token, command_line, &process_handle, &thread_handle))
    {
        LOG(LS_ERROR) << "startProcessWithToken failed";
        return nullptr;
    }

    return std::unique_ptr<DesktopSessionProcess>(
        new DesktopSessionProcess(std::move(process_handle), std::move(thread_handle)));
#elif defined(OS_LINUX)
    std::error_code ignored_error;
    std::filesystem::directory_iterator it("/usr/share/xsessions/", ignored_error);
    if (it == std::filesystem::end(it))
    {
        LOG(LS_ERROR) << "No X11 sessions";
        return nullptr;
    }

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("who", "r"), pclose);
    if (!pipe)
    {
        LOG(LS_ERROR) << "Unable to open pipe";
        return nullptr;
    }

    LOG(LS_INFO) << "WHO:";
    std::array<char, 512> buffer;
    while (fgets(buffer.data(), buffer.size(), pipe.get()))
    {
        std::u16string line = base::toLower(base::utf16FromLocal8Bit(buffer.data()));
        LOG(LS_INFO) << line;
        if (!base::contains(line, u":0"))
            continue;

        std::vector<std::u16string_view> splitted = base::splitStringView(
            line, u" ", base::TRIM_WHITESPACE, base::SPLIT_WANT_NONEMPTY);
        if (splitted.empty())
            continue;

        std::string user_name = base::local8BitFromUtf16(splitted.front());
        std::string command_line =
            base::stringPrintf("sudo DISPLAY=':0' -u %s %s --channel_id=%s &",
                               user_name.c_str(),
                               filePath().c_str(),
                               base::local8BitFromUtf16(channel_id).c_str());

        LOG(LS_INFO) << "Start desktop session agent: " << command_line;

        char sh_name[] = "sh";
        char sh_arguments[] = "-c";
        char* argv[] = { sh_name, sh_arguments, command_line.data(), nullptr };

        pid_t pid;
        if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, environ) != 0)
        {
            PLOG(LS_ERROR) << "Unable to start process";
            return nullptr;
        }

        return std::unique_ptr<DesktopSessionProcess>(new DesktopSessionProcess(pid));
    }

    LOG(LS_ERROR) << "Connected X sessions not found";

    std::string command_line =
        base::stringPrintf("sudo DISPLAY=':0' -u root %s --channel_id=%s &",
                           filePath().c_str(),
                           base::local8BitFromUtf16(channel_id).c_str());

    LOG(LS_INFO) << "Start desktop session agent: " << command_line;

    char sh_name[] = "sh";
    char sh_arguments[] = "-c";
    char* argv[] = { sh_name, sh_arguments, command_line.data(), nullptr };

    pid_t pid;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, environ) != 0)
    {
        PLOG(LS_ERROR) << "Unable to start process";
        return nullptr;
    }

    return std::unique_ptr<DesktopSessionProcess>(new DesktopSessionProcess(pid));
#else
    NOTIMPLEMENTED();
    return std::unique_ptr<DesktopSessionProcess>();
#endif
}

//--------------------------------------------------------------------------------------------------
// static
std::filesystem::path DesktopSessionProcess::filePath()
{
    std::filesystem::path file_path;
    if (!base::BasePaths::currentExecDir(&file_path))
    {
        LOG(LS_ERROR) << "currentExecDir failed";
        return std::filesystem::path();
    }

    file_path.append(kDesktopAgentFile);
    return file_path;
}

//--------------------------------------------------------------------------------------------------
void DesktopSessionProcess::kill()
{
#if defined(OS_WIN)
    if (!process_.isValid())
    {
        LOG(LS_ERROR) << "Invalid process handle";
        return;
    }

    if (!TerminateProcess(process_, 0))
    {
        PLOG(LS_ERROR) << "TerminateProcess failed";
    }
#elif defined(OS_LINUX)
    if (::kill(pid_, SIGKILL) != 0)
    {
        PLOG(LS_ERROR) << "kill failed";
    }
#else
    NOTIMPLEMENTED();
#endif
}

} // namespace host
