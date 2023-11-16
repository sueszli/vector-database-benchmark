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

#include "base/power_controller.h"

#include "base/win/scoped_impersonator.h"
#include "base/win/scoped_object.h"
#include "base/win/session_info.h"
#include "base/logging.h"

#include <Windows.h>
#include <WtsApi32.h>

namespace base {

namespace {

// Delay for shutdown and reboot.
const DWORD kActionDelayInSeconds = 0;

//--------------------------------------------------------------------------------------------------
bool copyProcessToken(DWORD desired_access, win::ScopedHandle* token_out)
{
    win::ScopedHandle process_token;
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
// Creates a copy of the current process with SE_SHUTDOWN_NAME privilege enabled.
bool createPrivilegedToken(win::ScopedHandle* token_out)
{
    const DWORD desired_access = TOKEN_ADJUST_PRIVILEGES | TOKEN_IMPERSONATE |
        TOKEN_DUPLICATE | TOKEN_QUERY;

    win::ScopedHandle privileged_token;
    if (!copyProcessToken(desired_access, &privileged_token))
    {
        LOG(LS_ERROR) << "copyProcessToken failed";
        return false;
    }

    // Get the LUID for the SE_SHUTDOWN_NAME privilege.
    TOKEN_PRIVILEGES state;
    state.PrivilegeCount = 1;
    state.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!LookupPrivilegeValueW(nullptr, SE_SHUTDOWN_NAME, &state.Privileges[0].Luid))
    {
        PLOG(LS_ERROR) << "LookupPrivilegeValueW failed";
        return false;
    }

    // Enable the SE_SHUTDOWN_NAME privilege.
    if (!AdjustTokenPrivileges(privileged_token, FALSE, &state, 0, nullptr, nullptr))
    {
        PLOG(LS_ERROR) << "AdjustTokenPrivileges failed";
        return false;
    }

    token_out->reset(privileged_token.release());
    return true;
}

} // namespace

//--------------------------------------------------------------------------------------------------
// static
bool PowerController::shutdown()
{
    const DWORD desired_access =
        TOKEN_ADJUST_DEFAULT | TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY;

    win::ScopedHandle process_token;
    if (!copyProcessToken(desired_access, &process_token))
    {
        LOG(LS_ERROR) << "copyProcessToken failed";
        return false;
    }

    win::ScopedHandle privileged_token;
    if (!createPrivilegedToken(&privileged_token))
    {
        LOG(LS_ERROR) << "createPrivilegedToken failed";
        return false;
    }

    win::ScopedImpersonator impersonator;
    if (!impersonator.loggedOnUser(privileged_token))
    {
        LOG(LS_ERROR) << "loggedOnUser failed";
        return false;
    }

    const DWORD reason = SHTDN_REASON_MAJOR_APPLICATION | SHTDN_REASON_MINOR_MAINTENANCE;

    bool result = !!InitiateSystemShutdownExW(nullptr,
                                              nullptr,
                                              kActionDelayInSeconds,
                                              TRUE,  // Force close apps.
                                              FALSE, // Shutdown.
                                              reason);
    if (!result)
    {
        PLOG(LS_ERROR) << "InitiateSystemShutdownExW failed";
    }

    return result;
}

//--------------------------------------------------------------------------------------------------
// static
bool PowerController::reboot()
{
    const DWORD desired_access =
        TOKEN_ADJUST_DEFAULT | TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY;

    win::ScopedHandle process_token;
    if (!copyProcessToken(desired_access, &process_token))
    {
        LOG(LS_ERROR) << "copyProcessToken failed";
        return false;
    }

    win::ScopedHandle privileged_token;
    if (!createPrivilegedToken(&privileged_token))
    {
        LOG(LS_ERROR) << "createPrivilegedToken failed";
        return false;
    }

    win::ScopedImpersonator impersonator;
    if (!impersonator.loggedOnUser(privileged_token))
    {
        LOG(LS_ERROR) << "loggedOnUser failed";
        return false;
    }

    const DWORD reason = SHTDN_REASON_MAJOR_APPLICATION | SHTDN_REASON_MINOR_MAINTENANCE;

    bool result = !!InitiateSystemShutdownExW(nullptr,
                                              nullptr,
                                              kActionDelayInSeconds,
                                              TRUE, // Force close apps.
                                              TRUE, // Reboot.
                                              reason);
    if (!result)
    {
        PLOG(LS_ERROR) << "InitiateSystemShutdownExW failed";
    }

    return result;
}

//--------------------------------------------------------------------------------------------------
// static
bool PowerController::logoff()
{
    DWORD session_id = base::kInvalidSessionId;
    if (!ProcessIdToSessionId(GetCurrentProcessId(), &session_id))
    {
        PLOG(LS_ERROR) << "ProcessIdToSessionId failed";
    }

    if (session_id != kInvalidSessionId)
    {
        base::win::SessionInfo session_info(session_id);
        if (session_info.connectState() != base::win::SessionInfo::ConnectState::ACTIVE)
        {
            LOG(LS_INFO) << "User session not in active state. Logoff not required";
            return true;
        }
    }

    if (!WTSLogoffSession(WTS_CURRENT_SERVER_HANDLE, session_id, FALSE))
    {
        PLOG(LS_ERROR) << "WTSLogoffSession failed";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
// static
bool PowerController::lock()
{
    if (!LockWorkStation())
    {
        PLOG(LS_ERROR) << "LockWorkStation failed";
        return false;
    }

    return true;
}

} // namespace base
