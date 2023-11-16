// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "ILogger.h"
#include "ILogService.h"

namespace skyline::service::lm {
    ILogService::ILogService(const DeviceState &state, ServiceManager &manager) : BaseService(state, manager) {}

    Result ILogService::OpenLogger(type::KSession &session, ipc::IpcRequest &request, ipc::IpcResponse &response) {
        manager.RegisterService(SRVREG(ILogger), session, response);
        return {};
    }
}
