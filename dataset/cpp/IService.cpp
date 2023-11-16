// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "IService.h"

namespace skyline::service::fatalsrv {
    IService::IService(const DeviceState &state, ServiceManager &manager) : BaseService(state, manager) {}

    Result IService::ThrowFatal(type::KSession &session, ipc::IpcRequest &request, ipc::IpcResponse &response) {
        throw exception("A fatal error with code: 0x{:X} has caused emulation to stop", request.Pop<u32>());
    }
}
