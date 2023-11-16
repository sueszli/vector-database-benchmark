// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/time/standard_local_system_clock_core.h"
#include "core/hle/service/time/standard_network_system_clock_core.h"
#include "core/hle/service/time/standard_user_system_clock_core.h"

namespace Service::Time::Clock {

StandardUserSystemClockCore::StandardUserSystemClockCore(
    StandardLocalSystemClockCore& local_system_clock_core_,
    StandardNetworkSystemClockCore& network_system_clock_core_, Core::System& system_)
    : SystemClockCore(local_system_clock_core_.GetSteadyClockCore()),
      local_system_clock_core{local_system_clock_core_},
      network_system_clock_core{network_system_clock_core_},
      auto_correction_time{SteadyClockTimePoint::GetRandom()}, service_context{
                                                                   system_,
                                                                   "StandardUserSystemClockCore"} {
    auto_correction_event =
        service_context.CreateEvent("StandardUserSystemClockCore:AutoCorrectionEvent");
}

StandardUserSystemClockCore::~StandardUserSystemClockCore() {
    service_context.CloseEvent(auto_correction_event);
}

Result StandardUserSystemClockCore::SetAutomaticCorrectionEnabled(Core::System& system,
                                                                  bool value) {
    if (const Result result{ApplyAutomaticCorrection(system, value)}; result != ResultSuccess) {
        return result;
    }

    auto_correction_enabled = value;

    return ResultSuccess;
}

Result StandardUserSystemClockCore::GetClockContext(Core::System& system,
                                                    SystemClockContext& ctx) const {
    if (const Result result{ApplyAutomaticCorrection(system, false)}; result != ResultSuccess) {
        return result;
    }

    return local_system_clock_core.GetClockContext(system, ctx);
}

Result StandardUserSystemClockCore::Flush(const SystemClockContext&) {
    UNIMPLEMENTED();
    return ERROR_NOT_IMPLEMENTED;
}

Result StandardUserSystemClockCore::SetClockContext(const SystemClockContext&) {
    UNIMPLEMENTED();
    return ERROR_NOT_IMPLEMENTED;
}

Result StandardUserSystemClockCore::ApplyAutomaticCorrection(Core::System& system,
                                                             bool value) const {
    if (auto_correction_enabled == value) {
        return ResultSuccess;
    }

    if (!network_system_clock_core.IsClockSetup(system)) {
        return ERROR_UNINITIALIZED_CLOCK;
    }

    SystemClockContext ctx{};
    if (const Result result{network_system_clock_core.GetClockContext(system, ctx)};
        result != ResultSuccess) {
        return result;
    }

    local_system_clock_core.SetClockContext(ctx);

    return ResultSuccess;
}

} // namespace Service::Time::Clock
