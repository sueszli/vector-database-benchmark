// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/scope_exit.h"
#include "common/scratch_buffer.h"
#include "core/core.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_server_session.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/svc_results.h"

namespace Kernel::Svc {

/// Makes a blocking IPC call to a service.
Result SendSyncRequest(Core::System& system, Handle handle) {
    // Get the client session from its handle.
    KScopedAutoObject session =
        GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KClientSession>(handle);
    R_UNLESS(session.IsNotNull(), ResultInvalidHandle);

    LOG_TRACE(Kernel_SVC, "called handle=0x{:08X}", handle);

    R_RETURN(session->SendSyncRequest());
}

Result SendSyncRequestWithUserBuffer(Core::System& system, uint64_t message_buffer,
                                     uint64_t message_buffer_size, Handle session_handle) {
    UNIMPLEMENTED();
    R_THROW(ResultNotImplemented);
}

Result SendAsyncRequestWithUserBuffer(Core::System& system, Handle* out_event_handle,
                                      uint64_t message_buffer, uint64_t message_buffer_size,
                                      Handle session_handle) {
    UNIMPLEMENTED();
    R_THROW(ResultNotImplemented);
}

Result ReplyAndReceive(Core::System& system, s32* out_index, uint64_t handles_addr, s32 num_handles,
                       Handle reply_target, s64 timeout_ns) {
    // Ensure number of handles is valid.
    R_UNLESS(0 <= num_handles && num_handles <= ArgumentHandleCountMax, ResultOutOfRange);

    // Get the synchronization context.
    auto& kernel = system.Kernel();
    auto& handle_table = GetCurrentProcess(kernel).GetHandleTable();
    auto objs = GetCurrentThread(kernel).GetSynchronizationObjectBuffer();
    auto handles = GetCurrentThread(kernel).GetHandleBuffer();

    // Copy user handles.
    if (num_handles > 0) {
        // Get the handles.
        R_UNLESS(GetCurrentMemory(kernel).ReadBlock(handles_addr, handles.data(),
                                                    sizeof(Handle) * num_handles),
                 ResultInvalidPointer);

        // Convert the handles to objects.
        R_UNLESS(handle_table.GetMultipleObjects<KSynchronizationObject>(
                     objs.data(), handles.data(), num_handles),
                 ResultInvalidHandle);
    }

    // Ensure handles are closed when we're done.
    SCOPE_EXIT({
        for (auto i = 0; i < num_handles; ++i) {
            objs[i]->Close();
        }
    });

    // Reply to the target, if one is specified.
    if (reply_target != InvalidHandle) {
        KScopedAutoObject session = handle_table.GetObject<KServerSession>(reply_target);
        R_UNLESS(session.IsNotNull(), ResultInvalidHandle);

        // If we fail to reply, we want to set the output index to -1.
        ON_RESULT_FAILURE {
            *out_index = -1;
        };

        // Send the reply.
        R_TRY(session->SendReply());
    }

    // Convert the timeout from nanoseconds to ticks.
    // NOTE: Nintendo does not use this conversion logic in WaitSynchronization...
    s64 timeout;
    if (timeout_ns > 0) {
        const s64 offset_tick(timeout_ns);
        if (offset_tick > 0) {
            timeout = kernel.HardwareTimer().GetTick() + offset_tick + 2;
            if (timeout <= 0) {
                timeout = std::numeric_limits<s64>::max();
            }
        } else {
            timeout = std::numeric_limits<s64>::max();
        }
    } else {
        timeout = timeout_ns;
    }

    // Wait for a message.
    while (true) {
        // Wait for an object.
        s32 index;
        Result result = KSynchronizationObject::Wait(kernel, std::addressof(index), objs.data(),
                                                     num_handles, timeout);
        if (result == ResultTimedOut) {
            R_RETURN(result);
        }

        // Receive the request.
        if (R_SUCCEEDED(result)) {
            KServerSession* session = objs[index]->DynamicCast<KServerSession*>();
            if (session != nullptr) {
                result = session->ReceiveRequest();
                if (result == ResultNotFound) {
                    continue;
                }
            }
        }

        *out_index = index;
        R_RETURN(result);
    }
}

Result ReplyAndReceiveWithUserBuffer(Core::System& system, int32_t* out_index,
                                     uint64_t message_buffer, uint64_t message_buffer_size,
                                     uint64_t handles, int32_t num_handles, Handle reply_target,
                                     int64_t timeout_ns) {
    UNIMPLEMENTED();
    R_THROW(ResultNotImplemented);
}

Result SendSyncRequest64(Core::System& system, Handle session_handle) {
    R_RETURN(SendSyncRequest(system, session_handle));
}

Result SendSyncRequestWithUserBuffer64(Core::System& system, uint64_t message_buffer,
                                       uint64_t message_buffer_size, Handle session_handle) {
    R_RETURN(
        SendSyncRequestWithUserBuffer(system, message_buffer, message_buffer_size, session_handle));
}

Result SendAsyncRequestWithUserBuffer64(Core::System& system, Handle* out_event_handle,
                                        uint64_t message_buffer, uint64_t message_buffer_size,
                                        Handle session_handle) {
    R_RETURN(SendAsyncRequestWithUserBuffer(system, out_event_handle, message_buffer,
                                            message_buffer_size, session_handle));
}

Result ReplyAndReceive64(Core::System& system, int32_t* out_index, uint64_t handles,
                         int32_t num_handles, Handle reply_target, int64_t timeout_ns) {
    R_RETURN(ReplyAndReceive(system, out_index, handles, num_handles, reply_target, timeout_ns));
}

Result ReplyAndReceiveWithUserBuffer64(Core::System& system, int32_t* out_index,
                                       uint64_t message_buffer, uint64_t message_buffer_size,
                                       uint64_t handles, int32_t num_handles, Handle reply_target,
                                       int64_t timeout_ns) {
    R_RETURN(ReplyAndReceiveWithUserBuffer(system, out_index, message_buffer, message_buffer_size,
                                           handles, num_handles, reply_target, timeout_ns));
}

Result SendSyncRequest64From32(Core::System& system, Handle session_handle) {
    R_RETURN(SendSyncRequest(system, session_handle));
}

Result SendSyncRequestWithUserBuffer64From32(Core::System& system, uint32_t message_buffer,
                                             uint32_t message_buffer_size, Handle session_handle) {
    R_RETURN(
        SendSyncRequestWithUserBuffer(system, message_buffer, message_buffer_size, session_handle));
}

Result SendAsyncRequestWithUserBuffer64From32(Core::System& system, Handle* out_event_handle,
                                              uint32_t message_buffer, uint32_t message_buffer_size,
                                              Handle session_handle) {
    R_RETURN(SendAsyncRequestWithUserBuffer(system, out_event_handle, message_buffer,
                                            message_buffer_size, session_handle));
}

Result ReplyAndReceive64From32(Core::System& system, int32_t* out_index, uint32_t handles,
                               int32_t num_handles, Handle reply_target, int64_t timeout_ns) {
    R_RETURN(ReplyAndReceive(system, out_index, handles, num_handles, reply_target, timeout_ns));
}

Result ReplyAndReceiveWithUserBuffer64From32(Core::System& system, int32_t* out_index,
                                             uint32_t message_buffer, uint32_t message_buffer_size,
                                             uint32_t handles, int32_t num_handles,
                                             Handle reply_target, int64_t timeout_ns) {
    R_RETURN(ReplyAndReceiveWithUserBuffer(system, out_index, message_buffer, message_buffer_size,
                                           handles, num_handles, reply_target, timeout_ns));
}

} // namespace Kernel::Svc
