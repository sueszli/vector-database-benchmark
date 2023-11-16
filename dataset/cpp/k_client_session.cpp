// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/scope_exit.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_server_session.h"
#include "core/hle/kernel/k_session.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/result.h"

namespace Kernel {

static constexpr u32 MessageBufferSize = 0x100;

KClientSession::KClientSession(KernelCore& kernel) : KAutoObjectWithSlabHeapAndContainer{kernel} {}
KClientSession::~KClientSession() = default;

void KClientSession::Destroy() {
    m_parent->OnClientClosed();
    m_parent->Close();
}

void KClientSession::OnServerClosed() {}

Result KClientSession::SendSyncRequest() {
    // Create a session request.
    KSessionRequest* request = KSessionRequest::Create(m_kernel);
    R_UNLESS(request != nullptr, ResultOutOfResource);
    SCOPE_EXIT({ request->Close(); });

    // Initialize the request.
    request->Initialize(nullptr, GetInteger(GetCurrentThread(m_kernel).GetTlsAddress()),
                        MessageBufferSize);

    // Send the request.
    R_RETURN(m_parent->GetServerSession().OnRequest(request));
}

} // namespace Kernel
