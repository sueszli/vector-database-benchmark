//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "CommonCorePch.h"
#include "Core/EtwTraceCore.h"

#ifdef ENABLE_JS_ETW
#ifndef ENABLE_JS_LTTNG
extern "C" {
    ETW_INLINE
        VOID EtwCallback(
        ULONG controlCode,
        PVOID callbackContext)
    {
        EtwCallbackApi::OnSessionChange(controlCode, callbackContext);
    }
}

bool EtwTraceCore::s_registered = false;

//
// Registers the ETW provider - this is usually done on Jscript DLL load
// After registration, we will receive callbacks when ETW tracing is enabled/disabled.
//
void EtwTraceCore::Register()
{
    if (!s_registered)
    {
        s_registered = true;

#ifdef NTBUILD
        JS_ETW(EventRegisterMicrosoft_IE());
#endif
        JS_ETW(EventRegisterMicrosoft_JScript());
#ifdef NTBUILD
        JS_ETW(EventRegisterMicrosoft_JScript_Internal());
#endif

        // This will be used to distinguish the provider we are getting the callback for.
        PROVIDER_JSCRIPT9_Context.RegistrationHandle = Microsoft_JScriptHandle;

#ifdef NTBUILD
        BERP_IE_Context.RegistrationHandle = Microsoft_IEHandle;
#endif
    }
}

//
// Unregister to ensure we do not get callbacks.
//
void EtwTraceCore::UnRegister()
{
    if (s_registered)
    {
        s_registered = false;

#ifdef NTBUILD
        JS_ETW(EventUnregisterMicrosoft_IE());
#endif
        JS_ETW(EventUnregisterMicrosoft_JScript());
#ifdef NTBUILD
        JS_ETW(EventUnregisterMicrosoft_JScript_Internal());
#endif
    }
}

#endif // !ENABLE_JS_LTTNG
#endif // ENABLE_JS_ETW
