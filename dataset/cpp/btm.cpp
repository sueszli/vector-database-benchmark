// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>

#include "common/logging/log.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/btm/btm.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/service.h"

namespace Service::BTM {

class IBtmUserCore final : public ServiceFramework<IBtmUserCore> {
public:
    explicit IBtmUserCore(Core::System& system_)
        : ServiceFramework{system_, "IBtmUserCore"}, service_context{system_, "IBtmUserCore"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &IBtmUserCore::AcquireBleScanEvent, "AcquireBleScanEvent"},
            {1, nullptr, "GetBleScanFilterParameter"},
            {2, nullptr, "GetBleScanFilterParameter2"},
            {3, nullptr, "StartBleScanForGeneral"},
            {4, nullptr, "StopBleScanForGeneral"},
            {5, nullptr, "GetBleScanResultsForGeneral"},
            {6, nullptr, "StartBleScanForPaired"},
            {7, nullptr, "StopBleScanForPaired"},
            {8, nullptr, "StartBleScanForSmartDevice"},
            {9, nullptr, "StopBleScanForSmartDevice"},
            {10, nullptr, "GetBleScanResultsForSmartDevice"},
            {17, &IBtmUserCore::AcquireBleConnectionEvent, "AcquireBleConnectionEvent"},
            {18, nullptr, "BleConnect"},
            {19, nullptr, "BleDisconnect"},
            {20, nullptr, "BleGetConnectionState"},
            {21, nullptr, "AcquireBlePairingEvent"},
            {22, nullptr, "BlePairDevice"},
            {23, nullptr, "BleUnPairDevice"},
            {24, nullptr, "BleUnPairDevice2"},
            {25, nullptr, "BleGetPairedDevices"},
            {26, &IBtmUserCore::AcquireBleServiceDiscoveryEvent, "AcquireBleServiceDiscoveryEvent"},
            {27, nullptr, "GetGattServices"},
            {28, nullptr, "GetGattService"},
            {29, nullptr, "GetGattIncludedServices"},
            {30, nullptr, "GetBelongingGattService"},
            {31, nullptr, "GetGattCharacteristics"},
            {32, nullptr, "GetGattDescriptors"},
            {33, &IBtmUserCore::AcquireBleMtuConfigEvent, "AcquireBleMtuConfigEvent"},
            {34, nullptr, "ConfigureBleMtu"},
            {35, nullptr, "GetBleMtu"},
            {36, nullptr, "RegisterBleGattDataPath"},
            {37, nullptr, "UnregisterBleGattDataPath"},
        };
        // clang-format on
        RegisterHandlers(functions);

        scan_event = service_context.CreateEvent("IBtmUserCore:ScanEvent");
        connection_event = service_context.CreateEvent("IBtmUserCore:ConnectionEvent");
        service_discovery_event = service_context.CreateEvent("IBtmUserCore:DiscoveryEvent");
        config_event = service_context.CreateEvent("IBtmUserCore:ConfigEvent");
    }

    ~IBtmUserCore() override {
        service_context.CloseEvent(scan_event);
        service_context.CloseEvent(connection_event);
        service_context.CloseEvent(service_discovery_event);
        service_context.CloseEvent(config_event);
    }

private:
    void AcquireBleScanEvent(HLERequestContext& ctx) {
        LOG_WARNING(Service_BTM, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 3, 1};
        rb.Push(ResultSuccess);
        rb.Push(true);
        rb.PushCopyObjects(scan_event->GetReadableEvent());
    }

    void AcquireBleConnectionEvent(HLERequestContext& ctx) {
        LOG_WARNING(Service_BTM, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 3, 1};
        rb.Push(ResultSuccess);
        rb.Push(true);
        rb.PushCopyObjects(connection_event->GetReadableEvent());
    }

    void AcquireBleServiceDiscoveryEvent(HLERequestContext& ctx) {
        LOG_WARNING(Service_BTM, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 3, 1};
        rb.Push(ResultSuccess);
        rb.Push(true);
        rb.PushCopyObjects(service_discovery_event->GetReadableEvent());
    }

    void AcquireBleMtuConfigEvent(HLERequestContext& ctx) {
        LOG_WARNING(Service_BTM, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 3, 1};
        rb.Push(ResultSuccess);
        rb.Push(true);
        rb.PushCopyObjects(config_event->GetReadableEvent());
    }

    KernelHelpers::ServiceContext service_context;

    Kernel::KEvent* scan_event;
    Kernel::KEvent* connection_event;
    Kernel::KEvent* service_discovery_event;
    Kernel::KEvent* config_event;
};

class BTM_USR final : public ServiceFramework<BTM_USR> {
public:
    explicit BTM_USR(Core::System& system_) : ServiceFramework{system_, "btm:u"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &BTM_USR::GetCore, "GetCore"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    void GetCore(HLERequestContext& ctx) {
        LOG_DEBUG(Service_BTM, "called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IBtmUserCore>(system);
    }
};

class BTM final : public ServiceFramework<BTM> {
public:
    explicit BTM(Core::System& system_) : ServiceFramework{system_, "btm"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "GetState"},
            {1, nullptr, "GetHostDeviceProperty"},
            {2, nullptr, "AcquireDeviceConditionEvent"},
            {3, nullptr, "GetDeviceCondition"},
            {4, nullptr, "SetBurstMode"},
            {5, nullptr, "SetSlotMode"},
            {6, nullptr, "SetBluetoothMode"},
            {7, nullptr, "SetWlanMode"},
            {8, nullptr, "AcquireDeviceInfoEvent"},
            {9, nullptr, "GetDeviceInfo"},
            {10, nullptr, "AddDeviceInfo"},
            {11, nullptr, "RemoveDeviceInfo"},
            {12, nullptr, "IncreaseDeviceInfoOrder"},
            {13, nullptr, "LlrNotify"},
            {14, nullptr, "EnableRadio"},
            {15, nullptr, "DisableRadio"},
            {16, nullptr, "HidDisconnect"},
            {17, nullptr, "HidSetRetransmissionMode"},
            {18, nullptr, "AcquireAwakeReqEvent"},
            {19, nullptr, "AcquireLlrStateEvent"},
            {20, nullptr, "IsLlrStarted"},
            {21, nullptr, "EnableSlotSaving"},
            {22, nullptr, "ProtectDeviceInfo"},
            {23, nullptr, "AcquireBleScanEvent"},
            {24, nullptr, "GetBleScanParameterGeneral"},
            {25, nullptr, "GetBleScanParameterSmartDevice"},
            {26, nullptr, "StartBleScanForGeneral"},
            {27, nullptr, "StopBleScanForGeneral"},
            {28, nullptr, "GetBleScanResultsForGeneral"},
            {29, nullptr, "StartBleScanForPairedDevice"},
            {30, nullptr, "StopBleScanForPairedDevice"},
            {31, nullptr, "StartBleScanForSmartDevice"},
            {32, nullptr, "StopBleScanForSmartDevice"},
            {33, nullptr, "GetBleScanResultsForSmartDevice"},
            {34, nullptr, "AcquireBleConnectionEvent"},
            {35, nullptr, "BleConnect"},
            {36, nullptr, "BleOverrideConnection"},
            {37, nullptr, "BleDisconnect"},
            {38, nullptr, "BleGetConnectionState"},
            {39, nullptr, "BleGetGattClientConditionList"},
            {40, nullptr, "AcquireBlePairingEvent"},
            {41, nullptr, "BlePairDevice"},
            {42, nullptr, "BleUnpairDeviceOnBoth"},
            {43, nullptr, "BleUnpairDevice"},
            {44, nullptr, "BleGetPairedAddresses"},
            {45, nullptr, "AcquireBleServiceDiscoveryEvent"},
            {46, nullptr, "GetGattServices"},
            {47, nullptr, "GetGattService"},
            {48, nullptr, "GetGattIncludedServices"},
            {49, nullptr, "GetBelongingService"},
            {50, nullptr, "GetGattCharacteristics"},
            {51, nullptr, "GetGattDescriptors"},
            {52, nullptr, "AcquireBleMtuConfigEvent"},
            {53, nullptr, "ConfigureBleMtu"},
            {54, nullptr, "GetBleMtu"},
            {55, nullptr, "RegisterBleGattDataPath"},
            {56, nullptr, "UnregisterBleGattDataPath"},
            {57, nullptr, "RegisterAppletResourceUserId"},
            {58, nullptr, "UnregisterAppletResourceUserId"},
            {59, nullptr, "SetAppletResourceUserId"},
            {60, nullptr, "Unknown60"},
            {61, nullptr, "Unknown61"},
            {62, nullptr, "Unknown62"},
            {63, nullptr, "Unknown63"},
            {64, nullptr, "Unknown64"},
            {65, nullptr, "Unknown65"},
            {66, nullptr, "Unknown66"},
            {67, nullptr, "Unknown67"},
            {68, nullptr, "Unknown68"},
            {69, nullptr, "Unknown69"},
            {70, nullptr, "Unknown70"},
            {71, nullptr, "Unknown71"},
            {72, nullptr, "Unknown72"},
            {73, nullptr, "Unknown73"},
            {74, nullptr, "Unknown74"},
            {75, nullptr, "Unknown75"},
            {76, nullptr, "Unknown76"},
            {100, nullptr, "Unknown100"},
            {101, nullptr, "Unknown101"},
            {110, nullptr, "Unknown110"},
            {111, nullptr, "Unknown111"},
            {112, nullptr, "Unknown112"},
            {113, nullptr, "Unknown113"},
            {114, nullptr, "Unknown114"},
            {115, nullptr, "Unknown115"},
        };
        // clang-format on

        RegisterHandlers(functions);
    }
};

class BTM_DBG final : public ServiceFramework<BTM_DBG> {
public:
    explicit BTM_DBG(Core::System& system_) : ServiceFramework{system_, "btm:dbg"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "AcquireDiscoveryEvent"},
            {1, nullptr, "StartDiscovery"},
            {2, nullptr, "CancelDiscovery"},
            {3, nullptr, "GetDeviceProperty"},
            {4, nullptr, "CreateBond"},
            {5, nullptr, "CancelBond"},
            {6, nullptr, "SetTsiMode"},
            {7, nullptr, "GeneralTest"},
            {8, nullptr, "HidConnect"},
            {9, nullptr, "GeneralGet"},
            {10, nullptr, "GetGattClientDisconnectionReason"},
            {11, nullptr, "GetBleConnectionParameter"},
            {12, nullptr, "GetBleConnectionParameterRequest"},
            {13, nullptr, "Unknown13"},
        };
        // clang-format on

        RegisterHandlers(functions);
    }
};

class IBtmSystemCore final : public ServiceFramework<IBtmSystemCore> {
public:
    explicit IBtmSystemCore(Core::System& system_) : ServiceFramework{system_, "IBtmSystemCore"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "StartGamepadPairing"},
            {1, nullptr, "CancelGamepadPairing"},
            {2, nullptr, "ClearGamepadPairingDatabase"},
            {3, nullptr, "GetPairedGamepadCount"},
            {4, nullptr, "EnableRadio"},
            {5, nullptr, "DisableRadio"},
            {6, nullptr, "GetRadioOnOff"},
            {7, nullptr, "AcquireRadioEvent"},
            {8, nullptr, "AcquireGamepadPairingEvent"},
            {9, nullptr, "IsGamepadPairingStarted"},
            {10, nullptr, "StartAudioDeviceDiscovery"},
            {11, nullptr, "StopAudioDeviceDiscovery"},
            {12, nullptr, "IsDiscoveryingAudioDevice"},
            {13, nullptr, "GetDiscoveredAudioDevice"},
            {14, nullptr, "AcquireAudioDeviceConnectionEvent"},
            {15, nullptr, "ConnectAudioDevice"},
            {16, nullptr, "IsConnectingAudioDevice"},
            {17, nullptr, "GetConnectedAudioDevices"},
            {18, nullptr, "DisconnectAudioDevice"},
            {19, nullptr, "AcquirePairedAudioDeviceInfoChangedEvent"},
            {20, nullptr, "GetPairedAudioDevices"},
            {21, nullptr, "RemoveAudioDevicePairing"},
            {22, nullptr, "RequestAudioDeviceConnectionRejection"},
            {23, nullptr, "CancelAudioDeviceConnectionRejection"}
        };
        // clang-format on

        RegisterHandlers(functions);
    }
};

class BTM_SYS final : public ServiceFramework<BTM_SYS> {
public:
    explicit BTM_SYS(Core::System& system_) : ServiceFramework{system_, "btm:sys"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &BTM_SYS::GetCore, "GetCore"},
        };
        // clang-format on

        RegisterHandlers(functions);
    }

private:
    void GetCore(HLERequestContext& ctx) {
        LOG_DEBUG(Service_BTM, "called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IBtmSystemCore>(system);
    }
};

void LoopProcess(Core::System& system) {
    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService("btm", std::make_shared<BTM>(system));
    server_manager->RegisterNamedService("btm:dbg", std::make_shared<BTM_DBG>(system));
    server_manager->RegisterNamedService("btm:sys", std::make_shared<BTM_SYS>(system));
    server_manager->RegisterNamedService("btm:u", std::make_shared<BTM_USR>(system));
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::BTM
