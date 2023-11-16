// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_funcs.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/nvdrv/devices/nvmap.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "core/hle/service/nvnflinger/binder.h"
#include "core/hle/service/nvnflinger/buffer_queue_producer.h"
#include "core/hle/service/nvnflinger/fb_share_buffer_manager.h"
#include "core/hle/service/nvnflinger/hos_binder_driver_server.h"
#include "core/hle/service/nvnflinger/nvnflinger.h"
#include "core/hle/service/nvnflinger/parcel.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/service.h"
#include "core/hle/service/vi/vi.h"
#include "core/hle/service/vi/vi_m.h"
#include "core/hle/service/vi/vi_results.h"
#include "core/hle/service/vi/vi_s.h"
#include "core/hle/service/vi/vi_u.h"

namespace Service::VI {

struct DisplayInfo {
    /// The name of this particular display.
    char display_name[0x40]{"Default"};

    /// Whether or not the display has a limited number of layers.
    u8 has_limited_layers{1};
    INSERT_PADDING_BYTES(7);

    /// Indicates the total amount of layers supported by the display.
    /// @note This is only valid if has_limited_layers is set.
    u64 max_layers{1};

    /// Maximum width in pixels.
    u64 width{1920};

    /// Maximum height in pixels.
    u64 height{1080};
};
static_assert(sizeof(DisplayInfo) == 0x60, "DisplayInfo has wrong size");

class NativeWindow final {
public:
    constexpr explicit NativeWindow(u32 id_) : id{id_} {}
    constexpr explicit NativeWindow(const NativeWindow& other) = default;

private:
    const u32 magic = 2;
    const u32 process_id = 1;
    const u64 id;
    INSERT_PADDING_WORDS(2);
    std::array<u8, 8> dispdrv = {'d', 'i', 's', 'p', 'd', 'r', 'v', '\0'};
    INSERT_PADDING_WORDS(2);
};
static_assert(sizeof(NativeWindow) == 0x28, "NativeWindow has wrong size");

class IHOSBinderDriver final : public ServiceFramework<IHOSBinderDriver> {
public:
    explicit IHOSBinderDriver(Core::System& system_, Nvnflinger::HosBinderDriverServer& server_)
        : ServiceFramework{system_, "IHOSBinderDriver"}, server(server_) {
        static const FunctionInfo functions[] = {
            {0, &IHOSBinderDriver::TransactParcel, "TransactParcel"},
            {1, &IHOSBinderDriver::AdjustRefcount, "AdjustRefcount"},
            {2, &IHOSBinderDriver::GetNativeHandle, "GetNativeHandle"},
            {3, &IHOSBinderDriver::TransactParcel, "TransactParcelAuto"},
        };
        RegisterHandlers(functions);
    }

private:
    void TransactParcel(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 id = rp.Pop<u32>();
        const auto transaction = static_cast<android::TransactionId>(rp.Pop<u32>());
        const u32 flags = rp.Pop<u32>();

        LOG_DEBUG(Service_VI, "called. id=0x{:08X} transaction={:X}, flags=0x{:08X}", id,
                  transaction, flags);

        server.TryGetProducer(id)->Transact(ctx, transaction, flags);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void AdjustRefcount(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 id = rp.Pop<u32>();
        const s32 addval = rp.PopRaw<s32>();
        const u32 type = rp.Pop<u32>();

        LOG_WARNING(Service_VI, "(STUBBED) called id={}, addval={:08X}, type={:08X}", id, addval,
                    type);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetNativeHandle(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 id = rp.Pop<u32>();
        const u32 unknown = rp.Pop<u32>();

        LOG_WARNING(Service_VI, "(STUBBED) called id={}, unknown={:08X}", id, unknown);

        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(server.TryGetProducer(id)->GetNativeHandle());
    }

private:
    Nvnflinger::HosBinderDriverServer& server;
};

class ISystemDisplayService final : public ServiceFramework<ISystemDisplayService> {
public:
    explicit ISystemDisplayService(Core::System& system_, Nvnflinger::Nvnflinger& nvnflinger_)
        : ServiceFramework{system_, "ISystemDisplayService"}, nvnflinger{nvnflinger_} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {1200, nullptr, "GetZOrderCountMin"},
            {1202, nullptr, "GetZOrderCountMax"},
            {1203, nullptr, "GetDisplayLogicalResolution"},
            {1204, nullptr, "SetDisplayMagnification"},
            {2201, nullptr, "SetLayerPosition"},
            {2203, nullptr, "SetLayerSize"},
            {2204, nullptr, "GetLayerZ"},
            {2205, &ISystemDisplayService::SetLayerZ, "SetLayerZ"},
            {2207, &ISystemDisplayService::SetLayerVisibility, "SetLayerVisibility"},
            {2209, nullptr, "SetLayerAlpha"},
            {2210, nullptr, "SetLayerPositionAndSize"},
            {2312, nullptr, "CreateStrayLayer"},
            {2400, nullptr, "OpenIndirectLayer"},
            {2401, nullptr, "CloseIndirectLayer"},
            {2402, nullptr, "FlipIndirectLayer"},
            {3000, nullptr, "ListDisplayModes"},
            {3001, nullptr, "ListDisplayRgbRanges"},
            {3002, nullptr, "ListDisplayContentTypes"},
            {3200, &ISystemDisplayService::GetDisplayMode, "GetDisplayMode"},
            {3201, nullptr, "SetDisplayMode"},
            {3202, nullptr, "GetDisplayUnderscan"},
            {3203, nullptr, "SetDisplayUnderscan"},
            {3204, nullptr, "GetDisplayContentType"},
            {3205, nullptr, "SetDisplayContentType"},
            {3206, nullptr, "GetDisplayRgbRange"},
            {3207, nullptr, "SetDisplayRgbRange"},
            {3208, nullptr, "GetDisplayCmuMode"},
            {3209, nullptr, "SetDisplayCmuMode"},
            {3210, nullptr, "GetDisplayContrastRatio"},
            {3211, nullptr, "SetDisplayContrastRatio"},
            {3214, nullptr, "GetDisplayGamma"},
            {3215, nullptr, "SetDisplayGamma"},
            {3216, nullptr, "GetDisplayCmuLuma"},
            {3217, nullptr, "SetDisplayCmuLuma"},
            {3218, nullptr, "SetDisplayCrcMode"},
            {6013, nullptr, "GetLayerPresentationSubmissionTimestamps"},
            {8225, &ISystemDisplayService::GetSharedBufferMemoryHandleId, "GetSharedBufferMemoryHandleId"},
            {8250, &ISystemDisplayService::OpenSharedLayer, "OpenSharedLayer"},
            {8251, nullptr, "CloseSharedLayer"},
            {8252, &ISystemDisplayService::ConnectSharedLayer, "ConnectSharedLayer"},
            {8253, nullptr, "DisconnectSharedLayer"},
            {8254, &ISystemDisplayService::AcquireSharedFrameBuffer, "AcquireSharedFrameBuffer"},
            {8255, &ISystemDisplayService::PresentSharedFrameBuffer, "PresentSharedFrameBuffer"},
            {8256, &ISystemDisplayService::GetSharedFrameBufferAcquirableEvent, "GetSharedFrameBufferAcquirableEvent"},
            {8257, nullptr, "FillSharedFrameBufferColor"},
            {8258, nullptr, "CancelSharedFrameBuffer"},
            {9000, nullptr, "GetDp2hdmiController"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    void GetSharedBufferMemoryHandleId(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 buffer_id = rp.PopRaw<u64>();

        LOG_INFO(Service_VI, "called. buffer_id={:#x}", buffer_id);

        struct OutputParameters {
            s32 nvmap_handle;
            u64 size;
        };

        OutputParameters out{};
        Nvnflinger::SharedMemoryPoolLayout layout{};
        const auto result = nvnflinger.GetSystemBufferManager().GetSharedBufferMemoryHandleId(
            &out.size, &out.nvmap_handle, &layout, buffer_id, 0);

        ctx.WriteBuffer(&layout, sizeof(layout));

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(result);
        rb.PushRaw(out);
    }

    void OpenSharedLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.PopRaw<u64>();

        LOG_INFO(Service_VI, "(STUBBED) called. layer_id={:#x}", layer_id);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void ConnectSharedLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.PopRaw<u64>();

        LOG_INFO(Service_VI, "(STUBBED) called. layer_id={:#x}", layer_id);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetSharedFrameBufferAcquirableEvent(HLERequestContext& ctx) {
        LOG_DEBUG(Service_VI, "called");

        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.PopRaw<u64>();

        Kernel::KReadableEvent* event{};
        const auto result = nvnflinger.GetSystemBufferManager().GetSharedFrameBufferAcquirableEvent(
            &event, layer_id);

        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(result);
        rb.PushCopyObjects(event);
    }

    void AcquireSharedFrameBuffer(HLERequestContext& ctx) {
        LOG_DEBUG(Service_VI, "called");

        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.PopRaw<u64>();

        struct OutputParameters {
            android::Fence fence;
            std::array<s32, 4> slots;
            s64 target_slot;
        };
        static_assert(sizeof(OutputParameters) == 0x40, "OutputParameters has wrong size");

        OutputParameters out{};
        const auto result = nvnflinger.GetSystemBufferManager().AcquireSharedFrameBuffer(
            &out.fence, out.slots, &out.target_slot, layer_id);

        IPC::ResponseBuilder rb{ctx, 18};
        rb.Push(result);
        rb.PushRaw(out);
    }

    void PresentSharedFrameBuffer(HLERequestContext& ctx) {
        LOG_DEBUG(Service_VI, "called");

        struct InputParameters {
            android::Fence fence;
            Common::Rectangle<s32> crop_region;
            u32 window_transform;
            s32 swap_interval;
            u64 layer_id;
            s64 surface_id;
        };
        static_assert(sizeof(InputParameters) == 0x50, "InputParameters has wrong size");

        IPC::RequestParser rp{ctx};
        auto input = rp.PopRaw<InputParameters>();

        const auto result = nvnflinger.GetSystemBufferManager().PresentSharedFrameBuffer(
            input.fence, input.crop_region, input.window_transform, input.swap_interval,
            input.layer_id, input.surface_id);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result);
    }

    void SetLayerZ(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.Pop<u64>();
        const u64 z_value = rp.Pop<u64>();

        LOG_WARNING(Service_VI, "(STUBBED) called. layer_id=0x{:016X}, z_value=0x{:016X}", layer_id,
                    z_value);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    // This function currently does nothing but return a success error code in
    // the vi library itself, so do the same thing, but log out the passed in values.
    void SetLayerVisibility(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.Pop<u64>();
        const bool visibility = rp.Pop<bool>();

        LOG_DEBUG(Service_VI, "called, layer_id=0x{:08X}, visibility={}", layer_id, visibility);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetDisplayMode(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(ResultSuccess);

        if (Settings::IsDockedMode()) {
            rb.Push(static_cast<u32>(Service::VI::DisplayResolution::DockedWidth));
            rb.Push(static_cast<u32>(Service::VI::DisplayResolution::DockedHeight));
        } else {
            rb.Push(static_cast<u32>(Service::VI::DisplayResolution::UndockedWidth));
            rb.Push(static_cast<u32>(Service::VI::DisplayResolution::UndockedHeight));
        }

        rb.PushRaw<float>(60.0f); // This wouldn't seem to be correct for 30 fps games.
        rb.Push<u32>(0);
    }

private:
    Nvnflinger::Nvnflinger& nvnflinger;
};

class IManagerDisplayService final : public ServiceFramework<IManagerDisplayService> {
public:
    explicit IManagerDisplayService(Core::System& system_, Nvnflinger::Nvnflinger& nv_flinger_)
        : ServiceFramework{system_, "IManagerDisplayService"}, nv_flinger{nv_flinger_} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {200, nullptr, "AllocateProcessHeapBlock"},
            {201, nullptr, "FreeProcessHeapBlock"},
            {1020, &IManagerDisplayService::CloseDisplay, "CloseDisplay"},
            {1102, nullptr, "GetDisplayResolution"},
            {2010, &IManagerDisplayService::CreateManagedLayer, "CreateManagedLayer"},
            {2011, nullptr, "DestroyManagedLayer"},
            {2012, nullptr, "CreateStrayLayer"},
            {2050, nullptr, "CreateIndirectLayer"},
            {2051, nullptr, "DestroyIndirectLayer"},
            {2052, nullptr, "CreateIndirectProducerEndPoint"},
            {2053, nullptr, "DestroyIndirectProducerEndPoint"},
            {2054, nullptr, "CreateIndirectConsumerEndPoint"},
            {2055, nullptr, "DestroyIndirectConsumerEndPoint"},
            {2060, nullptr, "CreateWatermarkCompositor"},
            {2062, nullptr, "SetWatermarkText"},
            {2063, nullptr, "SetWatermarkLayerStacks"},
            {2300, nullptr, "AcquireLayerTexturePresentingEvent"},
            {2301, nullptr, "ReleaseLayerTexturePresentingEvent"},
            {2302, nullptr, "GetDisplayHotplugEvent"},
            {2303, nullptr, "GetDisplayModeChangedEvent"},
            {2402, nullptr, "GetDisplayHotplugState"},
            {2501, nullptr, "GetCompositorErrorInfo"},
            {2601, nullptr, "GetDisplayErrorEvent"},
            {2701, nullptr, "GetDisplayFatalErrorEvent"},
            {4201, nullptr, "SetDisplayAlpha"},
            {4203, nullptr, "SetDisplayLayerStack"},
            {4205, nullptr, "SetDisplayPowerState"},
            {4206, nullptr, "SetDefaultDisplay"},
            {4207, nullptr, "ResetDisplayPanel"},
            {4208, nullptr, "SetDisplayFatalErrorEnabled"},
            {4209, nullptr, "IsDisplayPanelOn"},
            {4300, nullptr, "GetInternalPanelId"},
            {6000, &IManagerDisplayService::AddToLayerStack, "AddToLayerStack"},
            {6001, nullptr, "RemoveFromLayerStack"},
            {6002, &IManagerDisplayService::SetLayerVisibility, "SetLayerVisibility"},
            {6003, nullptr, "SetLayerConfig"},
            {6004, nullptr, "AttachLayerPresentationTracer"},
            {6005, nullptr, "DetachLayerPresentationTracer"},
            {6006, nullptr, "StartLayerPresentationRecording"},
            {6007, nullptr, "StopLayerPresentationRecording"},
            {6008, nullptr, "StartLayerPresentationFenceWait"},
            {6009, nullptr, "StopLayerPresentationFenceWait"},
            {6010, nullptr, "GetLayerPresentationAllFencesExpiredEvent"},
            {6011, nullptr, "EnableLayerAutoClearTransitionBuffer"},
            {6012, nullptr, "DisableLayerAutoClearTransitionBuffer"},
            {6013, nullptr, "SetLayerOpacity"},
            {6014, nullptr, "AttachLayerWatermarkCompositor"},
            {6015, nullptr, "DetachLayerWatermarkCompositor"},
            {7000, nullptr, "SetContentVisibility"},
            {8000, nullptr, "SetConductorLayer"},
            {8001, nullptr, "SetTimestampTracking"},
            {8100, nullptr, "SetIndirectProducerFlipOffset"},
            {8200, nullptr, "CreateSharedBufferStaticStorage"},
            {8201, nullptr, "CreateSharedBufferTransferMemory"},
            {8202, nullptr, "DestroySharedBuffer"},
            {8203, nullptr, "BindSharedLowLevelLayerToManagedLayer"},
            {8204, nullptr, "BindSharedLowLevelLayerToIndirectLayer"},
            {8207, nullptr, "UnbindSharedLowLevelLayer"},
            {8208, nullptr, "ConnectSharedLowLevelLayerToSharedBuffer"},
            {8209, nullptr, "DisconnectSharedLowLevelLayerFromSharedBuffer"},
            {8210, nullptr, "CreateSharedLayer"},
            {8211, nullptr, "DestroySharedLayer"},
            {8216, nullptr, "AttachSharedLayerToLowLevelLayer"},
            {8217, nullptr, "ForceDetachSharedLayerFromLowLevelLayer"},
            {8218, nullptr, "StartDetachSharedLayerFromLowLevelLayer"},
            {8219, nullptr, "FinishDetachSharedLayerFromLowLevelLayer"},
            {8220, nullptr, "GetSharedLayerDetachReadyEvent"},
            {8221, nullptr, "GetSharedLowLevelLayerSynchronizedEvent"},
            {8222, nullptr, "CheckSharedLowLevelLayerSynchronized"},
            {8223, nullptr, "RegisterSharedBufferImporterAruid"},
            {8224, nullptr, "UnregisterSharedBufferImporterAruid"},
            {8227, nullptr, "CreateSharedBufferProcessHeap"},
            {8228, nullptr, "GetSharedLayerLayerStacks"},
            {8229, nullptr, "SetSharedLayerLayerStacks"},
            {8291, nullptr, "PresentDetachedSharedFrameBufferToLowLevelLayer"},
            {8292, nullptr, "FillDetachedSharedFrameBufferColor"},
            {8293, nullptr, "GetDetachedSharedFrameBufferImage"},
            {8294, nullptr, "SetDetachedSharedFrameBufferImage"},
            {8295, nullptr, "CopyDetachedSharedFrameBufferImage"},
            {8296, nullptr, "SetDetachedSharedFrameBufferSubImage"},
            {8297, nullptr, "GetSharedFrameBufferContentParameter"},
            {8298, nullptr, "ExpandStartupLogoOnSharedFrameBuffer"},
        };
        // clang-format on

        RegisterHandlers(functions);
    }

private:
    void CloseDisplay(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 display = rp.Pop<u64>();

        const Result rc = nv_flinger.CloseDisplay(display) ? ResultSuccess : ResultUnknown;

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(rc);
    }

    void CreateManagedLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 unknown = rp.Pop<u32>();
        rp.Skip(1, false);
        const u64 display = rp.Pop<u64>();
        const u64 aruid = rp.Pop<u64>();

        LOG_WARNING(Service_VI,
                    "(STUBBED) called. unknown=0x{:08X}, display=0x{:016X}, aruid=0x{:016X}",
                    unknown, display, aruid);

        const auto layer_id = nv_flinger.CreateLayer(display);
        if (!layer_id) {
            LOG_ERROR(Service_VI, "Layer not found! display=0x{:016X}", display);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess);
        rb.Push(*layer_id);
    }

    void AddToLayerStack(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 stack = rp.Pop<u32>();
        const u64 layer_id = rp.Pop<u64>();

        LOG_WARNING(Service_VI, "(STUBBED) called. stack=0x{:08X}, layer_id=0x{:016X}", stack,
                    layer_id);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void SetLayerVisibility(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.Pop<u64>();
        const bool visibility = rp.Pop<bool>();

        LOG_WARNING(Service_VI, "(STUBBED) called, layer_id=0x{:X}, visibility={}", layer_id,
                    visibility);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    Nvnflinger::Nvnflinger& nv_flinger;
};

class IApplicationDisplayService final : public ServiceFramework<IApplicationDisplayService> {
public:
    IApplicationDisplayService(Core::System& system_, Nvnflinger::Nvnflinger& nv_flinger_,
                               Nvnflinger::HosBinderDriverServer& hos_binder_driver_server_)
        : ServiceFramework{system_, "IApplicationDisplayService"}, nv_flinger{nv_flinger_},
          hos_binder_driver_server{hos_binder_driver_server_} {

        static const FunctionInfo functions[] = {
            {100, &IApplicationDisplayService::GetRelayService, "GetRelayService"},
            {101, &IApplicationDisplayService::GetSystemDisplayService, "GetSystemDisplayService"},
            {102, &IApplicationDisplayService::GetManagerDisplayService,
             "GetManagerDisplayService"},
            {103, &IApplicationDisplayService::GetIndirectDisplayTransactionService,
             "GetIndirectDisplayTransactionService"},
            {1000, &IApplicationDisplayService::ListDisplays, "ListDisplays"},
            {1010, &IApplicationDisplayService::OpenDisplay, "OpenDisplay"},
            {1011, &IApplicationDisplayService::OpenDefaultDisplay, "OpenDefaultDisplay"},
            {1020, &IApplicationDisplayService::CloseDisplay, "CloseDisplay"},
            {1101, &IApplicationDisplayService::SetDisplayEnabled, "SetDisplayEnabled"},
            {1102, &IApplicationDisplayService::GetDisplayResolution, "GetDisplayResolution"},
            {2020, &IApplicationDisplayService::OpenLayer, "OpenLayer"},
            {2021, &IApplicationDisplayService::CloseLayer, "CloseLayer"},
            {2030, &IApplicationDisplayService::CreateStrayLayer, "CreateStrayLayer"},
            {2031, &IApplicationDisplayService::DestroyStrayLayer, "DestroyStrayLayer"},
            {2101, &IApplicationDisplayService::SetLayerScalingMode, "SetLayerScalingMode"},
            {2102, &IApplicationDisplayService::ConvertScalingMode, "ConvertScalingMode"},
            {2450, &IApplicationDisplayService::GetIndirectLayerImageMap,
             "GetIndirectLayerImageMap"},
            {2451, nullptr, "GetIndirectLayerImageCropMap"},
            {2460, &IApplicationDisplayService::GetIndirectLayerImageRequiredMemoryInfo,
             "GetIndirectLayerImageRequiredMemoryInfo"},
            {5202, &IApplicationDisplayService::GetDisplayVsyncEvent, "GetDisplayVsyncEvent"},
            {5203, nullptr, "GetDisplayVsyncEventForDebug"},
        };
        RegisterHandlers(functions);
    }

private:
    enum class ConvertedScaleMode : u64 {
        Freeze = 0,
        ScaleToWindow = 1,
        ScaleAndCrop = 2,
        None = 3,
        PreserveAspectRatio = 4,
    };

    enum class NintendoScaleMode : u32 {
        None = 0,
        Freeze = 1,
        ScaleToWindow = 2,
        ScaleAndCrop = 3,
        PreserveAspectRatio = 4,
    };

    void GetRelayService(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IHOSBinderDriver>(system, hos_binder_driver_server);
    }

    void GetSystemDisplayService(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<ISystemDisplayService>(system, nv_flinger);
    }

    void GetManagerDisplayService(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IManagerDisplayService>(system, nv_flinger);
    }

    void GetIndirectDisplayTransactionService(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IHOSBinderDriver>(system, hos_binder_driver_server);
    }

    void OpenDisplay(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        IPC::RequestParser rp{ctx};
        const auto name_buf = rp.PopRaw<std::array<char, 0x40>>();

        OpenDisplayImpl(ctx, std::string_view{name_buf.data(), name_buf.size()});
    }

    void OpenDefaultDisplay(HLERequestContext& ctx) {
        LOG_DEBUG(Service_VI, "called");

        OpenDisplayImpl(ctx, "Default");
    }

    void OpenDisplayImpl(HLERequestContext& ctx, std::string_view name) {
        const auto trim_pos = name.find('\0');

        if (trim_pos != std::string_view::npos) {
            name.remove_suffix(name.size() - trim_pos);
        }

        ASSERT_MSG(name == "Default", "Non-default displays aren't supported yet");

        const auto display_id = nv_flinger.OpenDisplay(name);
        if (!display_id) {
            LOG_ERROR(Service_VI, "Display not found! display_name={}", name);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess);
        rb.Push<u64>(*display_id);
    }

    void CloseDisplay(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 display_id = rp.Pop<u64>();

        const Result rc = nv_flinger.CloseDisplay(display_id) ? ResultSuccess : ResultUnknown;

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(rc);
    }

    // This literally does nothing internally in the actual service itself,
    // and just returns a successful result code regardless of the input.
    void SetDisplayEnabled(HLERequestContext& ctx) {
        LOG_DEBUG(Service_VI, "called.");

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetDisplayResolution(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 display_id = rp.Pop<u64>();

        LOG_DEBUG(Service_VI, "called. display_id=0x{:016X}", display_id);

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(ResultSuccess);

        // This only returns the fixed values of 1280x720 and makes no distinguishing
        // between docked and undocked dimensions. We take the liberty of applying
        // the resolution scaling factor here.
        rb.Push(static_cast<u64>(DisplayResolution::UndockedWidth));
        rb.Push(static_cast<u64>(DisplayResolution::UndockedHeight));
    }

    void SetLayerScalingMode(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto scaling_mode = rp.PopEnum<NintendoScaleMode>();
        const u64 unknown = rp.Pop<u64>();

        LOG_DEBUG(Service_VI, "called. scaling_mode=0x{:08X}, unknown=0x{:016X}", scaling_mode,
                  unknown);

        IPC::ResponseBuilder rb{ctx, 2};

        if (scaling_mode > NintendoScaleMode::PreserveAspectRatio) {
            LOG_ERROR(Service_VI, "Invalid scaling mode provided.");
            rb.Push(ResultOperationFailed);
            return;
        }

        if (scaling_mode != NintendoScaleMode::ScaleToWindow &&
            scaling_mode != NintendoScaleMode::PreserveAspectRatio) {
            LOG_ERROR(Service_VI, "Unsupported scaling mode supplied.");
            rb.Push(ResultNotSupported);
            return;
        }

        rb.Push(ResultSuccess);
    }

    void ListDisplays(HLERequestContext& ctx) {
        LOG_WARNING(Service_VI, "(STUBBED) called");

        const DisplayInfo display_info;
        ctx.WriteBuffer(&display_info, sizeof(DisplayInfo));
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess);
        rb.Push<u64>(1);
    }

    void OpenLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto name_buf = rp.PopRaw<std::array<u8, 0x40>>();
        const auto end = std::find(name_buf.begin(), name_buf.end(), '\0');

        const std::string display_name(name_buf.begin(), end);

        const u64 layer_id = rp.Pop<u64>();
        const u64 aruid = rp.Pop<u64>();

        LOG_DEBUG(Service_VI, "called. layer_id=0x{:016X}, aruid=0x{:016X}", layer_id, aruid);

        const auto display_id = nv_flinger.OpenDisplay(display_name);
        if (!display_id) {
            LOG_ERROR(Service_VI, "Layer not found! layer_id={}", layer_id);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        const auto buffer_queue_id = nv_flinger.FindBufferQueueId(*display_id, layer_id);
        if (!buffer_queue_id) {
            LOG_ERROR(Service_VI, "Buffer queue id not found! display_id={}", *display_id);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        android::OutputParcel parcel;
        parcel.WriteInterface(NativeWindow{*buffer_queue_id});

        const auto buffer_size = ctx.WriteBuffer(parcel.Serialize());

        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(ResultSuccess);
        rb.Push<u64>(buffer_size);
    }

    void CloseLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto layer_id{rp.Pop<u64>()};

        LOG_DEBUG(Service_VI, "called. layer_id=0x{:016X}", layer_id);

        nv_flinger.CloseLayer(layer_id);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void CreateStrayLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 flags = rp.Pop<u32>();
        rp.Pop<u32>(); // padding
        const u64 display_id = rp.Pop<u64>();

        LOG_DEBUG(Service_VI, "called. flags=0x{:08X}, display_id=0x{:016X}", flags, display_id);

        // TODO(Subv): What's the difference between a Stray and a Managed layer?

        const auto layer_id = nv_flinger.CreateLayer(display_id);
        if (!layer_id) {
            LOG_ERROR(Service_VI, "Layer not found! display_id={}", display_id);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        const auto buffer_queue_id = nv_flinger.FindBufferQueueId(display_id, *layer_id);
        if (!buffer_queue_id) {
            LOG_ERROR(Service_VI, "Buffer queue id not found! display_id={}", display_id);
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(ResultNotFound);
            return;
        }

        android::OutputParcel parcel;
        parcel.WriteInterface(NativeWindow{*buffer_queue_id});

        const auto buffer_size = ctx.WriteBuffer(parcel.Serialize());

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(ResultSuccess);
        rb.Push(*layer_id);
        rb.Push<u64>(buffer_size);
    }

    void DestroyStrayLayer(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 layer_id = rp.Pop<u64>();

        LOG_WARNING(Service_VI, "(STUBBED) called. layer_id=0x{:016X}", layer_id);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetDisplayVsyncEvent(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u64 display_id = rp.Pop<u64>();

        LOG_DEBUG(Service_VI, "called. display_id={}", display_id);

        Kernel::KReadableEvent* vsync_event{};
        const auto result = nv_flinger.FindVsyncEvent(&vsync_event, display_id);
        if (result != ResultSuccess) {
            if (result == ResultNotFound) {
                LOG_ERROR(Service_VI, "Vsync event was not found for display_id={}", display_id);
            }

            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(result);
            return;
        }

        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(vsync_event);
    }

    void ConvertScalingMode(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto mode = rp.PopEnum<NintendoScaleMode>();
        LOG_DEBUG(Service_VI, "called mode={}", mode);

        ConvertedScaleMode converted_mode{};
        const auto result = ConvertScalingModeImpl(&converted_mode, mode);

        if (result == ResultSuccess) {
            IPC::ResponseBuilder rb{ctx, 4};
            rb.Push(ResultSuccess);
            rb.PushEnum(converted_mode);
        } else {
            IPC::ResponseBuilder rb{ctx, 2};
            rb.Push(result);
        }
    }

    void GetIndirectLayerImageMap(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto width = rp.Pop<s64>();
        const auto height = rp.Pop<s64>();
        const auto indirect_layer_consumer_handle = rp.Pop<u64>();
        const auto applet_resource_user_id = rp.Pop<u64>();

        LOG_WARNING(Service_VI,
                    "(STUBBED) called, width={}, height={}, indirect_layer_consumer_handle={}, "
                    "applet_resource_user_id={}",
                    width, height, indirect_layer_consumer_handle, applet_resource_user_id);

        std::vector<u8> out_buffer(0x46);
        ctx.WriteBuffer(out_buffer);

        // TODO: Figure out what these are

        constexpr s64 unknown_result_1 = 0;
        constexpr s64 unknown_result_2 = 0;

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(unknown_result_1);
        rb.Push(unknown_result_2);
        rb.Push(ResultSuccess);
    }

    void GetIndirectLayerImageRequiredMemoryInfo(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto width = rp.Pop<u64>();
        const auto height = rp.Pop<u64>();
        LOG_DEBUG(Service_VI, "called width={}, height={}", width, height);

        constexpr u64 base_size = 0x20000;
        constexpr u64 alignment = 0x1000;
        const auto texture_size = width * height * 4;
        const auto out_size = (texture_size + base_size - 1) / base_size * base_size;

        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(ResultSuccess);
        rb.Push(out_size);
        rb.Push(alignment);
    }

    static Result ConvertScalingModeImpl(ConvertedScaleMode* out_scaling_mode,
                                         NintendoScaleMode mode) {
        switch (mode) {
        case NintendoScaleMode::None:
            *out_scaling_mode = ConvertedScaleMode::None;
            return ResultSuccess;
        case NintendoScaleMode::Freeze:
            *out_scaling_mode = ConvertedScaleMode::Freeze;
            return ResultSuccess;
        case NintendoScaleMode::ScaleToWindow:
            *out_scaling_mode = ConvertedScaleMode::ScaleToWindow;
            return ResultSuccess;
        case NintendoScaleMode::ScaleAndCrop:
            *out_scaling_mode = ConvertedScaleMode::ScaleAndCrop;
            return ResultSuccess;
        case NintendoScaleMode::PreserveAspectRatio:
            *out_scaling_mode = ConvertedScaleMode::PreserveAspectRatio;
            return ResultSuccess;
        default:
            LOG_ERROR(Service_VI, "Invalid scaling mode specified, mode={}", mode);
            return ResultOperationFailed;
        }
    }

    Nvnflinger::Nvnflinger& nv_flinger;
    Nvnflinger::HosBinderDriverServer& hos_binder_driver_server;
};

static bool IsValidServiceAccess(Permission permission, Policy policy) {
    if (permission == Permission::User) {
        return policy == Policy::User;
    }

    if (permission == Permission::System || permission == Permission::Manager) {
        return policy == Policy::User || policy == Policy::Compositor;
    }

    return false;
}

void detail::GetDisplayServiceImpl(HLERequestContext& ctx, Core::System& system,
                                   Nvnflinger::Nvnflinger& nv_flinger,
                                   Nvnflinger::HosBinderDriverServer& hos_binder_driver_server,
                                   Permission permission) {
    IPC::RequestParser rp{ctx};
    const auto policy = rp.PopEnum<Policy>();

    if (!IsValidServiceAccess(permission, policy)) {
        LOG_ERROR(Service_VI, "Permission denied for policy {}", policy);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultPermissionDenied);
        return;
    }

    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(ResultSuccess);
    rb.PushIpcInterface<IApplicationDisplayService>(system, nv_flinger, hos_binder_driver_server);
}

void LoopProcess(Core::System& system, Nvnflinger::Nvnflinger& nv_flinger,
                 Nvnflinger::HosBinderDriverServer& hos_binder_driver_server) {
    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService(
        "vi:m", std::make_shared<VI_M>(system, nv_flinger, hos_binder_driver_server));
    server_manager->RegisterNamedService(
        "vi:s", std::make_shared<VI_S>(system, nv_flinger, hos_binder_driver_server));
    server_manager->RegisterNamedService(
        "vi:u", std::make_shared<VI_U>(system, nv_flinger, hos_binder_driver_server));
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::VI
