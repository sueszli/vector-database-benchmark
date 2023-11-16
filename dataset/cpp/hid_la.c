#include <string.h>
#include "libapplet_internal.h"
#include "applets/hid_la.h"
#include "runtime/hosversion.h"

static Result _hidLaShow(const HidLaControllerSupportArgPrivate *private_arg, const void* arg, size_t arg_size, void* reply, size_t reply_size) {
    Result rc=0;
    LibAppletArgs commonargs;
    AppletHolder holder;
    u32 version=0x3; // [1.0.0+]

    if (hosversionAtLeast(11,0,0))
        version = 0x8;
    else if (hosversionAtLeast(8,0,0))
        version = 0x7;
    else if (hosversionAtLeast(6,0,0))
        version = 0x5;
    else if (hosversionAtLeast(3,0,0))
        version = 0x4;

    rc = appletCreateLibraryApplet(&holder, AppletId_LibraryAppletController, LibAppletMode_AllForeground);
    if (R_FAILED(rc)) return rc;

    libappletArgsCreate(&commonargs, version);
    libappletArgsSetPlayStartupSound(&commonargs, (private_arg->flag1!=0) && (private_arg->flag0!=0));

    if (R_SUCCEEDED(rc)) rc = libappletArgsPush(&commonargs, &holder);
    if (R_SUCCEEDED(rc)) rc = libappletPushInData(&holder, private_arg, sizeof(*private_arg));
    if (R_SUCCEEDED(rc)) rc = libappletPushInData(&holder, arg, arg_size);
    if (R_SUCCEEDED(rc)) rc = libappletStart(&holder);
    if (R_SUCCEEDED(rc)) libappletPopOutData(&holder, reply, reply_size, NULL); // Official sw ignores the rc/transfer_size.
    appletHolderClose(&holder);

    return rc;
}

static size_t _hidLaGetControllerSupportArgSize(void) {
    size_t arg_size = sizeof(HidLaControllerSupportArg);
    if (hosversionBefore(8,0,0)) arg_size = sizeof(HidLaControllerSupportArgV3);

    return arg_size;
}

static Result _hidLaShowControllerSupportCore(HidLaControllerSupportResultInfo *result_info, const HidLaControllerSupportArg *arg, const HidLaControllerSupportArgPrivate *private_arg) {
    Result rc=0;
    HidLaControllerSupportResultInfoInternal res={0};
    HidLaControllerSupportArgV3 arg_v3;
    const void* arg_ptr = arg;
    size_t arg_size = _hidLaGetControllerSupportArgSize();

    if (private_arg->mode == HidLaControllerSupportMode_ShowControllerFirmwareUpdate)
        return MAKERESULT(Module_Libnx, LibnxError_BadInput);

    if (hosversionBefore(8,0,0)) {
        memset(&arg_v3, 0, sizeof(arg_v3));
        arg_ptr = &arg_v3;

        memcpy(&arg_v3.hdr, &arg->hdr, sizeof(arg->hdr));
        memcpy(arg_v3.identification_color, arg->identification_color, sizeof(arg_v3.identification_color));
        arg_v3.enable_explain_text = arg->enable_explain_text;
        memcpy(arg_v3.explain_text, arg->explain_text, sizeof(arg_v3.explain_text));

        if (arg_v3.hdr.player_count_min > 4) arg_v3.hdr.player_count_min = 4;
        if (arg_v3.hdr.player_count_max > 4) arg_v3.hdr.player_count_max = 4;
    }

    rc = _hidLaShow(private_arg, arg_ptr, arg_size, &res, sizeof(res));
    if (R_SUCCEEDED(rc)) {
        if (result_info) {
            *result_info = res.info;
            result_info->selected_id = result_info->selected_id;
        }

        if (res.res != 0) {
            rc = MAKERESULT(Module_Libnx, LibnxError_LibAppletBadExit); // Official sw would return different values for 2/{other values}, but we won't do so.
        }
    }

    return rc;
}

static Result _hidLaShowControllerFirmwareUpdateCore(const HidLaControllerFirmwareUpdateArg *arg, const HidLaControllerSupportArgPrivate *private_arg) {
    Result rc=0;
    HidLaControllerSupportResultInfoInternal res={0};

    if (hosversionBefore(3,0,0)) return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    if (private_arg->mode != HidLaControllerSupportMode_ShowControllerFirmwareUpdate)
        return MAKERESULT(Module_Libnx, LibnxError_BadInput);

    rc = _hidLaShow(private_arg, arg, sizeof(*arg), &res, sizeof(res));
    if (R_SUCCEEDED(rc)) {
        if (res.res != 0) {
            rc = MAKERESULT(Module_Libnx, LibnxError_LibAppletBadExit);
        }
    }

    return rc;
}

static Result _hidLaShowControllerKeyRemappingCore(const HidLaControllerKeyRemappingArg *arg, const HidLaControllerSupportArgPrivate *private_arg) {
    Result rc=0;
    HidLaControllerSupportResultInfoInternal res={0};

    if (hosversionBefore(11,0,0)) return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    if (private_arg->mode != HidLaControllerSupportMode_ShowControllerKeyRemappingForSystem)
        return MAKERESULT(Module_Libnx, LibnxError_BadInput);

    rc = _hidLaShow(private_arg, arg, sizeof(*arg), &res, sizeof(res));
    if (R_SUCCEEDED(rc)) {
        if (res.res != 0) {
            rc = MAKERESULT(Module_Libnx, LibnxError_LibAppletBadExit);
        }
    }

    return rc;
}

static Result _hidLaSetupControllerSupportArgPrivate(HidLaControllerSupportArgPrivate *private_arg) {
    Result rc=0;
    u32 style_set;
    HidNpadJoyHoldType hold_type;

    rc = hidGetSupportedNpadStyleSet(&style_set);
    if (R_SUCCEEDED(rc)) rc = hidGetNpadJoyHoldType(&hold_type);

    if (R_SUCCEEDED(rc)) {
        private_arg->npad_style_set = style_set;
        private_arg->npad_joy_hold_type = hold_type;
    }

    return rc;
}

void hidLaCreateControllerSupportArg(HidLaControllerSupportArg *arg) {
    memset(arg, 0, sizeof(*arg));
    arg->hdr.player_count_min = 0;
    arg->hdr.player_count_max = 4;
    arg->hdr.enable_take_over_connection = 1;
    arg->hdr.enable_left_justify = 1;
    arg->hdr.enable_permit_joy_dual = 1;
}

void hidLaCreateControllerFirmwareUpdateArg(HidLaControllerFirmwareUpdateArg *arg) {
    memset(arg, 0, sizeof(*arg));
}

void hidLaCreateControllerKeyRemappingArg(HidLaControllerKeyRemappingArg *arg) {
    memset(arg, 0, sizeof(*arg));
}

Result hidLaSetExplainText(HidLaControllerSupportArg *arg, const char *str, HidNpadIdType id) {
    if (id >= 8)
        return MAKERESULT(Module_Libnx, LibnxError_BadInput);

    memset(arg->explain_text[id], 0, sizeof(arg->explain_text[id]));
    strncpy(arg->explain_text[id], str, sizeof(arg->explain_text[id])-1);

    return 0;
}

Result hidLaShowControllerSupport(HidLaControllerSupportResultInfo *result_info, const HidLaControllerSupportArg *arg) {
    Result rc=0;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(), .mode = HidLaControllerSupportMode_ShowControllerSupport
    };

    rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerSupportCore(result_info, arg, &private_arg);

    return rc;
}

Result hidLaShowControllerStrapGuide(void) {
    Result rc=0;
    HidLaControllerSupportArg arg;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(), .mode = HidLaControllerSupportMode_ShowControllerStrapGuide
    };

    if (hosversionBefore(3,0,0)) return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    hidLaCreateControllerSupportArg(&arg);

    rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerSupportCore(NULL, &arg, &private_arg);

    return rc;
}

Result hidLaShowControllerFirmwareUpdate(const HidLaControllerFirmwareUpdateArg *arg) {
    Result rc=0;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(), .mode = HidLaControllerSupportMode_ShowControllerFirmwareUpdate
    };

    rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerFirmwareUpdateCore(arg, &private_arg);

    return rc;
}

Result hidLaShowControllerSupportForSystem(HidLaControllerSupportResultInfo *result_info, const HidLaControllerSupportArg *arg, bool flag) {
    Result rc=0;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(),
        .flag0 = flag!=0, .flag1 = 1, .mode = HidLaControllerSupportMode_ShowControllerSupport
    };

    if (hosversionAtLeast(3,0,0)) {
        rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);
    }
    else {
        private_arg.npad_style_set = 0;
        private_arg.npad_joy_hold_type = HidNpadJoyHoldType_Horizontal;
    }

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerSupportCore(result_info, arg, &private_arg);

    return rc;
}

Result hidLaShowControllerFirmwareUpdateForSystem(const HidLaControllerFirmwareUpdateArg *arg, HidLaControllerSupportCaller caller) {
    Result rc=0;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(),
        .flag1 = 1, .mode = HidLaControllerSupportMode_ShowControllerFirmwareUpdate, .controller_support_caller = caller
    };

    rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerFirmwareUpdateCore(arg, &private_arg);

    return rc;
}

Result hidLaShowControllerKeyRemappingForSystem(const HidLaControllerKeyRemappingArg *arg, HidLaControllerSupportCaller caller) {
    Result rc=0;
    HidLaControllerSupportArgPrivate private_arg = {
        .private_size = sizeof(private_arg), .arg_size = _hidLaGetControllerSupportArgSize(),
        .flag1 = 1, .mode = HidLaControllerSupportMode_ShowControllerKeyRemappingForSystem, .controller_support_caller = caller
    };

    rc = _hidLaSetupControllerSupportArgPrivate(&private_arg);

    if (R_SUCCEEDED(rc)) rc = _hidLaShowControllerKeyRemappingCore(arg, &private_arg);

    return rc;
}

