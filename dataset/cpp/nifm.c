#include <string.h>
#include "service_guard.h"
#include "services/nifm.h"
#include "runtime/hosversion.h"

static NifmServiceType g_nifmServiceType;

static Service g_nifmSrv;
static Service g_nifmIGS;

static Result _nifmCreateGeneralService(Service* srv_out);
static Result _nifmCreateGeneralServiceOld(Service* srv_out);

static Result _nifmRequestGetSystemEventReadableHandles(NifmRequest* r, bool autoclear);

NX_GENERATE_SERVICE_GUARD_PARAMS(nifm, (NifmServiceType service_type), (service_type));

Result _nifmInitialize(NifmServiceType service_type) {
    Result rc = MAKERESULT(Module_Libnx, LibnxError_BadInput);
    g_nifmServiceType = service_type;
    switch (g_nifmServiceType) {
        case NifmServiceType_User:
            rc = smGetService(&g_nifmSrv, "nifm:u");
            break;
        case NifmServiceType_System:
            rc = smGetService(&g_nifmSrv, "nifm:s");
            break;
        case NifmServiceType_Admin:
            rc = smGetService(&g_nifmSrv, "nifm:a");
            break;
    }
    
    if (R_SUCCEEDED(rc)) rc = serviceConvertToDomain(&g_nifmSrv);

    if (R_SUCCEEDED(rc)) {
        if (hosversionAtLeast(3,0,0))
            rc = _nifmCreateGeneralService(&g_nifmIGS);
        else
            rc = _nifmCreateGeneralServiceOld(&g_nifmIGS);
    }

    return rc;
}

void _nifmCleanup(void) {
    serviceClose(&g_nifmIGS);
    serviceClose(&g_nifmSrv);
}

Service* nifmGetServiceSession_StaticService(void) {
    return &g_nifmSrv;
}

Service* nifmGetServiceSession_GeneralService(void) {
    return &g_nifmIGS;
}

static Result _nifmCmdNoIO(Service* srv, u32 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatch(srv, cmd_id);
}

static Result _nifmCmdGetSession(Service* srv, Service* srv_out, u32 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatch(srv, cmd_id,
        .out_num_objects = 1,
        .out_objects = srv_out,
    );
}

/*static Result _nifmCmdNoInOutU32(Service* srv, u32 *out, u32 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatchOut(srv, cmd_id, *out);
}*/

static Result _nifmCmdNoInOutU8(Service* srv, u8 *out, u32 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatchOut(srv, cmd_id, *out);
}

static Result _nifmCmdNoInOutBool(Service* srv, bool *out, u32 cmd_id) {
    u8 tmp=0;
    Result rc = _nifmCmdNoInOutU8(srv, &tmp, cmd_id);
    if (R_SUCCEEDED(rc) && out) *out = tmp & 1;
    return rc;
}

static Result _nifmCmdNoInOutU32(Service* srv, u32 *out, u32 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatchOut(srv, cmd_id, *out);
}

static Result _nifmCmdInU8NoOut(Service* srv, u8 inval, u64 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatchIn(srv, cmd_id, inval);
}

static Result _nifmCmdInBoolNoOut(Service* srv, bool inval, u32 cmd_id) {
    return _nifmCmdInU8NoOut(srv, inval!=0, cmd_id);
}

static Result _nifmCmdInU32NoOut(Service* srv, u32 inval, u64 cmd_id) {
    serviceAssumeDomain(srv);
    return serviceDispatchIn(srv, cmd_id, inval);
}

static Result _nifmCreateGeneralServiceOld(Service* srv_out) {
    return _nifmCmdGetSession(&g_nifmSrv, srv_out, 4);
}

static Result _nifmCreateGeneralService(Service* srv_out) {
    u64 reserved=0;
    serviceAssumeDomain(&g_nifmSrv);
    return serviceDispatchIn(&g_nifmSrv, 5, reserved,
        .in_send_pid = true,
        .out_num_objects = 1,
        .out_objects = srv_out,
    );
}

static void _nifmConvertSfToNetworkProfileData(const NifmSfNetworkProfileData *in, NifmNetworkProfileData *out) {
    memset(out, 0, sizeof(*out));

    out->uuid = in->uuid;
    memcpy(out->network_name, in->network_name, sizeof(in->network_name));
    out->network_name[sizeof(out->network_name)-1] = 0;

    out->unk_x50 = in->unk_x112;
    out->unk_x54 = in->unk_x113;
    out->unk_x58 = in->unk_x114;
    out->unk_x59 = in->unk_x115;

    out->wireless_setting_data.ssid_len = in->wireless_setting_data.ssid_len;
    if (out->wireless_setting_data.ssid_len > sizeof(out->wireless_setting_data.ssid)-1) out->wireless_setting_data.ssid_len = sizeof(out->wireless_setting_data.ssid)-1;
    if (out->wireless_setting_data.ssid_len) memcpy(out->wireless_setting_data.ssid, in->wireless_setting_data.ssid, out->wireless_setting_data.ssid_len);
    out->wireless_setting_data.unk_x22 = in->wireless_setting_data.unk_x21;
    out->wireless_setting_data.unk_x24 = in->wireless_setting_data.unk_x22;
    out->wireless_setting_data.unk_x28 = in->wireless_setting_data.unk_x23;
    memcpy(out->wireless_setting_data.passphrase, in->wireless_setting_data.passphrase, sizeof(out->wireless_setting_data.passphrase));

    memcpy(&out->ip_setting_data, &in->ip_setting_data, sizeof(out->ip_setting_data));
}

static void _nifmConvertSfFromNetworkProfileData(const NifmNetworkProfileData *in, NifmSfNetworkProfileData *out) {
    memset(out, 0, sizeof(*out));

    out->uuid = in->uuid;
    memcpy(out->network_name, in->network_name, sizeof(out->network_name));
    out->network_name[sizeof(out->network_name)-1] = 0;

    out->unk_x112 = in->unk_x50;
    out->unk_x113 = in->unk_x54;
    out->unk_x114 = in->unk_x58;
    out->unk_x115 = in->unk_x59;

    out->wireless_setting_data.ssid_len = in->wireless_setting_data.ssid_len;
    memcpy(out->wireless_setting_data.ssid, in->wireless_setting_data.ssid, sizeof(out->wireless_setting_data.ssid)-1);
    out->wireless_setting_data.unk_x21 = in->wireless_setting_data.unk_x22;
    out->wireless_setting_data.unk_x22 = in->wireless_setting_data.unk_x24;
    out->wireless_setting_data.unk_x23 = in->wireless_setting_data.unk_x28;
    memcpy(out->wireless_setting_data.passphrase, in->wireless_setting_data.passphrase, sizeof(out->wireless_setting_data.passphrase));

    memcpy(&out->ip_setting_data, &in->ip_setting_data, sizeof(out->ip_setting_data));
}

NifmClientId nifmGetClientId(void) {
    NifmClientId id={0};
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatch(&g_nifmIGS, 1,
        .buffer_attrs = { SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_Out },
        .buffers = { { &id, sizeof(id) } },
    );
    if (R_FAILED(rc)) id.id = 0;
    return id;
}

static Result _nifmCreateRequest(Service* srv_out, s32 inval) {
    serviceAssumeDomain(&g_nifmIGS);
    return serviceDispatchIn(&g_nifmIGS, 4, inval,
        .out_num_objects = 1,
        .out_objects = srv_out,
    );
}

Result nifmCreateRequest(NifmRequest* r, bool autoclear) {
    Result rc=0;

    memset(r, 0, sizeof(*r));

    rc = _nifmCreateRequest(&r->s, 0x2);

    if (R_SUCCEEDED(rc)) {
        rc = _nifmRequestGetSystemEventReadableHandles(r, autoclear);

        if (R_FAILED(rc)) {
            serviceAssumeDomain(&r->s);
            serviceClose(&r->s);
        }
    }

    if (R_SUCCEEDED(rc)) {
        r->request_state = NifmRequestState_Unknown1;
        r->res = MAKERESULT(110, 311);
    }

    return rc;
}

Result nifmGetCurrentNetworkProfile(NifmNetworkProfileData *profile) {
    NifmSfNetworkProfileData tmp={0};
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatch(&g_nifmIGS, 5,
        .buffer_attrs = { SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_Out },
        .buffers = { { &tmp, sizeof(tmp) } },
    );
    if (R_SUCCEEDED(rc)) _nifmConvertSfToNetworkProfileData(&tmp, profile);
    return rc;
}

Result nifmGetNetworkProfile(Uuid uuid, NifmNetworkProfileData *profile) {
    NifmSfNetworkProfileData tmp={0};
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchIn(&g_nifmIGS, 8, uuid,
        .buffer_attrs = { SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_Out },
        .buffers = { { &tmp, sizeof(tmp) } },
    );
    if (R_SUCCEEDED(rc)) _nifmConvertSfToNetworkProfileData(&tmp, profile);
    return rc;
}

Result nifmSetNetworkProfile(const NifmNetworkProfileData *profile, Uuid *uuid) {
    NifmSfNetworkProfileData tmp={0};
    _nifmConvertSfFromNetworkProfileData(profile, &tmp);
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchOut(&g_nifmIGS, 9, *uuid,
        .buffer_attrs = { SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_In },
        .buffers = { { &tmp, sizeof(tmp) } },
    );
    return rc;
}

Result nifmGetCurrentIpAddress(u32* out) {
    NifmIpV4Address tmp={0};
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchOut(&g_nifmIGS, 12, tmp);
    if (R_SUCCEEDED(rc) && out) *out = *((u32*)tmp.addr);
    return rc;
}

Result nifmGetCurrentIpConfigInfo(u32 *current_addr, u32 *subnet_mask, u32 *gateway, u32 *primary_dns_server, u32 *secondary_dns_server) {
    struct {
        NifmIpAddressSetting ip_setting;
        NifmDnsSetting dns_setting;
    } out;

    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchOut(&g_nifmIGS, 15, out);
    if (R_SUCCEEDED(rc)) {
        if (current_addr) *current_addr = *((u32*)out.ip_setting.current_addr.addr);
        if (subnet_mask) *subnet_mask = *((u32*)out.ip_setting.subnet_mask.addr);
        if (gateway) *gateway = *((u32*)out.ip_setting.gateway.addr);
        if (primary_dns_server) *primary_dns_server = *((u32*)out.dns_setting.primary_dns_server.addr);
        if (secondary_dns_server) *secondary_dns_server = *((u32*)out.dns_setting.secondary_dns_server.addr);
    }
    return rc;
}

Result nifmSetWirelessCommunicationEnabled(bool enable) {
    if (g_nifmServiceType < NifmServiceType_System)
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    return _nifmCmdInBoolNoOut(&g_nifmIGS, enable, 16);
}

Result nifmIsWirelessCommunicationEnabled(bool* out) {
    return _nifmCmdNoInOutBool(&g_nifmIGS, out, 17);
}

Result nifmGetInternetConnectionStatus(NifmInternetConnectionType* connectionType, u32* wifiStrength, NifmInternetConnectionStatus* connectionStatus) {
    struct {
        u8 out1;
        u8 out2;
        u8 out3;
    } out;

    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchOut(&g_nifmIGS, 18, out);
    if (R_SUCCEEDED(rc)) {
        if (connectionType) *connectionType = out.out1;
        if (wifiStrength) *wifiStrength = out.out2;
        if (connectionStatus) *connectionStatus = out.out3;
    }
    return rc;
}

Result nifmIsEthernetCommunicationEnabled(bool* out) {
    return _nifmCmdNoInOutBool(&g_nifmIGS, out, 20);
}

bool nifmIsAnyInternetRequestAccepted(NifmClientId id) {
    u8 tmp=0;
    serviceAssumeDomain(&g_nifmIGS);
    Result rc = serviceDispatchOut(&g_nifmIGS, 21, tmp,
        .buffer_attrs = { SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_In },
        .buffers = { { &id, sizeof(id) } },
    );
    return R_SUCCEEDED(rc) ? tmp & 1 : 0;
}

Result nifmIsAnyForegroundRequestAccepted(bool* out) {
    return _nifmCmdNoInOutBool(&g_nifmIGS, out, 22);
}

Result nifmPutToSleep(void) {
    return _nifmCmdNoIO(&g_nifmIGS, 23);
}

Result nifmWakeUp(void) {
    return _nifmCmdNoIO(&g_nifmIGS, 24);
}

Result nifmSetWowlDelayedWakeTime(s32 val) {
    if (hosversionBefore(9,0,0))
        return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    return _nifmCmdInU32NoOut(&g_nifmIGS, (u32)val, 43);
}

// IRequest

void nifmRequestClose(NifmRequest* r) {
    eventClose(&r->event1);
    eventClose(&r->event_request_state);

    serviceAssumeDomain(&r->s);
    serviceClose(&r->s);
}

static Result _nifmRequestGetSystemEventReadableHandles(NifmRequest* r, bool autoclear) {
    Result rc=0;
    Handle tmp_handles[2] = {INVALID_HANDLE, INVALID_HANDLE};

    serviceAssumeDomain(&r->s);
    rc = serviceDispatch(&r->s, 2,
        .out_handle_attrs = { SfOutHandleAttr_HipcCopy, SfOutHandleAttr_HipcCopy },
        .out_handles = tmp_handles,
    );

    if (R_SUCCEEDED(rc)) {
        eventLoadRemote(&r->event_request_state, tmp_handles[0], true);
        eventLoadRemote(&r->event1, tmp_handles[1], autoclear);
    }

    return rc;
}

static void _nifmUpdateState(NifmRequest* r) {
    Result rc=0;
    u32 tmp=0;

    rc = _nifmCmdNoInOutU32(&r->s, &tmp, 0); // GetRequestState
    r->request_state = R_SUCCEEDED(rc) ? tmp : 0; // sdknso ignores error other than this.

    rc = _nifmCmdNoIO(&r->s, 1); // GetResult
    r->res = rc;
}

Result nifmGetRequestState(NifmRequest* r, NifmRequestState *out) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    if (R_FAILED(eventWait(&r->event_request_state, 0))) {
        if (out) *out = r->request_state;
        return 0;
    }

    // sdknso would clear the event here, but it's autoclear anyway.

    _nifmUpdateState(r);
    if (out) *out = r->request_state;

    return 0;
}

Result nifmGetResult(NifmRequest* r) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    if (R_FAILED(eventWait(&r->event_request_state, 0))) return r->res;

    // sdknso would clear the event here, but it's autoclear anyway.

    _nifmUpdateState(r);
    return r->res;
}

Result nifmRequestCancel(NifmRequest* r) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    return _nifmCmdNoIO(&r->s, 3);
}

Result nifmRequestSubmit(NifmRequest* r) {
    Result rc=0;
    NifmRequestState tmp;

    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    rc = nifmGetRequestState(r, &tmp);

    if (R_SUCCEEDED(rc) && (tmp == NifmRequestState_Unknown1 || tmp == NifmRequestState_OnHold || tmp == NifmRequestState_Available || tmp == NifmRequestState_Unknown5)) {
        _nifmCmdNoIO(&r->s, 4); // Submit (sdknso ignores error)
        _nifmUpdateState(r);
    }

    return rc;
}

Result nifmRequestSubmitAndWait(NifmRequest* r) {
    Result rc=0;
    NifmRequestState tmp;

    rc = nifmRequestSubmit(r);
    if (R_FAILED(rc)) return rc;

    while(1) {
        rc = nifmGetRequestState(r, &tmp);
        if (R_FAILED(rc)) return rc;

        if (tmp != NifmRequestState_OnHold) return rc;

        if (R_SUCCEEDED(eventWait(&r->event_request_state, 10000000000ULL))) break;
    }

    // sdknso would clear the event here, but it's autoclear anyway.

    _nifmUpdateState(r);

    return rc;
}

Result nifmRequestGetAppletInfo(NifmRequest* r, u32 theme_color, void* buffer, size_t size, u32 *applet_id, u32 *mode, u32 *out_size) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);

    struct {
        u32 applet_id;
        u32 mode;
        u32 out_size;
    } out;

    serviceAssumeDomain(&r->s);
    Result rc = serviceDispatchInOut(&r->s, 21, theme_color, out,
        .buffer_attrs = { SfBufferAttr_HipcMapAlias | SfBufferAttr_Out },
        .buffers = { { buffer, size } },
    );
    if (R_SUCCEEDED(rc)) {
        if (applet_id) *applet_id = out.applet_id;
        if (mode) *mode = out.mode;
        if (out_size) *out_size = out.out_size;
    }
    return rc;
}

Result nifmRequestSetKeptInSleep(NifmRequest* r, bool flag) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);
    if (hosversionBefore(3,0,0))
        return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    return _nifmCmdInBoolNoOut(&r->s, flag, 23);
}

Result nifmRequestRegisterSocketDescriptor(NifmRequest* r, int sockfd) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);
    if (hosversionBefore(3,0,0))
        return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    return _nifmCmdInU32NoOut(&r->s, (u32)sockfd, 24);
}

Result nifmRequestUnregisterSocketDescriptor(NifmRequest* r, int sockfd) {
    if (!serviceIsActive(&r->s))
        return MAKERESULT(Module_Libnx, LibnxError_NotInitialized);
    if (hosversionBefore(3,0,0))
        return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);

    return _nifmCmdInU32NoOut(&r->s, (u32)sockfd, 25);
}

