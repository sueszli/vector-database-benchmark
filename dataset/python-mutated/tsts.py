import struct
from datetime import datetime
from ldap3.protocol.formatters.formatters import format_sid
from impacket.dcerpc.v5 import transport
from impacket.uuid import uuidtup_to_bin, bin_to_string, string_to_bin
from impacket.dcerpc.v5.ndr import NDR, NDRCALL, NDRSTRUCT, NDRENUM, NDRUNION, NDRUniConformantArray, NDRPOINTER, NDRUniConformantVaryingArray, UNKNOWNDATA
from impacket.dcerpc.v5.dtypes import NULL, BOOL, BOOLEAN, STR, WSTR, LPWSTR, WIDESTR, RPC_UNICODE_STRING, LONG, UINT, ULONG, PULONG, LPDWORD, LARGE_INTEGER, DWORD, NDRHYPER, USHORT, UCHAR, PCHAR, BYTE, PBYTE, UUID, GUID
from impacket import system_errors
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.rpcrt import DCERPCException, RPC_C_AUTHN_GSS_NEGOTIATE, RPC_C_AUTHN_LEVEL_PKT_PRIVACY
TermSrvSession_UUID = uuidtup_to_bin(('484809d6-4239-471b-b5bc-61df8c23ac48', '1.0'))
TermSrvNotification_UUID = uuidtup_to_bin(('11899a43-2b68-4a76-92e3-a3d6ad8c26ce', '1.0'))
TermSrvEnumeration_UUID = uuidtup_to_bin(('88143fd0-c28d-4b2b-8fef-8d882f6a9390', '1.0'))
RCMPublic_UUID = uuidtup_to_bin(('bde95fdf-eee0-45de-9e12-e5a61cd0d4fe', '1.0'))
RcmListener_UUID = uuidtup_to_bin(('497d95a6-2d27-4bf5-9bbd-a6046957133c', '1.0'))
LegacyAPI_UUID = uuidtup_to_bin(('5ca4a760-ebb1-11cf-8611-00a0245420ed', '1.0'))
AUDIODRIVENAME_LENGTH = 9
WDPREFIX_LENGTH = 12
STACK_ADDRESS_LENGTH = 128
MAX_BR_NAME = 65
DIRECTORY_LENGTH = 256
INITIALPROGRAM_LENGTH = 256
USERNAME_LENGTH = 20
DOMAIN_LENGTH = 17
PASSWORD_LENGTH = 14
NASISPECIFICNAME_LENGTH = 14
NASIUSERNAME_LENGTH = 47
NASIPASSWORD_LENGTH = 24
NASISESSIONNAME_LENGTH = 16
NASIFILESERVER_LENGTH = 47
CLIENTDATANAME_LENGTH = 7
CLIENTNAME_LENGTH = 20
CLIENTADDRESS_LENGTH = 30
IMEFILENAME_LENGTH = 32
DIRECTORY_LENGTH = 256
CLIENTLICENSE_LENGTH = 32
CLIENTMODEM_LENGTH = 40
CLIENT_PRODUCT_ID_LENGTH = 32
MAX_COUNTER_EXTENSIONS = 2
WINSTATIONNAME_LENGTH = 32
PROTOCOL_CONSOLE = 0
PROTOCOL_ICA = 1
PROTOCOL_TSHARE = 2
PROTOCOL_RDP = 2
PDNAME_LENGTH = 32
WDNAME_LENGTH = 32
CDNAME_LENGTH = 32
DEVICENAME_LENGTH = 128
MODEMNAME_LENGTH = DEVICENAME_LENGTH
CALLBACK_LENGTH = 50
DLLNAME_LENGTH = 32
WINSTATIONCOMMENT_LENGTH = 60
MAX_LICENSE_SERVER_LENGTH = 1024
LOGONID_CURRENT = ULONG
MAX_PDCONFIG = 10
TERMSRV_TOTAL_SESSIONS = 1
TERMSRV_DISC_SESSIONS = 2
TERMSRV_RECON_SESSIONS = 3
TERMSRV_CURRENT_ACTIVE_SESSIONS = 4
TERMSRV_CURRENT_DISC_SESSIONS = 5
TERMSRV_PENDING_SESSIONS = 6
TERMSRV_SUCC_TOTAL_LOGONS = 7
TERMSRV_SUCC_LOCAL_LOGONS = 8
TERMSRV_SUCC_REMOTE_LOGONS = 9
TERMSRV_SUCC_SESSION0_LOGONS = 10
TERMSRV_CURRENT_TERMINATING_SESSIONS = 11
TERMSRV_CURRENT_LOGGEDON_SESSIONS = 12
NO_FALLBACK_DRIVERS = 0
FALLBACK_BESTGUESS = 1
FALLBACK_PCL = 2
FALLBACK_PS = 3
FALLBACK_PCLANDPS = 4
VIRTUALCHANNELNAME_LENGTH = 7
WINSTATION_QUERY = 1
WINSTATION_SET = 2
WINSTATION_RESET = 4
WINSTATION_VIRTUAL = 8
WINSTATION_SHADOW = 16
WINSTATION_LOGON = 32
WINSTATION_LOGOFF = 64
WINSTATION_MSG = 128
WINSTATION_CONNECT = 256
WINSTATION_DISCONNECT = 512
_NDRENUM = NDRENUM

class NDRENUM(_NDRENUM):

    def dump(self, msg=None, indent=0):
        if False:
            while True:
                i = 10
        if msg is None:
            msg = self.__class__.__name__
        if msg != '':
            print(msg, end=' ')
        try:
            print(' %s' % self.enumItems(self.fields['Data']).name, end=' ')
        except:
            print(' %s' % hex(self.fields['Data']), end=' ')

class TS_WCHAR(WSTR):
    commonHdr = (('ActualCount', '<L=len(Data)//2'),)
    commonHdr64 = (('ActualCount', '<Q=len(Data)//2'),)
    structure = (('Data', ':'),)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if key == 'Data':
            return self.fields[key].decode('utf-16le')
        else:
            return NDR.__getitem__(self, key)

class TS_LPWCHAR(NDRPOINTER):
    referent = (('Data', TS_WCHAR),)

class TS_CHAR(STR):
    commonHdr = (('ActualCount', '<L=len(Data)'),)
    commonHdr64 = (('ActualCount', '<Q=len(Data)'),)
    structure = (('Data', ':'),)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'Data':
            return self.fields[key]
        else:
            return NDR.__getitem__(self, key)

class SYSTEM_TIMESTAMP(NDRHYPER):

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'Data':
            return datetime.fromtimestamp(getUnixTime(int(str(self.fields[key]))))
        else:
            return NDR.__getitem__(self, key)

class TS_UNICODE_STRING(NDRSTRUCT):
    """
    typedef struct _TS_UNICODE_STRING {
        USHORT Length;
        USHORT MaximumLength;
        #ifdef __midl
            [size_is(MaximumLength),length_is(Length)]PWSTR Buffer;
        #else
            PWSTR Buffer;
        #endif
    } TS_UNICODE_STRING;
    """
    structure = (('Length', USHORT), ('MaximumLength', USHORT), ('Buffer', LPWSTR))

class TS_LPCHAR(NDRPOINTER):
    referent = (('Data', TS_CHAR),)
TS_PBYTE = TS_LPCHAR

class TS_WCHAR_STRIPPED(TS_WCHAR):

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if key == 'Data':
            return self.fields[key].decode('utf-16le').strip('\x00')
        else:
            return NDR.__getitem__(self, key)

class WIDESTR_STRIPPED(WIDESTR):
    length = None

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'Data':
            return self.fields[key].decode('utf-16le').rstrip('\x00')
        else:
            return NDR.__getitem__(self, key)

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        if self.length is None:
            return super().getDataLen(data, offset)
        return self.length * 2

class WSTR_STRIPPED(WSTR):

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'Data':
            return self.fields[key].decode('utf-16le').rstrip('\x00')
        else:
            return NDR.__getitem__(self, key)

class LPWCHAR_STRIPPED(NDRPOINTER):
    referent = (('Data', WIDESTR_STRIPPED),)

class LONG_ARRAY(NDRUniConformantArray):
    item = 'L'

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        if key == 'Data':
            return b''.join(self.fields[key])
        else:
            return NDR.__getitem__(self, key)

class UCHAR_ARRAY(NDRUniConformantArray):
    item = 'c'

class LPUCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', UCHAR_ARRAY),)

class WCHAR_ARRAY_32(WIDESTR_STRIPPED):
    length = 32

class WCHAR_ARRAY_256(WIDESTR_STRIPPED):
    length = 256

class WCHAR_ARRAY_33(WIDESTR_STRIPPED):
    length = 33

class WCHAR_ARRAY_21(WIDESTR_STRIPPED):
    length = 21

class WCHAR_ARRAY_18(WIDESTR_STRIPPED):
    length = 18

class WCHAR_ARRAY_4(WIDESTR_STRIPPED):
    length = 4

class WCHAR_CLIENTNAME_LENGTH(WIDESTR_STRIPPED):
    length = CLIENTNAME_LENGTH + 1

class WCHAR_DOMAIN_LENGTH(WIDESTR_STRIPPED):
    length = DOMAIN_LENGTH + 1

class WCHAR_USERNAME_LENGTH(WIDESTR_STRIPPED):
    length = USERNAME_LENGTH + 1

class WCHAR_PASSWORD_LENGTH(WIDESTR_STRIPPED):
    length = PASSWORD_LENGTH + 1

class WCHAR_DIRECTORY_LENGTH(WIDESTR_STRIPPED):
    length = DIRECTORY_LENGTH + 1

class WCHAR_INITIALPROGRAM_LENGTH(WIDESTR_STRIPPED):
    length = INITIALPROGRAM_LENGTH + 1

class WCHAR_CLIENTADDRESS_LENGTH(WIDESTR_STRIPPED):
    length = CLIENTADDRESS_LENGTH + 1

class WCHAR_IMEFILENAME_LENGTH(WIDESTR_STRIPPED):
    length = IMEFILENAME_LENGTH + 1

class WCHAR_CLIENTLICENSE_LENGTH(WIDESTR_STRIPPED):
    length = CLIENTLICENSE_LENGTH + 1

class WCHAR_CLIENTMODEM_LENGTH(WIDESTR_STRIPPED):
    length = CLIENTMODEM_LENGTH + 1

class WCHAR_AUDIODRIVENAME_LENGTH(WIDESTR_STRIPPED):
    length = AUDIODRIVENAME_LENGTH

class WCHAR_CLIENT_PRODUCT_ID_LENGTH(WIDESTR_STRIPPED):
    length = CLIENT_PRODUCT_ID_LENGTH

class WCHAR_NASIFILESERVER_LENGTH(WIDESTR_STRIPPED):
    length = NASIFILESERVER_LENGTH + 1

class WCHAR_CALLBACK_LENGTH(WIDESTR_STRIPPED):
    length = CALLBACK_LENGTH + 1

class WCHAR_MAX_BR_NAME(WIDESTR_STRIPPED):
    length = MAX_BR_NAME

class WCHAR_WINSTATIONCOMMENT_LENGTH(WIDESTR_STRIPPED):
    length = WINSTATIONCOMMENT_LENGTH + 1

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            return 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            while True:
                i = 10
        key = self.error_code & 65535
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'TSTS SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'TSTS SessionError: unknown error code: 0x%x' % self.error_code

def ZEROPAD(data, size=None):
    if False:
        i = 10
        return i + 15
    if size is None:
        size = len(data) + 1
    assert len(data) <= size, 'Invalid data size!'
    data += '\x00' * (size - len(data))
    return data

def getUnixTime(t):
    if False:
        print('Hello World!')
    t -= 116444736000000000
    t /= 10000000
    return t

def enum2value(enum, key):
    if False:
        return 10
    return enum.enumItems._value2member_map_[key]._name_

class SID(TS_CHAR):

    def known_sid(self, sid):
        if False:
            print('Hello World!')
        knownSids = {'S-1-5-10': 'SELF', 'S-1-5-13': 'TERMINAL SERVER USER', 'S-1-5-11': 'Authenticated Users', 'S-1-5-12': 'RESTRICTED', 'S-1-5-14': 'Authenticated Users', 'S-1-5-15': 'This Organization', 'S-1-5-17': 'IUSR', 'S-1-5-18': 'SYSTEM', 'S-1-5-19': 'LOCAL SERVICE', 'S-1-5-20': 'NETWORK SERVICE'}
        if sid.startswith('S-1-5-90-0-') and len(sid.split('-')) == 6:
            return 'DWM-{}'.format(int(sid.split('-')[-1]))
        elif sid.startswith('S-1-5-96-0-') and len(sid.split('-')) == 6:
            return 'UMFD-{}'.format(int(sid.split('-')[-1]))
        elif sid in knownSids:
            return knownSids[sid]
        return sid

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'Data':
            sid = format_sid(self.fields[key])
            if not len(sid):
                return ''
            return self.known_sid(sid)
        else:
            return NDR.__getitem__(self, key)

class context_handle(NDRSTRUCT):
    structure = (('context_handle_attributes', ULONG), ('context_handle_uuid', UUID))

    def getUUID(self):
        if False:
            return 10
        return bin_to_string(self['context_handle_uuid'])

    def tuple(self):
        if False:
            i = 10
            return i + 15
        return (bin_to_string(self['context_handle_uuid']), self['context_handle_attributes'])

    def from_tuple(self, tup):
        if False:
            return 10
        (self['context_handle_uuid'], self['context_handle_attributes']) = (string_to_bin(tup[0]), tup[1])

    def __init__(self, data=None, isNDR64=False):
        if False:
            i = 10
            return i + 15
        NDRSTRUCT.__init__(self, data, isNDR64)
        self['context_handle_uuid'] = b'\x00' * 16

    def isNull(self):
        if False:
            while True:
                i = 10
        return self['context_handle_uuid'] == b'\x00' * 16

    def __str__(self):
        if False:
            return 10
        return bin_to_string(self['context_handle_uuid'])

class handle_t(NDRSTRUCT):
    structure = (('Data', '20s=b""'),)

    def getAlignment(self):
        if False:
            return 10
        if self._isNDR64 is True:
            return 8
        else:
            return 4
ENUM_HANDLE = context_handle

class pHandle(NDRPOINTER):
    referent = (('Data', handle_t),)
HLISTENER = context_handle
SERVER_HANDLE = context_handle
NOTIFY_HANDLE = context_handle
SESSION_HANDLE = context_handle

class MSGBOX_ENUM(NDRENUM):

    class enumItems(Enum):
        IDABORT = 3
        IDCANCEL = 2
        IDIGNORE = 5
        IDNO = 7
        IDOK = 1
        IDRETRY = 4
        IDYES = 6
        IDASYNC = 32001
        IDTIMEOUT = 32000

class ShutdownFlags(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        WSD_LOGOFF = 1
        WSD_SHUTDOWN = 2
        WSD_REBOOT = 4
        WSD_POWEROFF = 8

class HotKeyModifiers(NDRENUM):
    structure = (('Data', '<H'),)
    NONE = 0
    Alt = 1
    Control = 2
    Shift = 4
    WindowsKey = 8

class EventFlags(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        WEVENT_NONE = 0
        WEVENT_CREATE = 1
        WEVENT_DELETE = 2
        WEVENT_RENAME = 4
        WEVENT_CONNECT = 8
        WEVENT_DISCONNECT = 16
        WEVENT_LOGON = 32
        WEVENT_LOGOFF = 64
        WEVENT_STATECHANGE = 128
        WEVENT_LICENSE = 256
        WEVENT_ALL = 2147483647
        WEVENT_FLUSH = 2147483648

class ADDRESSFAMILY_ENUM(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        AppleTalk = 16
        Atm = 22
        Banyan = 21
        Ccitt = 10
        Chaos = 5
        Cluster = 24
        ControllerAreaNetwork = 65537
        DataKit = 9
        DataLink = 13
        DecNet = 12
        Ecma = 8
        FireFox = 19
        HyperChannel = 15
        Ieee12844 = 25
        ImpLink = 3
        InterNetwork = 2
        InterNetworkV6 = 23
        Ipx = 6
        Irda = 26
        Iso = 7
        Lat = 14
        Max = 29
        NetBios = 17
        NetworkDesigners = 28
        NS = 6
        Osi = 7
        Packet = 65536
        Pup = 4
        Sna = 11
        Unix = 1
        Unspecified = 0
        VoiceView = 18

class WINSTATIONNAME(WIDESTR_STRIPPED):
    length = WINSTATIONNAME_LENGTH + 1

class DLLNAME(WIDESTR):

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        return DLLNAME_LENGTH + 1

class PDLLNAME(NDRPOINTER):
    referent = (('Data', DLLNAME),)

class DEVICENAME(WIDESTR):

    def getDataLen(self, data, offset=0):
        if False:
            while True:
                i = 10
        return DEVICENAME_LENGTH + 1

class PDEVICENAME(NDRPOINTER):
    referent = (('Data', DEVICENAME),)

class CLIENTDATANAME(STR):

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        return CLIENTDATANAME_LENGTH + 1

class PCLIENTDATANAME(NDRPOINTER):
    referent = (('Data', CLIENTDATANAME),)

class WINSTATIONINFOCLASS(NDRENUM):

    class enumItems(Enum):
        WinStationCreateData = 0
        WinStationConfiguration = 1
        WinStationPdParams = 2
        WinStationWd = 3
        WinStationPd = 4
        WinStationPrinter = 5
        WinStationClient = 6
        WinStationModules = 7
        WinStationInformation = 8
        WinStationTrace = 9
        WinStationBeep = 10
        WinStationEncryptionOff = 11
        WinStationEncryptionPerm = 12
        WinStationNtSecurity = 13
        WinStationUserToken = 14
        WinStationUnused1 = 15
        WinStationVideoData = 16
        WinStationInitialProgram = 17
        WinStationCd = 18
        WinStationSystemTrace = 19
        WinStationVirtualData = 20
        WinStationClientData = 21
        WinStationSecureDesktopEnter = 22
        WinStationSecureDesktopExit = 23
        WinStationLoadBalanceSessionTarget = 24
        WinStationLoadIndicator = 25
        WinStationShadowInfo = 26
        WinStationDigProductId = 27
        WinStationLockedState = 28
        WinStationRemoteAddress = 29
        WinStationIdleTime = 30
        WinStationLastReconnectType = 31
        WinStationDisallowAutoReconnect = 32
        WinStationUnused2 = 33
        WinStationUnused3 = 34
        WinStationUnused4 = 35
        WinStationUnused5 = 36
        WinStationReconnectedFromId = 37
        WinStationEffectsPolicy = 38
        WinStationType = 39
        WinStationInformationEx = 40

class WINSTATIONSTATECLASS(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        State_Active = 0
        State_Connected = 1
        State_ConnectQuery = 2
        State_Shadow = 3
        State_Disconnected = 4
        State_Idle = 5
        State_Listen = 6
        State_Reset = 7
        State_Down = 8
        State_Init = 9

class SDCLASS(NDRENUM):

    class enumItems(Enum):
        SdNone = 0
        SdConsole = 1
        SdNetwork = 2
        SdAsync = 3
        SdOemTransport = 4

class SHADOWCLASS(NDRENUM):

    class enumItems(Enum):
        Shadow_Disable = 0
        Shadow_EnableInputNotify = 1
        Shadow_EnableInputNoNotify = 2
        Shadow_EnableNoInputNotify = 3
        Shadow_EnableNoInputNoNotify = 4

class RECONNECT_TYPE(NDRENUM):

    class enumItems(Enum):
        NeverReconnected = 0
        ManualReconnect = 1
        AutoReconnect = 2

class PRECONNECT_TYPE(NDRPOINTER):
    referent = (('Data', RECONNECT_TYPE),)
BOUNDED_ULONG = ULONG

class UINT_PTR(NDRPOINTER):
    referent = (('Data', UINT),)

class SESSIONTYPE(NDRENUM):

    class enumItems(Enum):
        SESSIONTYPE_UNKNOWN = 0
        SESSIONTYPE_SERVICES = 1
        SESSIONTYPE_LISTENER = 2
        SESSIONTYPE_REGULARDESKTOP = 3
        SESSIONTYPE_ALTERNATESHELL = 4
        SESSIONTYPE_REMOTEAPP = 5
        SESSIONTYPE_MEDIACENTEREXT = 6

class SHADOW_CONTROL_REQUEST(NDRENUM):

    class enumItems(Enum):
        SHADOW_CONTROL_REQUEST_VIEW = 0
        SHADOW_CONTROL_REQUEST_TAKECONTROL = 1
        SHADOW_CONTROL_REQUEST_Count = 2

class SHADOW_PERMISSION_REQUEST(NDRENUM):

    class enumItems(Enum):
        SHADOW_PERMISSION_REQUEST_SILENT = 0
        SHADOW_PERMISSION_REQUEST_REQUESTPERMISSION = 1
        SHADOW_PERMISSION_REQUEST_Count = 2

class SHADOW_REQUEST_RESPONSE(NDRENUM):

    class enumItems(Enum):
        SHADOW_REQUEST_RESPONSE_ALLOW = 0
        SHADOW_REQUEST_RESPONSE_DECLINE = 1
        SHADOW_REQUEST_RESPONSE_POLICY_PERMISSION_REQUIRED = 2
        SHADOW_REQUEST_RESPONSE_POLICY_DISABLED = 3
        SHADOW_REQUEST_RESPONSE_POLICY_VIEW_ONLY = 4
        SHADOW_REQUEST_RESPONSE_POLICY_VIEW_ONLY_PERMISSION_REQUIRED = 5
        SHADOW_REQUEST_RESPONSE_SESSION_ALREADY_CONTROLLED = 6

class SESSION_FILTER(NDRENUM):

    class enumItems(Enum):
        SF_SERVICES_SESSION_POPUP = 0

class PROTOCOLSTATUS_INFO_TYPE(NDRENUM):

    class enumItems(Enum):
        PROTOCOLSTATUS_INFO_BASIC = 0
        PROTOCOLSTATUS_INFO_EXTENDED = 1

class QUERY_SESSION_DATA_TYPE(NDRENUM):

    class enumItems(Enum):
        QUERY_SESSION_DATA_MODULE = 0
        QUERY_SESSION_DATA_WDCONFIG = 1
        QUERY_SESSION_DATA_VIRTUALDATA = 2
        QUERY_SESSION_DATA_LICENSE = 3
        QUERY_SESSION_DATA_DEVICEID = 4
        QUERY_SESSION_DATA_LICENSE_VALIDATION = 5

class SESSIONENUM_LEVEL1(NDRSTRUCT):
    structure = (('SessionId', LONG), ('State', WINSTATIONSTATECLASS), ('Name', WCHAR_ARRAY_33))

class SESSIONENUM_LEVEL2(NDRSTRUCT):
    structure = (('SessionId', LONG), ('State', WINSTATIONSTATECLASS), ('Name', WCHAR_ARRAY_33), ('Source', ULONG), ('bFullDesktop', BOOLEAN), ('SessionType', GUID))

class SESSIONENUM_LEVEL3(NDRSTRUCT):
    structure = (('SessionId', LONG), ('State', WINSTATIONSTATECLASS), ('Name', WCHAR_ARRAY_33), ('Source', ULONG), ('bFullDesktop', BOOLEAN), ('SessionType', GUID), ('ProtoDataSize', ULONG), ('pProtocolData', UCHAR))

class SessionInfo(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('SessionEnum_Level1', SESSIONENUM_LEVEL1), 2: ('SessionEnum_Level2', SESSIONENUM_LEVEL2), 3: ('SessionEnum_Level3', SESSIONENUM_LEVEL3)}

class SessionInfo_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('SessionInfo', SessionInfo))

class SESSIONENUM(NDRUniConformantArray):
    item = SessionInfo_STRUCT

class PSESSIONENUM(NDRPOINTER):
    referent = (('Data', SESSIONENUM),)
SessionInfo_Ex = SessionInfo
PSESSIONENUM_EX = SESSIONENUM

class EXECENVDATA_LEVEL1(NDRSTRUCT):
    structure = (('ExecEnvId', LONG), ('State', WINSTATIONSTATECLASS), ('SessionName', WCHAR_ARRAY_33))

class PEXECENVDATA_LEVEL1(NDRPOINTER):
    referent = (('Data', EXECENVDATA_LEVEL1),)

class EXECENVDATA_LEVEL2(NDRSTRUCT):
    structure = (('ExecEnvId', LONG), ('State', WINSTATIONSTATECLASS), ('SessionName', WCHAR_ARRAY_33), ('AbsSessionId', LONG), ('HostName', WCHAR_ARRAY_33), ('UserName', WCHAR_ARRAY_33), ('DomainName', WCHAR_ARRAY_33), ('FarmName', WCHAR_ARRAY_33))

class PEXECENVDATA_LEVEL2(NDRPOINTER):
    referent = (('Data', EXECENVDATA_LEVEL2),)

class ExecEnvData(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('ExecEnvEnum_Level1', EXECENVDATA_LEVEL1), 2: ('ExecEnvEnum_Level2', EXECENVDATA_LEVEL2)}

class ExecEnvData_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('ExecEnvData', ExecEnvData))

class EXECENVDATA(NDRUniConformantArray):
    item = ExecEnvData_STRUCT

class PEXECENVDATA(NDRPOINTER):
    referent = (('Data', EXECENVDATA),)

class EXECENVDATAEX_LEVEL1(NDRSTRUCT):
    """
        structure = (
            ('ExecEnvId', LONG),
            ('State', WINSTATIONSTATECLASS),
            ('AbsSessionId', LONG),
            ('pszSessionName', WIDESTR),
            ('pszHostName', WIDESTR),
            ('pszUserName', WIDESTR),
            ('pszFarmName', WIDESTR),
        )
    """
    pass

class ExecEnvDataEx(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('ExecEnvEnum_Level1', EXECENVDATAEX_LEVEL1)}

class EXECENVDATAEX(NDRUniConformantArray):
    item = ExecEnvDataEx

class PEXECENVDATAEX(NDRPOINTER):
    referent = (('Data', EXECENVDATAEX),)

class LISTENERENUM_LEVEL1(NDRSTRUCT):
    structure = (('Id', LONG), ('bListening', BOOL), ('Name', WCHAR_ARRAY_33))

class ListenerInfo(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('ListenerEnum_Level1', LISTENERENUM_LEVEL1)}

class ListenerInfo_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('ListenerInfo', ListenerInfo))

class LISTENERENUM(NDRUniConformantArray):
    item = ListenerInfo_STRUCT

class PLISTENERENUM(NDRPOINTER):
    referent = (('Data', LISTENERENUM),)

class LSMSESSIONINFORMATION(NDRSTRUCT):
    structure = (('pszUserName', LPWCHAR_STRIPPED), ('pszDomain', LPWCHAR_STRIPPED), ('pszTerminalName', LPWCHAR_STRIPPED), ('SessionState', WINSTATIONSTATECLASS), ('DesktopLocked', BOOLEAN), ('ConnectTime', SYSTEM_TIMESTAMP), ('DisconnectTime', SYSTEM_TIMESTAMP), ('LogonTime', SYSTEM_TIMESTAMP))

class TS_SYSTEMTIME(NDRSTRUCT):
    structure = (('wYear', USHORT), ('wMonth', USHORT), ('wDayOfWeek', USHORT), ('wDay', USHORT), ('wHour', USHORT), ('wMinute', USHORT), ('wSecond', USHORT), ('wMilliseconds', USHORT))

class TS_TIME_ZONE_INFORMATION(NDRSTRUCT):
    structure = (('Bias', ULONG), ('StandardName', WCHAR_ARRAY_32), ('StandardDate', TS_SYSTEMTIME), ('StandardBias', ULONG), ('DaylightName', WCHAR_ARRAY_32), ('DaylightDate', TS_SYSTEMTIME), ('DaylightBias', ULONG))

class WINSTATIONCLIENT(NDRSTRUCT):

    class FLAGS(NDRSTRUCT):
        structure = (('flags', '6s=b""'),)

        def __getitem__(self, key):
            if False:
                i = 10
                return i + 15
            if key == 'flags':
                flagsInt = int.from_bytes(self.fields[key][2:], 'little')
                keys = {'fTextOnly': False, 'fDisableCtrlAltDel': False, 'fMouse': False, 'fDoubleClickDetect': False, 'fINetClient': False, 'fPromptForPassword': False, 'fMaximizeShell': False, 'fEnableWindowsKey': False, 'fRemoteConsoleAudio': False, 'fPasswordIsScPin': False, 'fNoAudioPlayback': False, 'fUsingSavedCreds': False, 'fRestrictedLogon': False}
                for k in keys:
                    keys[k] = bool(flagsInt & 1)
                    flagsInt >>= 1
                return keys
            else:
                return NDR.__getitem__(self, key)
    structure = (('flags', FLAGS), ('ClientName', WCHAR_CLIENTNAME_LENGTH), ('Domain', WCHAR_DOMAIN_LENGTH), ('UserName', WCHAR_USERNAME_LENGTH), ('Password', WCHAR_PASSWORD_LENGTH), ('WorkDirectory', WCHAR_DIRECTORY_LENGTH), ('InitialProgram', WCHAR_INITIALPROGRAM_LENGTH), ('SerialNumber', ULONG), ('EncryptionLevel', BYTE), ('ClientAddressFamily', ADDRESSFAMILY_ENUM), ('ClientAddress', WCHAR_CLIENTADDRESS_LENGTH), ('HRes', USHORT), ('VRes', USHORT), ('ColorDepth', USHORT), ('ProtocolType', USHORT), ('KeyboardLayout', ULONG), ('KeyboardType', ULONG), ('KeyboardSubType', ULONG), ('KeyboardFunctionKey', ULONG), ('imeFileName', WCHAR_IMEFILENAME_LENGTH), ('ClientDirectory', WCHAR_DIRECTORY_LENGTH), ('ClientLicense', WCHAR_CLIENTLICENSE_LENGTH), ('ClientModem', WCHAR_CLIENTMODEM_LENGTH), ('ClientBuildNumber', ULONG), ('ClientHardwareId', ULONG), ('ClientProductId', USHORT), ('OutBufCountHost', USHORT), ('OutBufCountClient', USHORT), ('OutBufLength', USHORT), ('AudioDriverName', WCHAR_AUDIODRIVENAME_LENGTH), ('ClientTimeZone', TS_TIME_ZONE_INFORMATION), ('ClientSessionId', ULONG), ('clientDigProductId', WCHAR_CLIENT_PRODUCT_ID_LENGTH), ('PerformanceFlags', ULONG), ('ActiveInputLocale', ULONG))

class PWINSTATIONCLIENT(NDRPOINTER):
    referent = (('Data', WINSTATIONCLIENT),)

class TS_COUNTER_HEADER(NDRSTRUCT):
    structure = (('dwCounterID', DWORD), ('bResult', BOOLEAN))

class TS_COUNTER(NDRSTRUCT):
    structure = (('counterHead', TS_COUNTER_HEADER), ('dwValue', DWORD), ('startTime', LARGE_INTEGER))

class TS_COUNTER_ARRAY(NDRUniConformantArray):
    item = TS_COUNTER

class PTS_COUNTER(NDRPOINTER):
    referent = (('Data', TS_COUNTER_ARRAY),)

class SESSIONFLAGS(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        WTS_SESSIONSTATE_UNKNOWN = 4294967295
        WTS_SESSIONSTATE_LOCK = 0
        WTS_SESSIONSTATE_UNLOCK = 1

class LSM_SESSIONINFO_EX_LEVEL1(NDRSTRUCT):
    structure = (('SessionState', WINSTATIONSTATECLASS), ('SessionFlags', SESSIONFLAGS), ('SessionName', WCHAR_ARRAY_33), ('DomainName', WCHAR_ARRAY_18), ('UserName', WCHAR_ARRAY_21), ('ConnectTime', SYSTEM_TIMESTAMP), ('DisconnectTime', SYSTEM_TIMESTAMP), ('LogonTime', SYSTEM_TIMESTAMP), ('LastInputTime', SYSTEM_TIMESTAMP), ('ProtocolDataSize', ULONG), ('ProtocolData', TS_LPCHAR))

class LSM_SESSIONINFO_EX(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('LSM_SessionInfo_Level1', LSM_SESSIONINFO_EX_LEVEL1)}

class PLSMSESSIONINFORMATION_EX(NDRPOINTER):
    referent = (('Data', LSM_SESSIONINFO_EX),)

class TNotificationId(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        WTS_NOTIFY_NONE = 0
        WTS_NOTIFY_CREATE = 1
        WTS_NOTIFY_CONNECT = 2
        WTS_NOTIFY_DISCONNECT = 4
        WTS_NOTIFY_LOGON = 8
        WTS_NOTIFY_LOGOFF = 16
        WTS_NOTIFY_SHADOW_START = 32
        WTS_NOTIFY_SHADOW_STOP = 64
        WTS_NOTIFY_TERMINATE = 128
        WTS_NOTIFY_CONSOLE_CONNECT = 256
        WTS_NOTIFY_CONSOLE_DISCONNECT = 512
        WTS_NOTIFY_LOCK = 1024
        WTS_NOTIFY_UNLOCK = 2048
        WTS_NOTIFY_ALL = 4294967295

class SESSION_CHANGE(NDRSTRUCT):
    structure = (('SessionId', LONG), ('TNotificationId', TNotificationId))

class SESSION_CHANGE_ARRAY(NDRUniConformantArray):
    item = SESSION_CHANGE

class PSESSION_CHANGE(NDRPOINTER):
    referent = (('Data', SESSION_CHANGE_ARRAY),)

class CALLBACKCLASS(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        Callback_Disable = 0
        Callback_Roving = 1
        Callback_Fixed = 2

class USERCONFIG(NDRSTRUCT):

    class FLAGS(NDRSTRUCT):
        structure = (('flags', '7s=b""'),)

        def __getitem__(self, key):
            if False:
                i = 10
                return i + 15
            if key == 'flags':
                flagsInt = int.from_bytes(self.fields[key][:], 'little')
                tmp = [('fInheritAutoLogon', 1), ('fInheritResetBroken', 1), ('fInheritReconnectSame', 1), ('fInheritInitialProgram', 1), ('fInheritCallback', 1), ('fInheritCallbackNumber', 1), ('fInheritShadow', 1), ('fInheritMaxSessionTime', 1), ('fInheritMaxDisconnectionTime', 1), ('fInheritMaxIdleTime', 1), ('fInheritAutoClient', 1), ('fInheritSecurity', 1), ('fPromptForPassword', 1), ('fResetBroken', 1), ('fReconnectSame', 1), ('fLogonDisabled', 1), ('fWallPaperDisabled', 1), ('fAutoClientDrives', 1), ('fAutoClientLpts', 1), ('fForceClientLptDef', 1), ('fRequireEncryption', 1), ('fDisableEncryption', 1), ('fUnused1', 1), ('fHomeDirectoryMapRoot', 1), ('fUseDefaultGina', 1), ('fCursorBlinkDisabled', 1), ('fPublishedApp', 1), ('fHideTitleBar', 1), ('fMaximize', 1), ('fDisableCpm', 1), ('fDisableCdm', 1), ('fDisableCcm', 1), ('fDisableLPT', 1), ('fDisableClip', 1), ('fDisableExe', 1), ('fDisableCam', 1), ('fDisableAutoReconnect', 1), ('ColorDepth', 3), ('fInheritColorDepth', 1), ('fErrorInvalidProfile', 1), ('fPasswordIsScPin', 1), ('fDisablePNPRedir', 1)]
                keys = {}
                for (k, bits) in tmp:
                    if bits == 1:
                        keys[k] = flagsInt & 1
                    else:
                        keys[k] = flagsInt & (1 << bits) - 1
                    flagsInt >>= bits
                return keys
            else:
                return NDR.__getitem__(self, key)
    structure = (('flags', FLAGS), ('UserName', WCHAR_USERNAME_LENGTH), ('Domain', WCHAR_DOMAIN_LENGTH), ('Password', WCHAR_PASSWORD_LENGTH), ('WorkDirectory', WCHAR_DIRECTORY_LENGTH), ('InitialProgram', WCHAR_INITIALPROGRAM_LENGTH), ('CallbackNumber', WCHAR_CALLBACK_LENGTH), ('Callback', CALLBACKCLASS), ('Shadow', SHADOWCLASS), ('MaxConnectionTime', ULONG), ('MaxDisconnectionTime', ULONG), ('MaxIdleTime', ULONG), ('KeyboardLayout', ULONG), ('MinEncryptionLevel', BYTE), ('NWLogonServer', WCHAR_NASIFILESERVER_LENGTH), ('PublishedName', WCHAR_MAX_BR_NAME), ('WFProfilePath', WCHAR_DIRECTORY_LENGTH), ('WFHomeDir', WCHAR_DIRECTORY_LENGTH), ('WFHomeDirDrive', WCHAR_ARRAY_4))

class OEMId(NDRSTRUCT):
    structure = (('OEMId', '4s=""'),)

class WINSTATIONCONFIG(NDRSTRUCT):
    pass
    structure = (('Comment', WCHAR_WINSTATIONCOMMENT_LENGTH), ('User', USERCONFIG), ('OEMId', OEMId))

class PWINSTATIONCONFIG(NDRPOINTER):
    referent = (('Data', WINSTATIONCONFIG),)

class PROTOCOLCOUNTERS(NDRSTRUCT):
    pass

class CACHE_STATISTICS(NDRSTRUCT):
    pass

class PROTOCOLSTATUS(NDRSTRUCT):
    pass

class PPROTOCOLSTATUS(NDRPOINTER):
    referent = (('Data', PROTOCOLSTATUS),)

class IPv4ADDRESS(NDRSTRUCT):
    structure = (('Data', '<L'),)

    def __getitem__(self, key):
        if False:
            return 10
        if key == 'Data':
            x = self.fields[key]
            y = []
            while x:
                y += [str(x & 255)]
                x >>= 8
            return '.'.join(y)
        else:
            return super().__getitem__(key)

class RCM_REMOTEADDRESS_UNION_CASE_IPV4(NDRSTRUCT):

    class _4CHAR(NDRSTRUCT):
        structure = (('sin_zero', '4s=b""'),)
    structure = (('sin_port', USHORT), ('sin_port2', USHORT), ('in_addr', IPv4ADDRESS), ('sin_zero', _4CHAR))

class RCM_REMOTEADDRESS_UNION_CASE_IPV6(NDRSTRUCT):

    class _8CHAR(NDRSTRUCT):
        structure = (('sin_zero', '8s=b""'),)
    structure = (('sin_port', USHORT), ('in_addr', ULONG), ('sin_zero', _8CHAR), ('sin6_scope_id', ULONG))

class RCM_REMOTEADDRESS(NDRUNION):
    commonHdr = (('tag', USHORT),)
    union = {2: ('ipv4', RCM_REMOTEADDRESS_UNION_CASE_IPV4), 23: ('ipv6', RCM_REMOTEADDRESS_UNION_CASE_IPV6)}

class pResult_ENUM(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        STATUS_SUCCESS = 0
        STATUS_INVALID_PARAMETER = 3221225485
        STATUS_CANCELLED = 3221225760
        STATUS_INVALID_INFO_CLASS = 3221225475
        STATUS_NO_MEMORY = 3221225495
        STATUS_ACCESS_DENIED = 3221225506
        STATUS_BUFFER_TOO_SMALL = 3221225507
        STATUS_NOT_IMPLEMENTED = 3221225474
        STATUS_INFO_LENGTH_MISMATCH = 3221225476
        STATUS_UNSUCCESSFUL = 3221225473
        STATUS_CTX_WINSTATION_NOT_FOUND = 3221880853
        STATUS_WRONG_PASSWORD = 3221225578
        DOES_NOT_EXISTS_OR_INSUFFICIENT_PERMISSIONS = 2147949422
        INVALID_PARAMETER2 = 2147942487
        ERROR_ACCESS_DENIED = 2147942405
        ERROR_INVALID_STATE = 2147947423
        ERROR_LOGON_FAILURE = 2147943726
        ERROR_FILE_NOT_FOUND = 2147942402
        ERROR_STATUS_BUFFER_TOO_SMALL = 2147942522

class TS_SYS_PROCESS_INFORMATION(NDRSTRUCT):
    structure = (('NextEntryOffset', ULONG), ('NumberOfThreads', ULONG), ('SpareLi1', LARGE_INTEGER), ('SpareLi2', LARGE_INTEGER), ('SpareLi3', LARGE_INTEGER), ('CreateTime', LARGE_INTEGER), ('UserTime', LARGE_INTEGER), ('KernelTime', LARGE_INTEGER), ('ImageNameSize', RPC_UNICODE_STRING), ('BasePriority', LONG), ('UniqueProcessId', DWORD), ('InheritedFromUniqueProcessId', DWORD), ('HandleCount', ULONG), ('SessionId', ULONG), ('SpareUl3', ULONG), ('PeakVirtualSize', ULONG), ('VirtualSize', ULONG), ('PageFaultCount', ULONG), ('PeakWorkingSetSize', ULONG), ('WorkingSetSize', ULONG), ('QuotaPeakPagedPoolUsage', ULONG), ('QuotaPagedPoolUsage', ULONG), ('QuotaPeakNonPagedPoolUsage', ULONG), ('QuotaNonPagedPoolUsage', ULONG), ('PagefileUsage', ULONG), ('PeakPagefileUsage', ULONG), ('PrivatePageCount', ULONG), ('ImageName', WSTR_STRIPPED), ('pSid', SID))

class PTS_SYS_PROCESS_INFORMATION(NDRPOINTER):
    referent = (('Data', TS_SYS_PROCESS_INFORMATION),)

class TS_ALL_PROCESSES_INFO(NDRSTRUCT):
    structure = (('pTsProcessInfo', TS_SYS_PROCESS_INFORMATION), ('SizeOfSid', DWORD), ('pSid', TS_CHAR))

class TS_ALL_PROCESSES_INFO_ARRAY(NDRUniConformantVaryingArray):
    item = TS_SYS_PROCESS_INFORMATION

class PTS_ALL_PROCESSES_INFO(NDRPOINTER):
    referent = (('Data', TS_ALL_PROCESSES_INFO_ARRAY),)

class WINSTATIONCONFIG2(NDRSTRUCT):
    pass

class CLIENT_STACK_ADDRESS(NDRSTRUCT):
    pass

class RpcOpenSession(NDRCALL):
    opnum = 0
    structure = (('SessionId', ULONG), ('phSession', handle_t))

class RpcOpenSessionResponse(NDRCALL):
    structure = (('phSession', SESSION_HANDLE), ('ErrorCode', ULONG))

class RpcCloseSession(NDRCALL):
    opnum = 1
    structure = (('phSession', SESSION_HANDLE),)

class RpcCloseSessionResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcConnect(NDRCALL):
    opnum = 2
    structure = (('hSession', SESSION_HANDLE), ('TargetSessionId', LONG), ('szPassword', WSTR))

class RpcConnectResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcDisconnect(NDRCALL):
    opnum = 3
    structure = (('hSession', SESSION_HANDLE),)

class RpcDisconnectResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcLogoff(NDRCALL):
    opnum = 4
    structure = (('hSession', SESSION_HANDLE),)

class RpcLogoffResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcGetUserName(NDRCALL):
    opnum = 5
    structure = (('hSession', SESSION_HANDLE),)

class RpcGetUserNameResponse(NDRCALL):
    structure = (('pszUserName', LPWCHAR_STRIPPED), ('pszDomain', LPWCHAR_STRIPPED), ('ErrorCode', ULONG))

class RpcGetTerminalName(NDRCALL):
    opnum = 6
    structure = (('hSession', SESSION_HANDLE),)

class RpcGetTerminalNameResponse(NDRCALL):
    structure = (('pszTerminalName', LPWCHAR_STRIPPED), ('ErrorCode', ULONG))

class RpcGetState(NDRCALL):
    opnum = 7
    structure = (('hSession', SESSION_HANDLE),)

class RpcGetStateResponse(NDRCALL):
    structure = (('plState', WINSTATIONSTATECLASS), ('ErrorCode', ULONG))

class RpcIsSessionDesktopLocked(NDRCALL):
    opnum = 8
    structure = (('hSession', SESSION_HANDLE),)

class RpcIsSessionDesktopLockedResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcShowMessageBox(NDRCALL):
    opnum = 9
    structure = (('hSession', SESSION_HANDLE), ('szTitle', WSTR), ('szMessage', WSTR), ('ulStyle', ULONG), ('ulTimeout', ULONG), ('bDoNotWait', BOOL))

class RpcShowMessageBoxResponse(NDRCALL):
    structure = (('pulResponse', MSGBOX_ENUM), ('ErrorCode', ULONG))

class RpcGetTimes(NDRCALL):
    opnum = 10
    structure = (('hSession', SESSION_HANDLE),)

class RpcGetTimesResponse(NDRCALL):
    structure = (('pConnectTime', SYSTEM_TIMESTAMP), ('pDisconnectTime', SYSTEM_TIMESTAMP), ('pLogonTime', SYSTEM_TIMESTAMP), ('ErrorCode', ULONG))

class RpcGetSessionCounters(NDRCALL):
    opnum = 11
    structure = (('hBinding', handle_t), ('uEntries', LONG))

class RpcGetSessionCountersResponse(NDRCALL):
    structure = (('pCounter', PTS_COUNTER), ('ErrorCode', ULONG))

class RpcGetSessionInformation(NDRCALL):
    opnum = 12
    structure = (('SessionId', LONG),)

class RpcGetSessionInformationResponse(NDRCALL):
    structure = (('pSessionInfo', LSMSESSIONINFORMATION), ('ErrorCode', ULONG))

class RpcSwitchToServicesSession(NDRCALL):
    opnum = 13
    structure = (('hBinding', handle_t),)

class RpcSwitchToServicesSessionResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcRevertFromServicesSession(NDRCALL):
    opnum = 14
    structure = (('hBinding', handle_t),)

class RpcRevertFromServicesSessionResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcGetLoggedOnCount(NDRCALL):
    opnum = 15
    structure = (('hBinding', handle_t),)

class RpcGetLoggedOnCountResponse(NDRCALL):
    structure = (('pUserSessions', ULONG), ('pDeviceSessions', ULONG), ('ErrorCode', ULONG))

class RpcGetSessionType(NDRCALL):
    opnum = 16
    structure = (('SessionId', LONG),)

class RpcGetSessionTypeResponse(NDRCALL):
    structure = (('pSessionType', SESSIONTYPE), ('ErrorCode', ULONG))

class RpcGetSessionInformationEx(NDRCALL):
    opnum = 17
    structure = (('SessionId', LONG), ('Level', DWORD))

class RpcGetSessionInformationExResponse(NDRCALL):
    structure = (('LSMSessionInfoExPtr', PLSMSESSIONINFORMATION_EX), ('ErrorCode', ULONG))

class RpcWaitForSessionState(NDRCALL):
    opnum = 0
    structure = (('SessionId', LONG), ('State', LONG), ('Timeout', ULONG))

class RpcWaitForSessionStateResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcRegisterAsyncNotification(NDRCALL):
    opnum = 1
    structure = (('SessionId', LONG), ('Mask', ULONG))

class RpcRegisterAsyncNotificationResponse(NDRCALL):
    structure = (('phNotify', NOTIFY_HANDLE), ('ErrorCode', ULONG))

class RpcWaitAsyncNotification(NDRCALL):
    opnum = 2
    structure = (('hNotify', context_handle),)

class RpcWaitAsyncNotificationResponse(NDRCALL):
    structure = (('SessionChange', PSESSION_CHANGE), ('pEntries', ULONG), ('ErrorCode', ULONG))

class RpcUnRegisterAsyncNotification(NDRCALL):
    opnum = 3
    structure = (('hNotify', NOTIFY_HANDLE),)

class RpcUnRegisterAsyncNotificationResponse(NDRCALL):
    structure = (('hNotify', NOTIFY_HANDLE), ('ErrorCode', ULONG))

class RpcOpenEnum(NDRCALL):
    opnum = 0
    structure = (('hBinding', handle_t),)

class RpcOpenEnumResponse(NDRCALL):
    structure = (('phEnum', ENUM_HANDLE), ('ErrorCode', ULONG))

class RpcCloseEnum(NDRCALL):
    opnum = 1
    structure = (('phEnum', ENUM_HANDLE),)

class RpcCloseEnumResponse(NDRCALL):
    structure = (('phEnum', ENUM_HANDLE), ('ErrorCode', ULONG))

class RpcFilterByState(NDRCALL):
    opnum = 2
    structure = (('hEnum', ENUM_HANDLE), ('State', LONG), ('bInvert', BOOL))

class RpcFilterByStateResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcFilterByCallersName(NDRCALL):
    opnum = 3
    structure = (('hEnum', ENUM_HANDLE),)

class RpcFilterByCallersNameResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcEnumAddFilter(NDRCALL):
    opnum = 4
    structure = (('hEnum', ENUM_HANDLE), ('hSubEnum', ENUM_HANDLE))

class RpcEnumAddFilterResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcGetEnumResult(NDRCALL):
    opnum = 5
    structure = (('hEnum', ENUM_HANDLE), ('Level', DWORD))

class RpcGetEnumResultResponse(NDRCALL):
    structure = (('ppSessionEnumResult', PSESSIONENUM), ('pEntries', ULONG), ('ErrorCode', ULONG))

class RpcFilterBySessionType(NDRCALL):
    opnum = 6
    structure = (('hEnum', ENUM_HANDLE), ('pSessionType', GUID))

class RpcFilterBySessionTypeResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcGetSessionIds(NDRCALL):
    opnum = 8
    structure = (('handle_t', handle_t), ('Filter', SESSION_FILTER), ('MaxEntries', ULONG))

class RpcGetSessionIdsResponse(NDRCALL):
    structure = (('pSessionIds', LONG_ARRAY), ('pcSessionIds', ULONG), ('ErrorCode', ULONG))

class RpcGetEnumResultEx(NDRCALL):
    opnum = 9
    structure = (('hEnum', ENUM_HANDLE), ('Level', DWORD))

class RpcGetEnumResultExResponse(NDRCALL):
    structure = (('ppSessionEnumResult', PSESSIONENUM), ('pEntries', ULONG), ('ErrorCode', ULONG))

class RpcGetAllSessions(NDRCALL):
    opnum = 10
    structure = (('pLevel', ULONG),)

class RpcGetAllSessionsResponse(NDRCALL):
    structure = (('pLevel', ULONG), ('ppSessionData', PEXECENVDATA), ('pcEntries', ULONG), ('ErrorCode', ULONG))

class RpcGetAllSessionsEx(NDRCALL):
    opnum = 11
    structure = (('Level', ULONG),)

class RpcGetAllSessionsExResponse(NDRCALL):
    structure = (('Buffer', UNKNOWNDATA),)

class RpcGetClientData(NDRCALL):
    opnum = 0
    structure = (('SessionId', ULONG),)

class RpcGetClientDataResponse(NDRCALL):
    structure = (('ppBuff', PWINSTATIONCLIENT), ('pOutBuffByteLen', ULONG), ('ErrorCode', ULONG))

class RpcGetConfigData(NDRCALL):
    opnum = 1
    structure = (('SessionId', ULONG),)

class RpcGetConfigDataResponse(NDRCALL):
    structure = (('ppBuff', PWINSTATIONCONFIG), ('pOutBuffByteLen', ULONG), ('ErrorCode', ULONG))

class RpcGetProtocolStatus(NDRCALL):
    opnum = 2
    structure = (('SessionId', ULONG), ('InfoType', PROTOCOLSTATUS_INFO_TYPE))

class RpcGetProtocolStatusResponse(NDRCALL):
    structure = (('ppProtoStatus', PROTOCOLSTATUS_INFO_TYPE), ('pcbProtoStatus', PPROTOCOLSTATUS), ('ErrorCode', ULONG))

class RpcGetLastInputTime(NDRCALL):
    opnum = 3
    structure = (('SessionId', ULONG),)

class RpcGetLastInputTimeResponse(NDRCALL):
    structure = (('pLastInputTime', SYSTEM_TIMESTAMP), ('ErrorCode', ULONG))

class RpcGetRemoteAddress(NDRCALL):
    opnum = 4
    structure = (('SessionId', ULONG),)

class RpcGetRemoteAddressResponse(NDRCALL):
    structure = (('pRemoteAddress', RCM_REMOTEADDRESS), ('ErrorCode', ULONG))

class RpcShadow(NDRCALL):
    opnum = 5
    structure = (('szTargetServerName', WSTR), ('TargetSessionId', ULONG), ('HotKeyVk', BYTE), ('HotkeyModifiers', USHORT))

class RpcShadowResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcShadowTarget(NDRCALL):
    """
    HRESULT RpcShadowTarget(
    [in] handle_t hBinding,
    [in] ULONG SessionId,
    [in, size_is(ConfigSize)] PBYTE pConfig,
    [in, range(0, 0x8000)] ULONG ConfigSize,
    [in, size_is(AddressSize)] PBYTE pAddress,
    [in, range(0, 0x1000)] ULONG AddressSize,
    [in, size_is(ModuleDataSize)] PBYTE pModuleData,
    [in, range(0, 0x1000)] ULONG ModuleDataSize,
    [in, size_is(ThinwireDataSize)]
    PBYTE pThinwireData,
    [in, range(0, 0x1000)] ULONG ThinwireDataSize,
    [in, string] WCHAR* szClientName
    );
    """
    opnum = 6

class RpcShadowTargetResponse(NDRCALL):
    structure = (('Buffer', UNKNOWNDATA),)

class RpcShadowStop(NDRCALL):
    opnum = 7
    structure = (('SessionId', ULONG),)

class RpcShadowStopResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcGetAllListeners(NDRCALL):
    opnum = 8
    structure = (('Level', DWORD),)

class RpcGetAllListenersResponse(NDRCALL):
    structure = (('ppListeners', PLISTENERENUM), ('pNumListeners', ULONG), ('ErrorCode', ULONG))

class RpcGetSessionProtocolLastInputTime(NDRCALL):
    opnum = 9
    '\n    HRESULT RpcGetSessionProtocolLastInputTime(\n    [in] handle_t hBinding,\n    [in] ULONG SessionId,\n    [in] PROTOCOLSTATUS_INFO_TYPE InfoType,\n    [out, size_is(,*pcbProtoStatus )]\n    unsigned char** ppProtoStatus,\n    [out] ULONG* pcbProtoStatus,\n    [out] hyper* pLastInputTime\n    );\n    '

class RpcGetSessionProtocolLastInputTimeResponse(NDRCALL):
    structure = (('Data', UNKNOWNDATA),)

class RpcGetUserCertificates(NDRCALL):
    opnum = 10
    '\n    HRESULT RpcGetUserCertificates(\n    [in] handle_t hBinding,\n    [in] ULONG SessionId,\n    [out] ULONG* pcCerts,\n    [out, size_is(, *pcbCerts)] byte** ppbCerts,\n    [out] ULONG* pcbCerts\n    );'

class RpcGetUserCertificatesResponse(NDRCALL):
    structure = (('Data', UNKNOWNDATA),)

class RpcQuerySessionData(NDRCALL):
    """
    HRESULT RpcQuerySessionData(
        [in] handle_t hBinding,
        [in] ULONG SessionId,
        [in] QUERY_SESSION_DATA_TYPE type,
        [in, unique, size_is(cbInputData )] byte* pbInputData,
        [in, range(0, 8192)] DWORD cbInputData,
        [out, ref, size_is(cbSessionData), length_is(*pcbReturnLength)] byte* pbSessionData,
        [in, range(0, 8192)] ULONG cbSessionData,
        [out, ref] ULONG* pcbReturnLength,
        [out, ref] ULONG* pcbRequireBufferSize
    );
    """
    opnum = 11

class RpcQuerySessionDataResponse(NDRCALL):
    structure = (('Buffer', UNKNOWNDATA),)

class RpcOpenListener(NDRCALL):
    opnum = 0
    structure = (('szListenerName', WSTR),)

class RpcOpenListenerResponse(NDRCALL):
    structure = (('phListener', HLISTENER), ('ErrorCode', ULONG))

class RpcCloseListener(NDRCALL):
    opnum = 1
    structure = (('phListener', HLISTENER),)

class RpcCloseListenerResponse(NDRCALL):
    structure = (('phListener', HLISTENER), ('ErrorCode', ULONG))

class RpcStopListener(NDRCALL):
    opnum = 2
    structure = (('phListener', HLISTENER),)

class RpcStopListenerResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcStartListener(NDRCALL):
    opnum = 3
    structure = (('phListener', HLISTENER),)

class RpcStartListenerResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class RpcIsListening(NDRCALL):
    opnum = 4
    structure = (('phListener', HLISTENER),)

class RpcIsListeningResponse(NDRCALL):
    structure = (('pbIsListening', BOOLEAN), ('ErrorCode', ULONG))

class RpcWinStationOpenServer(NDRCALL):
    opnum = 0
    structure = (('hBinding', handle_t),)

class RpcWinStationOpenServerResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('phServer', SERVER_HANDLE), ('ErrorCode', BOOLEAN))

class RpcWinStationCloseServer(NDRCALL):
    opnum = 1
    structure = (('hServer', SERVER_HANDLE),)

class RpcWinStationCloseServerResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcIcaServerPing(NDRCALL):
    opnum = 2
    structure = (('hServer', SERVER_HANDLE),)

class RpcIcaServerPingResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationEnumerate(NDRCALL):
    """
    BOOLEAN RpcWinStationEnumerate(
    [in] SERVER_HANDLE hServer,
    [out] DWORD* pResult,
    [in, out] PULONG pEntries,
    [in, out, unique, size_is(*pByteCount)]
    PCHAR pLogonId,
    [in, out] PULONG pByteCount,
    [in, out] PULONG pIndex
    );
    """
    opnum = 3
    structure = (('hServer', SERVER_HANDLE), ('pEntries', PULONG), ('pLogonId', PCHAR), ('pByteCount', PULONG), ('pIndex', PULONG))

class RpcWinStationEnumerateResponse(NDRCALL):
    structure = (('pResult', UNKNOWNDATA),)

class RpcWinStationRename(NDRCALL):
    """
    BOOLEAN RpcWinStationRename(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in, size_is(NameOldSize)] PWCHAR pWinStationNameOld,
        [in, range(0, 256)] DWORD NameOldSize,
        [in, size_is(NameNewSize)] PWCHAR pWinStationNameNew,
        [in, range(0, 256)] DWORD NameNewSize
    );
    """
    opnum = 4
    structure = (('hServer', SERVER_HANDLE), ('pWinStationNameOld', TS_WCHAR), ('NameOldSize', '<L=len(pWinStationNameOld["Data"])'), ('pWinStationNameNew', TS_WCHAR), ('NameNewSize', '<L=len(pWinStationNameNew["Data"])'))

class RpcWinStationRenameResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationQueryInformation(NDRCALL):
    """
    BOOLEAN RpcWinStationQueryInformation(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in] DWORD LogonId,
        [in] DWORD WinStationInformationClass,
        [in, out, unique, size_is(WinStationInformationLength)]
        PCHAR pWinStationInformation,
        [in, range(0, 0x8000)] DWORD WinStationInformationLength,
        [out] DWORD* pReturnLength
    );
    """
    opnum = 5
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('WinStationInformationClass', DWORD), ('buff', ':'))

class RpcWinStationQueryInformationResponse(NDRCALL):
    structure = (('Buffer', UNKNOWNDATA),)

class RpcWinStationSetInformation(NDRCALL):
    """
    BOOLEAN RpcWinStationSetInformation(
    [in] SERVER_HANDLE hServer,
    [out] DWORD* pResult,
    [in] DWORD LogonId,
    [in] DWORD WinStationInformationClass,
    [in, out, unique, size_is(WinStationInformationLength)]
    PCHAR pWinStationInformation,
    [in, range(0, 0x8000)] DWORD WinStationInformationLength
    );
    """
    opnum = 6

class RpcWinStationSetInformationResponse(NDRCALL):
    structure = (('Buffer', UNKNOWNDATA),)

class RpcWinStationSendMessage(NDRCALL):
    opnum = 7
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('pTitle', TS_WCHAR), ('TitleLength', '<L=len(pTitle["Data"])'), ('pMessage', TS_WCHAR), ('MessageLength', '<L=len(pMessage["Data"])'), ('Style', DWORD), ('Timeout', DWORD), ('DoNotWait', BOOLEAN))

class RpcWinStationSendMessageResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pResponse', DWORD), ('ErrorCode', BOOLEAN))

class RpcLogonIdFromWinStationName(NDRCALL):
    opnum = 8
    structure = (('hServer', SERVER_HANDLE), ('pWinStationName', TS_WCHAR), ('NameSize', '<L=len(pWinStationName["Data"])'))

class RpcLogonIdFromWinStationNameResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pLogonId', DWORD), ('ErrorCode', BOOLEAN))

class RpcWinStationNameFromLogonId(NDRCALL):
    opnum = 9
    structure = (('hServer', SERVER_HANDLE), ('LoginId', DWORD), ('pWinStationName', TS_WCHAR), ('NameSize', '<L=%d' % (WINSTATIONNAME_LENGTH + 1)))

class RpcWinStationNameFromLogonIdResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pWinStationName', TS_WCHAR_STRIPPED), ('ErrorCode', BOOLEAN))

class RpcWinStationConnect(NDRCALL):
    opnum = 10
    structure = (('hServer', SERVER_HANDLE), ('ClientLogonId', DWORD), ('ConnectLogonId', DWORD), ('TargetLogonId', DWORD), ('pPassword', TS_WCHAR), ('PasswordSize', '<L=len(pPassword["Data"])'), ('Wait', BOOLEAN))

class RpcWinStationConnectResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationVirtualOpen(NDRCALL):
    opnum = 11
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('Pid', DWORD), ('pVirtualName', TS_CHAR), ('NameSize', '<L=len(pVirtualName["Data"])'))

class RpcWinStationVirtualOpenResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pHandle', ULONG), ('ErrorCode', BOOLEAN))

class RpcWinStationBeepOpen(NDRCALL):
    opnum = 12
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('Pid', DWORD))

class RpcWinStationBeepOpenResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pHandle', ULONG), ('ErrorCode', BOOLEAN))

class RpcWinStationDisconnect(NDRCALL):
    opnum = 13
    structure = (('hServer', SERVER_HANDLE), ('LoginId', DWORD), ('bWait', BOOLEAN))

class RpcWinStationDisconnectResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationReset(NDRCALL):
    opnum = 14
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('bWait', BOOLEAN))

class RpcWinStationResetResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationShutdownSystem(NDRCALL):
    opnum = 15
    structure = (('hServer', SERVER_HANDLE), ('ClientLogonId', DWORD), ('ShutdownFlags', DWORD))

class RpcWinStationShutdownSystemResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationWaitSystemEvent(NDRCALL):
    opnum = 16
    structure = (('hServer', SERVER_HANDLE), ('EventMask', DWORD))

class RpcWinStationWaitSystemEventResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pEventFlags', DWORD), ('ErrorCode', BOOLEAN))

class RpcWinStationShadow(NDRCALL):
    opnum = 17
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('pTargetServerName', TS_LPWCHAR), ('NameSize', '<L=len(pTargetServerName["Data"])'), ('TargetLogonId', DWORD), ('HotKeyVk', BYTE), ('HotkeyModifiers', USHORT))

class RpcWinStationShadowResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationShadowTargetSetup(NDRCALL):
    opnum = 18
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD))

class RpcWinStationShadowTargetSetupResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationShadowTarget(NDRCALL):
    opnum = 19
    '\n    BOOLEAN RpcWinStationShadowTarget(\n        [in] SERVER_HANDLE hServer,\n        [out] DWORD* pResult,\n        [in] DWORD LogonId,\n        [in, size_is(ConfigSize)] PBYTE pConfig,\n        [in, range(0, 0x8000)] DWORD ConfigSize,\n        [in, size_is(AddressSize)] PBYTE pAddress,\n        [in, range(0, 0x1000 )] DWORD AddressSize,\n        [in, size_is(ModuleDataSize)] PBYTE pModuleData,\n        [in, range(0, 0x1000 )] DWORD ModuleDataSize,\n        [in, size_is(ThinwireDataSize)]\n        PBYTE pThinwireData,\n        [in] DWORD ThinwireDataSize,\n        [in, size_is(ClientNameSize)] PBYTE pClientName,\n        [in, range(0, 1024 )] DWORD ClientNameSize\n    );\n    '
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('pConfig', PBYTE), ('ConfigSize', DWORD), ('pAddress', PBYTE), ('AddressSize', DWORD), ('pModuleData', PBYTE), ('ModuleDataSize', DWORD), ('pThinwireData', PBYTE), ('ThinwireDataSize', DWORD), ('pClientName', STR), ('ClientNameSize', DWORD))

class RpcWinStationShadowTargetResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationSetPoolCount(NDRCALL):
    opnum = 26
    structure = (('hServer', SERVER_HANDLE), ('pLicense', TS_CHAR), ('LicenseSize', '<L=len(pLicense["Data"])'))

class RpcWinStationSetPoolCountResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationQueryUpdateRequired(NDRCALL):
    opnum = 27
    structure = (('hServer', SERVER_HANDLE),)

class RpcWinStationQueryUpdateRequiredResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pUpdateFlag', DWORD), ('ErrorCode', BOOLEAN))

class RpcWinStationCallback(NDRCALL):
    opnum = 28
    '\n    BOOLEAN RpcWinStationCallback(\n        [in] SERVER_HANDLE hServer,\n        [out] DWORD* pResult,\n        [in] DWORD LogonId,\n        [in, size_is(PhoneNumberSize)]\n        PWCHAR pPhoneNumber,\n        [in, range(0, 0x1000 )] DWORD PhoneNumberSize\n    );\n    '
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('pPhoneNumber', TS_WCHAR), ('PhoneNumberSize', '<L=len(pPhoneNumber["Data"])'))

class RpcWinStationCallbackResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationBreakPoint(NDRCALL):
    opnum = 29
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('KernelFlag', BOOLEAN))

class RpcWinStationBreakPointResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationReadRegistry(NDRCALL):
    opnum = 30
    structure = (('hServer', SERVER_HANDLE),)

class RpcWinStationReadRegistryResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationWaitForConnect(NDRCALL):
    opnum = 31
    structure = (('hServer', SERVER_HANDLE), ('ClientLogonId', DWORD), ('ClientProcessId', DWORD))

class RpcWinStationWaitForConnectResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationNotifyLogon(NDRCALL):
    opnum = 32
    structure = (('hServer', SERVER_HANDLE), ('ClientLogonId', DWORD), ('ClientProcessId', DWORD), ('fUserIsAdmin', BOOLEAN), ('UserToken', DWORD), ('pDomain', TS_WCHAR), ('DomainSize', '<L=len(pDomain["Data"])'), ('pUserName', TS_WCHAR), ('UserNameSize', '<L=len(pUserName["Data"])'), ('pPassword', TS_WCHAR), ('PasswordSize', '<L=len(pPassword["Data"])'), ('Seed', UCHAR), ('pUserConfig', TS_CHAR), ('ConfigSize', '<L=len(pUserConfig["Data"])'), ('pfIsRedirected', DWORD))

class RpcWinStationNotifyLogonResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pfIsRedirected', BOOLEAN), ('ErrorCode', BOOLEAN))

class RpcWinStationNotifyLogoff(NDRCALL):
    opnum = 33
    structure = (('hServer', SERVER_HANDLE), ('ClientLogonId', DWORD), ('ClientProcessId', DWORD))

class RpcWinStationNotifyLogoffResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class OldRpcWinStationEnumerateProcesses(NDRCALL):
    opnum = 34
    structure = (('hServer', SERVER_HANDLE), ('ByteCount', DWORD))

class OldRpcWinStationEnumerateProcessesResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pProcessBuffer', TS_CHAR), ('ErrorCode', BOOLEAN))

class RpcWinStationAnnoyancePopup(NDRCALL):
    opnum = 35
    structure = (('hServer', SERVER_HANDLE), ('LogonIdld', DWORD))

class RpcWinStationAnnoyancePopupResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN), ('buff', UNKNOWNDATA))

class RpcWinStationEnumerateProcesses(NDRCALL):
    opnum = 36
    structure = (('hServer', SERVER_HANDLE), ('ByteCount', DWORD))

class RpcWinStationEnumerateProcessesResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pProcessBuffer', TS_CHAR), ('ErrorCode', BOOLEAN))

class RpcWinStationTerminateProcess(NDRCALL):
    opnum = 37
    structure = (('hServer', SERVER_HANDLE), ('ProcessId', DWORD), ('ExitCode', DWORD))

class RpcWinStationTerminateProcessResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationNtsdDebug(NDRCALL):
    opnum = 42
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('ProcessId', LONG), ('DbgProcessId', ULONG), ('DbgThreadId', ULONG), ('AttachCompletionRoutine', LPDWORD))

class RpcWinStationNtsdDebugResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationGetAllProcesses(NDRCALL):
    opnum = 43
    structure = (('hServer', SERVER_HANDLE), ('Level', ULONG), ('pNumberOfProcesses', BOUNDED_ULONG))

class RpcWinStationGetAllProcessesResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pNumberOfProcesses', BOUNDED_ULONG), ('buffer', ':'))

class RpcWinStationGetProcessSid(NDRCALL):
    opnum = 44
    structure = (('hServer', SERVER_HANDLE), ('dwUniqueProcessId', DWORD), ('ProcessStartTime', LARGE_INTEGER), ('pProcessUserSid', TS_PBYTE), ('dwSidSize', '<L=len(pProcessUserSid["Data"])'), ('pdwSizeNeeded', DWORD))

class RpcWinStationGetProcessSidResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pProcessUserSid', TS_PBYTE), ('pdwSizeNeeded', DWORD), ('ErrorCode', BOOLEAN))

class RpcWinStationGetTermSrvCountersValue(NDRCALL):
    """
    BOOLEAN RpcWinStationGetTermSrvCountersValue(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in, range(0, 0x1000)] DWORD dwEntries,
        [in, out, size_is(dwEntries)] PTS_COUNTER pCounter
    );
    """
    opnum = 45
    structure = (('hServer', SERVER_HANDLE), ('dwEntries', DWORD), ('pCounter', PTS_COUNTER))

class RpcWinStationGetTermSrvCountersValueResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('pCounter', PTS_COUNTER), ('ErrorCode', BOOLEAN))

class RpcWinStationReInitializeSecurity(NDRCALL):
    opnum = 46
    structure = (('hServer', SERVER_HANDLE),)

class RpcWinStationReInitializeSecurityResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))
'\nLONG RpcWinStationBroadcastSystemMessage(\n [in] SERVER_HANDLE hServer,\n [in] ULONG sessionID,\n [in] ULONG timeOut,\n [in] DWORD dwFlags,\n [in, out, ptr] DWORD* lpdwRecipients,\n [in] ULONG uiMessage,\n [in] UINT_PTR wParam,\n [in] LONG_PTR lParam,\n [in, size_is(bufferSize)] PBYTE pBuffer,\n [in, range(0, 0x8000 )] ULONG bufferSize,\n [in] BOOLEAN fBufferHasValidData,\n [out] LONG* pResponse\n);\n'
'\nLONG RpcWinStationSendWindowMessage(\n [in] SERVER_HANDLE hServer,\n [in] ULONG sessionID,\n [in] ULONG timeOut,\n [in] ULONG hWnd,\n [in] ULONG Msg,\n [in] UINT_PTR wParam,\n [in] LONG_PTR lParam,\n [in, size_is(bufferSize)] PBYTE pBuffer,\n [in, range(0, 0x8000 )] ULONG bufferSize,\n [in] BOOLEAN fBufferHasValidData,\n [out] LONG* pResponse\n);\n'
'\nBOOLEAN RpcWinStationNotifyNewSession(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] DWORD ClientLogonId\n);\n'

class RpcWinStationGetLanAdapterName(NDRCALL):
    """
    BOOLEAN RpcWinStationGetLanAdapterName(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in, range(0, 0x1000)] DWORD PdNameSize,
        [in, size_is(PdNameSize)] PWCHAR pPdName,
        [in, range(0, 1024)] ULONG LanAdapter,
        [out] ULONG* pLength,
        [out, size_is(,*pLength)] PWCHAR* ppLanAdapter
    );
    """
    opnum = 53
    structure = (('hServer', SERVER_HANDLE), ('PdNameSize', '<L=len(pPdName["Data"])'), ('pPdName', TS_WCHAR), ('LanAdapter', ULONG))

class RpcWinStationGetLanAdapterNameResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ppLanAdapter', TS_WCHAR), ('ErrorCode', BOOLEAN))
'\nBOOLEAN RpcWinStationUpdateUserConfig(\n [in] SERVER_HANDLE hServer,\n [in] DWORD ClientLogonId,\n [in] DWORD ClientProcessId,\n [in] DWORD UserToken,\n [out] DWORD* pResult\n);\n'

class RpcWinStationQueryLogonCredentials(NDRCALL):
    """
    BOOLEAN RpcWinStationQueryLogonCredentials(
        [in] SERVER_HANDLE hServer,
        [in] ULONG LogonId,
        [out, size_is(,*pcbCredentials)]
        PCHAR* ppCredentials,
        [in, out] ULONG* pcbCredentials
    );
    """
    opnum = 55
    structure = (('hServer', SERVER_HANDLE), ('LogonId', ULONG), ('pcbCredentials', ULONG))

class RpcWinStationQueryLogonCredentialsResponse(NDRCALL):
    structure = (('pResult', UNKNOWNDATA),)
'\nBOOLEAN RpcWinStationRegisterConsoleNotification(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] ULONG SessionId,\n [in] ULONG_PTR hWnd,\n [in] DWORD dwFlags,\n [in] DWORD dwMask\n);\n'
'\nBOOLEAN RpcWinStationUnRegisterConsoleNotification(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] ULONG SessionId,\n [in] ULONG hWnd\n);\n'

class RpcWinStationUpdateSettings(NDRCALL):
    """
    BOOLEAN RpcWinStationUnRegisterConsoleNotification(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in] ULONG SessionId,
        [in] ULONG hWnd
    );
    """
    opnum = 58
    structure = (('hServer', SERVER_HANDLE), ('SettingsClass', DWORD), ('SettingsParameters', DWORD))

class RpcWinStationUpdateSettingsResponse(NDRCALL):
    structure = (('pResult', UNKNOWNDATA),)

class RpcWinStationShadowStop(NDRCALL):
    opnum = 59
    structure = (('hServer', SERVER_HANDLE), ('LogonId', DWORD), ('bWait', BOOLEAN))

class RpcWinStationShadowStopResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationCloseServerEx(NDRCALL):
    opnum = 60
    structure = (('hServer', SERVER_HANDLE),)

class RpcWinStationCloseServerExResponse(NDRCALL):
    structure = (('phServer', SERVER_HANDLE), ('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

class RpcWinStationIsHelpAssistantSession(NDRCALL):
    opnum = 61
    structure = (('hServer', SERVER_HANDLE), ('SessionId', ULONG))

class RpcWinStationIsHelpAssistantSessionResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))
'\nBOOLEAN RpcWinStationGetMachinePolicy(\n [in] SERVER_HANDLE hServer,\n [in, out, size_is(bufferSize)] PBYTE pPolicy,\n [in, range(0, 0x8000 )] ULONG bufferSize\n);\n'
'\nBOOLEAN RpcWinStationCheckLoopBack(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] DWORD ClientLogonId,\n [in] DWORD TargetLogonId,\n [in, size_is(NameSize)] PWCHAR pTargetServerName,\n [in, range(0, 1024)] DWORD NameSize\n);\n'

class RpcConnectCallback(NDRCALL):
    """
    BOOLEAN RpcConnectCallback(
        [in] SERVER_HANDLE hServer,
        [out] DWORD* pResult,
        [in] DWORD TimeOut,
        [in] ULONG AddressType,
        [in, size_is(AddressSize)] PBYTE pAddress,
        [in, range(0, 0x1000 )] ULONG AddressSize
    );
    """
    opnum = 61
    structure = (('hServer', SERVER_HANDLE), ('TimeOut', DWORD), ('AddressType', ULONG), ('pAddress', TS_PBYTE), ('AddressSize', '<L=len(pAddress["Data"])'))

class RpcConnectCallbackResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN), ('out', UNKNOWNDATA))
'\nBOOLEAN RpcRemoteAssistancePrepareSystemRestore(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult\n);\n'
'\nBOOLEAN RpcWinStationGetAllProcesses_NT6(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] ULONG Level,\n [in, out] BOUNDED_ULONG* pNumberOfProcesses,\n [out, size_is(,*pNumberOfProcesses)]\n PTS_ALL_PROCESSES_INFO_NT6* ppTsAllProcessesInfo\n);\n'
'\nBOOLEAN RpcWinStationRegisterNotificationEvent(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [out] REGISTRATION_HANDLE* pNotificationId,\n [in] ULONG_PTR EventHandle,\n [in] DWORD TargetSessionId,\n [in] DWORD dwMask,\n [in] DWORD dwProcessId\n);\n'
'\nBOOLEAN RpcWinStationUnRegisterNotificationEvent(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n[in, out] REGISTRATION_HANDLE* NotificationId\n);\n'
'\nBOOLEAN RpcWinStationAutoReconnect(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] DWORD LogonId,\n [in] DWORD flags\n);\n'
'\nBOOLEAN RpcWinStationCheckAccess(\n [in] SERVER_HANDLE hServer,\n [out] DWORD* pResult,\n [in] DWORD ClientLogonId,\n [in] DWORD UserToken,\n [in] ULONG LogonId,\n [in] ULONG AccessMask\n);\n'

class RpcWinStationOpenSessionDirectory(NDRCALL):
    opnum = 75
    structure = (('hServer', SERVER_HANDLE), ('pszServerName', WSTR))

class RpcWinStationOpenSessionDirectoryResponse(NDRCALL):
    structure = (('pResult', pResult_ENUM), ('ErrorCode', BOOLEAN))

def hRpcOpenSession(dce, SessionId):
    if False:
        i = 10
        return i + 15
    request = RpcOpenSession()
    request['SessionId'] = SessionId
    return dce.request(request)['phSession']

def hRpcCloseSession(dce, phSession):
    if False:
        print('Hello World!')
    request = RpcCloseSession()
    request['phSession'] = phSession
    return dce.request(request)

def hRpcConnect(dce, hSession, TargetSessionId, Password=None):
    if False:
        return 10
    if Password is None:
        Password = ''
    request = RpcConnect()
    request['hSession'] = hSession
    request['TargetSessionId'] = TargetSessionId
    request['szPassword'] = Password + '\x00'
    try:
        return dce.request(request)
    except DCERPCSessionError as e:
        if e.error_code == 1:
            resp = RpcConnectResponse()
            resp['ErrorCode'] = 0
            return resp
        raise e

def hRpcDisconnect(dce, hSession):
    if False:
        while True:
            i = 10
    request = RpcDisconnect()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcLogoff(dce, hSession):
    if False:
        while True:
            i = 10
    request = RpcLogoff()
    request['hSession'] = hSession
    try:
        return dce.request(request)
    except DCERPCSessionError as e:
        if e.error_code == 268435456:
            resp = RpcLogoffResponse()
            resp['ErrorCode'] = 0
            return resp
        raise e
    return dce.request(request)

def hRpcGetUserName(dce, hSession):
    if False:
        while True:
            i = 10
    request = RpcGetUserName()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcGetTerminalName(dce, hSession):
    if False:
        return 10
    request = RpcGetTerminalName()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcGetState(dce, hSession):
    if False:
        for i in range(10):
            print('nop')
    request = RpcGetState()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcIsSessionDesktopLocked(dce, hSession):
    if False:
        return 10
    request = RpcIsSessionDesktopLocked()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcShowMessageBox(dce, hSession, Title, Message, Style=0, Timeout=0, DoNotWait=True):
    if False:
        i = 10
        return i + 15
    Title = Title if Title is not None else ' '
    Message = Message if Message is not None else ''
    request = RpcShowMessageBox()
    request['hSession'] = hSession
    request['szTitle'] = Title + '\x00'
    request['szMessage'] = Message + '\x00'
    request['ulStyle'] = Style
    request['ulTimeout'] = Timeout
    request['bDoNotWait'] = DoNotWait
    return dce.request(request)

def hRpcGetTimes(dce, hSession):
    if False:
        return 10
    request = RpcGetTimes()
    request['hSession'] = hSession
    return dce.request(request)

def hRpcGetSessionCounters(dce, Entries):
    if False:
        print('Hello World!')
    request = RpcGetSessionCounters()
    request['uEntries'] = Entries
    return dce.request(request)

def hRpcGetSessionInformation(dce, SessionId):
    if False:
        while True:
            i = 10
    request = RpcGetSessionInformation()
    request['SessionId'] = SessionId
    return dce.request(request)

def hRpcGetLoggedOnCount(dce):
    if False:
        while True:
            i = 10
    request = RpcGetLoggedOnCount()
    return dce.request(request)

def hRpcGetSessionType(dce, SessionId):
    if False:
        return 10
    request = RpcGetSessionType()
    request['SessionId'] = SessionId
    return dce.request(request)

def hRpcGetSessionInformationEx(dce, SessionId):
    if False:
        return 10
    request = RpcGetSessionInformationEx()
    request['SessionId'] = SessionId
    request['Level'] = 1
    return dce.request(request)
    "\n    RpcGetSessionInformationExResponse \n    LSMSessionInfoExPtr:            \n    tag:                             1 \n    LSM_SessionInfo_Level1:         \n        SessionState:                    State_Active \n        SessionFlags:                    WTS_SESSIONSTATE_UNLOCK \n        SessionName:                     'RDP-Tcp#0' \n        DomainName:                      'W11-WKS' \n        UserName:                        'john' \n        ConnectTime:                     datetime.datetime(2022, 5, 9, 2, 34, 48, 700543) \n        DisconnectTime:                  datetime.datetime(2022, 5, 9, 2, 34, 48, 547684) \n        LogonTime:                       datetime.datetime(2022, 5, 9, 2, 23, 31, 119361) \n        LastInputTime:                   datetime.datetime(1601, 1, 1, 2, 20, 54) \n        ProtocolDataSize:                1816 \n        ProtocolData:                    \n    "

def hRpcWaitForSessionState(dce, SessionId, State, Timeout):
    if False:
        return 10
    request = RpcWaitForSessionState()
    request['SessionId'] = SessionId
    request['State'] = State
    request['Timeout'] = Timeout
    return dce.request(request)

def hRpcRegisterAsyncNotification(dce, SessionId, Mask):
    if False:
        i = 10
        return i + 15
    request = RpcRegisterAsyncNotification()
    request['SessionId'] = SessionId
    request['Mask'] = Mask
    return dce.request(request)['phNotify']

def hRpcWaitAsyncNotification(dce, hNotify):
    if False:
        print('Hello World!')
    request = RpcWaitAsyncNotification()
    request['hNotify'] = hNotify
    return dce.request(request)

def hRpcUnRegisterAsyncNotification(dce, hNotify):
    if False:
        for i in range(10):
            print('nop')
    request = RpcUnRegisterAsyncNotification()
    request['hNotify'] = hNotify
    return dce.request(request)

def hRpcOpenEnum(dce):
    if False:
        for i in range(10):
            print('nop')
    request = RpcOpenEnum()
    return dce.request(request)['phEnum']

def hRpcCloseEnum(dce, phEnum):
    if False:
        for i in range(10):
            print('nop')
    request = RpcCloseEnum()
    request['phEnum'] = phEnum
    return dce.request(request)

def hRpcGetEnumResult(dce, hEnum, Level=1):
    if False:
        while True:
            i = 10
    request = RpcGetEnumResult()
    request['hEnum'] = hEnum
    request['Level'] = Level
    return dce.request(request)

def hRpcGetEnumResultEx(dce, hEnum, Level=1):
    if False:
        i = 10
        return i + 15
    request = RpcGetEnumResultEx()
    request['hEnum'] = hEnum
    request['Level'] = Level
    return dce.request(request)

def hRpcGetAllSessions(dce, Level=1):
    if False:
        return 10
    request = RpcGetAllSessions()
    request['pLevel'] = Level
    return dce.request(request)

def hRpcGetClientData(dce, SessionId):
    if False:
        for i in range(10):
            print('nop')
    request = RpcGetClientData()
    request['SessionId'] = SessionId
    try:
        return dce.request(request)
    except:
        return None

def hRpcGetConfigData(dce, SessionId):
    if False:
        while True:
            i = 10
    request = RpcGetConfigData()
    request['SessionId'] = SessionId
    return dce.request(request)

def hRpcGetLastInputTime(dce, SessionId):
    if False:
        for i in range(10):
            print('nop')
    request = RpcGetLastInputTime()
    request['SessionId'] = SessionId
    return dce.request(request)

def hRpcGetRemoteAddress(dce, SessionId):
    if False:
        while True:
            i = 10
    request = RpcGetRemoteAddress()
    request['SessionId'] = SessionId
    try:
        return dce.request(request)
    except:
        return None

def hRpcGetAllListeners(dce):
    if False:
        i = 10
        return i + 15
    request = RpcGetAllListeners()
    request['Level'] = 1
    return dce.request(request)

def hRpcOpenListener(dce, ListenerName):
    if False:
        for i in range(10):
            print('nop')
    request = RpcOpenListener()
    request['szListenerName'] = ListenerName + '\x00'
    return dce.request(request)['phListener']

def hRpcCloseListener(dce, phListener):
    if False:
        for i in range(10):
            print('nop')
    request = RpcCloseListener()
    request['phListener'] = phListener
    return dce.request(request)

def hRpcStopListener(dce, phListener):
    if False:
        return 10
    request = RpcStopListener()
    request['phListener'] = phListener
    return dce.request(request)

def hRpcStartListener(dce, phListener):
    if False:
        while True:
            i = 10
    request = RpcStartListener()
    request['phListener'] = phListener
    return dce.request(request)

def hRpcIsListening(dce, phListener):
    if False:
        while True:
            i = 10
    request = RpcIsListening()
    request['phListener'] = phListener
    return dce.request(request)

def hRpcWinStationOpenServer(dce):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationOpenServer()
    resp = dce.request(request, checkError=False)
    if resp['ErrorCode']:
        return resp['phServer']
    return None

def hRpcWinStationCloseServer(dce, hServer):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationCloseServer()
    request['hServer'] = hServer
    return dce.request(request, checkError=False)

def hRpcIcaServerPing(dce, hServer):
    if False:
        return 10
    request = RpcIcaServerPing()
    request['hServer'] = hServer
    return dce.request(request, checkError=False)

def hRpcWinStationSendMessage(dce, hServer, LogonId, Title, Message, DoNotWait=True):
    if False:
        return 10
    request = RpcWinStationSendMessage()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    request['pTitle'] = ZEROPAD(Title, 1024)
    request['pMessage'] = ZEROPAD(Message, 1024)
    request['DoNotWait'] = DoNotWait
    return dce.request(request, checkError=False)

def hRpcLogonIdFromWinStationName(dce, hServer, WinStationName):
    if False:
        for i in range(10):
            print('nop')
    request = RpcLogonIdFromWinStationName()
    request['hServer'] = hServer
    request['pWinStationName'] = ZEROPAD(WinStationName, WINSTATIONNAME_LENGTH + 1)
    return dce.request(request, checkError=False)

def hRpcWinStationNameFromLogonId(dce, hServer, LoginId):
    if False:
        print('Hello World!')
    request = RpcWinStationNameFromLogonId()
    request['hServer'] = hServer
    request['LoginId'] = LoginId
    request['pWinStationName'] = ZEROPAD('', WINSTATIONNAME_LENGTH + 1)
    return dce.request(request, checkError=False)

def hRpcWinStationConnect(dce, hServer, ClientLogonId, ConnectLogonId, TargetLogonId, Password, Wait=False):
    if False:
        print('Hello World!')
    request = RpcWinStationConnect()
    request['hServer'] = hServer
    request['ClientLogonId'] = ClientLogonId
    request['ConnectLogonId'] = ConnectLogonId
    request['TargetLogonId'] = TargetLogonId
    request['pPassword'] = Password + '\x00'
    request['Wait'] = Wait
    return dce.request(request, checkError=False)

def hRpcWinStationDisconnect(dce, hServer, LoginId, bWait=False):
    if False:
        return 10
    request = RpcWinStationDisconnect()
    request['hServer'] = hServer
    request['LoginId'] = LoginId
    request['bWait'] = bWait
    return dce.request(request, checkError=False)

def hRpcWinStationReset(dce, hServer, LogonId, bWait=False):
    if False:
        return 10
    request = RpcWinStationReset()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    request['bWait'] = bWait
    return dce.request(request, checkError=False)

def hRpcWinStationShutdownSystem(dce, hServer, ClientLogonId, ShutdownFlags):
    if False:
        print('Hello World!')
    request = RpcWinStationShutdownSystem()
    request['hServer'] = hServer
    request['ClientLogonId'] = ClientLogonId
    request['ShutdownFlags'] = ShutdownFlags
    return dce.request(request, checkError=False)

def hRpcWinStationWaitSystemEvent(dce, hServer, EventMask):
    if False:
        while True:
            i = 10
    request = RpcWinStationWaitSystemEvent()
    request['hServer'] = hServer
    request['EventMask'] = EventMask
    return dce.request(request, checkError=False)

def hRpcWinStationShadow(dce, hServer, LogonId, pTargetServerName, TargetLogonId, HotKeyVk, HotkeyModifiers):
    if False:
        return 10
    request = RpcWinStationShadow()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    request['pTargetServerName'] = pTargetServerName
    request['TargetLogonId'] = TargetLogonId
    request['HotKeyVk'] = HotKeyVk
    request['HotkeyModifiers'] = HotkeyModifiers
    return dce.request(request, checkError=False)

def hRpcWinStationShadowTargetSetup(dce, hServer, LogonId):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationShadowTargetSetup()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    return dce.request(request, checkError=False)

def hRpcWinStationBreakPoint(dce, hServer, LogonId, KernelFlag):
    if False:
        print('Hello World!')
    request = RpcWinStationBreakPoint()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    request['KernelFlag'] = KernelFlag
    return dce.request(request, checkError=False)

def hRpcWinStationReadRegistry(dce, hServer):
    if False:
        return 10
    request = RpcWinStationReadRegistry()
    request['hServer'] = hServer
    return dce.request(request, checkError=False)

def hOldRpcWinStationEnumerateProcesses(dce, hServer, ByteCount):
    if False:
        return 10
    request = OldRpcWinStationEnumerateProcesses()
    request['hServer'] = hServer
    request['ByteCount'] = ByteCount
    return dce.request(request, checkError=False)

def hRpcWinStationEnumerateProcesses(dce, hServer, ByteCount):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationEnumerateProcesses()
    request['hServer'] = hServer
    request['ByteCount'] = ByteCount
    return dce.request(request, checkError=False)

def hRpcWinStationTerminateProcess(dce, hServer, ProcessId, ExitCode=0):
    if False:
        while True:
            i = 10
    request = RpcWinStationTerminateProcess()
    request['hServer'] = hServer
    request['ProcessId'] = ProcessId
    request['ExitCode'] = ExitCode
    return dce.request(request, checkError=False)

def hRpcWinStationGetAllProcesses(dce, hServer):
    if False:
        return 10
    request = RpcWinStationGetAllProcesses()
    request['hServer'] = hServer
    request['Level'] = 0
    request['pNumberOfProcesses'] = 32768
    resp = dce.request(request, checkError=False)
    data = resp.getData()
    bResult = bool(data[-1])
    if not bResult:
        raise DCERPCSessionError(error_code=resp['pResult'])
    data = data[:-1]
    procs = []
    if not resp['pNumberOfProcesses']:
        return procs
    offset = 0
    arrayOffset = 0
    while 1:
        offset = data.find(b'\x02\x00')
        if offset > 12:
            break
        data = data[offset + 2:]
        arrayOffset = arrayOffset + offset + 2
    procInfo = ''
    while len(data) > 1:
        if len(data[len(procInfo):]) < 16:
            break
        (b, c, d, e) = struct.unpack('<LLLL', data[len(procInfo):len(procInfo) + 16])
        if b:
            data = data[len(procInfo) - 4:]
        elif c:
            data = data[len(procInfo):]
        elif d:
            data = data[len(procInfo) + 4:]
        elif e:
            data = data[len(procInfo) + 8:]
        procInfo = TS_SYS_PROCESS_INFORMATION()
        procInfo.fromString(data)
        procs.append(procInfo)
    return procs

def hRpcWinStationGetProcessSid(dce, hServer, dwUniqueProcessId, ProcessStartTime):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationGetProcessSid()
    request['hServer'] = hServer
    request['dwUniqueProcessId'] = dwUniqueProcessId
    request['ProcessStartTime'] = ProcessStartTime
    request['pProcessUserSid'] = b'\x00' * 28
    resp = dce.request(request, checkError=False)
    if resp['pResult'] == pResult_ENUM.ERROR_STATUS_BUFFER_TOO_SMALL:
        sizeNeeded = resp['pdwSizeNeeded']
        request['pProcessUserSid'] = b'\x00' * sizeNeeded
        request['dwSidSize'] = sizeNeeded
        resp = dce.request(request, checkError=False)
    if resp['ErrorCode']:
        return format_sid(resp['pProcessUserSid'])

def hRpcWinStationReInitializeSecurity(dce, hServer):
    if False:
        while True:
            i = 10
    request = RpcWinStationReInitializeSecurity()
    request['hServer'] = hServer
    return dce.request(request, checkError=False)

def hRpcWinStationGetLanAdapterName(dce, hServer, pPdName, LanAdapter):
    if False:
        while True:
            i = 10
    request = RpcWinStationGetLanAdapterName()
    request['hServer'] = hServer
    request['pPdName'] = hServer
    request['LanAdapter'] = hServer
    return dce.request(request, checkError=False)

def hRpcWinStationUpdateSettings(dce, hServer, SettingsClass, SettingsParameters):
    if False:
        print('Hello World!')
    request = RpcWinStationUpdateSettings()
    request['hServer'] = hServer
    request['SettingsClass'] = hServer
    request['SettingsParameters'] = hServer
    return dce.request(request, checkError=False)

def hRpcWinStationShadowStop(dce, hServer, LogonId, bWait):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationShadowStop()
    request['hServer'] = hServer
    request['LogonId'] = LogonId
    request['bWait'] = bWait
    return dce.request(request, checkError=False)

def hRpcWinStationCloseServerEx(dce, hServer):
    if False:
        while True:
            i = 10
    request = RpcWinStationShadowStop()
    request['hServer'] = hServer
    return dce.request(request, checkError=False)

def hRpcWinStationIsHelpAssistantSession(dce, hServer, SessionId):
    if False:
        i = 10
        return i + 15
    request = RpcWinStationShadowStop()
    request['hServer'] = hServer
    request['SessionId'] = SessionId
    return dce.request(request, checkError=False)

def hRpcWinStationOpenSessionDirectory(dce, hServer, pszServerName):
    if False:
        print('Hello World!')
    request = RpcWinStationShadowStop()
    request['hServer'] = hServer
    request['pszServerName'] = pszServerName
    return dce.request(request, checkError=False)

class TSTSEndpoint:

    def __init__(self, smb, target_ip, stringbinding, endpoint, kerberos=False):
        if False:
            for i in range(10):
                print('nop')
        self._stringbinding = stringbinding.format(target_ip)
        self._endpoint = endpoint
        self._smbconnection = smb
        self._bind()
        self.request = self._dce.request

    def _bind(self):
        if False:
            return 10
        self._rpctransport = transport.DCERPCTransportFactory(self._stringbinding)
        self._rpctransport.set_smb_connection(self._smbconnection)
        self._dce = self._rpctransport.get_dce_rpc()
        self._dce.set_credentials(*self._rpctransport.get_credentials())
        self._dce.connect()
        self._dce.set_auth_level(RPC_C_AUTHN_LEVEL_PKT_PRIVACY)
        self._dce.bind(self._endpoint)
        return self._dce

    def _disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        self._dce.disconnect()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, type, value, traceback):
        if False:
            return 10
        self._disconnect()

class TermSrvSession(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            while True:
                i = 10
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\LSM_API_service]', endpoint=TermSrvSession_UUID)
    hRpcOpenSession = hRpcOpenSession
    hRpcCloseSession = hRpcCloseSession
    hRpcConnect = hRpcConnect
    hRpcDisconnect = hRpcDisconnect
    hRpcLogoff = hRpcLogoff
    hRpcGetUserName = hRpcGetUserName
    hRpcGetTerminalName = hRpcGetTerminalName
    hRpcGetState = hRpcGetState
    hRpcIsSessionDesktopLocked = hRpcIsSessionDesktopLocked
    hRpcShowMessageBox = hRpcShowMessageBox
    hRpcGetTimes = hRpcGetTimes
    hRpcGetSessionCounters = hRpcGetSessionCounters
    hRpcGetSessionInformation = hRpcGetSessionInformation
    hRpcGetLoggedOnCount = hRpcGetLoggedOnCount
    hRpcGetSessionType = hRpcGetSessionType
    hRpcGetSessionInformationEx = hRpcGetSessionInformationEx

class TermSrvNotification(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            print('Hello World!')
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\LSM_API_service]', endpoint=TermSrvNotification_UUID)
    hRpcWaitForSessionState = hRpcWaitForSessionState
    hRpcRegisterAsyncNotification = hRpcRegisterAsyncNotification
    hRpcWaitAsyncNotification = hRpcWaitAsyncNotification
    hRpcUnRegisterAsyncNotification = hRpcUnRegisterAsyncNotification

class TermSrvEnumeration(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            while True:
                i = 10
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\LSM_API_service]', endpoint=TermSrvEnumeration_UUID)
    hRpcOpenEnum = hRpcOpenEnum
    hRpcCloseEnum = hRpcCloseEnum
    hRpcGetEnumResult = hRpcGetEnumResult
    hRpcGetEnumResultEx = hRpcGetEnumResultEx
    hRpcGetAllSessions = hRpcGetAllSessions

class RCMPublic(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            i = 10
            return i + 15
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\TermSrv_API_service]', endpoint=RCMPublic_UUID)
    hRpcGetClientData = hRpcGetClientData
    hRpcGetConfigData = hRpcGetConfigData
    hRpcGetLastInputTime = hRpcGetLastInputTime
    hRpcGetRemoteAddress = hRpcGetRemoteAddress
    hRpcGetAllListeners = hRpcGetAllListeners

class RcmListener(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\TermSrv_API_service]', endpoint=RcmListener_UUID)
    hRpcOpenListener = hRpcOpenListener
    hRpcCloseListener = hRpcCloseListener
    hRpcStopListener = hRpcStopListener
    hRpcStartListener = hRpcStartListener
    hRpcIsListening = hRpcIsListening

class LegacyAPI(TSTSEndpoint):

    def __init__(self, smb, target_ip):
        if False:
            return 10
        super().__init__(smb, target_ip, stringbinding='ncacn_np:{}[\\pipe\\Ctx_WinStation_API_service]', endpoint=LegacyAPI_UUID)
    hRpcWinStationOpenServer = hRpcWinStationOpenServer
    hRpcWinStationCloseServer = hRpcWinStationCloseServer
    hRpcIcaServerPing = hRpcIcaServerPing
    hRpcWinStationSendMessage = hRpcWinStationSendMessage
    hRpcLogonIdFromWinStationName = hRpcLogonIdFromWinStationName
    hRpcWinStationNameFromLogonId = hRpcWinStationNameFromLogonId
    hRpcWinStationConnect = hRpcWinStationConnect
    hRpcWinStationDisconnect = hRpcWinStationDisconnect
    hRpcWinStationReset = hRpcWinStationReset
    hRpcWinStationShutdownSystem = hRpcWinStationShutdownSystem
    hRpcWinStationWaitSystemEvent = hRpcWinStationWaitSystemEvent
    hRpcWinStationShadow = hRpcWinStationShadow
    hRpcWinStationShadowTargetSetup = hRpcWinStationShadowTargetSetup
    hRpcWinStationBreakPoint = hRpcWinStationBreakPoint
    hRpcWinStationReadRegistry = hRpcWinStationReadRegistry
    hOldRpcWinStationEnumerateProcesses = hOldRpcWinStationEnumerateProcesses
    hRpcWinStationEnumerateProcesses = hRpcWinStationEnumerateProcesses
    hRpcWinStationTerminateProcess = hRpcWinStationTerminateProcess
    hRpcWinStationGetAllProcesses = hRpcWinStationGetAllProcesses
    hRpcWinStationGetProcessSid = hRpcWinStationGetProcessSid
    hRpcWinStationReInitializeSecurity = hRpcWinStationReInitializeSecurity
    hRpcWinStationGetLanAdapterName = hRpcWinStationGetLanAdapterName
    hRpcWinStationUpdateSettings = hRpcWinStationUpdateSettings
    hRpcWinStationShadowStop = hRpcWinStationShadowStop
    hRpcWinStationCloseServerEx = hRpcWinStationCloseServerEx
    hRpcWinStationIsHelpAssistantSession = hRpcWinStationIsHelpAssistantSession