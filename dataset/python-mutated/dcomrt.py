from __future__ import division
from __future__ import print_function
import socket
from struct import pack
from threading import Timer, current_thread
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRPOINTER, NDRUniConformantArray, NDRTLSTRUCT, UNKNOWNDATA
from impacket.dcerpc.v5.dtypes import LPWSTR, ULONGLONG, HRESULT, GUID, USHORT, WSTR, DWORD, LPLONG, LONG, PGUID, ULONG, UUID, WIDESTR, NULL
from impacket import hresult_errors, LOG
from impacket.uuid import string_to_bin, uuidtup_to_bin, generate
from impacket.dcerpc.v5.rpcrt import TypeSerialization1, RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_NONE, RPC_C_AUTHN_LEVEL_PKT_PRIVACY, RPC_C_AUTHN_GSS_NEGOTIATE, RPC_C_AUTHN_WINNT, DCERPCException
from impacket.dcerpc.v5 import transport
CLSID_ActivationContextInfo = string_to_bin('000001a5-0000-0000-c000-000000000046')
CLSID_ActivationPropertiesIn = string_to_bin('00000338-0000-0000-c000-000000000046')
CLSID_ActivationPropertiesOut = string_to_bin('00000339-0000-0000-c000-000000000046')
CLSID_CONTEXT_EXTENSION = string_to_bin('00000334-0000-0000-c000-000000000046')
CLSID_ContextMarshaler = string_to_bin('0000033b-0000-0000-c000-000000000046')
CLSID_ERROR_EXTENSION = string_to_bin('0000031c-0000-0000-c000-000000000046')
CLSID_ErrorObject = string_to_bin('0000031b-0000-0000-c000-000000000046')
CLSID_InstanceInfo = string_to_bin('000001ad-0000-0000-c000-000000000046')
CLSID_InstantiationInfo = string_to_bin('000001ab-0000-0000-c000-000000000046')
CLSID_PropsOutInfo = string_to_bin('00000339-0000-0000-c000-000000000046')
CLSID_ScmReplyInfo = string_to_bin('000001b6-0000-0000-c000-000000000046')
CLSID_ScmRequestInfo = string_to_bin('000001aa-0000-0000-c000-000000000046')
CLSID_SecurityInfo = string_to_bin('000001a6-0000-0000-c000-000000000046')
CLSID_ServerLocationInfo = string_to_bin('000001a4-0000-0000-c000-000000000046')
CLSID_SpecialSystemProperties = string_to_bin('000001b9-0000-0000-c000-000000000046')
IID_IActivation = uuidtup_to_bin(('4d9f4ab8-7d1c-11cf-861e-0020af6e7c57', '0.0'))
IID_IActivationPropertiesIn = uuidtup_to_bin(('000001A2-0000-0000-C000-000000000046', '0.0'))
IID_IActivationPropertiesOut = uuidtup_to_bin(('000001A3-0000-0000-C000-000000000046', '0.0'))
IID_IContext = uuidtup_to_bin(('000001c0-0000-0000-C000-000000000046', '0.0'))
IID_IObjectExporter = uuidtup_to_bin(('99fcfec4-5260-101b-bbcb-00aa0021347a', '0.0'))
IID_IRemoteSCMActivator = uuidtup_to_bin(('000001A0-0000-0000-C000-000000000046', '0.0'))
IID_IRemUnknown = uuidtup_to_bin(('00000131-0000-0000-C000-000000000046', '0.0'))
IID_IRemUnknown2 = uuidtup_to_bin(('00000143-0000-0000-C000-000000000046', '0.0'))
IID_IUnknown = uuidtup_to_bin(('00000000-0000-0000-C000-000000000046', '0.0'))
IID_IClassFactory = uuidtup_to_bin(('00000001-0000-0000-C000-000000000046', '0.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            for i in range(10):
                print('nop')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            return 10
        if self.error_code in hresult_errors.ERROR_MESSAGES:
            error_msg_short = hresult_errors.ERROR_MESSAGES[self.error_code][0]
            error_msg_verbose = hresult_errors.ERROR_MESSAGES[self.error_code][1]
            return 'DCOM SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'DCOM SessionError: unknown error code: 0x%x' % self.error_code
OID = ULONGLONG

class OID_ARRAY(NDRUniConformantArray):
    item = OID

class POID_ARRAY(NDRPOINTER):
    referent = (('Data', OID_ARRAY),)
SETID = ULONGLONG
error_status_t = ULONG
CID = GUID
CLSID = GUID
IID = GUID
PIID = PGUID
IPID = GUID
OXID = ULONGLONG
FLAGS_OBJREF_STANDARD = 1
FLAGS_OBJREF_HANDLER = 2
FLAGS_OBJREF_CUSTOM = 4
FLAGS_OBJREF_EXTENDED = 8
SORF_NOPING = 4096
CTXMSHLFLAGS_BYVAL = 2
CPFLAG_PROPAGATE = 1
CPFLAG_EXPOSE = 2
CPFLAG_ENVOY = 4
ACTVFLAGS_DISABLE_AAA = 2
ACTVFLAGS_ACTIVATE_32_BIT_SERVER = 4
ACTVFLAGS_ACTIVATE_64_BIT_SERVER = 8
ACTVFLAGS_NO_FAILURE_LOG = 32
SPD_FLAG_USE_CONSOLE_SESSION = 1
MAX_REQUESTED_INTERFACES = 32768
MAX_REQUESTED_PROTSEQS = 32768
MIN_ACTPROP_LIMIT = 1
MAX_ACTPROP_LIMIT = 10

class handle_t(NDRSTRUCT):
    structure = (('context_handle_attributes', ULONG), ('context_handle_uuid', UUID))

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        self['context_handle_uuid'] = b'\x00' * 16

    def isNull(self):
        if False:
            while True:
                i = 10
        return self['context_handle_uuid'] == b'\x00' * 16

class COMVERSION(NDRSTRUCT):
    default_major_version = 5
    default_minor_version = 7
    structure = (('MajorVersion', USHORT), ('MinorVersion', USHORT))

    @classmethod
    def set_default_version(cls, major_version=None, minor_version=None):
        if False:
            print('Hello World!')
        if major_version is not None:
            cls.default_major_version = major_version
        if minor_version is not None:
            cls.default_minor_version = minor_version

    def __init__(self, data=None, isNDR64=False):
        if False:
            return 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        if data is None:
            self['MajorVersion'] = self.default_major_version
            self['MinorVersion'] = self.default_minor_version

class PCOMVERSION(NDRPOINTER):
    referent = (('Data', COMVERSION),)

class BYTE_ARRAY(NDRUniConformantArray):
    item = 'c'

class ORPC_EXTENT(NDRSTRUCT):
    structure = (('id', GUID), ('size', ULONG), ('data', BYTE_ARRAY))

class PORPC_EXTENT(NDRPOINTER):
    referent = (('Data', ORPC_EXTENT),)

class EXTENT_ARRAY(NDRUniConformantArray):
    item = PORPC_EXTENT

class PEXTENT_ARRAY(NDRPOINTER):
    referent = (('Data', EXTENT_ARRAY),)

class ORPC_EXTENT_ARRAY(NDRSTRUCT):
    structure = (('size', ULONG), ('reserved', ULONG), ('extent', PEXTENT_ARRAY))

class PORPC_EXTENT_ARRAY(NDRPOINTER):
    referent = (('Data', ORPC_EXTENT_ARRAY),)

class ORPCTHIS(NDRSTRUCT):
    structure = (('version', COMVERSION), ('flags', ULONG), ('reserved1', ULONG), ('cid', CID), ('extensions', PORPC_EXTENT_ARRAY))

class ORPCTHAT(NDRSTRUCT):
    structure = (('flags', ULONG), ('extensions', PORPC_EXTENT_ARRAY))

class MInterfacePointer(NDRSTRUCT):
    structure = (('ulCntData', ULONG), ('abData', BYTE_ARRAY))

class PMInterfacePointerInternal(NDRPOINTER):
    referent = (('Data', MInterfacePointer),)

class PMInterfacePointer(NDRPOINTER):
    referent = (('Data', MInterfacePointer),)

class PPMInterfacePointer(NDRPOINTER):
    referent = (('Data', PMInterfacePointer),)

class OBJREF(NDRSTRUCT):
    commonHdr = (('signature', ULONG), ('flags', ULONG), ('iid', GUID))

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        if data is None:
            self['signature'] = 1464812877

class STDOBJREF(NDRSTRUCT):
    structure = (('flags', ULONG), ('cPublicRefs', ULONG), ('oxid', OXID), ('oid', OID), ('ipid', IPID))

class OBJREF_STANDARD(OBJREF):
    structure = (('std', STDOBJREF), ('saResAddr', ':'))

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        OBJREF.__init__(self, data, isNDR64)
        if data is None:
            self['flags'] = FLAGS_OBJREF_STANDARD

class OBJREF_HANDLER(OBJREF):
    structure = (('std', STDOBJREF), ('clsid', CLSID), ('saResAddr', ':'))

    def __init__(self, data=None, isNDR64=False):
        if False:
            return 10
        OBJREF.__init__(self, data, isNDR64)
        if data is None:
            self['flags'] = FLAGS_OBJREF_HANDLER

class OBJREF_CUSTOM(OBJREF):
    structure = (('clsid', CLSID), ('cbExtension', ULONG), ('ObjectReferenceSize', ULONG), ('pObjectData', ':'))

    def __init__(self, data=None, isNDR64=False):
        if False:
            for i in range(10):
                print('nop')
        OBJREF.__init__(self, data, isNDR64)
        if data is None:
            self['flags'] = FLAGS_OBJREF_CUSTOM

class DATAELEMENT(NDRSTRUCT):
    structure = (('dataID', GUID), ('cbSize', ULONG), ('cbRounded', ULONG), ('Data', ':'))

class DUALSTRINGARRAYPACKED(NDRSTRUCT):
    structure = (('wNumEntries', USHORT), ('wSecurityOffset', USHORT), ('aStringArray', ':'))

    def getDataLen(self, data, offset=0):
        if False:
            return 10
        return self['wNumEntries'] * 2

class OBJREF_EXTENDED(OBJREF):
    structure = (('std', STDOBJREF), ('Signature1', ULONG), ('saResAddr', DUALSTRINGARRAYPACKED), ('nElms', ULONG), ('Signature2', ULONG), ('ElmArray', DATAELEMENT))

    def __init__(self, data=None, isNDR64=False):
        if False:
            for i in range(10):
                print('nop')
        OBJREF.__init__(self, data, isNDR64)
        if data is None:
            self['flags'] = FLAGS_OBJREF_EXTENDED
            self['Signature1'] = 1314085206
            self['Signature1'] = 1314085206
            self['nElms'] = 1314085206

class USHORT_ARRAY(NDRUniConformantArray):
    item = '<H'

class PUSHORT_ARRAY(NDRPOINTER):
    referent = (('Data', USHORT_ARRAY),)

class DUALSTRINGARRAY(NDRSTRUCT):
    structure = (('wNumEntries', USHORT), ('wSecurityOffset', USHORT), ('aStringArray', USHORT_ARRAY))

class PDUALSTRINGARRAY(NDRPOINTER):
    referent = (('Data', DUALSTRINGARRAY),)

class STRINGBINDING(NDRSTRUCT):
    structure = (('wTowerId', USHORT), ('aNetworkAddr', WIDESTR))

class SECURITYBINDING(NDRSTRUCT):
    structure = (('wAuthnSvc', USHORT), ('Reserved', USHORT), ('aPrincName', WIDESTR))

class PROPMARSHALHEADER(NDRSTRUCT):
    structure = (('clsid', CLSID), ('policyId', GUID), ('flags', ULONG), ('cb', ULONG), ('ctxProperty', ':'))

class PROPMARSHALHEADER_ARRAY(NDRUniConformantArray):
    item = PROPMARSHALHEADER

class Context(NDRSTRUCT):
    structure = (('MajorVersion', USHORT), ('MinVersion', USHORT), ('ContextId', GUID), ('Flags', ULONG), ('Reserved', ULONG), ('dwNumExtents', ULONG), ('cbExtents', ULONG), ('MshlFlags', ULONG), ('Count', ULONG), ('Frozen', ULONG), ('PropMarshalHeader', PROPMARSHALHEADER_ARRAY))

class ErrorInfoString(NDRSTRUCT):
    structure = (('dwMax', ULONG), ('dwOffSet', ULONG), ('dwActual', IID), ('Name', WSTR))

class ORPC_ERROR_INFORMATION(NDRSTRUCT):
    structure = (('dwVersion', ULONG), ('dwHelpContext', ULONG), ('iid', IID), ('dwSourceSignature', ULONG), ('Source', ErrorInfoString), ('dwDescriptionSignature', ULONG), ('Description', ErrorInfoString), ('dwHelpFileSignature', ULONG), ('HelpFile', ErrorInfoString))

class EntryHeader(NDRSTRUCT):
    structure = (('Signature', ULONG), ('cbEHBuffer', ULONG), ('cbSize', ULONG), ('reserved', ULONG), ('policyID', GUID))

class EntryHeader_ARRAY(NDRUniConformantArray):
    item = EntryHeader

class ORPC_CONTEXT(NDRSTRUCT):
    structure = (('SignatureVersion', ULONG), ('Version', ULONG), ('cPolicies', ULONG), ('cbBuffer', ULONG), ('cbSize', ULONG), ('hr', ULONG), ('hrServer', ULONG), ('reserved', ULONG), ('EntryHeader', EntryHeader_ARRAY), ('PolicyData', ':'))

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        if data is None:
            self['SignatureVersion'] = 1095652683

class CLSID_ARRAY(NDRUniConformantArray):
    item = CLSID

class PCLSID_ARRAY(NDRPOINTER):
    referent = (('Data', CLSID_ARRAY),)

class DWORD_ARRAY(NDRUniConformantArray):
    item = DWORD

class PDWORD_ARRAY(NDRPOINTER):
    referent = (('Data', DWORD_ARRAY),)

class CustomHeader(TypeSerialization1):
    structure = (('totalSize', DWORD), ('headerSize', DWORD), ('dwReserved', DWORD), ('destCtx', DWORD), ('cIfs', DWORD), ('classInfoClsid', CLSID), ('pclsid', PCLSID_ARRAY), ('pSizes', PDWORD_ARRAY), ('pdwReserved', LPLONG))

    def getData(self, soFar=0):
        if False:
            return 10
        self['headerSize'] = len(TypeSerialization1.getData(self, soFar)) + len(TypeSerialization1.getDataReferents(self, soFar))
        self['cIfs'] = len(self['pclsid'])
        return TypeSerialization1.getData(self, soFar)

class ACTIVATION_BLOB(NDRTLSTRUCT):
    structure = (('dwSize', ULONG), ('dwReserved', ULONG), ('CustomHeader', CustomHeader), ('Property', UNKNOWNDATA))

    def getData(self, soFar=0):
        if False:
            i = 10
            return i + 15
        self['dwSize'] = len(self['CustomHeader'].getData(soFar)) + len(self['CustomHeader'].getDataReferents(soFar)) + len(self['Property'])
        self['CustomHeader']['totalSize'] = self['dwSize']
        return NDRTLSTRUCT.getData(self)

class IID_ARRAY(NDRUniConformantArray):
    item = IID

class PIID_ARRAY(NDRPOINTER):
    referent = (('Data', IID_ARRAY),)

class InstantiationInfoData(TypeSerialization1):
    structure = (('classId', CLSID), ('classCtx', DWORD), ('actvflags', DWORD), ('fIsSurrogate', LONG), ('cIID', DWORD), ('instFlag', DWORD), ('pIID', PIID_ARRAY), ('thisSize', DWORD), ('clientCOMVersion', COMVERSION))

class SpecialPropertiesData(TypeSerialization1):
    structure = (('dwSessionId', ULONG), ('fRemoteThisSessionId', LONG), ('fClientImpersonating', LONG), ('fPartitionIDPresent', LONG), ('dwDefaultAuthnLvl', DWORD), ('guidPartition', GUID), ('dwPRTFlags', DWORD), ('dwOrigClsctx', DWORD), ('dwFlags', DWORD), ('Reserved0', DWORD), ('Reserved0', DWORD), ('Reserved', '32s=""'))

class InstanceInfoData(TypeSerialization1):
    structure = (('fileName', LPWSTR), ('mode', DWORD), ('ifdROT', PMInterfacePointer), ('ifdStg', PMInterfacePointer))

class customREMOTE_REQUEST_SCM_INFO(NDRSTRUCT):
    structure = (('ClientImpLevel', DWORD), ('cRequestedProtseqs', USHORT), ('pRequestedProtseqs', PUSHORT_ARRAY))

class PcustomREMOTE_REQUEST_SCM_INFO(NDRPOINTER):
    referent = (('Data', customREMOTE_REQUEST_SCM_INFO),)

class ScmRequestInfoData(TypeSerialization1):
    structure = (('pdwReserved', LPLONG), ('remoteRequest', PcustomREMOTE_REQUEST_SCM_INFO))

class ActivationContextInfoData(TypeSerialization1):
    structure = (('clientOK', LONG), ('bReserved1', LONG), ('dwReserved1', DWORD), ('dwReserved2', DWORD), ('pIFDClientCtx', PMInterfacePointer), ('pIFDPrototypeCtx', PMInterfacePointer))

class LocationInfoData(TypeSerialization1):
    structure = (('machineName', LPWSTR), ('processId', DWORD), ('apartmentId', DWORD), ('contextId', DWORD))

class COSERVERINFO(NDRSTRUCT):
    structure = (('dwReserved1', DWORD), ('pwszName', LPWSTR), ('pdwReserved', LPLONG), ('dwReserved2', DWORD))

class PCOSERVERINFO(NDRPOINTER):
    referent = (('Data', COSERVERINFO),)

class SecurityInfoData(TypeSerialization1):
    structure = (('dwAuthnFlags', DWORD), ('pServerInfo', PCOSERVERINFO), ('pdwReserved', LPLONG))

class customREMOTE_REPLY_SCM_INFO(NDRSTRUCT):
    structure = (('Oxid', OXID), ('pdsaOxidBindings', PDUALSTRINGARRAY), ('ipidRemUnknown', IPID), ('authnHint', DWORD), ('serverVersion', COMVERSION))

class PcustomREMOTE_REPLY_SCM_INFO(NDRPOINTER):
    referent = (('Data', customREMOTE_REPLY_SCM_INFO),)

class ScmReplyInfoData(TypeSerialization1):
    structure = (('pdwReserved', DWORD), ('remoteReply', PcustomREMOTE_REPLY_SCM_INFO))

class HRESULT_ARRAY(NDRUniConformantArray):
    item = HRESULT

class PHRESULT_ARRAY(NDRPOINTER):
    referent = (('Data', HRESULT_ARRAY),)

class MInterfacePointer_ARRAY(NDRUniConformantArray):
    item = MInterfacePointer

class PMInterfacePointer_ARRAY(NDRUniConformantArray):
    item = PMInterfacePointer

class PPMInterfacePointer_ARRAY(NDRPOINTER):
    referent = (('Data', PMInterfacePointer_ARRAY),)

class PropsOutInfo(TypeSerialization1):
    structure = (('cIfs', DWORD), ('piid', PIID_ARRAY), ('phresults', PHRESULT_ARRAY), ('ppIntfData', PPMInterfacePointer_ARRAY))

class REMINTERFACEREF(NDRSTRUCT):
    structure = (('ipid', IPID), ('cPublicRefs', LONG), ('cPrivateRefs', LONG))

class REMINTERFACEREF_ARRAY(NDRUniConformantArray):
    item = REMINTERFACEREF

class REMQIRESULT(NDRSTRUCT):
    structure = (('hResult', HRESULT), ('std', STDOBJREF))

class PREMQIRESULT(NDRPOINTER):
    referent = (('Data', REMQIRESULT),)
REFIPID = GUID

class DCOMCALL(NDRCALL):
    commonHdr = (('ORPCthis', ORPCTHIS),)

class DCOMANSWER(NDRCALL):
    commonHdr = (('ORPCthat', ORPCTHAT),)

class ResolveOxid(NDRCALL):
    opnum = 0
    structure = (('pOxid', OXID), ('cRequestedProtseqs', USHORT), ('arRequestedProtseqs', USHORT_ARRAY))

class ResolveOxidResponse(NDRCALL):
    structure = (('ppdsaOxidBindings', PDUALSTRINGARRAY), ('pipidRemUnknown', IPID), ('pAuthnHint', DWORD), ('ErrorCode', error_status_t))

class SimplePing(NDRCALL):
    opnum = 1
    structure = (('pSetId', SETID),)

class SimplePingResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class ComplexPing(NDRCALL):
    opnum = 2
    structure = (('pSetId', SETID), ('SequenceNum', USHORT), ('cAddToSet', USHORT), ('cDelFromSet', USHORT), ('AddToSet', POID_ARRAY), ('DelFromSet', POID_ARRAY))

class ComplexPingResponse(NDRCALL):
    structure = (('pSetId', SETID), ('pPingBackoffFactor', USHORT), ('ErrorCode', error_status_t))

class ServerAlive(NDRCALL):
    opnum = 3
    structure = ()

class ServerAliveResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class ResolveOxid2(NDRCALL):
    opnum = 4
    structure = (('pOxid', OXID), ('cRequestedProtseqs', USHORT), ('arRequestedProtseqs', USHORT_ARRAY))

class ResolveOxid2Response(NDRCALL):
    structure = (('ppdsaOxidBindings', PDUALSTRINGARRAY), ('pipidRemUnknown', IPID), ('pAuthnHint', DWORD), ('pComVersion', COMVERSION), ('ErrorCode', error_status_t))

class ServerAlive2(NDRCALL):
    opnum = 5
    structure = ()

class ServerAlive2Response(NDRCALL):
    structure = (('pComVersion', COMVERSION), ('ppdsaOrBindings', PDUALSTRINGARRAY), ('pReserved', LPLONG), ('ErrorCode', error_status_t))

class RemoteActivation(NDRCALL):
    opnum = 0
    structure = (('ORPCthis', ORPCTHIS), ('Clsid', GUID), ('pwszObjectName', LPWSTR), ('pObjectStorage', PMInterfacePointer), ('ClientImpLevel', DWORD), ('Mode', DWORD), ('Interfaces', DWORD), ('pIIDs', PIID_ARRAY), ('cRequestedProtseqs', USHORT), ('aRequestedProtseqs', USHORT_ARRAY))

class RemoteActivationResponse(NDRCALL):
    structure = (('ORPCthat', ORPCTHAT), ('pOxid', OXID), ('ppdsaOxidBindings', PDUALSTRINGARRAY), ('pipidRemUnknown', IPID), ('pAuthnHint', DWORD), ('pServerVersion', COMVERSION), ('phr', HRESULT), ('ppInterfaceData', PMInterfacePointer_ARRAY), ('pResults', HRESULT_ARRAY), ('ErrorCode', error_status_t))

class RemoteGetClassObject(NDRCALL):
    opnum = 3
    structure = (('ORPCthis', ORPCTHIS), ('pActProperties', PMInterfacePointer))

class RemoteGetClassObjectResponse(NDRCALL):
    structure = (('ORPCthat', ORPCTHAT), ('ppActProperties', PMInterfacePointer), ('ErrorCode', error_status_t))

class RemoteCreateInstance(NDRCALL):
    opnum = 4
    structure = (('ORPCthis', ORPCTHIS), ('pUnkOuter', PMInterfacePointer), ('pActProperties', PMInterfacePointer))

class RemoteCreateInstanceResponse(NDRCALL):
    structure = (('ORPCthat', ORPCTHAT), ('ppActProperties', PMInterfacePointer), ('ErrorCode', error_status_t))

class RemQueryInterface(DCOMCALL):
    opnum = 3
    structure = (('ripid', REFIPID), ('cRefs', ULONG), ('cIids', USHORT), ('iids', IID_ARRAY))

class RemQueryInterfaceResponse(DCOMANSWER):
    structure = (('ppQIResults', PREMQIRESULT), ('ErrorCode', error_status_t))

class RemAddRef(DCOMCALL):
    opnum = 4
    structure = (('cInterfaceRefs', USHORT), ('InterfaceRefs', REMINTERFACEREF_ARRAY))

class RemAddRefResponse(DCOMANSWER):
    structure = (('pResults', DWORD_ARRAY), ('ErrorCode', error_status_t))

class RemRelease(DCOMCALL):
    opnum = 5
    structure = (('cInterfaceRefs', USHORT), ('InterfaceRefs', REMINTERFACEREF_ARRAY))

class RemReleaseResponse(DCOMANSWER):
    structure = (('ErrorCode', error_status_t),)
OPNUMS = {}

class DCOMConnection:
    """
    This class represents a DCOM Connection. It is in charge of establishing the 
    DCE connection against the portmap, and then launch a thread that will be 
    pinging the objects created against the target.
    In theory, there should be a single instance of this class for every target
    """
    PINGTIMER = None
    OID_ADD = {}
    OID_DEL = {}
    OID_SET = {}
    PORTMAPS = {}

    def __init__(self, target, username='', password='', domain='', lmhash='', nthash='', aesKey='', TGT=None, TGS=None, authLevel=RPC_C_AUTHN_LEVEL_PKT_PRIVACY, oxidResolver=False, doKerberos=False, kdcHost=None):
        if False:
            i = 10
            return i + 15
        self.__target = target
        self.__userName = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = lmhash
        self.__nthash = nthash
        self.__aesKey = aesKey
        self.__TGT = TGT
        self.__TGS = TGS
        self.__authLevel = authLevel
        self.__portmap = None
        self.__oxidResolver = oxidResolver
        self.__doKerberos = doKerberos
        self.__kdcHost = kdcHost
        self.initConnection()

    @classmethod
    def addOid(cls, target, oid):
        if False:
            return 10
        if (target in DCOMConnection.OID_ADD) is False:
            DCOMConnection.OID_ADD[target] = set()
        DCOMConnection.OID_ADD[target].add(oid)
        if (target in DCOMConnection.OID_SET) is False:
            DCOMConnection.OID_SET[target] = {}
            DCOMConnection.OID_SET[target]['oids'] = set()
            DCOMConnection.OID_SET[target]['setid'] = 0

    @classmethod
    def delOid(cls, target, oid):
        if False:
            return 10
        if (target in DCOMConnection.OID_DEL) is False:
            DCOMConnection.OID_DEL[target] = set()
        DCOMConnection.OID_DEL[target].add(oid)
        if (target in DCOMConnection.OID_SET) is False:
            DCOMConnection.OID_SET[target] = {}
            DCOMConnection.OID_SET[target]['oids'] = set()
            DCOMConnection.OID_SET[target]['setid'] = 0

    @classmethod
    def pingServer(cls):
        if False:
            return 10
        try:
            for target in DCOMConnection.OID_SET:
                addedOids = set()
                deletedOids = set()
                if target in DCOMConnection.OID_ADD:
                    addedOids = DCOMConnection.OID_ADD[target]
                    del DCOMConnection.OID_ADD[target]
                if target in DCOMConnection.OID_DEL:
                    deletedOids = DCOMConnection.OID_DEL[target]
                    del DCOMConnection.OID_DEL[target]
                objExporter = IObjectExporter(DCOMConnection.PORTMAPS[target])
                if len(addedOids) > 0 or len(deletedOids) > 0:
                    if 'setid' in DCOMConnection.OID_SET[target]:
                        setId = DCOMConnection.OID_SET[target]['setid']
                    else:
                        setId = 0
                    resp = objExporter.ComplexPing(setId, 0, addedOids, deletedOids)
                    DCOMConnection.OID_SET[target]['oids'] -= deletedOids
                    DCOMConnection.OID_SET[target]['oids'] |= addedOids
                    DCOMConnection.OID_SET[target]['setid'] = resp['pSetId']
                else:
                    objExporter.SimplePing(DCOMConnection.OID_SET[target]['setid'])
        except Exception as e:
            LOG.error(str(e))
            pass
        DCOMConnection.PINGTIMER = Timer(120, DCOMConnection.pingServer)
        try:
            DCOMConnection.PINGTIMER.start()
        except Exception as e:
            if str(e).find('threads can only be started once') < 0:
                raise e

    def initTimer(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__oxidResolver is True:
            if DCOMConnection.PINGTIMER is None:
                DCOMConnection.PINGTIMER = Timer(120, DCOMConnection.pingServer)
            try:
                DCOMConnection.PINGTIMER.start()
            except Exception as e:
                if str(e).find('threads can only be started once') < 0:
                    raise e

    def initConnection(self):
        if False:
            i = 10
            return i + 15
        stringBinding = 'ncacn_ip_tcp:%s' % self.__target
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if hasattr(rpctransport, 'set_credentials') and len(self.__userName) >= 0:
            rpctransport.set_credentials(self.__userName, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, self.__TGT, self.__TGS)
            rpctransport.set_kerberos(self.__doKerberos, self.__kdcHost)
        self.__portmap = rpctransport.get_dce_rpc()
        self.__portmap.set_auth_level(self.__authLevel)
        if self.__doKerberos is True:
            self.__portmap.set_auth_type(RPC_C_AUTHN_GSS_NEGOTIATE)
        self.__portmap.connect()
        DCOMConnection.PORTMAPS[self.__target] = self.__portmap

    def CoCreateInstanceEx(self, clsid, iid):
        if False:
            return 10
        scm = IRemoteSCMActivator(self.__portmap)
        iInterface = scm.RemoteCreateInstance(clsid, iid)
        self.initTimer()
        return iInterface

    def get_dce_rpc(self):
        if False:
            print('Hello World!')
        return DCOMConnection.PORTMAPS[self.__target]

    def disconnect(self):
        if False:
            i = 10
            return i + 15
        if DCOMConnection.PINGTIMER is not None:
            del DCOMConnection.PORTMAPS[self.__target]
            del DCOMConnection.OID_SET[self.__target]
            if len(DCOMConnection.PORTMAPS) == 0:
                DCOMConnection.PINGTIMER.cancel()
                DCOMConnection.PINGTIMER.join()
                DCOMConnection.PINGTIMER = None
        if self.__target in INTERFACE.CONNECTIONS:
            del INTERFACE.CONNECTIONS[self.__target][current_thread().name]
        self.__portmap.disconnect()

class CLASS_INSTANCE:

    def __init__(self, ORPCthis, stringBinding):
        if False:
            i = 10
            return i + 15
        self.__stringBindings = stringBinding
        self.__ORPCthis = ORPCthis
        self.__authType = RPC_C_AUTHN_WINNT
        self.__authLevel = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def get_ORPCthis(self):
        if False:
            return 10
        return self.__ORPCthis

    def get_string_bindings(self):
        if False:
            print('Hello World!')
        return self.__stringBindings

    def get_auth_level(self):
        if False:
            print('Hello World!')
        if RPC_C_AUTHN_LEVEL_NONE < self.__authLevel < RPC_C_AUTHN_LEVEL_PKT_PRIVACY:
            if self.__authType == RPC_C_AUTHN_WINNT:
                return RPC_C_AUTHN_LEVEL_PKT_INTEGRITY
            else:
                return RPC_C_AUTHN_LEVEL_PKT_PRIVACY
        return self.__authLevel

    def set_auth_level(self, level):
        if False:
            i = 10
            return i + 15
        self.__authLevel = level

    def get_auth_type(self):
        if False:
            print('Hello World!')
        return self.__authType

    def set_auth_type(self, authType):
        if False:
            while True:
                i = 10
        self.__authType = authType

class INTERFACE:
    CONNECTIONS = {}

    def __init__(self, cinstance=None, objRef=None, ipidRemUnknown=None, iPid=None, oxid=None, oid=None, target=None, interfaceInstance=None):
        if False:
            while True:
                i = 10
        if interfaceInstance is not None:
            self.__target = interfaceInstance.get_target()
            self.__iPid = interfaceInstance.get_iPid()
            self.__oid = interfaceInstance.get_oid()
            self.__oxid = interfaceInstance.get_oxid()
            self.__cinstance = interfaceInstance.get_cinstance()
            self.__objRef = interfaceInstance.get_objRef()
            self.__ipidRemUnknown = interfaceInstance.get_ipidRemUnknown()
        else:
            if target is None:
                raise Exception('No target')
            self.__target = target
            self.__iPid = iPid
            self.__oid = oid
            self.__oxid = oxid
            self.__cinstance = cinstance
            self.__objRef = objRef
            self.__ipidRemUnknown = ipidRemUnknown
            if (self.__target in INTERFACE.CONNECTIONS) is not True:
                INTERFACE.CONNECTIONS[self.__target] = {}
                INTERFACE.CONNECTIONS[self.__target][current_thread().name] = {}
            if objRef is not None:
                self.process_interface(objRef)

    def process_interface(self, data):
        if False:
            for i in range(10):
                print('nop')
        objRefType = OBJREF(data)['flags']
        objRef = None
        if objRefType == FLAGS_OBJREF_CUSTOM:
            objRef = OBJREF_CUSTOM(data)
        elif objRefType == FLAGS_OBJREF_HANDLER:
            objRef = OBJREF_HANDLER(data)
        elif objRefType == FLAGS_OBJREF_STANDARD:
            objRef = OBJREF_STANDARD(data)
        elif objRefType == FLAGS_OBJREF_EXTENDED:
            objRef = OBJREF_EXTENDED(data)
        else:
            LOG.error('Unknown OBJREF Type! 0x%x' % objRefType)
        if objRefType != FLAGS_OBJREF_CUSTOM:
            if objRef['std']['flags'] & SORF_NOPING == 0:
                DCOMConnection.addOid(self.__target, objRef['std']['oid'])
            self.__iPid = objRef['std']['ipid']
            self.__oid = objRef['std']['oid']
            self.__oxid = objRef['std']['oxid']
            if self.__oxid is None:
                objRef.dump()
                raise Exception('OXID is None')

    def get_oxid(self):
        if False:
            i = 10
            return i + 15
        return self.__oxid

    def set_oxid(self, oxid):
        if False:
            i = 10
            return i + 15
        self.__oxid = oxid

    def get_oid(self):
        if False:
            while True:
                i = 10
        return self.__oid

    def set_oid(self, oid):
        if False:
            while True:
                i = 10
        self.__oid = oid

    def get_target(self):
        if False:
            while True:
                i = 10
        return self.__target

    def get_iPid(self):
        if False:
            return 10
        return self.__iPid

    def set_iPid(self, iPid):
        if False:
            i = 10
            return i + 15
        self.__iPid = iPid

    def get_objRef(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__objRef

    def set_objRef(self, objRef):
        if False:
            while True:
                i = 10
        self.__objRef = objRef

    def get_ipidRemUnknown(self):
        if False:
            i = 10
            return i + 15
        return self.__ipidRemUnknown

    def get_dce_rpc(self):
        if False:
            i = 10
            return i + 15
        return INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['dce']

    def get_cinstance(self):
        if False:
            while True:
                i = 10
        return self.__cinstance

    def set_cinstance(self, cinstance):
        if False:
            return 10
        self.__cinstance = cinstance

    def is_fqdn(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            socket.inet_aton(self.__target)
        except:
            try:
                self.__target.index(':')
            except:
                return True
        return False

    def connect(self, iid=None):
        if False:
            for i in range(10):
                print('nop')
        if (self.__target in INTERFACE.CONNECTIONS) is True:
            if current_thread().name in INTERFACE.CONNECTIONS[self.__target] and (self.__oxid in INTERFACE.CONNECTIONS[self.__target][current_thread().name]) is True:
                dce = INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['dce']
                currentBinding = INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['currentBinding']
                if currentBinding == iid:
                    pass
                else:
                    newDce = dce.alter_ctx(iid)
                    INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['dce'] = newDce
                    INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['currentBinding'] = iid
            else:
                stringBindings = self.get_cinstance().get_string_bindings()
                stringBinding = None
                isTargetFQDN = self.is_fqdn()
                LOG.debug('Target system is %s and isFQDN is %s' % (self.get_target(), isTargetFQDN))
                for strBinding in stringBindings:
                    LOG.debug('StringBinding: %s' % strBinding['aNetworkAddr'])
                    if strBinding['wTowerId'] == 7:
                        if strBinding['aNetworkAddr'].find('[') >= 0:
                            (binding, _, bindingPort) = strBinding['aNetworkAddr'].partition('[')
                            bindingPort = '[' + bindingPort
                        else:
                            binding = strBinding['aNetworkAddr']
                            bindingPort = ''
                        if binding.upper().find(self.get_target().upper()) >= 0:
                            stringBinding = 'ncacn_ip_tcp:' + strBinding['aNetworkAddr'][:-1]
                            break
                        elif isTargetFQDN and binding.upper().find(self.get_target().upper().partition('.')[0]) >= 0:
                            stringBinding = 'ncacn_ip_tcp:%s%s' % (self.get_target(), bindingPort)
                            break
                LOG.debug('StringBinding chosen: %s' % stringBinding)
                if stringBinding is None:
                    raise Exception("Can't find a valid stringBinding to connect")
                dcomInterface = transport.DCERPCTransportFactory(stringBinding)
                if hasattr(dcomInterface, 'set_credentials'):
                    dcomInterface.set_credentials(*DCOMConnection.PORTMAPS[self.__target].get_credentials())
                    dcomInterface.set_kerberos(DCOMConnection.PORTMAPS[self.__target].get_rpc_transport().get_kerberos(), DCOMConnection.PORTMAPS[self.__target].get_rpc_transport().get_kdcHost())
                dcomInterface.set_connect_timeout(300)
                dce = dcomInterface.get_dce_rpc()
                if iid is None:
                    raise Exception('IID is None')
                else:
                    dce.set_auth_level(self.__cinstance.get_auth_level())
                    dce.set_auth_type(self.__cinstance.get_auth_type())
                dce.connect()
                if iid is None:
                    raise Exception('IID is None')
                else:
                    dce.bind(iid)
                if self.__oxid is None:
                    raise Exception('OXID NONE, something wrong!!!')
                INTERFACE.CONNECTIONS[self.__target][current_thread().name] = {}
                INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid] = {}
                INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['dce'] = dce
                INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['currentBinding'] = iid
        else:
            raise Exception('No connection created')

    def request(self, req, iid=None, uuid=None):
        if False:
            i = 10
            return i + 15
        req['ORPCthis'] = self.get_cinstance().get_ORPCthis()
        req['ORPCthis']['flags'] = 0
        self.connect(iid)
        dce = self.get_dce_rpc()
        try:
            resp = dce.request(req, uuid)
        except Exception as e:
            if str(e).find('RPC_E_DISCONNECTED') >= 0:
                msg = str(e) + '\n'
                msg += "DCOM keep-alive pinging it might not be working as expected. You can't be idle for more than 14 minutes!\n"
                msg += 'You should exit the app and start again\n'
                raise DCERPCException(msg)
            else:
                raise
        return resp

    def disconnect(self):
        if False:
            print('Hello World!')
        return INTERFACE.CONNECTIONS[self.__target][current_thread().name][self.__oxid]['dce'].disconnect()

class IRemUnknown(INTERFACE):

    def __init__(self, interface):
        if False:
            i = 10
            return i + 15
        self._iid = IID_IRemUnknown
        INTERFACE.__init__(self, interfaceInstance=interface)
        self.set_oxid(interface.get_oxid())

    def RemQueryInterface(self, cRefs, iids):
        if False:
            i = 10
            return i + 15
        request = RemQueryInterface()
        request['ORPCthis'] = self.get_cinstance().get_ORPCthis()
        request['ORPCthis']['flags'] = 0
        request['ripid'] = self.get_iPid()
        request['cRefs'] = cRefs
        request['cIids'] = len(iids)
        for iid in iids:
            _iid = IID()
            _iid['Data'] = iid
            request['iids'].append(_iid)
        resp = self.request(request, IID_IRemUnknown, self.get_ipidRemUnknown())
        return IRemUnknown2(INTERFACE(self.get_cinstance(), None, self.get_ipidRemUnknown(), resp['ppQIResults']['std']['ipid'], oxid=resp['ppQIResults']['std']['oxid'], oid=resp['ppQIResults']['std']['oxid'], target=self.get_target()))

    def RemAddRef(self):
        if False:
            print('Hello World!')
        request = RemAddRef()
        request['ORPCthis'] = self.get_cinstance().get_ORPCthis()
        request['ORPCthis']['flags'] = 0
        request['cInterfaceRefs'] = 1
        element = REMINTERFACEREF()
        element['ipid'] = self.get_iPid()
        element['cPublicRefs'] = 1
        request['InterfaceRefs'].append(element)
        resp = self.request(request, IID_IRemUnknown, self.get_ipidRemUnknown())
        return resp

    def RemRelease(self):
        if False:
            print('Hello World!')
        request = RemRelease()
        request['ORPCthis'] = self.get_cinstance().get_ORPCthis()
        request['ORPCthis']['flags'] = 0
        request['cInterfaceRefs'] = 1
        element = REMINTERFACEREF()
        element['ipid'] = self.get_iPid()
        element['cPublicRefs'] = 1
        request['InterfaceRefs'].append(element)
        resp = self.request(request, IID_IRemUnknown, self.get_ipidRemUnknown())
        DCOMConnection.delOid(self.get_target(), self.get_oid())
        return resp

class IRemUnknown2(IRemUnknown):

    def __init__(self, interface):
        if False:
            for i in range(10):
                print('nop')
        IRemUnknown.__init__(self, interface)
        self._iid = IID_IRemUnknown2

class IObjectExporter:

    def __init__(self, dce):
        if False:
            print('Hello World!')
        self.__portmap = dce

    def ResolveOxid(self, pOxid, arRequestedProtseqs):
        if False:
            for i in range(10):
                print('nop')
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = ResolveOxid()
        request['pOxid'] = pOxid
        request['cRequestedProtseqs'] = len(arRequestedProtseqs)
        for protSeq in arRequestedProtseqs:
            request['arRequestedProtseqs'].append(protSeq)
        resp = self.__portmap.request(request)
        Oxids = b''.join((pack('<H', x) for x in resp['ppdsaOxidBindings']['aStringArray']))
        strBindings = Oxids[:resp['ppdsaOxidBindings']['wSecurityOffset'] * 2]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        return stringBindings

    def SimplePing(self, setId):
        if False:
            i = 10
            return i + 15
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = SimplePing()
        request['pSetId'] = setId
        resp = self.__portmap.request(request)
        return resp

    def ComplexPing(self, setId=0, sequenceNum=0, addToSet=[], delFromSet=[]):
        if False:
            return 10
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = ComplexPing()
        request['pSetId'] = setId
        request['SequenceNum'] = setId
        request['cAddToSet'] = len(addToSet)
        request['cDelFromSet'] = len(delFromSet)
        if len(addToSet) > 0:
            for oid in addToSet:
                oidn = OID()
                oidn['Data'] = oid
                request['AddToSet'].append(oidn)
        else:
            request['AddToSet'] = NULL
        if len(delFromSet) > 0:
            for oid in delFromSet:
                oidn = OID()
                oidn['Data'] = oid
                request['DelFromSet'].append(oidn)
        else:
            request['DelFromSet'] = NULL
        resp = self.__portmap.request(request)
        return resp

    def ServerAlive(self):
        if False:
            while True:
                i = 10
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = ServerAlive()
        resp = self.__portmap.request(request)
        return resp

    def ResolveOxid2(self, pOxid, arRequestedProtseqs):
        if False:
            for i in range(10):
                print('nop')
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = ResolveOxid2()
        request['pOxid'] = pOxid
        request['cRequestedProtseqs'] = len(arRequestedProtseqs)
        for protSeq in arRequestedProtseqs:
            request['arRequestedProtseqs'].append(protSeq)
        resp = self.__portmap.request(request)
        Oxids = b''.join((pack('<H', x) for x in resp['ppdsaOxidBindings']['aStringArray']))
        strBindings = Oxids[:resp['ppdsaOxidBindings']['wSecurityOffset'] * 2]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        return stringBindings

    def ServerAlive2(self):
        if False:
            i = 10
            return i + 15
        self.__portmap.connect()
        self.__portmap.bind(IID_IObjectExporter)
        request = ServerAlive2()
        resp = self.__portmap.request(request)
        Oxids = b''.join((pack('<H', x) for x in resp['ppdsaOrBindings']['aStringArray']))
        strBindings = Oxids[:resp['ppdsaOrBindings']['wSecurityOffset'] * 2]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        return stringBindings

class IActivation:

    def __init__(self, dce):
        if False:
            while True:
                i = 10
        self.__portmap = dce

    def RemoteActivation(self, clsId, iid):
        if False:
            while True:
                i = 10
        self.__portmap.bind(IID_IActivation)
        ORPCthis = ORPCTHIS()
        ORPCthis['cid'] = generate()
        ORPCthis['extensions'] = NULL
        ORPCthis['flags'] = 1
        request = RemoteActivation()
        request['Clsid'] = clsId
        request['pwszObjectName'] = NULL
        request['pObjectStorage'] = NULL
        request['ClientImpLevel'] = 2
        request['Mode'] = 0
        request['Interfaces'] = 1
        _iid = IID()
        _iid['Data'] = iid
        request['pIIDs'].append(_iid)
        request['cRequestedProtseqs'] = 1
        request['aRequestedProtseqs'].append(7)
        resp = self.__portmap.request(request)
        ipidRemUnknown = resp['pipidRemUnknown']
        Oxids = b''.join((pack('<H', x) for x in resp['ppdsaOxidBindings']['aStringArray']))
        strBindings = Oxids[:resp['ppdsaOxidBindings']['wSecurityOffset'] * 2]
        securityBindings = Oxids[resp['ppdsaOxidBindings']['wSecurityOffset'] * 2:]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        done = False
        while not done:
            if len(securityBindings) < 2:
                done = True
            elif securityBindings[0:1] == b'\x00' and securityBindings[1:2] == b'\x00':
                done = True
            else:
                secBinding = SECURITYBINDING(securityBindings)
                securityBindings = securityBindings[len(secBinding):]
        classInstance = CLASS_INSTANCE(ORPCthis, stringBindings)
        return IRemUnknown2(INTERFACE(classInstance, b''.join(resp['ppInterfaceData'][0]['abData']), ipidRemUnknown, target=self.__portmap.get_rpc_transport().getRemoteHost()))

class IRemoteSCMActivator:

    def __init__(self, dce):
        if False:
            while True:
                i = 10
        self.__portmap = dce

    def RemoteGetClassObject(self, clsId, iid):
        if False:
            for i in range(10):
                print('nop')
        self.__portmap.bind(IID_IRemoteSCMActivator)
        ORPCthis = ORPCTHIS()
        ORPCthis['cid'] = generate()
        ORPCthis['extensions'] = NULL
        ORPCthis['flags'] = 1
        request = RemoteGetClassObject()
        request['ORPCthis'] = ORPCthis
        activationBLOB = ACTIVATION_BLOB()
        activationBLOB['CustomHeader']['destCtx'] = 2
        activationBLOB['CustomHeader']['pdwReserved'] = NULL
        clsid = CLSID()
        clsid['Data'] = CLSID_InstantiationInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ActivationContextInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ServerLocationInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ScmRequestInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        properties = b''
        instantiationInfo = InstantiationInfoData()
        instantiationInfo['classId'] = clsId
        instantiationInfo['cIID'] = 1
        _iid = IID()
        _iid['Data'] = iid
        instantiationInfo['pIID'].append(_iid)
        dword = DWORD()
        marshaled = instantiationInfo.getData() + instantiationInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        instantiationInfo['thisSize'] = dword['Data']
        properties += marshaled + b'\xfa' * pad
        activationInfo = ActivationContextInfoData()
        activationInfo['pIFDClientCtx'] = NULL
        activationInfo['pIFDPrototypeCtx'] = NULL
        dword = DWORD()
        marshaled = activationInfo.getData() + activationInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += marshaled + b'\xfa' * pad
        locationInfo = LocationInfoData()
        locationInfo['machineName'] = NULL
        dword = DWORD()
        dword['Data'] = len(locationInfo.getData())
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += locationInfo.getData() + locationInfo.getDataReferents()
        scmInfo = ScmRequestInfoData()
        scmInfo['pdwReserved'] = NULL
        scmInfo['remoteRequest']['cRequestedProtseqs'] = 1
        scmInfo['remoteRequest']['pRequestedProtseqs'].append(7)
        dword = DWORD()
        marshaled = scmInfo.getData() + scmInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += marshaled + b'\xfa' * pad
        activationBLOB['Property'] = properties
        objrefcustom = OBJREF_CUSTOM()
        objrefcustom['iid'] = IID_IActivationPropertiesIn[:-4]
        objrefcustom['clsid'] = CLSID_ActivationPropertiesIn
        objrefcustom['pObjectData'] = activationBLOB.getData()
        objrefcustom['ObjectReferenceSize'] = len(objrefcustom['pObjectData']) + 8
        request['pActProperties']['ulCntData'] = len(objrefcustom.getData())
        request['pActProperties']['abData'] = list(objrefcustom.getData())
        resp = self.__portmap.request(request)
        objRefType = OBJREF(b''.join(resp['ppActProperties']['abData']))['flags']
        objRef = None
        if objRefType == FLAGS_OBJREF_CUSTOM:
            objRef = OBJREF_CUSTOM(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_HANDLER:
            objRef = OBJREF_HANDLER(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_STANDARD:
            objRef = OBJREF_STANDARD(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_EXTENDED:
            objRef = OBJREF_EXTENDED(b''.join(resp['ppActProperties']['abData']))
        else:
            LOG.error('Unknown OBJREF Type! 0x%x' % objRefType)
        activationBlob = ACTIVATION_BLOB(objRef['pObjectData'])
        propOutput = activationBlob['Property'][:activationBlob['CustomHeader']['pSizes'][0]['Data']]
        scmReply = activationBlob['Property'][activationBlob['CustomHeader']['pSizes'][0]['Data']:activationBlob['CustomHeader']['pSizes'][0]['Data'] + activationBlob['CustomHeader']['pSizes'][1]['Data']]
        scmr = ScmReplyInfoData()
        size = scmr.fromString(scmReply)
        scmr.fromStringReferents(scmReply[size:])
        ipidRemUnknown = scmr['remoteReply']['ipidRemUnknown']
        Oxids = b''.join((pack('<H', x) for x in scmr['remoteReply']['pdsaOxidBindings']['aStringArray']))
        strBindings = Oxids[:scmr['remoteReply']['pdsaOxidBindings']['wSecurityOffset'] * 2]
        securityBindings = Oxids[scmr['remoteReply']['pdsaOxidBindings']['wSecurityOffset'] * 2:]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        done = False
        while not done:
            if len(securityBindings) < 2:
                done = True
            elif securityBindings[0:1] == b'\x00' and securityBindings[1:2] == b'\x00':
                done = True
            else:
                secBinding = SECURITYBINDING(securityBindings)
                securityBindings = securityBindings[len(secBinding):]
        propsOut = PropsOutInfo()
        size = propsOut.fromString(propOutput)
        propsOut.fromStringReferents(propOutput[size:])
        classInstance = CLASS_INSTANCE(ORPCthis, stringBindings)
        classInstance.set_auth_level(scmr['remoteReply']['authnHint'])
        classInstance.set_auth_type(self.__portmap.get_auth_type())
        return IRemUnknown2(INTERFACE(classInstance, b''.join(propsOut['ppIntfData'][0]['abData']), ipidRemUnknown, target=self.__portmap.get_rpc_transport().getRemoteHost()))

    def RemoteCreateInstance(self, clsId, iid):
        if False:
            i = 10
            return i + 15
        self.__portmap.bind(IID_IRemoteSCMActivator)
        ORPCthis = ORPCTHIS()
        ORPCthis['cid'] = generate()
        ORPCthis['extensions'] = NULL
        ORPCthis['flags'] = 1
        request = RemoteCreateInstance()
        request['ORPCthis'] = ORPCthis
        request['pUnkOuter'] = NULL
        activationBLOB = ACTIVATION_BLOB()
        activationBLOB['CustomHeader']['destCtx'] = 2
        activationBLOB['CustomHeader']['pdwReserved'] = NULL
        clsid = CLSID()
        clsid['Data'] = CLSID_InstantiationInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ActivationContextInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ServerLocationInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        clsid = CLSID()
        clsid['Data'] = CLSID_ScmRequestInfo
        activationBLOB['CustomHeader']['pclsid'].append(clsid)
        properties = b''
        instantiationInfo = InstantiationInfoData()
        instantiationInfo['classId'] = clsId
        instantiationInfo['cIID'] = 1
        _iid = IID()
        _iid['Data'] = iid
        instantiationInfo['pIID'].append(_iid)
        dword = DWORD()
        marshaled = instantiationInfo.getData() + instantiationInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        instantiationInfo['thisSize'] = dword['Data']
        properties += marshaled + b'\xfa' * pad
        activationInfo = ActivationContextInfoData()
        activationInfo['pIFDClientCtx'] = NULL
        activationInfo['pIFDPrototypeCtx'] = NULL
        dword = DWORD()
        marshaled = activationInfo.getData() + activationInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += marshaled + b'\xfa' * pad
        locationInfo = LocationInfoData()
        locationInfo['machineName'] = NULL
        dword = DWORD()
        dword['Data'] = len(locationInfo.getData())
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += locationInfo.getData() + locationInfo.getDataReferents()
        scmInfo = ScmRequestInfoData()
        scmInfo['pdwReserved'] = NULL
        scmInfo['remoteRequest']['cRequestedProtseqs'] = 1
        scmInfo['remoteRequest']['pRequestedProtseqs'].append(7)
        dword = DWORD()
        marshaled = scmInfo.getData() + scmInfo.getDataReferents()
        pad = (8 - len(marshaled) % 8) % 8
        dword['Data'] = len(marshaled) + pad
        activationBLOB['CustomHeader']['pSizes'].append(dword)
        properties += marshaled + b'\xfa' * pad
        activationBLOB['Property'] = properties
        objrefcustom = OBJREF_CUSTOM()
        objrefcustom['iid'] = IID_IActivationPropertiesIn[:-4]
        objrefcustom['clsid'] = CLSID_ActivationPropertiesIn
        objrefcustom['pObjectData'] = activationBLOB.getData()
        objrefcustom['ObjectReferenceSize'] = len(objrefcustom['pObjectData']) + 8
        request['pActProperties']['ulCntData'] = len(objrefcustom.getData())
        request['pActProperties']['abData'] = list(objrefcustom.getData())
        resp = self.__portmap.request(request)
        objRefType = OBJREF(b''.join(resp['ppActProperties']['abData']))['flags']
        objRef = None
        if objRefType == FLAGS_OBJREF_CUSTOM:
            objRef = OBJREF_CUSTOM(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_HANDLER:
            objRef = OBJREF_HANDLER(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_STANDARD:
            objRef = OBJREF_STANDARD(b''.join(resp['ppActProperties']['abData']))
        elif objRefType == FLAGS_OBJREF_EXTENDED:
            objRef = OBJREF_EXTENDED(b''.join(resp['ppActProperties']['abData']))
        else:
            LOG.error('Unknown OBJREF Type! 0x%x' % objRefType)
        activationBlob = ACTIVATION_BLOB(objRef['pObjectData'])
        propOutput = activationBlob['Property'][:activationBlob['CustomHeader']['pSizes'][0]['Data']]
        scmReply = activationBlob['Property'][activationBlob['CustomHeader']['pSizes'][0]['Data']:activationBlob['CustomHeader']['pSizes'][0]['Data'] + activationBlob['CustomHeader']['pSizes'][1]['Data']]
        scmr = ScmReplyInfoData()
        size = scmr.fromString(scmReply)
        scmr.fromStringReferents(scmReply[size:])
        ipidRemUnknown = scmr['remoteReply']['ipidRemUnknown']
        Oxids = b''.join((pack('<H', x) for x in scmr['remoteReply']['pdsaOxidBindings']['aStringArray']))
        strBindings = Oxids[:scmr['remoteReply']['pdsaOxidBindings']['wSecurityOffset'] * 2]
        securityBindings = Oxids[scmr['remoteReply']['pdsaOxidBindings']['wSecurityOffset'] * 2:]
        done = False
        stringBindings = list()
        while not done:
            if strBindings[0:1] == b'\x00' and strBindings[1:2] == b'\x00':
                done = True
            else:
                binding = STRINGBINDING(strBindings)
                stringBindings.append(binding)
                strBindings = strBindings[len(binding):]
        done = False
        while not done:
            if len(securityBindings) < 2:
                done = True
            elif securityBindings[0:1] == b'\x00' and securityBindings[1:2] == b'\x00':
                done = True
            else:
                secBinding = SECURITYBINDING(securityBindings)
                securityBindings = securityBindings[len(secBinding):]
        propsOut = PropsOutInfo()
        size = propsOut.fromString(propOutput)
        propsOut.fromStringReferents(propOutput[size:])
        classInstance = CLASS_INSTANCE(ORPCthis, stringBindings)
        classInstance.set_auth_level(scmr['remoteReply']['authnHint'])
        classInstance.set_auth_type(self.__portmap.get_auth_type())
        return IRemUnknown2(INTERFACE(classInstance, b''.join(propsOut['ppIntfData'][0]['abData']), ipidRemUnknown, target=self.__portmap.get_rpc_transport().getRemoteHost()))