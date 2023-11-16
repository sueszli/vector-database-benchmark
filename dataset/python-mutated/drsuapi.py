from __future__ import division
from __future__ import print_function
from builtins import bytes
import hashlib
from struct import pack
import six
from six import PY2
from impacket import LOG
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRPOINTER, NDRUniConformantArray, NDRUNION, NDR, NDRENUM
from impacket.dcerpc.v5.dtypes import PUUID, DWORD, NULL, GUID, LPWSTR, BOOL, ULONG, UUID, LONGLONG, ULARGE_INTEGER, LARGE_INTEGER
from impacket import hresult_errors, system_errors
from impacket.structure import Structure
from impacket.uuid import uuidtup_to_bin, string_to_bin
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.krb5 import crypto
from pyasn1.type import univ
from pyasn1.codec.ber import decoder
from impacket.crypto import transformKey
try:
    from Cryptodome.Cipher import ARC4, DES
except Exception:
    LOG.critical("Warning: You don't have any crypto installed. You need pycryptodomex")
    LOG.critical('See https://pypi.org/project/pycryptodomex/')
MSRPC_UUID_DRSUAPI = uuidtup_to_bin(('E3514235-4B06-11D1-AB04-00C04FC2DCD2', '4.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            print('Hello World!')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            return 10
        key = self.error_code
        if key in hresult_errors.ERROR_MESSAGES:
            error_msg_short = hresult_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = hresult_errors.ERROR_MESSAGES[key][1]
            return 'DRSR SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        elif key & 65535 in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key & 65535][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key & 65535][1]
            return 'DRSR SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'DRSR SessionError: unknown error code: 0x%x' % self.error_code

class EXOP_ERR(NDRENUM):
    align = 4
    align64 = 4
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        EXOP_ERR_SUCCESS = 1
        EXOP_ERR_UNKNOWN_OP = 2
        EXOP_ERR_FSMO_NOT_OWNER = 3
        EXOP_ERR_UPDATE_ERR = 4
        EXOP_ERR_EXCEPTION = 5
        EXOP_ERR_UNKNOWN_CALLER = 6
        EXOP_ERR_RID_ALLOC = 7
        EXOP_ERR_FSMO_OWNER_DELETED = 8
        EXOP_ERR_FSMO_PENDING_OP = 9
        EXOP_ERR_MISMATCH = 10
        EXOP_ERR_COULDNT_CONTACT = 11
        EXOP_ERR_FSMO_REFUSING_ROLES = 12
        EXOP_ERR_DIR_ERROR = 13
        EXOP_ERR_FSMO_MISSING_SETTINGS = 14
        EXOP_ERR_ACCESS_DENIED = 15
        EXOP_ERR_PARAM_ERROR = 16

    def dump(self, msg=None, indent=0):
        if False:
            i = 10
            return i + 15
        if msg is None:
            msg = self.__class__.__name__
        if msg != '':
            print(msg, end=' ')
        try:
            print(' %s' % self.enumItems(self.fields['Data']).name, end=' ')
        except ValueError:
            print(' %d' % self.fields['Data'])
EXOP_FSMO_REQ_ROLE = 1
EXOP_FSMO_REQ_RID_ALLOC = 2
EXOP_FSMO_RID_REQ_ROLE = 3
EXOP_FSMO_REQ_PDC = 4
EXOP_FSMO_ABANDON_ROLE = 5
EXOP_REPL_OBJ = 6
EXOP_REPL_SECRETS = 7
ATTRTYP = ULONG
DSTIME = LONGLONG
DRS_EXT_BASE = 1
DRS_EXT_ASYNCREPL = 2
DRS_EXT_REMOVEAPI = 4
DRS_EXT_MOVEREQ_V2 = 8
DRS_EXT_GETCHG_DEFLATE = 16
DRS_EXT_DCINFO_V1 = 32
DRS_EXT_RESTORE_USN_OPTIMIZATION = 64
DRS_EXT_ADDENTRY = 128
DRS_EXT_KCC_EXECUTE = 256
DRS_EXT_ADDENTRY_V2 = 512
DRS_EXT_LINKED_VALUE_REPLICATION = 1024
DRS_EXT_DCINFO_V2 = 2048
DRS_EXT_INSTANCE_TYPE_NOT_REQ_ON_MOD = 4096
DRS_EXT_CRYPTO_BIND = 8192
DRS_EXT_GET_REPL_INFO = 16384
DRS_EXT_STRONG_ENCRYPTION = 32768
DRS_EXT_DCINFO_VFFFFFFFF = 65536
DRS_EXT_TRANSITIVE_MEMBERSHIP = 131072
DRS_EXT_ADD_SID_HISTORY = 262144
DRS_EXT_POST_BETA3 = 524288
DRS_EXT_GETCHGREQ_V5 = 1048576
DRS_EXT_GETMEMBERSHIPS2 = 2097152
DRS_EXT_GETCHGREQ_V6 = 4194304
DRS_EXT_NONDOMAIN_NCS = 8388608
DRS_EXT_GETCHGREQ_V8 = 16777216
DRS_EXT_GETCHGREPLY_V5 = 33554432
DRS_EXT_GETCHGREPLY_V6 = 67108864
DRS_EXT_GETCHGREPLY_V9 = 256
DRS_EXT_WHISTLER_BETA3 = 134217728
DRS_EXT_W2K3_DEFLATE = 268435456
DRS_EXT_GETCHGREQ_V10 = 536870912
DRS_EXT_RESERVED_FOR_WIN2K_OR_DOTNET_PART2 = 1073741824
DRS_EXT_RESERVED_FOR_WIN2K_OR_DOTNET_PART3 = 2147483648
DRS_EXT_ADAM = 1
DRS_EXT_LH_BETA2 = 2
DRS_EXT_RECYCLE_BIN = 4
DRS_ASYNC_OP = 1
DRS_GETCHG_CHECK = 2
DRS_UPDATE_NOTIFICATION = 2
DRS_ADD_REF = 4
DRS_SYNC_ALL = 8
DRS_DEL_REF = 8
DRS_WRIT_REP = 16
DRS_INIT_SYNC = 32
DRS_PER_SYNC = 64
DRS_MAIL_REP = 128
DRS_ASYNC_REP = 256
DRS_IGNORE_ERROR = 256
DRS_TWOWAY_SYNC = 512
DRS_CRITICAL_ONLY = 1024
DRS_GET_ANC = 2048
DRS_GET_NC_SIZE = 4096
DRS_LOCAL_ONLY = 4096
DRS_NONGC_RO_REP = 8192
DRS_SYNC_BYNAME = 16384
DRS_REF_OK = 16384
DRS_FULL_SYNC_NOW = 32768
DRS_NO_SOURCE = 32768
DRS_FULL_SYNC_IN_PROGRESS = 65536
DRS_FULL_SYNC_PACKET = 131072
DRS_SYNC_REQUEUE = 262144
DRS_SYNC_URGENT = 524288
DRS_REF_GCSPN = 1048576
DRS_NO_DISCARD = 1048576
DRS_NEVER_SYNCED = 2097152
DRS_SPECIAL_SECRET_PROCESSING = 4194304
DRS_INIT_SYNC_NOW = 8388608
DRS_PREEMPTED = 16777216
DRS_SYNC_FORCED = 33554432
DRS_DISABLE_AUTO_SYNC = 67108864
DRS_DISABLE_PERIODIC_SYNC = 134217728
DRS_USE_COMPRESSION = 268435456
DRS_NEVER_NOTIFY = 536870912
DRS_SYNC_PAS = 1073741824
DRS_GET_ALL_GROUP_MEMBERSHIP = 2147483648
BND = 1
SSL = 2
UDP = 4
GC = 8
GSS = 16
NGO = 32
SPL = 64
MD5 = 128
SGN = 256
SL = 512
NTDSAPI_CLIENT_GUID = string_to_bin('e24d201a-4fd6-11d1-a3da-0000f875ae0d')
NULLGUID = string_to_bin('00000000-0000-0000-0000-000000000000')
USN = LONGLONG
DS_NAME_FLAG_GCVERIFY = 4
DS_NAME_FLAG_TRUST_REFERRAL = 8
DS_NAME_FLAG_PRIVATE_RESOLVE_FPOS = 2147483648
DS_LIST_SITES = 4294967295
DS_LIST_SERVERS_IN_SITE = 4294967294
DS_LIST_DOMAINS_IN_SITE = 4294967293
DS_LIST_SERVERS_FOR_DOMAIN_IN_SITE = 4294967292
DS_LIST_INFO_FOR_SERVER = 4294967291
DS_LIST_ROLES = 4294967290
DS_NT4_ACCOUNT_NAME_SANS_DOMAIN = 4294967289
DS_MAP_SCHEMA_GUID = 4294967288
DS_LIST_DOMAINS = 4294967287
DS_LIST_NCS = 4294967286
DS_ALT_SECURITY_IDENTITIES_NAME = 4294967285
DS_STRING_SID_NAME = 4294967284
DS_LIST_SERVERS_WITH_DCS_IN_SITE = 4294967283
DS_LIST_GLOBAL_CATALOG_SERVERS = 4294967281
DS_NT4_ACCOUNT_NAME_SANS_DOMAIN_EX = 4294967280
DS_USER_PRINCIPAL_NAME_AND_ALTSECID = 4294967279
DS_USER_PRINCIPAL_NAME_FOR_LOGON = 4294967282
ENTINF_FROM_MASTER = 1
ENTINF_DYNAMIC_OBJECT = 2
ENTINF_REMOTE_MODIFY = 65536
DRS_VERIFY_DSNAMES = 0
DRS_VERIFY_SIDS = 1
DRS_VERIFY_SAM_ACCOUNT_NAMES = 2
DRS_VERIFY_FPOS = 3
DRS_NT4_CHGLOG_GET_CHANGE_LOG = 1
DRS_NT4_CHGLOG_GET_SERIAL_NUMBERS = 2
DRS_MSG_GETCHGREPLY_NATIVE_VERSION_NUMBER = 9

class ENCRYPTED_PAYLOAD(Structure):
    structure = (('Salt', '16s'), ('CheckSum', '<L'), ('EncryptedData', ':'))

class NT4SID(NDRSTRUCT):
    structure = (('Data', '28s=b""'),)

    def getAlignment(self):
        if False:
            while True:
                i = 10
        return 4

class DRS_HANDLE(NDRSTRUCT):
    structure = (('Data', '20s=b""'),)

    def getAlignment(self):
        if False:
            while True:
                i = 10
        return 4

class PDRS_HANDLE(NDRPOINTER):
    referent = (('Data', DRS_HANDLE),)

class BYTE_ARRAY(NDRUniConformantArray):
    item = 'c'

class PBYTE_ARRAY(NDRPOINTER):
    referent = (('Data', BYTE_ARRAY),)

class DRS_EXTENSIONS(NDRSTRUCT):
    structure = (('cb', DWORD), ('rgb', BYTE_ARRAY))

class PDRS_EXTENSIONS(NDRPOINTER):
    referent = (('Data', DRS_EXTENSIONS),)

class DRS_EXTENSIONS_INT(Structure):
    structure = (('dwFlags', '<L=0'), ('SiteObjGuid', '16s=b""'), ('Pid', '<L=0'), ('dwReplEpoch', '<L=0'), ('dwFlagsExt', '<L=0'), ('ConfigObjGUID', '16s=b""'), ('dwExtCaps', '<L=0'))

class DRS_MSG_DCINFOREQ_V1(NDRSTRUCT):
    structure = (('Domain', LPWSTR), ('InfoLevel', DWORD))

class DRS_MSG_DCINFOREQ(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_DCINFOREQ_V1)}

class DS_DOMAIN_CONTROLLER_INFO_1W(NDRSTRUCT):
    structure = (('NetbiosName', LPWSTR), ('DnsHostName', LPWSTR), ('SiteName', LPWSTR), ('ComputerObjectName', LPWSTR), ('ServerObjectName', LPWSTR), ('fIsPdc', BOOL), ('fDsEnabled', BOOL))

class DS_DOMAIN_CONTROLLER_INFO_1W_ARRAY(NDRUniConformantArray):
    item = DS_DOMAIN_CONTROLLER_INFO_1W

class PDS_DOMAIN_CONTROLLER_INFO_1W_ARRAY(NDRPOINTER):
    referent = (('Data', DS_DOMAIN_CONTROLLER_INFO_1W_ARRAY),)

class DRS_MSG_DCINFOREPLY_V1(NDRSTRUCT):
    structure = (('cItems', DWORD), ('rItems', PDS_DOMAIN_CONTROLLER_INFO_1W_ARRAY))

class DS_DOMAIN_CONTROLLER_INFO_2W(NDRSTRUCT):
    structure = (('NetbiosName', LPWSTR), ('DnsHostName', LPWSTR), ('SiteName', LPWSTR), ('SiteObjectName', LPWSTR), ('ComputerObjectName', LPWSTR), ('ServerObjectName', LPWSTR), ('NtdsDsaObjectName', LPWSTR), ('fIsPdc', BOOL), ('fDsEnabled', BOOL), ('fIsGc', BOOL), ('SiteObjectGuid', GUID), ('ComputerObjectGuid', GUID), ('ServerObjectGuid', GUID), ('NtdsDsaObjectGuid', GUID))

class DS_DOMAIN_CONTROLLER_INFO_2W_ARRAY(NDRUniConformantArray):
    item = DS_DOMAIN_CONTROLLER_INFO_2W

class PDS_DOMAIN_CONTROLLER_INFO_2W_ARRAY(NDRPOINTER):
    referent = (('Data', DS_DOMAIN_CONTROLLER_INFO_2W_ARRAY),)

class DRS_MSG_DCINFOREPLY_V2(NDRSTRUCT):
    structure = (('cItems', DWORD), ('rItems', PDS_DOMAIN_CONTROLLER_INFO_2W_ARRAY))

class DS_DOMAIN_CONTROLLER_INFO_3W(NDRSTRUCT):
    structure = (('NetbiosName', LPWSTR), ('DnsHostName', LPWSTR), ('SiteName', LPWSTR), ('SiteObjectName', LPWSTR), ('ComputerObjectName', LPWSTR), ('ServerObjectName', LPWSTR), ('NtdsDsaObjectName', LPWSTR), ('fIsPdc', BOOL), ('fDsEnabled', BOOL), ('fIsGc', BOOL), ('fIsRodc', BOOL), ('SiteObjectGuid', GUID), ('ComputerObjectGuid', GUID), ('ServerObjectGuid', GUID), ('NtdsDsaObjectGuid', GUID))

class DS_DOMAIN_CONTROLLER_INFO_3W_ARRAY(NDRUniConformantArray):
    item = DS_DOMAIN_CONTROLLER_INFO_3W

class PDS_DOMAIN_CONTROLLER_INFO_3W_ARRAY(NDRPOINTER):
    referent = (('Data', DS_DOMAIN_CONTROLLER_INFO_3W_ARRAY),)

class DRS_MSG_DCINFOREPLY_V3(NDRSTRUCT):
    structure = (('cItems', DWORD), ('rItems', PDS_DOMAIN_CONTROLLER_INFO_3W_ARRAY))

class DS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW(NDRSTRUCT):
    structure = (('IPAddress', DWORD), ('NotificationCount', DWORD), ('secTimeConnected', DWORD), ('Flags', DWORD), ('TotalRequests', DWORD), ('Reserved1', DWORD), ('UserName', LPWSTR))

class DS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW_ARRAY(NDRUniConformantArray):
    item = DS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW

class PDS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW_ARRAY(NDRPOINTER):
    referent = (('Data', DS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW_ARRAY),)

class DRS_MSG_DCINFOREPLY_VFFFFFFFF(NDRSTRUCT):
    structure = (('cItems', DWORD), ('rItems', PDS_DOMAIN_CONTROLLER_INFO_FFFFFFFFW_ARRAY))

class DRS_MSG_DCINFOREPLY(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_DCINFOREPLY_V1), 2: ('V2', DRS_MSG_DCINFOREPLY_V2), 3: ('V3', DRS_MSG_DCINFOREPLY_V3), 4294967295: ('V1', DRS_MSG_DCINFOREPLY_VFFFFFFFF)}

class LPWSTR_ARRAY(NDRUniConformantArray):
    item = LPWSTR

class PLPWSTR_ARRAY(NDRPOINTER):
    referent = (('Data', LPWSTR_ARRAY),)

class DRS_MSG_CRACKREQ_V1(NDRSTRUCT):
    structure = (('CodePage', ULONG), ('LocaleId', ULONG), ('dwFlags', DWORD), ('formatOffered', DWORD), ('formatDesired', DWORD), ('cNames', DWORD), ('rpNames', PLPWSTR_ARRAY))

class DRS_MSG_CRACKREQ(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_CRACKREQ_V1)}

class DS_NAME_FORMAT(NDRENUM):

    class enumItems(Enum):
        DS_UNKNOWN_NAME = 0
        DS_FQDN_1779_NAME = 1
        DS_NT4_ACCOUNT_NAME = 2
        DS_DISPLAY_NAME = 3
        DS_UNIQUE_ID_NAME = 6
        DS_CANONICAL_NAME = 7
        DS_USER_PRINCIPAL_NAME = 8
        DS_CANONICAL_NAME_EX = 9
        DS_SERVICE_PRINCIPAL_NAME = 10
        DS_SID_OR_SID_HISTORY_NAME = 11
        DS_DNS_DOMAIN_NAME = 12

class DS_NAME_RESULT_ITEMW(NDRSTRUCT):
    structure = (('status', DWORD), ('pDomain', LPWSTR), ('pName', LPWSTR))

class DS_NAME_RESULT_ITEMW_ARRAY(NDRUniConformantArray):
    item = DS_NAME_RESULT_ITEMW

class PDS_NAME_RESULT_ITEMW_ARRAY(NDRPOINTER):
    referent = (('Data', DS_NAME_RESULT_ITEMW_ARRAY),)

class DS_NAME_RESULTW(NDRSTRUCT):
    structure = (('cItems', DWORD), ('rItems', PDS_NAME_RESULT_ITEMW_ARRAY))

class PDS_NAME_RESULTW(NDRPOINTER):
    referent = (('Data', DS_NAME_RESULTW),)

class DRS_MSG_CRACKREPLY_V1(NDRSTRUCT):
    structure = (('pResult', PDS_NAME_RESULTW),)

class DRS_MSG_CRACKREPLY(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_CRACKREPLY_V1)}

class UPTODATE_CURSOR_V1(NDRSTRUCT):
    structure = (('uuidDsa', UUID), ('usnHighPropUpdate', USN))

class UPTODATE_CURSOR_V1_ARRAY(NDRUniConformantArray):
    item = UPTODATE_CURSOR_V1

class UPTODATE_VECTOR_V1_EXT(NDRSTRUCT):
    structure = (('dwVersion', DWORD), ('dwReserved1', DWORD), ('cNumCursors', DWORD), ('dwReserved2', DWORD), ('rgCursors', UPTODATE_CURSOR_V1_ARRAY))

class PUPTODATE_VECTOR_V1_EXT(NDRPOINTER):
    referent = (('Data', UPTODATE_VECTOR_V1_EXT),)

class USN_VECTOR(NDRSTRUCT):
    structure = (('usnHighObjUpdate', USN), ('usnReserved', USN), ('usnHighPropUpdate', USN))

class WCHAR_ARRAY(NDRUniConformantArray):
    item = 'H'

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.fields['MaximumCount'] = None
        self.data = None
        return NDRUniConformantArray.__setitem__(self, key, [ord(c) for c in value])

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if key == 'Data':
            try:
                return ''.join([six.unichr(i) for i in self.fields[key]])
            except ValueError as e:
                LOG.debug('ValueError Exception', exc_info=True)
                LOG.error(str(e))
        else:
            return NDR.__getitem__(self, key)

class DSNAME(NDRSTRUCT):
    structure = (('structLen', ULONG), ('SidLen', ULONG), ('Guid', GUID), ('Sid', NT4SID), ('NameLen', ULONG), ('StringName', WCHAR_ARRAY))

    def getDataLen(self, data, offset=0):
        if False:
            while True:
                i = 10
        return self['NameLen']

    def getData(self, soFar=0):
        if False:
            i = 10
            return i + 15
        return NDRSTRUCT.getData(self, soFar)

class PDSNAME(NDRPOINTER):
    referent = (('Data', DSNAME),)

class PDSNAME_ARRAY(NDRUniConformantArray):
    item = PDSNAME

class PPDSNAME_ARRAY(NDRPOINTER):
    referent = (('Data', PDSNAME_ARRAY),)

class ATTRTYP_ARRAY(NDRUniConformantArray):
    item = ATTRTYP

class PARTIAL_ATTR_VECTOR_V1_EXT(NDRSTRUCT):
    structure = (('dwVersion', DWORD), ('dwReserved1', DWORD), ('cAttrs', DWORD), ('rgPartialAttr', ATTRTYP_ARRAY))

class PPARTIAL_ATTR_VECTOR_V1_EXT(NDRPOINTER):
    referent = (('Data', PARTIAL_ATTR_VECTOR_V1_EXT),)

class OID_t(NDRSTRUCT):
    structure = (('length', ULONG), ('elements', PBYTE_ARRAY))

class PrefixTableEntry(NDRSTRUCT):
    structure = (('ndx', ULONG), ('prefix', OID_t))

class PrefixTableEntry_ARRAY(NDRUniConformantArray):
    item = PrefixTableEntry

class PPrefixTableEntry_ARRAY(NDRPOINTER):
    referent = (('Data', PrefixTableEntry_ARRAY),)

class SCHEMA_PREFIX_TABLE(NDRSTRUCT):
    structure = (('PrefixCount', DWORD), ('pPrefixEntry', PPrefixTableEntry_ARRAY))

class DRS_MSG_GETCHGREQ_V3(NDRSTRUCT):
    structure = (('uuidDsaObjDest', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('pUpToDateVecDestV1', PUPTODATE_VECTOR_V1_EXT), ('pPartialAttrVecDestV1', PPARTIAL_ATTR_VECTOR_V1_EXT), ('PrefixTableDest', SCHEMA_PREFIX_TABLE), ('ulFlags', ULONG), ('cMaxObjects', ULONG), ('cMaxBytes', ULONG), ('ulExtendedOp', ULONG))

class MTX_ADDR(NDRSTRUCT):
    structure = (('mtx_namelen', ULONG), ('mtx_name', PBYTE_ARRAY))

class PMTX_ADDR(NDRPOINTER):
    referent = (('Data', MTX_ADDR),)

class DRS_MSG_GETCHGREQ_V4(NDRSTRUCT):
    structure = (('uuidTransportObj', UUID), ('pmtxReturnAddress', PMTX_ADDR), ('V3', DRS_MSG_GETCHGREQ_V3))

class DRS_MSG_GETCHGREQ_V5(NDRSTRUCT):
    structure = (('uuidDsaObjDest', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('pUpToDateVecDestV1', PUPTODATE_VECTOR_V1_EXT), ('ulFlags', ULONG), ('cMaxObjects', ULONG), ('cMaxBytes', ULONG), ('ulExtendedOp', ULONG), ('liFsmoInfo', ULARGE_INTEGER))

class DRS_MSG_GETCHGREQ_V7(NDRSTRUCT):
    structure = (('uuidTransportObj', UUID), ('pmtxReturnAddress', PMTX_ADDR), ('V3', DRS_MSG_GETCHGREQ_V3), ('pPartialAttrSet', PPARTIAL_ATTR_VECTOR_V1_EXT), ('pPartialAttrSetEx1', PPARTIAL_ATTR_VECTOR_V1_EXT), ('PrefixTableDest', SCHEMA_PREFIX_TABLE))

class DRS_MSG_GETCHGREQ_V8(NDRSTRUCT):
    structure = (('uuidDsaObjDest', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('pUpToDateVecDest', PUPTODATE_VECTOR_V1_EXT), ('ulFlags', ULONG), ('cMaxObjects', ULONG), ('cMaxBytes', ULONG), ('ulExtendedOp', ULONG), ('liFsmoInfo', ULARGE_INTEGER), ('pPartialAttrSet', PPARTIAL_ATTR_VECTOR_V1_EXT), ('pPartialAttrSetEx1', PPARTIAL_ATTR_VECTOR_V1_EXT), ('PrefixTableDest', SCHEMA_PREFIX_TABLE))

class DRS_MSG_GETCHGREQ_V10(NDRSTRUCT):
    structure = (('uuidDsaObjDest', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('pUpToDateVecDest', PUPTODATE_VECTOR_V1_EXT), ('ulFlags', ULONG), ('cMaxObjects', ULONG), ('cMaxBytes', ULONG), ('ulExtendedOp', ULONG), ('liFsmoInfo', ULARGE_INTEGER), ('pPartialAttrSet', PPARTIAL_ATTR_VECTOR_V1_EXT), ('pPartialAttrSetEx1', PPARTIAL_ATTR_VECTOR_V1_EXT), ('PrefixTableDest', SCHEMA_PREFIX_TABLE), ('ulMoreFlags', ULONG))

class DRS_MSG_GETCHGREQ(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {4: ('V4', DRS_MSG_GETCHGREQ_V4), 5: ('V5', DRS_MSG_GETCHGREQ_V5), 7: ('V7', DRS_MSG_GETCHGREQ_V7), 8: ('V8', DRS_MSG_GETCHGREQ_V8), 10: ('V10', DRS_MSG_GETCHGREQ_V10)}

class ATTRVAL(NDRSTRUCT):
    structure = (('valLen', ULONG), ('pVal', PBYTE_ARRAY))

class ATTRVAL_ARRAY(NDRUniConformantArray):
    item = ATTRVAL

class PATTRVAL_ARRAY(NDRPOINTER):
    referent = (('Data', ATTRVAL_ARRAY),)

class ATTRVALBLOCK(NDRSTRUCT):
    structure = (('valCount', ULONG), ('pAVal', PATTRVAL_ARRAY))

class ATTR(NDRSTRUCT):
    structure = (('attrTyp', ATTRTYP), ('AttrVal', ATTRVALBLOCK))

class ATTR_ARRAY(NDRUniConformantArray):
    item = ATTR

class PATTR_ARRAY(NDRPOINTER):
    referent = (('Data', ATTR_ARRAY),)

class ATTRBLOCK(NDRSTRUCT):
    structure = (('attrCount', ULONG), ('pAttr', PATTR_ARRAY))

class ENTINF(NDRSTRUCT):
    structure = (('pName', PDSNAME), ('ulFlags', ULONG), ('AttrBlock', ATTRBLOCK))

class ENTINF_ARRAY(NDRUniConformantArray):
    item = ENTINF

class PENTINF_ARRAY(NDRPOINTER):
    referent = (('Data', ENTINF_ARRAY),)

class PROPERTY_META_DATA_EXT(NDRSTRUCT):
    structure = (('dwVersion', DWORD), ('timeChanged', DSTIME), ('uuidDsaOriginating', UUID), ('usnOriginating', USN))

class PROPERTY_META_DATA_EXT_ARRAY(NDRUniConformantArray):
    item = PROPERTY_META_DATA_EXT

class PROPERTY_META_DATA_EXT_VECTOR(NDRSTRUCT):
    structure = (('cNumProps', DWORD), ('rgMetaData', PROPERTY_META_DATA_EXT_ARRAY))

class PPROPERTY_META_DATA_EXT_VECTOR(NDRPOINTER):
    referent = (('Data', PROPERTY_META_DATA_EXT_VECTOR),)

class REPLENTINFLIST(NDRSTRUCT):
    structure = (('pNextEntInf', NDRPOINTER), ('Entinf', ENTINF), ('fIsNCPrefix', BOOL), ('pParentGuidm', PUUID), ('pMetaDataExt', PPROPERTY_META_DATA_EXT_VECTOR))

    def fromString(self, data, soFar=0):
        if False:
            i = 10
            return i + 15
        self.fields['pNextEntInf'] = PREPLENTINFLIST(isNDR64=self._isNDR64)
        retVal = NDRSTRUCT.fromString(self, data, soFar)
        return retVal

class PREPLENTINFLIST(NDRPOINTER):
    referent = (('Data', REPLENTINFLIST),)

class DRS_MSG_GETCHGREPLY_V1(NDRSTRUCT):
    structure = (('uuidDsaObjSrc', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('usnvecTo', USN_VECTOR), ('pUpToDateVecSrcV1', PUPTODATE_VECTOR_V1_EXT), ('PrefixTableSrc', SCHEMA_PREFIX_TABLE), ('ulExtendedRet', EXOP_ERR), ('cNumObjects', ULONG), ('cNumBytes', ULONG), ('pObjects', PREPLENTINFLIST), ('fMoreData', BOOL))

class DRS_COMPRESSED_BLOB(NDRSTRUCT):
    structure = (('cbUncompressedSize', DWORD), ('cbCompressedSize', DWORD), ('pbCompressedData', BYTE_ARRAY))

class DRS_MSG_GETCHGREPLY_V2(NDRSTRUCT):
    structure = (('CompressedV1', DRS_COMPRESSED_BLOB),)

class UPTODATE_CURSOR_V2(NDRSTRUCT):
    structure = (('uuidDsa', UUID), ('usnHighPropUpdate', USN), ('timeLastSyncSuccess', DSTIME))

class UPTODATE_CURSOR_V2_ARRAY(NDRUniConformantArray):
    item = UPTODATE_CURSOR_V2

class UPTODATE_VECTOR_V2_EXT(NDRSTRUCT):
    structure = (('dwVersion', DWORD), ('dwReserved1', DWORD), ('cNumCursors', DWORD), ('dwReserved2', DWORD), ('rgCursors', UPTODATE_CURSOR_V2_ARRAY))

class PUPTODATE_VECTOR_V2_EXT(NDRPOINTER):
    referent = (('Data', UPTODATE_VECTOR_V2_EXT),)

class VALUE_META_DATA_EXT_V1(NDRSTRUCT):
    structure = (('timeCreated', DSTIME), ('MetaData', PROPERTY_META_DATA_EXT))

class VALUE_META_DATA_EXT_V3(NDRSTRUCT):
    structure = (('timeCreated', DSTIME), ('MetaData', PROPERTY_META_DATA_EXT), ('unused1', DWORD), ('unused1', DWORD), ('unused1', DWORD), ('timeExpired', DSTIME))

class REPLVALINF_V1(NDRSTRUCT):
    structure = (('pObject', PDSNAME), ('attrTyp', ATTRTYP), ('Aval', ATTRVAL), ('fIsPresent', BOOL), ('MetaData', VALUE_META_DATA_EXT_V1))

    def fromString(self, data, soFar=0):
        if False:
            print('Hello World!')
        retVal = NDRSTRUCT.fromString(self, data, soFar)
        return retVal

class REPLVALINF_V1_ARRAY(NDRUniConformantArray):
    item = REPLVALINF_V1

class PREPLVALINF_V1_ARRAY(NDRPOINTER):
    referent = (('Data', REPLVALINF_V1_ARRAY),)

class REPLVALINF_V3(NDRSTRUCT):
    structure = (('pObject', PDSNAME), ('attrTyp', ATTRTYP), ('Aval', ATTRVAL), ('fIsPresent', BOOL), ('MetaData', VALUE_META_DATA_EXT_V3))

    def fromString(self, data, soFar=0):
        if False:
            i = 10
            return i + 15
        retVal = NDRSTRUCT.fromString(self, data, soFar)
        return retVal

class REPLVALINF_V3_ARRAY(NDRUniConformantArray):
    item = REPLVALINF_V3

class PREPLVALINF_V3_ARRAY(NDRPOINTER):
    referent = (('Data', REPLVALINF_V3_ARRAY),)
REPLVALINF_NATIVE = REPLVALINF_V3

class DRS_MSG_GETCHGREPLY_V6(NDRSTRUCT):
    structure = (('uuidDsaObjSrc', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('usnvecTo', USN_VECTOR), ('pUpToDateVecSrc', PUPTODATE_VECTOR_V2_EXT), ('PrefixTableSrc', SCHEMA_PREFIX_TABLE), ('ulExtendedRet', EXOP_ERR), ('cNumObjects', ULONG), ('cNumBytes', ULONG), ('pObjects', PREPLENTINFLIST), ('fMoreData', BOOL), ('cNumNcSizeObjectsc', ULONG), ('cNumNcSizeValues', ULONG), ('cNumValues', DWORD), ('rgValues', DWORD), ('dwDRSError', DWORD))

class DRS_COMP_ALG_TYPE(NDRENUM):

    class enumItems(Enum):
        DRS_COMP_ALG_NONE = 0
        DRS_COMP_ALG_UNUSED = 1
        DRS_COMP_ALG_MSZIP = 2
        DRS_COMP_ALG_WIN2K3 = 3

class DRS_MSG_GETCHGREPLY_V7(NDRSTRUCT):
    structure = (('dwCompressedVersion', DWORD), ('CompressionAlg', DRS_COMP_ALG_TYPE), ('CompressedAny', DRS_COMPRESSED_BLOB))

class DRS_MSG_GETCHGREPLY_V9(NDRSTRUCT):
    structure = (('uuidDsaObjSrc', UUID), ('uuidInvocIdSrc', UUID), ('pNC', PDSNAME), ('usnvecFrom', USN_VECTOR), ('usnvecTo', USN_VECTOR), ('pUpToDateVecSrc', PUPTODATE_VECTOR_V2_EXT), ('PrefixTableSrc', SCHEMA_PREFIX_TABLE), ('ulExtendedRet', EXOP_ERR), ('cNumObjects', ULONG), ('cNumBytes', ULONG), ('pObjects', PREPLENTINFLIST), ('fMoreData', BOOL), ('cNumNcSizeObjectsc', ULONG), ('cNumNcSizeValues', ULONG), ('cNumValues', DWORD), ('rgValues', DWORD), ('dwDRSError', DWORD))
DRS_MSG_GETCHGREPLY_NATIVE = DRS_MSG_GETCHGREPLY_V9

class DRS_MSG_GETCHGREPLY(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_GETCHGREPLY_V1), 2: ('V2', DRS_MSG_GETCHGREPLY_V2), 6: ('V6', DRS_MSG_GETCHGREPLY_V6), 7: ('V7', DRS_MSG_GETCHGREPLY_V7), 9: ('V9', DRS_MSG_GETCHGREPLY_V9)}

class DRS_MSG_VERIFYREQ_V1(NDRSTRUCT):
    structure = (('dwFlags', DWORD), ('cNames', DWORD), ('rpNames', PPDSNAME_ARRAY), ('RequiredAttrs', ATTRBLOCK), ('PrefixTable', SCHEMA_PREFIX_TABLE))

class DRS_MSG_VERIFYREQ(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_VERIFYREQ_V1)}

class DRS_MSG_VERIFYREPLY_V1(NDRSTRUCT):
    structure = (('error', DWORD), ('cNames', DWORD), ('rpEntInf', PENTINF_ARRAY), ('PrefixTable', SCHEMA_PREFIX_TABLE))

class DRS_MSG_VERIFYREPLY(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_VERIFYREPLY_V1)}

class DRS_MSG_NT4_CHGLOG_REQ_V1(NDRSTRUCT):
    structure = (('dwFlags', DWORD), ('PreferredMaximumLength', DWORD), ('cbRestart', DWORD), ('pRestart', PBYTE_ARRAY))

class DRS_MSG_NT4_CHGLOG_REQ(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_NT4_CHGLOG_REQ_V1)}

class NT4_REPLICATION_STATE(NDRSTRUCT):
    structure = (('SamSerialNumber', LARGE_INTEGER), ('SamCreationTime', LARGE_INTEGER), ('BuiltinSerialNumber', LARGE_INTEGER), ('BuiltinCreationTime', LARGE_INTEGER), ('LsaSerialNumber', LARGE_INTEGER), ('LsaCreationTime', LARGE_INTEGER))

class DRS_MSG_NT4_CHGLOG_REPLY_V1(NDRSTRUCT):
    structure = (('cbRestart', DWORD), ('cbLog', DWORD), ('ReplicationState', NT4_REPLICATION_STATE), ('ActualNtStatus', DWORD), ('pRestart', PBYTE_ARRAY), ('pLog', PBYTE_ARRAY))

class DRS_MSG_NT4_CHGLOG_REPLY(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', DRS_MSG_NT4_CHGLOG_REPLY_V1)}

class DRSBind(NDRCALL):
    opnum = 0
    structure = (('puuidClientDsa', PUUID), ('pextClient', PDRS_EXTENSIONS))

class DRSBindResponse(NDRCALL):
    structure = (('ppextServer', PDRS_EXTENSIONS), ('phDrs', DRS_HANDLE), ('ErrorCode', DWORD))

class DRSUnbind(NDRCALL):
    opnum = 1
    structure = (('phDrs', DRS_HANDLE),)

class DRSUnbindResponse(NDRCALL):
    structure = (('phDrs', DRS_HANDLE), ('ErrorCode', DWORD))

class DRSGetNCChanges(NDRCALL):
    opnum = 3
    structure = (('hDrs', DRS_HANDLE), ('dwInVersion', DWORD), ('pmsgIn', DRS_MSG_GETCHGREQ))

class DRSGetNCChangesResponse(NDRCALL):
    structure = (('pdwOutVersion', DWORD), ('pmsgOut', DRS_MSG_GETCHGREPLY), ('ErrorCode', DWORD))

class DRSVerifyNames(NDRCALL):
    opnum = 8
    structure = (('hDrs', DRS_HANDLE), ('dwInVersion', DWORD), ('pmsgIn', DRS_MSG_VERIFYREQ))

class DRSVerifyNamesResponse(NDRCALL):
    structure = (('pdwOutVersion', DWORD), ('pmsgOut', DRS_MSG_VERIFYREPLY), ('ErrorCode', DWORD))

class DRSGetNT4ChangeLog(NDRCALL):
    opnum = 11
    structure = (('hDrs', DRS_HANDLE), ('dwInVersion', DWORD), ('pmsgIn', DRS_MSG_NT4_CHGLOG_REQ))

class DRSGetNT4ChangeLogResponse(NDRCALL):
    structure = (('pdwOutVersion', DWORD), ('pmsgOut', DRS_MSG_NT4_CHGLOG_REPLY), ('ErrorCode', DWORD))

class DRSCrackNames(NDRCALL):
    opnum = 12
    structure = (('hDrs', DRS_HANDLE), ('dwInVersion', DWORD), ('pmsgIn', DRS_MSG_CRACKREQ))

class DRSCrackNamesResponse(NDRCALL):
    structure = (('pdwOutVersion', DWORD), ('pmsgOut', DRS_MSG_CRACKREPLY), ('ErrorCode', DWORD))

class DRSDomainControllerInfo(NDRCALL):
    opnum = 16
    structure = (('hDrs', DRS_HANDLE), ('dwInVersion', DWORD), ('pmsgIn', DRS_MSG_DCINFOREQ))

class DRSDomainControllerInfoResponse(NDRCALL):
    structure = (('pdwOutVersion', DWORD), ('pmsgOut', DRS_MSG_DCINFOREPLY), ('ErrorCode', DWORD))
OPNUMS = {0: (DRSBind, DRSBindResponse), 1: (DRSUnbind, DRSUnbindResponse), 3: (DRSGetNCChanges, DRSGetNCChangesResponse), 12: (DRSCrackNames, DRSCrackNamesResponse), 16: (DRSDomainControllerInfo, DRSDomainControllerInfoResponse)}

def checkNullString(string):
    if False:
        i = 10
        return i + 15
    if string == NULL:
        return string
    if string[-1:] != '\x00':
        return string + '\x00'
    else:
        return string

def hDRSUnbind(dce, hDrs):
    if False:
        for i in range(10):
            print('nop')
    request = DRSUnbind()
    request['phDrs'] = hDrs
    return dce.request(request)

def hDRSDomainControllerInfo(dce, hDrs, domain, infoLevel):
    if False:
        while True:
            i = 10
    request = DRSDomainControllerInfo()
    request['hDrs'] = hDrs
    request['dwInVersion'] = 1
    request['pmsgIn']['tag'] = 1
    request['pmsgIn']['V1']['Domain'] = checkNullString(domain)
    request['pmsgIn']['V1']['InfoLevel'] = infoLevel
    return dce.request(request)

def hDRSCrackNames(dce, hDrs, flags, formatOffered, formatDesired, rpNames=()):
    if False:
        for i in range(10):
            print('nop')
    request = DRSCrackNames()
    request['hDrs'] = hDrs
    request['dwInVersion'] = 1
    request['pmsgIn']['tag'] = 1
    request['pmsgIn']['V1']['CodePage'] = 0
    request['pmsgIn']['V1']['LocaleId'] = 0
    request['pmsgIn']['V1']['dwFlags'] = flags
    request['pmsgIn']['V1']['formatOffered'] = formatOffered
    request['pmsgIn']['V1']['formatDesired'] = formatDesired
    request['pmsgIn']['V1']['cNames'] = len(rpNames)
    for name in rpNames:
        record = LPWSTR()
        record['Data'] = checkNullString(name)
        request['pmsgIn']['V1']['rpNames'].append(record)
    return dce.request(request)

def deriveKey(baseKey):
    if False:
        while True:
            i = 10
    key = pack('<L', baseKey)
    key1 = [key[0], key[1], key[2], key[3], key[0], key[1], key[2]]
    key2 = [key[3], key[0], key[1], key[2], key[3], key[0], key[1]]
    if PY2:
        return (transformKey(b''.join(key1)), transformKey(b''.join(key2)))
    else:
        return (transformKey(bytes(key1)), transformKey(bytes(key2)))

def removeDESLayer(cryptedHash, rid):
    if False:
        for i in range(10):
            print('nop')
    (Key1, Key2) = deriveKey(rid)
    Crypt1 = DES.new(Key1, DES.MODE_ECB)
    Crypt2 = DES.new(Key2, DES.MODE_ECB)
    decryptedHash = Crypt1.decrypt(cryptedHash[:8]) + Crypt2.decrypt(cryptedHash[8:])
    return decryptedHash

def DecryptAttributeValue(dce, attribute):
    if False:
        i = 10
        return i + 15
    sessionKey = dce.get_session_key()
    if isinstance(sessionKey, crypto.Key):
        sessionKey = sessionKey.contents
    encryptedPayload = ENCRYPTED_PAYLOAD(attribute)
    md5 = hashlib.new('md5')
    md5.update(sessionKey)
    md5.update(encryptedPayload['Salt'])
    finalMD5 = md5.digest()
    cipher = ARC4.new(finalMD5)
    plainText = cipher.decrypt(attribute[16:])
    return plainText[4:]

def MakeAttid(prefixTable, oid):
    if False:
        return 10
    lastValue = int(oid.split('.')[-1])
    from pyasn1.type import univ
    from pyasn1.codec.ber import encoder
    binaryOID = encoder.encode(univ.ObjectIdentifier(oid))[2:]
    if lastValue < 128:
        oidPrefix = list(binaryOID[:-1])
    else:
        oidPrefix = list(binaryOID[:-2])
    fToAdd = True
    pos = len(prefixTable)
    for (j, item) in enumerate(prefixTable):
        if item['prefix']['elements'] == oidPrefix:
            fToAdd = False
            pos = j
            break
    if fToAdd is True:
        entry = PrefixTableEntry()
        entry['ndx'] = pos
        entry['prefix']['length'] = len(oidPrefix)
        entry['prefix']['elements'] = oidPrefix
        prefixTable.append(entry)
    lowerWord = lastValue % 16384
    if lastValue >= 16384:
        lowerWord += 32768
    upperWord = pos
    attrTyp = ATTRTYP()
    attrTyp['Data'] = (upperWord << 16) + lowerWord
    return attrTyp

def OidFromAttid(prefixTable, attr):
    if False:
        print('Hello World!')
    upperWord = attr // 65536
    lowerWord = attr % 65536
    binaryOID = None
    for (j, item) in enumerate(prefixTable):
        if item['ndx'] == upperWord:
            binaryOID = item['prefix']['elements'][:item['prefix']['length']]
            if lowerWord < 128:
                binaryOID.append(pack('B', lowerWord))
            else:
                if lowerWord >= 32768:
                    lowerWord -= 32768
                binaryOID.append(pack('B', lowerWord // 128 % 128 + 128))
                binaryOID.append(pack('B', lowerWord % 128))
            break
    if binaryOID is None:
        return None
    return str(decoder.decode(b'\x06' + pack('B', len(binaryOID)) + b''.join(binaryOID), asn1Spec=univ.ObjectIdentifier())[0])
if __name__ == '__main__':
    prefixTable = []
    oid0 = '1.2.840.113556.1.4.94'
    oid1 = '2.5.6.2'
    oid2 = '1.2.840.113556.1.2.1'
    oid3 = '1.2.840.113556.1.3.223'
    oid4 = '1.2.840.113556.1.5.7000.53'
    o0 = MakeAttid(prefixTable, oid0)
    print(hex(o0))
    o1 = MakeAttid(prefixTable, oid1)
    print(hex(o1))
    o2 = MakeAttid(prefixTable, oid2)
    print(hex(o2))
    o3 = MakeAttid(prefixTable, oid3)
    print(hex(o3))
    o4 = MakeAttid(prefixTable, oid4)
    print(hex(o4))
    jj = OidFromAttid(prefixTable, o0)
    print(jj)
    jj = OidFromAttid(prefixTable, o1)
    print(jj)
    jj = OidFromAttid(prefixTable, o2)
    print(jj)
    jj = OidFromAttid(prefixTable, o3)
    print(jj)
    jj = OidFromAttid(prefixTable, o4)
    print(jj)