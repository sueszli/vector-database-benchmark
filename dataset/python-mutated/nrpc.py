import time
from struct import pack, unpack
from six import b
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRENUM, NDRUNION, NDRPOINTER, NDRUniConformantArray, NDRUniFixedArray, NDRUniConformantVaryingArray
from impacket.dcerpc.v5.dtypes import WSTR, LPWSTR, DWORD, ULONG, USHORT, PGUID, NTSTATUS, NULL, LONG, UCHAR, PRPC_SID, GUID, RPC_UNICODE_STRING, SECURITY_INFORMATION, LPULONG, ULONGLONG
from impacket import system_errors, nt_errors
from impacket.uuid import uuidtup_to_bin
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.samr import OLD_LARGE_INTEGER
from impacket.dcerpc.v5.lsad import PLSA_FOREST_TRUST_INFORMATION
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.structure import Structure
from impacket import ntlm, crypto, LOG
import hmac
import hashlib
try:
    from Cryptodome.Cipher import DES, AES, ARC4
except ImportError:
    LOG.critical("Warning: You don't have any crypto installed. You need pycryptodomex")
    LOG.critical('See https://pypi.org/project/pycryptodomex/')
MSRPC_UUID_NRPC = uuidtup_to_bin(('12345678-1234-ABCD-EF00-01234567CFFB', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            for i in range(10):
                print('nop')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        key = self.error_code
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'NRPC SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        elif key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'NRPC SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'NRPC SessionError: unknown error code: 0x%x' % self.error_code
NlDnsLdapAtSite = 22
NlDnsGcAtSite = 25
NlDnsDsaCname = 28
NlDnsKdcAtSite = 30
NlDnsDcAtSite = 32
NlDnsRfc1510KdcAtSite = 34
NlDnsGenericGcAtSite = 36
NlDnsDomainName = 1
NlDnsDomainNameAlias = 2
NlDnsForestName = 3
NlDnsForestNameAlias = 4
NlDnsNdncDomainName = 5
NlDnsRecordName = 6
VER_SUITE_BACKOFFICE = 4
VER_SUITE_BLADE = 1024
VER_SUITE_COMPUTE_SERVER = 16384
VER_SUITE_DATACENTER = 128
VER_SUITE_ENTERPRISE = 2
VER_SUITE_EMBEDDEDNT = 64
VER_SUITE_PERSONAL = 512
VER_SUITE_SINGLEUSERTS = 256
VER_SUITE_SMALLBUSINESS = 1
VER_SUITE_SMALLBUSINESS_RESTRICTED = 32
VER_SUITE_STORAGE_SERVER = 8192
VER_SUITE_TERMINAL = 16
VER_NT_DOMAIN_CONTROLLER = 2
VER_NT_SERVER = 3
VER_NT_WORKSTATION = 1
NETLOGON_UAS_LOGON_ACCESS = 1
NETLOGON_UAS_LOGOFF_ACCESS = 2
NETLOGON_CONTROL_ACCESS = 4
NETLOGON_QUERY_ACCESS = 8
NETLOGON_SERVICE_ACCESS = 16
NETLOGON_FTINFO_ACCESS = 32
NETLOGON_WKSTA_RPC_ACCESS = 64
NETLOGON_CONTROL_QUERY = 1
NETLOGON_CONTROL_REPLICATE = 2
NETLOGON_CONTROL_SYNCHRONIZE = 3
NETLOGON_CONTROL_PDC_REPLICATE = 4
NETLOGON_CONTROL_REDISCOVER = 5
NETLOGON_CONTROL_TC_QUERY = 6
NETLOGON_CONTROL_TRANSPORT_NOTIFY = 7
NETLOGON_CONTROL_FIND_USER = 8
NETLOGON_CONTROL_CHANGE_PASSWORD = 9
NETLOGON_CONTROL_TC_VERIFY = 10
NETLOGON_CONTROL_FORCE_DNS_REG = 11
NETLOGON_CONTROL_QUERY_DNS_REG = 12
NETLOGON_CONTROL_BACKUP_CHANGE_LOG = 65532
NETLOGON_CONTROL_TRUNCATE_LOG = 65533
NETLOGON_CONTROL_SET_DBFLAG = 65534
NETLOGON_CONTROL_BREAKPOINT = 65535
LOGONSRV_HANDLE = WSTR
PLOGONSRV_HANDLE = LPWSTR

class CYPHER_BLOCK(NDRSTRUCT):
    structure = (('Data', '8s=b""'),)

    def getAlignment(self):
        if False:
            i = 10
            return i + 15
        return 1
NET_API_STATUS = DWORD
from impacket.dcerpc.v5.lsad import STRING

class CYPHER_BLOCK_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            print('Hello World!')
        return len(CYPHER_BLOCK()) * 2

class LM_OWF_PASSWORD(NDRSTRUCT):
    structure = (('Data', CYPHER_BLOCK_ARRAY),)
NT_OWF_PASSWORD = LM_OWF_PASSWORD
ENCRYPTED_NT_OWF_PASSWORD = NT_OWF_PASSWORD

class UCHAR_FIXED_ARRAY(NDRUniFixedArray):
    align = 1

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        return len(CYPHER_BLOCK())

class NETLOGON_CREDENTIAL(NDRSTRUCT):
    structure = (('Data', UCHAR_FIXED_ARRAY),)

    def getAlignment(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

class NETLOGON_AUTHENTICATOR(NDRSTRUCT):
    structure = (('Credential', NETLOGON_CREDENTIAL), ('Timestamp', DWORD))

class PNETLOGON_AUTHENTICATOR(NDRPOINTER):
    referent = (('Data', NETLOGON_AUTHENTICATOR),)

class DOMAIN_CONTROLLER_INFOW(NDRSTRUCT):
    structure = (('DomainControllerName', LPWSTR), ('DomainControllerAddress', LPWSTR), ('DomainControllerAddressType', ULONG), ('DomainGuid', GUID), ('DomainName', LPWSTR), ('DnsForestName', LPWSTR), ('Flags', ULONG), ('DcSiteName', LPWSTR), ('ClientSiteName', LPWSTR))

class PDOMAIN_CONTROLLER_INFOW(NDRPOINTER):
    referent = (('Data', DOMAIN_CONTROLLER_INFOW),)

class RPC_UNICODE_STRING_ARRAY(NDRUniConformantArray):
    item = RPC_UNICODE_STRING

class PRPC_UNICODE_STRING_ARRAY(NDRPOINTER):
    referent = (('Data', RPC_UNICODE_STRING_ARRAY),)

class NL_SITE_NAME_ARRAY(NDRSTRUCT):
    structure = (('EntryCount', ULONG), ('SiteNames', PRPC_UNICODE_STRING_ARRAY))

class PNL_SITE_NAME_ARRAY(NDRPOINTER):
    referent = (('Data', NL_SITE_NAME_ARRAY),)

class RPC_UNICODE_STRING_ARRAY(NDRUniConformantArray):
    item = RPC_UNICODE_STRING

class NL_SITE_NAME_EX_ARRAY(NDRSTRUCT):
    structure = (('EntryCount', ULONG), ('SiteNames', PRPC_UNICODE_STRING_ARRAY), ('SubnetNames', PRPC_UNICODE_STRING_ARRAY))

class PNL_SITE_NAME_EX_ARRAY(NDRPOINTER):
    referent = (('Data', NL_SITE_NAME_EX_ARRAY),)

class IPv4Address(Structure):
    structure = (('AddressFamily', '<H=0'), ('Port', '<H=0'), ('Address', '<L=0'), ('Padding', '<L=0'))

class UCHAR_ARRAY(NDRUniConformantArray):
    item = 'c'

class PUCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', UCHAR_ARRAY),)

class NL_SOCKET_ADDRESS(NDRSTRUCT):
    structure = (('lpSockaddr', PUCHAR_ARRAY), ('iSockaddrLength', ULONG))

class NL_SOCKET_ADDRESS_ARRAY(NDRUniConformantArray):
    item = NL_SOCKET_ADDRESS

class NL_DNS_NAME_INFO(NDRSTRUCT):
    structure = (('Type', ULONG), ('DnsDomainInfoType', WSTR), ('Priority', ULONG), ('Weight', ULONG), ('Port', ULONG), ('Register', UCHAR), ('Status', ULONG))

class NL_DNS_NAME_INFO_ARRAY(NDRUniConformantArray):
    item = NL_DNS_NAME_INFO

class PNL_DNS_NAME_INFO_ARRAY(NDRPOINTER):
    referent = (('Data', NL_DNS_NAME_INFO_ARRAY),)

class NL_DNS_NAME_INFO_ARRAY(NDRSTRUCT):
    structure = (('EntryCount', ULONG), ('DnsNamesInfo', PNL_DNS_NAME_INFO_ARRAY))

class NETLOGON_LSA_POLICY_INFO(NDRSTRUCT):
    structure = (('LsaPolicySize', ULONG), ('LsaPolicy', PUCHAR_ARRAY))

class PNETLOGON_LSA_POLICY_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_LSA_POLICY_INFO),)

class NETLOGON_WORKSTATION_INFO(NDRSTRUCT):
    structure = (('LsaPolicy', NETLOGON_LSA_POLICY_INFO), ('DnsHostName', LPWSTR), ('SiteName', LPWSTR), ('Dummy1', LPWSTR), ('Dummy2', LPWSTR), ('Dummy3', LPWSTR), ('Dummy4', LPWSTR), ('OsVersion', RPC_UNICODE_STRING), ('OsName', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('WorkstationFlags', ULONG), ('KerberosSupportedEncryptionTypes', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_WORKSTATION_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_WORKSTATION_INFO),)

class NL_TRUST_PASSWORD_FIXED_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        return 512 + 4

    def getAlignment(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

class WCHAR_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            return 10
        return 512

class NL_TRUST_PASSWORD(NDRSTRUCT):
    structure = (('Buffer', WCHAR_ARRAY), ('Length', ULONG))

class PNL_TRUST_PASSWORD(NDRPOINTER):
    referent = (('Data', NL_TRUST_PASSWORD),)

class NL_PASSWORD_VERSION(NDRSTRUCT):
    structure = (('ReservedField', ULONG), ('PasswordVersionNumber', ULONG), ('PasswordVersionPresent', ULONG))

class NETLOGON_WORKSTATION_INFORMATION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('WorkstationInfo', PNETLOGON_WORKSTATION_INFO), 2: ('LsaPolicyInfo', PNETLOGON_LSA_POLICY_INFO)}

class NETLOGON_ONE_DOMAIN_INFO(NDRSTRUCT):
    structure = (('DomainName', RPC_UNICODE_STRING), ('DnsDomainName', RPC_UNICODE_STRING), ('DnsForestName', RPC_UNICODE_STRING), ('DomainGuid', GUID), ('DomainSid', PRPC_SID), ('TrustExtension', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class NETLOGON_ONE_DOMAIN_INFO_ARRAY(NDRUniConformantArray):
    item = NETLOGON_ONE_DOMAIN_INFO

class PNETLOGON_ONE_DOMAIN_INFO_ARRAY(NDRPOINTER):
    referent = (('Data', NETLOGON_ONE_DOMAIN_INFO_ARRAY),)

class NETLOGON_DOMAIN_INFO(NDRSTRUCT):
    structure = (('PrimaryDomain', NETLOGON_ONE_DOMAIN_INFO), ('TrustedDomainCount', ULONG), ('TrustedDomains', PNETLOGON_ONE_DOMAIN_INFO_ARRAY), ('LsaPolicy', NETLOGON_LSA_POLICY_INFO), ('DnsHostNameInDs', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('WorkstationFlags', ULONG), ('SupportedEncTypes', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DOMAIN_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_DOMAIN_INFO),)

class NETLOGON_DOMAIN_INFORMATION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('DomainInfo', PNETLOGON_DOMAIN_INFO), 2: ('LsaPolicyInfo', PNETLOGON_LSA_POLICY_INFO)}

class NETLOGON_SECURE_CHANNEL_TYPE(NDRENUM):

    class enumItems(Enum):
        NullSecureChannel = 0
        MsvApSecureChannel = 1
        WorkstationSecureChannel = 2
        TrustedDnsDomainSecureChannel = 3
        TrustedDomainSecureChannel = 4
        UasServerSecureChannel = 5
        ServerSecureChannel = 6
        CdcServerSecureChannel = 7

class NETLOGON_CAPABILITIES(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('ServerCapabilities', ULONG)}

class UCHAR_FIXED_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            for i in range(10):
                print('nop')
        return 128

class NL_OSVERSIONINFO_V1(NDRSTRUCT):
    structure = (('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', UCHAR_FIXED_ARRAY), ('wServicePackMajor', USHORT), ('wServicePackMinor', USHORT), ('wSuiteMask', USHORT), ('wProductType', UCHAR), ('wReserved', UCHAR))

class PNL_OSVERSIONINFO_V1(NDRPOINTER):
    referent = (('Data', NL_OSVERSIONINFO_V1),)

class PLPWSTR(NDRPOINTER):
    referent = (('Data', LPWSTR),)

class NL_IN_CHAIN_SET_CLIENT_ATTRIBUTES_V1(NDRSTRUCT):
    structure = (('ClientDnsHostName', PLPWSTR), ('OsVersionInfo', PNL_OSVERSIONINFO_V1), ('OsName', PLPWSTR))

class NL_IN_CHAIN_SET_CLIENT_ATTRIBUTES(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', NL_IN_CHAIN_SET_CLIENT_ATTRIBUTES_V1)}

class NL_OUT_CHAIN_SET_CLIENT_ATTRIBUTES_V1(NDRSTRUCT):
    structure = (('HubName', PLPWSTR), ('OldDnsHostName', PLPWSTR), ('SupportedEncTypes', LPULONG))

class NL_OUT_CHAIN_SET_CLIENT_ATTRIBUTES(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('V1', NL_OUT_CHAIN_SET_CLIENT_ATTRIBUTES_V1)}

class CHAR_FIXED_8_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            for i in range(10):
                print('nop')
        return 8

class LM_CHALLENGE(NDRSTRUCT):
    structure = (('Data', CHAR_FIXED_8_ARRAY),)

class NETLOGON_LOGON_IDENTITY_INFO(NDRSTRUCT):
    structure = (('LogonDomainName', RPC_UNICODE_STRING), ('ParameterControl', ULONG), ('Reserved', OLD_LARGE_INTEGER), ('UserName', RPC_UNICODE_STRING), ('Workstation', RPC_UNICODE_STRING))

class PNETLOGON_LOGON_IDENTITY_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_LOGON_IDENTITY_INFO),)

class NETLOGON_GENERIC_INFO(NDRSTRUCT):
    structure = (('Identity', NETLOGON_LOGON_IDENTITY_INFO), ('PackageName', RPC_UNICODE_STRING), ('DataLength', ULONG), ('LogonData', PUCHAR_ARRAY))

class PNETLOGON_GENERIC_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_GENERIC_INFO),)

class NETLOGON_INTERACTIVE_INFO(NDRSTRUCT):
    structure = (('Identity', NETLOGON_LOGON_IDENTITY_INFO), ('LmOwfPassword', LM_OWF_PASSWORD), ('NtOwfPassword', NT_OWF_PASSWORD))

class PNETLOGON_INTERACTIVE_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_INTERACTIVE_INFO),)

class NETLOGON_SERVICE_INFO(NDRSTRUCT):
    structure = (('Identity', NETLOGON_LOGON_IDENTITY_INFO), ('LmOwfPassword', LM_OWF_PASSWORD), ('NtOwfPassword', NT_OWF_PASSWORD))

class PNETLOGON_SERVICE_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_SERVICE_INFO),)

class NETLOGON_NETWORK_INFO(NDRSTRUCT):
    structure = (('Identity', NETLOGON_LOGON_IDENTITY_INFO), ('LmChallenge', LM_CHALLENGE), ('NtChallengeResponse', STRING), ('LmChallengeResponse', STRING))

class PNETLOGON_NETWORK_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_NETWORK_INFO),)

class NETLOGON_LOGON_INFO_CLASS(NDRENUM):

    class enumItems(Enum):
        NetlogonInteractiveInformation = 1
        NetlogonNetworkInformation = 2
        NetlogonServiceInformation = 3
        NetlogonGenericInformation = 4
        NetlogonInteractiveTransitiveInformation = 5
        NetlogonNetworkTransitiveInformation = 6
        NetlogonServiceTransitiveInformation = 7

class NETLOGON_LEVEL(NDRUNION):
    union = {NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation: ('LogonInteractive', PNETLOGON_INTERACTIVE_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveTransitiveInformation: ('LogonInteractiveTransitive', PNETLOGON_INTERACTIVE_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonServiceInformation: ('LogonService', PNETLOGON_SERVICE_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonServiceTransitiveInformation: ('LogonServiceTransitive', PNETLOGON_SERVICE_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonNetworkInformation: ('LogonNetwork', PNETLOGON_NETWORK_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonNetworkTransitiveInformation: ('LogonNetworkTransitive', PNETLOGON_NETWORK_INFO), NETLOGON_LOGON_INFO_CLASS.NetlogonGenericInformation: ('LogonGeneric', PNETLOGON_GENERIC_INFO)}

class NETLOGON_SID_AND_ATTRIBUTES(NDRSTRUCT):
    structure = (('Sid', PRPC_SID), ('Attributes', ULONG))

class NETLOGON_VALIDATION_GENERIC_INFO2(NDRSTRUCT):
    structure = (('DataLength', ULONG), ('ValidationData', PUCHAR_ARRAY))

class PNETLOGON_VALIDATION_GENERIC_INFO2(NDRPOINTER):
    referent = (('Data', NETLOGON_VALIDATION_GENERIC_INFO2),)
USER_SESSION_KEY = LM_OWF_PASSWORD

class GROUP_MEMBERSHIP(NDRSTRUCT):
    structure = (('RelativeId', ULONG), ('Attributes', ULONG))

class GROUP_MEMBERSHIP_ARRAY(NDRUniConformantArray):
    item = GROUP_MEMBERSHIP

class PGROUP_MEMBERSHIP_ARRAY(NDRPOINTER):
    referent = (('Data', GROUP_MEMBERSHIP_ARRAY),)

class LONG_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            print('Hello World!')
        return 4 * 10

class NETLOGON_VALIDATION_SAM_INFO(NDRSTRUCT):
    structure = (('LogonTime', OLD_LARGE_INTEGER), ('LogoffTime', OLD_LARGE_INTEGER), ('KickOffTime', OLD_LARGE_INTEGER), ('PasswordLastSet', OLD_LARGE_INTEGER), ('PasswordCanChange', OLD_LARGE_INTEGER), ('PasswordMustChange', OLD_LARGE_INTEGER), ('EffectiveName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('LogonScript', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('LogonCount', USHORT), ('BadPasswordCount', USHORT), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('GroupCount', ULONG), ('GroupIds', PGROUP_MEMBERSHIP_ARRAY), ('UserFlags', ULONG), ('UserSessionKey', USER_SESSION_KEY), ('LogonServer', RPC_UNICODE_STRING), ('LogonDomainName', RPC_UNICODE_STRING), ('LogonDomainId', PRPC_SID), ('ExpansionRoom', LONG_ARRAY))

class PNETLOGON_VALIDATION_SAM_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_VALIDATION_SAM_INFO),)

class NETLOGON_SID_AND_ATTRIBUTES_ARRAY(NDRUniConformantArray):
    item = NETLOGON_SID_AND_ATTRIBUTES

class PNETLOGON_SID_AND_ATTRIBUTES_ARRAY(NDRPOINTER):
    referent = (('Data', NETLOGON_SID_AND_ATTRIBUTES_ARRAY),)

class NETLOGON_VALIDATION_SAM_INFO2(NDRSTRUCT):
    structure = (('LogonTime', OLD_LARGE_INTEGER), ('LogoffTime', OLD_LARGE_INTEGER), ('KickOffTime', OLD_LARGE_INTEGER), ('PasswordLastSet', OLD_LARGE_INTEGER), ('PasswordCanChange', OLD_LARGE_INTEGER), ('PasswordMustChange', OLD_LARGE_INTEGER), ('EffectiveName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('LogonScript', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('LogonCount', USHORT), ('BadPasswordCount', USHORT), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('GroupCount', ULONG), ('GroupIds', PGROUP_MEMBERSHIP_ARRAY), ('UserFlags', ULONG), ('UserSessionKey', USER_SESSION_KEY), ('LogonServer', RPC_UNICODE_STRING), ('LogonDomainName', RPC_UNICODE_STRING), ('LogonDomainId', PRPC_SID), ('ExpansionRoom', LONG_ARRAY), ('SidCount', ULONG), ('ExtraSids', PNETLOGON_SID_AND_ATTRIBUTES_ARRAY))

class PNETLOGON_VALIDATION_SAM_INFO2(NDRPOINTER):
    referent = (('Data', NETLOGON_VALIDATION_SAM_INFO2),)

class NETLOGON_VALIDATION_SAM_INFO4(NDRSTRUCT):
    structure = (('LogonTime', OLD_LARGE_INTEGER), ('LogoffTime', OLD_LARGE_INTEGER), ('KickOffTime', OLD_LARGE_INTEGER), ('PasswordLastSet', OLD_LARGE_INTEGER), ('PasswordCanChange', OLD_LARGE_INTEGER), ('PasswordMustChange', OLD_LARGE_INTEGER), ('EffectiveName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('LogonScript', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('LogonCount', USHORT), ('BadPasswordCount', USHORT), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('GroupCount', ULONG), ('GroupIds', PGROUP_MEMBERSHIP_ARRAY), ('UserFlags', ULONG), ('UserSessionKey', USER_SESSION_KEY), ('LogonServer', RPC_UNICODE_STRING), ('LogonDomainName', RPC_UNICODE_STRING), ('LogonDomainId', PRPC_SID), ('LMKey', CHAR_FIXED_8_ARRAY), ('UserAccountControl', ULONG), ('SubAuthStatus', ULONG), ('LastSuccessfulILogon', OLD_LARGE_INTEGER), ('LastFailedILogon', OLD_LARGE_INTEGER), ('FailedILogonCount', ULONG), ('Reserved4', ULONG), ('SidCount', ULONG), ('ExtraSids', PNETLOGON_SID_AND_ATTRIBUTES_ARRAY), ('DnsLogonDomainName', RPC_UNICODE_STRING), ('Upn', RPC_UNICODE_STRING), ('ExpansionString1', RPC_UNICODE_STRING), ('ExpansionString2', RPC_UNICODE_STRING), ('ExpansionString3', RPC_UNICODE_STRING), ('ExpansionString4', RPC_UNICODE_STRING), ('ExpansionString5', RPC_UNICODE_STRING), ('ExpansionString6', RPC_UNICODE_STRING), ('ExpansionString7', RPC_UNICODE_STRING), ('ExpansionString8', RPC_UNICODE_STRING), ('ExpansionString9', RPC_UNICODE_STRING), ('ExpansionString10', RPC_UNICODE_STRING))

class PNETLOGON_VALIDATION_SAM_INFO4(NDRPOINTER):
    referent = (('Data', NETLOGON_VALIDATION_SAM_INFO4),)

class NETLOGON_VALIDATION_INFO_CLASS(NDRENUM):

    class enumItems(Enum):
        NetlogonValidationUasInfo = 1
        NetlogonValidationSamInfo = 2
        NetlogonValidationSamInfo2 = 3
        NetlogonValidationGenericInfo = 4
        NetlogonValidationGenericInfo2 = 5
        NetlogonValidationSamInfo4 = 6

class NETLOGON_VALIDATION(NDRUNION):
    union = {NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo: ('ValidationSam', PNETLOGON_VALIDATION_SAM_INFO), NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo2: ('ValidationSam2', PNETLOGON_VALIDATION_SAM_INFO2), NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationGenericInfo2: ('ValidationGeneric2', PNETLOGON_VALIDATION_GENERIC_INFO2), NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo4: ('ValidationSam4', PNETLOGON_VALIDATION_SAM_INFO4)}

class NLPR_QUOTA_LIMITS(NDRSTRUCT):
    structure = (('PagedPoolLimit', ULONG), ('NonPagedPoolLimit', ULONG), ('MinimumWorkingSetSize', ULONG), ('MaximumWorkingSetSize', ULONG), ('PagefileLimit', ULONG), ('Reserved', OLD_LARGE_INTEGER))

class ULONG_ARRAY(NDRUniConformantArray):
    item = ULONG

class PULONG_ARRAY(NDRPOINTER):
    referent = (('Data', ULONG_ARRAY),)

class NETLOGON_DELTA_ACCOUNTS(NDRSTRUCT):
    structure = (('PrivilegeEntries', ULONG), ('PrivilegeControl', ULONG), ('PrivilegeAttributes', PULONG_ARRAY), ('PrivilegeNames', PRPC_UNICODE_STRING_ARRAY), ('QuotaLimits', NLPR_QUOTA_LIMITS), ('SystemAccessFlags', ULONG), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_ACCOUNTS(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_ACCOUNTS),)

class NLPR_SID_INFORMATION(NDRSTRUCT):
    structure = (('SidPointer', PRPC_SID),)

class NLPR_SID_INFORMATION_ARRAY(NDRUniConformantArray):
    item = NLPR_SID_INFORMATION

class PNLPR_SID_INFORMATION_ARRAY(NDRPOINTER):
    referent = (('Data', NLPR_SID_INFORMATION_ARRAY),)

class NLPR_SID_ARRAY(NDRSTRUCT):
    referent = (('Count', ULONG), ('Sids', PNLPR_SID_INFORMATION_ARRAY))

class NETLOGON_DELTA_ALIAS_MEMBER(NDRSTRUCT):
    structure = (('Members', NLPR_SID_ARRAY), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_ALIAS_MEMBER(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_ALIAS_MEMBER),)

class NETLOGON_DELTA_DELETE_GROUP(NDRSTRUCT):
    structure = (('AccountName', LPWSTR), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_DELETE_GROUP(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_DELETE_GROUP),)

class NETLOGON_DELTA_DELETE_USER(NDRSTRUCT):
    structure = (('AccountName', LPWSTR), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_DELETE_USER(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_DELETE_USER),)

class NETLOGON_DELTA_DOMAIN(NDRSTRUCT):
    structure = (('DomainName', RPC_UNICODE_STRING), ('OemInformation', RPC_UNICODE_STRING), ('ForceLogoff', OLD_LARGE_INTEGER), ('MinPasswordLength', USHORT), ('PasswordHistoryLength', USHORT), ('MaxPasswordAge', OLD_LARGE_INTEGER), ('MinPasswordAge', OLD_LARGE_INTEGER), ('DomainModifiedCount', OLD_LARGE_INTEGER), ('DomainCreationTime', OLD_LARGE_INTEGER), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('DomainLockoutInformation', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('PasswordProperties', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_DOMAIN(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_DOMAIN),)

class NETLOGON_DELTA_GROUP(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('RelativeId', ULONG), ('Attributes', ULONG), ('AdminComment', RPC_UNICODE_STRING), ('SecurityInformation', USHORT), ('SecuritySize', ULONG), ('SecurityDescriptor', SECURITY_INFORMATION), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_GROUP(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_GROUP),)

class NETLOGON_RENAME_GROUP(NDRSTRUCT):
    structure = (('OldName', RPC_UNICODE_STRING), ('NewName', RPC_UNICODE_STRING), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_RENAME_GROUP(NDRPOINTER):
    referent = (('Data', NETLOGON_RENAME_GROUP),)
from impacket.dcerpc.v5.samr import SAMPR_LOGON_HOURS
NLPR_LOGON_HOURS = SAMPR_LOGON_HOURS

class NLPR_USER_PRIVATE_INFO(NDRSTRUCT):
    structure = (('SensitiveData', UCHAR), ('DataLength', ULONG), ('Data', PUCHAR_ARRAY))

class NETLOGON_DELTA_USER(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('ScriptPath', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING), ('WorkStations', RPC_UNICODE_STRING), ('LastLogon', OLD_LARGE_INTEGER), ('LastLogoff', OLD_LARGE_INTEGER), ('LogonHours', NLPR_LOGON_HOURS), ('BadPasswordCount', USHORT), ('LogonCount', USHORT), ('PasswordLastSet', OLD_LARGE_INTEGER), ('AccountExpires', OLD_LARGE_INTEGER), ('UserAccountControl', ULONG), ('EncryptedNtOwfPassword', PUCHAR_ARRAY), ('EncryptedLmOwfPassword', PUCHAR_ARRAY), ('NtPasswordPresent', UCHAR), ('LmPasswordPresent', UCHAR), ('PasswordExpired', UCHAR), ('UserComment', RPC_UNICODE_STRING), ('Parameters', RPC_UNICODE_STRING), ('CountryCode', USHORT), ('CodePage', USHORT), ('PrivateData', NLPR_USER_PRIVATE_INFO), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('ProfilePath', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_USER(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_USER),)

class NETLOGON_RENAME_USER(NDRSTRUCT):
    structure = (('OldName', RPC_UNICODE_STRING), ('NewName', RPC_UNICODE_STRING), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_RENAME_USER(NDRPOINTER):
    referent = (('Data', NETLOGON_RENAME_USER),)

class NETLOGON_DELTA_GROUP_MEMBER(NDRSTRUCT):
    structure = (('Members', PULONG_ARRAY), ('Attributes', PULONG_ARRAY), ('MemberCount', ULONG), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_GROUP_MEMBER(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_GROUP_MEMBER),)

class NETLOGON_DELTA_ALIAS(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('RelativeId', ULONG), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('Comment', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_ALIAS(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_ALIAS),)

class NETLOGON_RENAME_ALIAS(NDRSTRUCT):
    structure = (('OldName', RPC_UNICODE_STRING), ('NewName', RPC_UNICODE_STRING), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_RENAME_ALIAS(NDRPOINTER):
    referent = (('Data', NETLOGON_RENAME_ALIAS),)

class NETLOGON_DELTA_POLICY(NDRSTRUCT):
    structure = (('MaximumLogSize', ULONG), ('AuditRetentionPeriod', OLD_LARGE_INTEGER), ('AuditingMode', UCHAR), ('MaximumAuditEventCount', ULONG), ('EventAuditingOptions', PULONG_ARRAY), ('PrimaryDomainName', RPC_UNICODE_STRING), ('PrimaryDomainSid', PRPC_SID), ('QuotaLimits', NLPR_QUOTA_LIMITS), ('ModifiedId', OLD_LARGE_INTEGER), ('DatabaseCreationTime', OLD_LARGE_INTEGER), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_POLICY(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_POLICY),)

class NETLOGON_DELTA_TRUSTED_DOMAINS(NDRSTRUCT):
    structure = (('DomainName', RPC_UNICODE_STRING), ('NumControllerEntries', ULONG), ('ControllerNames', PRPC_UNICODE_STRING_ARRAY), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_TRUSTED_DOMAINS(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_TRUSTED_DOMAINS),)

class UCHAR_ARRAY2(NDRUniConformantVaryingArray):
    item = UCHAR

class PUCHAR_ARRAY2(NDRPOINTER):
    referent = (('Data', UCHAR_ARRAY2),)

class NLPR_CR_CIPHER_VALUE(NDRSTRUCT):
    structure = (('Length', ULONG), ('MaximumLength', ULONG), ('Buffer', PUCHAR_ARRAY2))

class NETLOGON_DELTA_SECRET(NDRSTRUCT):
    structure = (('CurrentValue', NLPR_CR_CIPHER_VALUE), ('CurrentValueSetTime', OLD_LARGE_INTEGER), ('OldValue', NLPR_CR_CIPHER_VALUE), ('OldValueSetTime', OLD_LARGE_INTEGER), ('SecurityInformation', SECURITY_INFORMATION), ('SecuritySize', ULONG), ('SecurityDescriptor', PUCHAR_ARRAY), ('DummyString1', RPC_UNICODE_STRING), ('DummyString2', RPC_UNICODE_STRING), ('DummyString3', RPC_UNICODE_STRING), ('DummyString4', RPC_UNICODE_STRING), ('DummyLong1', ULONG), ('DummyLong2', ULONG), ('DummyLong3', ULONG), ('DummyLong4', ULONG))

class PNETLOGON_DELTA_SECRET(NDRPOINTER):
    referent = (('Data', NETLOGON_DELTA_SECRET),)

class NLPR_MODIFIED_COUNT(NDRSTRUCT):
    structure = (('ModifiedCount', OLD_LARGE_INTEGER),)

class PNLPR_MODIFIED_COUNT(NDRPOINTER):
    referent = (('Data', NLPR_MODIFIED_COUNT),)

class NETLOGON_DELTA_TYPE(NDRENUM):

    class enumItems(Enum):
        AddOrChangeDomain = 1
        AddOrChangeGroup = 2
        DeleteGroup = 3
        RenameGroup = 4
        AddOrChangeUser = 5
        DeleteUser = 6
        RenameUser = 7
        ChangeGroupMembership = 8
        AddOrChangeAlias = 9
        DeleteAlias = 10
        RenameAlias = 11
        ChangeAliasMembership = 12
        AddOrChangeLsaPolicy = 13
        AddOrChangeLsaTDomain = 14
        DeleteLsaTDomain = 15
        AddOrChangeLsaAccount = 16
        DeleteLsaAccount = 17
        AddOrChangeLsaSecret = 18
        DeleteLsaSecret = 19
        DeleteGroupByName = 20
        DeleteUserByName = 21
        SerialNumberSkip = 22

class NETLOGON_DELTA_UNION(NDRUNION):
    union = {NETLOGON_DELTA_TYPE.AddOrChangeDomain: ('DeltaDomain', PNETLOGON_DELTA_DOMAIN), NETLOGON_DELTA_TYPE.AddOrChangeGroup: ('DeltaGroup', PNETLOGON_DELTA_GROUP), NETLOGON_DELTA_TYPE.RenameGroup: ('DeltaRenameGroup', PNETLOGON_DELTA_RENAME_GROUP), NETLOGON_DELTA_TYPE.AddOrChangeUser: ('DeltaUser', PNETLOGON_DELTA_USER), NETLOGON_DELTA_TYPE.RenameUser: ('DeltaRenameUser', PNETLOGON_DELTA_RENAME_USER), NETLOGON_DELTA_TYPE.ChangeGroupMembership: ('DeltaGroupMember', PNETLOGON_DELTA_GROUP_MEMBER), NETLOGON_DELTA_TYPE.AddOrChangeAlias: ('DeltaAlias', PNETLOGON_DELTA_ALIAS), NETLOGON_DELTA_TYPE.RenameAlias: ('DeltaRenameAlias', PNETLOGON_DELTA_RENAME_ALIAS), NETLOGON_DELTA_TYPE.ChangeAliasMembership: ('DeltaAliasMember', PNETLOGON_DELTA_ALIAS_MEMBER), NETLOGON_DELTA_TYPE.AddOrChangeLsaPolicy: ('DeltaPolicy', PNETLOGON_DELTA_POLICY), NETLOGON_DELTA_TYPE.AddOrChangeLsaTDomain: ('DeltaTDomains', PNETLOGON_DELTA_TRUSTED_DOMAINS), NETLOGON_DELTA_TYPE.AddOrChangeLsaAccount: ('DeltaAccounts', PNETLOGON_DELTA_ACCOUNTS), NETLOGON_DELTA_TYPE.AddOrChangeLsaSecret: ('DeltaSecret', PNETLOGON_DELTA_SECRET), NETLOGON_DELTA_TYPE.DeleteGroupByName: ('DeltaDeleteGroup', PNETLOGON_DELTA_DELETE_GROUP), NETLOGON_DELTA_TYPE.DeleteUserByName: ('DeltaDeleteUser', PNETLOGON_DELTA_DELETE_USER), NETLOGON_DELTA_TYPE.SerialNumberSkip: ('DeltaSerialNumberSkip', PNLPR_MODIFIED_COUNT)}

class NETLOGON_DELTA_ID_UNION(NDRUNION):
    union = {NETLOGON_DELTA_TYPE.AddOrChangeDomain: ('Rid', ULONG), NETLOGON_DELTA_TYPE.AddOrChangeGroup: ('Rid', ULONG), NETLOGON_DELTA_TYPE.DeleteGroup: ('Rid', ULONG), NETLOGON_DELTA_TYPE.RenameGroup: ('Rid', ULONG), NETLOGON_DELTA_TYPE.AddOrChangeUser: ('Rid', ULONG), NETLOGON_DELTA_TYPE.DeleteUser: ('Rid', ULONG), NETLOGON_DELTA_TYPE.RenameUser: ('Rid', ULONG), NETLOGON_DELTA_TYPE.ChangeGroupMembership: ('Rid', ULONG), NETLOGON_DELTA_TYPE.AddOrChangeAlias: ('Rid', ULONG), NETLOGON_DELTA_TYPE.DeleteAlias: ('Rid', ULONG), NETLOGON_DELTA_TYPE.RenameAlias: ('Rid', ULONG), NETLOGON_DELTA_TYPE.ChangeAliasMembership: ('Rid', ULONG), NETLOGON_DELTA_TYPE.DeleteGroupByName: ('Rid', ULONG), NETLOGON_DELTA_TYPE.DeleteUserByName: ('Rid', ULONG), NETLOGON_DELTA_TYPE.AddOrChangeLsaPolicy: ('Sid', PRPC_SID), NETLOGON_DELTA_TYPE.AddOrChangeLsaTDomain: ('Sid', PRPC_SID), NETLOGON_DELTA_TYPE.DeleteLsaTDomain: ('Sid', PRPC_SID), NETLOGON_DELTA_TYPE.AddOrChangeLsaAccount: ('Sid', PRPC_SID), NETLOGON_DELTA_TYPE.DeleteLsaAccount: ('Sid', PRPC_SID), NETLOGON_DELTA_TYPE.AddOrChangeLsaSecret: ('Name', LPWSTR), NETLOGON_DELTA_TYPE.DeleteLsaSecret: ('Name', LPWSTR)}

class NETLOGON_DELTA_ENUM(NDRSTRUCT):
    structure = (('DeltaType', NETLOGON_DELTA_TYPE), ('DeltaID', NETLOGON_DELTA_ID_UNION), ('DeltaUnion', NETLOGON_DELTA_UNION))

class NETLOGON_DELTA_ENUM_ARRAY_ARRAY(NDRUniConformantArray):
    item = NETLOGON_DELTA_ENUM

class PNETLOGON_DELTA_ENUM_ARRAY_ARRAY(NDRSTRUCT):
    referent = (('Data', NETLOGON_DELTA_ENUM_ARRAY_ARRAY),)

class PNETLOGON_DELTA_ENUM_ARRAY(NDRPOINTER):
    structure = (('CountReturned', DWORD), ('Deltas', PNETLOGON_DELTA_ENUM_ARRAY_ARRAY))

class SYNC_STATE(NDRENUM):

    class enumItems(Enum):
        NormalState = 0
        DomainState = 1
        GroupState = 2
        UasBuiltInGroupState = 3
        UserState = 4
        GroupMemberState = 5
        AliasState = 6
        AliasMemberState = 7
        SamDoneState = 8

class DOMAIN_NAME_BUFFER(NDRSTRUCT):
    structure = (('DomainNameByteCount', ULONG), ('DomainNames', PUCHAR_ARRAY))

class DS_DOMAIN_TRUSTSW(NDRSTRUCT):
    structure = (('NetbiosDomainName', LPWSTR), ('DnsDomainName', LPWSTR), ('Flags', ULONG), ('ParentIndex', ULONG), ('TrustType', ULONG), ('TrustAttributes', ULONG), ('DomainSid', PRPC_SID), ('DomainGuid', GUID))

class DS_DOMAIN_TRUSTSW_ARRAY(NDRUniConformantArray):
    item = DS_DOMAIN_TRUSTSW

class PDS_DOMAIN_TRUSTSW_ARRAY(NDRPOINTER):
    referent = (('Data', DS_DOMAIN_TRUSTSW_ARRAY),)

class NETLOGON_TRUSTED_DOMAIN_ARRAY(NDRSTRUCT):
    structure = (('DomainCount', DWORD), ('Domains', PDS_DOMAIN_TRUSTSW_ARRAY))

class NL_GENERIC_RPC_DATA(NDRSTRUCT):
    structure = (('UlongEntryCount', ULONG), ('UlongData', PULONG_ARRAY), ('UnicodeStringEntryCount', ULONG), ('UnicodeStringData', PRPC_UNICODE_STRING_ARRAY))

class PNL_GENERIC_RPC_DATA(NDRPOINTER):
    referent = (('Data', NL_GENERIC_RPC_DATA),)

class NETLOGON_CONTROL_DATA_INFORMATION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {5: ('TrustedDomainName', LPWSTR), 6: ('TrustedDomainName', LPWSTR), 9: ('TrustedDomainName', LPWSTR), 10: ('TrustedDomainName', LPWSTR), 65534: ('DebugFlag', DWORD), 8: ('UserName', LPWSTR)}

class NETLOGON_INFO_1(NDRSTRUCT):
    structure = (('netlog1_flags', DWORD), ('netlog1_pdc_connection_status', NET_API_STATUS))

class PNETLOGON_INFO_1(NDRPOINTER):
    referent = (('Data', NETLOGON_INFO_1),)

class NETLOGON_INFO_2(NDRSTRUCT):
    structure = (('netlog2_flags', DWORD), ('netlog2_pdc_connection_status', NET_API_STATUS), ('netlog2_trusted_dc_name', LPWSTR), ('netlog2_tc_connection_status', NET_API_STATUS))

class PNETLOGON_INFO_2(NDRPOINTER):
    referent = (('Data', NETLOGON_INFO_2),)

class NETLOGON_INFO_3(NDRSTRUCT):
    structure = (('netlog3_flags', DWORD), ('netlog3_logon_attempts', DWORD), ('netlog3_reserved1', DWORD), ('netlog3_reserved2', DWORD), ('netlog3_reserved3', DWORD), ('netlog3_reserved4', DWORD), ('netlog3_reserved5', DWORD))

class PNETLOGON_INFO_3(NDRPOINTER):
    referent = (('Data', NETLOGON_INFO_3),)

class NETLOGON_INFO_4(NDRSTRUCT):
    structure = (('netlog4_trusted_dc_name', LPWSTR), ('netlog4_trusted_domain_name', LPWSTR))

class PNETLOGON_INFO_4(NDRPOINTER):
    referent = (('Data', NETLOGON_INFO_4),)

class NETLOGON_CONTROL_QUERY_INFORMATION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('NetlogonInfo1', PNETLOGON_INFO_1), 2: ('NetlogonInfo2', PNETLOGON_INFO_2), 3: ('NetlogonInfo3', PNETLOGON_INFO_3), 4: ('NetlogonInfo4', PNETLOGON_INFO_4)}

class NETLOGON_VALIDATION_UAS_INFO(NDRSTRUCT):
    structure = (('usrlog1_eff_name', DWORD), ('usrlog1_priv', DWORD), ('usrlog1_auth_flags', DWORD), ('usrlog1_num_logons', DWORD), ('usrlog1_bad_pw_count', DWORD), ('usrlog1_last_logon', DWORD), ('usrlog1_last_logoff', DWORD), ('usrlog1_logoff_time', DWORD), ('usrlog1_kickoff_time', DWORD), ('usrlog1_password_age', DWORD), ('usrlog1_pw_can_change', DWORD), ('usrlog1_pw_must_change', DWORD), ('usrlog1_computer', LPWSTR), ('usrlog1_domain', LPWSTR), ('usrlog1_script_path', LPWSTR), ('usrlog1_reserved1', DWORD))

class PNETLOGON_VALIDATION_UAS_INFO(NDRPOINTER):
    referent = (('Data', NETLOGON_VALIDATION_UAS_INFO),)

class NETLOGON_LOGOFF_UAS_INFO(NDRSTRUCT):
    structure = (('Duration', DWORD), ('LogonCount', USHORT))

class UAS_INFO_0(NDRSTRUCT):
    structure = (('ComputerName', '16s=""'), ('TimeCreated', ULONG), ('SerialNumber', ULONG))

    def getAlignment(self):
        if False:
            i = 10
            return i + 15
        return 4

class NETLOGON_DUMMY1(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {1: ('Dummy', ULONG)}

class CHAR_FIXED_16_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            i = 10
            return i + 15
        return 16
NL_AUTH_MESSAGE_NETBIOS_DOMAIN = 1
NL_AUTH_MESSAGE_NETBIOS_HOST = 2
NL_AUTH_MESSAGE_DNS_DOMAIN = 4
NL_AUTH_MESSAGE_DNS_HOST = 8
NL_AUTH_MESSAGE_NETBIOS_HOST_UTF8 = 16
NL_AUTH_MESSAGE_REQUEST = 0
NL_AUTH_MESSAGE_RESPONSE = 1
NL_SIGNATURE_HMAC_MD5 = 119
NL_SIGNATURE_HMAC_SHA256 = 19
NL_SEAL_NOT_ENCRYPTED = 65535
NL_SEAL_RC4 = 122
NL_SEAL_AES128 = 26

class NL_AUTH_MESSAGE(Structure):
    structure = (('MessageType', '<L=0'), ('Flags', '<L=0'), ('Buffer', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            i = 10
            return i + 15
        Structure.__init__(self, data, alignment)
        if data is None:
            self['Buffer'] = b'\x00' * 4

class NL_AUTH_SIGNATURE(Structure):
    structure = (('SignatureAlgorithm', '<H=0'), ('SealAlgorithm', '<H=0'), ('Pad', '<H=0xffff'), ('Flags', '<H=0'), ('SequenceNumber', '8s=""'), ('Checksum', '8s=""'), ('_Confounder', '_-Confounder', '8'), ('Confounder', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            while True:
                i = 10
        Structure.__init__(self, data, alignment)
        if data is None:
            self['Confounder'] = ''

class NL_AUTH_SHA2_SIGNATURE(Structure):
    structure = (('SignatureAlgorithm', '<H=0'), ('SealAlgorithm', '<H=0'), ('Pad', '<H=0xffff'), ('Flags', '<H=0'), ('SequenceNumber', '8s=""'), ('Checksum', '32s=""'), ('_Confounder', '_-Confounder', '8'), ('Confounder', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            i = 10
            return i + 15
        Structure.__init__(self, data, alignment)
        if data is None:
            self['Confounder'] = ''

def ComputeNetlogonCredential(inputData, Sk):
    if False:
        while True:
            i = 10
    k1 = Sk[:7]
    k3 = crypto.transformKey(k1)
    k2 = Sk[7:14]
    k4 = crypto.transformKey(k2)
    Crypt1 = DES.new(k3, DES.MODE_ECB)
    Crypt2 = DES.new(k4, DES.MODE_ECB)
    cipherText = Crypt1.encrypt(inputData)
    return Crypt2.encrypt(cipherText)

def ComputeNetlogonCredentialAES(inputData, Sk):
    if False:
        while True:
            i = 10
    IV = b'\x00' * 16
    Crypt1 = AES.new(Sk, AES.MODE_CFB, IV)
    return Crypt1.encrypt(inputData)

def ComputeSessionKeyAES(sharedSecret, clientChallenge, serverChallenge, sharedSecretHash=None):
    if False:
        print('Hello World!')
    if sharedSecretHash is None:
        M4SS = ntlm.NTOWFv1(sharedSecret)
    else:
        M4SS = sharedSecretHash
    hm = hmac.new(key=M4SS, digestmod=hashlib.sha256)
    hm.update(clientChallenge)
    hm.update(serverChallenge)
    sessionKey = hm.digest()
    return sessionKey[:16]

def ComputeSessionKeyStrongKey(sharedSecret, clientChallenge, serverChallenge, sharedSecretHash=None):
    if False:
        while True:
            i = 10
    if sharedSecretHash is None:
        M4SS = ntlm.NTOWFv1(sharedSecret)
    else:
        M4SS = sharedSecretHash
    md5 = hashlib.new('md5')
    md5.update(b'\x00' * 4)
    md5.update(clientChallenge)
    md5.update(serverChallenge)
    finalMD5 = md5.digest()
    hm = hmac.new(M4SS, digestmod=hashlib.md5)
    hm.update(finalMD5)
    return hm.digest()

def deriveSequenceNumber(sequenceNum):
    if False:
        print('Hello World!')
    sequenceLow = sequenceNum & 4294967295
    sequenceHigh = sequenceNum >> 32 & 4294967295
    sequenceHigh |= 2147483648
    res = pack('>L', sequenceLow)
    res += pack('>L', sequenceHigh)
    return res

def ComputeNetlogonSignatureAES(authSignature, message, confounder, sessionKey):
    if False:
        while True:
            i = 10
    hm = hmac.new(key=sessionKey, digestmod=hashlib.sha256)
    hm.update(authSignature.getData()[:8])
    hm.update(confounder)
    hm.update(bytes(message))
    return hm.digest()[:8] + '\x00' * 24

def ComputeNetlogonSignatureMD5(authSignature, message, confounder, sessionKey):
    if False:
        i = 10
        return i + 15
    md5 = hashlib.new('md5')
    md5.update(b'\x00' * 4)
    md5.update(authSignature.getData()[:8])
    md5.update(confounder)
    md5.update(bytes(message))
    finalMD5 = md5.digest()
    hm = hmac.new(sessionKey, digestmod=hashlib.md5)
    hm.update(finalMD5)
    return hm.digest()[:8]

def ComputeNetlogonAuthenticator(clientStoredCredential, sessionKey):
    if False:
        while True:
            i = 10
    timestamp = int(time.time())
    authenticator = NETLOGON_AUTHENTICATOR()
    authenticator['Timestamp'] = timestamp
    credential = unpack('<I', clientStoredCredential[:4])[0] + timestamp
    if credential > 4294967295:
        credential &= 4294967295
    credential = pack('<I', credential)
    authenticator['Credential'] = ComputeNetlogonCredential(credential + clientStoredCredential[4:], sessionKey)
    return authenticator

def encryptSequenceNumberRC4(sequenceNum, checkSum, sessionKey):
    if False:
        return 10
    hm = hmac.new(sessionKey, digestmod=hashlib.md5)
    hm.update(b'\x00' * 4)
    hm2 = hmac.new(hm.digest(), digestmod=hashlib.md5)
    hm2.update(checkSum)
    encryptionKey = hm2.digest()
    cipher = ARC4.new(encryptionKey)
    return cipher.encrypt(sequenceNum)

def decryptSequenceNumberRC4(sequenceNum, checkSum, sessionKey):
    if False:
        print('Hello World!')
    return encryptSequenceNumberRC4(sequenceNum, checkSum, sessionKey)

def encryptSequenceNumberAES(sequenceNum, checkSum, sessionKey):
    if False:
        return 10
    IV = checkSum[:8] + checkSum[:8]
    Cipher = AES.new(sessionKey, AES.MODE_CFB, IV)
    return Cipher.encrypt(sequenceNum)

def decryptSequenceNumberAES(sequenceNum, checkSum, sessionKey):
    if False:
        while True:
            i = 10
    IV = checkSum[:8] + checkSum[:8]
    Cipher = AES.new(sessionKey, AES.MODE_CFB, IV)
    return Cipher.decrypt(sequenceNum)

def SIGN(data, confounder, sequenceNum, key, aes=False):
    if False:
        for i in range(10):
            print('nop')
    if aes is False:
        signature = NL_AUTH_SIGNATURE()
        signature['SignatureAlgorithm'] = NL_SIGNATURE_HMAC_MD5
        if confounder == '':
            signature['SealAlgorithm'] = NL_SEAL_NOT_ENCRYPTED
        else:
            signature['SealAlgorithm'] = NL_SEAL_RC4
        signature['Checksum'] = ComputeNetlogonSignatureMD5(signature, data, confounder, key)
        signature['SequenceNumber'] = encryptSequenceNumberRC4(deriveSequenceNumber(sequenceNum), signature['Checksum'], key)
        return signature
    else:
        signature = NL_AUTH_SIGNATURE()
        signature['SignatureAlgorithm'] = NL_SIGNATURE_HMAC_SHA256
        if confounder == '':
            signature['SealAlgorithm'] = NL_SEAL_NOT_ENCRYPTED
        else:
            signature['SealAlgorithm'] = NL_SEAL_AES128
        signature['Checksum'] = ComputeNetlogonSignatureAES(signature, data, confounder, key)
        signature['SequenceNumber'] = encryptSequenceNumberAES(deriveSequenceNumber(sequenceNum), signature['Checksum'], key)
        return signature

def SEAL(data, confounder, sequenceNum, key, aes=False):
    if False:
        for i in range(10):
            print('nop')
    signature = SIGN(data, confounder, sequenceNum, key, aes)
    sequenceNum = deriveSequenceNumber(sequenceNum)
    XorKey = bytearray(key)
    for i in range(len(XorKey)):
        XorKey[i] = XorKey[i] ^ 240
    XorKey = bytes(XorKey)
    if aes is False:
        hm = hmac.new(XorKey, digestmod=hashlib.md5)
        hm.update(b'\x00' * 4)
        hm2 = hmac.new(hm.digest(), digestmod=hashlib.md5)
        hm2.update(sequenceNum)
        encryptionKey = hm2.digest()
        cipher = ARC4.new(encryptionKey)
        cfounder = cipher.encrypt(confounder)
        cipher = ARC4.new(encryptionKey)
        encrypted = cipher.encrypt(data)
        signature['Confounder'] = cfounder
        return (encrypted, signature)
    else:
        IV = sequenceNum + sequenceNum
        cipher = AES.new(XorKey, AES.MODE_CFB, IV)
        cfounder = cipher.encrypt(confounder)
        encrypted = cipher.encrypt(data)
        signature['Confounder'] = cfounder
        return (encrypted, signature)

def UNSEAL(data, auth_data, key, aes=False):
    if False:
        return 10
    auth_data = NL_AUTH_SIGNATURE(auth_data)
    XorKey = bytearray(key)
    for i in range(len(XorKey)):
        XorKey[i] = XorKey[i] ^ 240
    XorKey = bytes(XorKey)
    if aes is False:
        sequenceNum = decryptSequenceNumberRC4(auth_data['SequenceNumber'], auth_data['Checksum'], key)
        hm = hmac.new(XorKey, digestmod=hashlib.md5)
        hm.update(b'\x00' * 4)
        hm2 = hmac.new(hm.digest(), digestmod=hashlib.md5)
        hm2.update(sequenceNum)
        encryptionKey = hm2.digest()
        cipher = ARC4.new(encryptionKey)
        cfounder = cipher.encrypt(auth_data['Confounder'])
        cipher = ARC4.new(encryptionKey)
        plain = cipher.encrypt(data)
        return (plain, cfounder)
    else:
        sequenceNum = decryptSequenceNumberAES(auth_data['SequenceNumber'], auth_data['Checksum'], key)
        IV = sequenceNum + sequenceNum
        cipher = AES.new(XorKey, AES.MODE_CFB, IV)
        cfounder = cipher.decrypt(auth_data['Confounder'])
        plain = cipher.decrypt(data)
        return (plain, cfounder)

def getSSPType1(workstation='', domain='', signingRequired=False):
    if False:
        while True:
            i = 10
    auth = NL_AUTH_MESSAGE()
    auth['Flags'] = 0
    auth['Buffer'] = b''
    auth['Flags'] |= NL_AUTH_MESSAGE_NETBIOS_DOMAIN
    if domain != '':
        auth['Buffer'] = auth['Buffer'] + b(domain) + b'\x00'
    else:
        auth['Buffer'] += b'WORKGROUP\x00'
    auth['Flags'] |= NL_AUTH_MESSAGE_NETBIOS_HOST
    if workstation != '':
        auth['Buffer'] = auth['Buffer'] + b(workstation) + b'\x00'
    else:
        auth['Buffer'] += b'MYHOST\x00'
    auth['Flags'] |= NL_AUTH_MESSAGE_NETBIOS_HOST_UTF8
    if workstation != '':
        auth['Buffer'] += pack('<B', len(workstation)) + b(workstation) + b'\x00'
    else:
        auth['Buffer'] += b'\x06MYHOST\x00'
    return auth

class DsrGetDcNameEx2(NDRCALL):
    opnum = 34
    structure = (('ComputerName', PLOGONSRV_HANDLE), ('AccountName', LPWSTR), ('AllowableAccountControlBits', ULONG), ('DomainName', LPWSTR), ('DomainGuid', PGUID), ('SiteName', LPWSTR), ('Flags', ULONG))

class DsrGetDcNameEx2Response(NDRCALL):
    structure = (('DomainControllerInfo', PDOMAIN_CONTROLLER_INFOW), ('ErrorCode', NET_API_STATUS))

class DsrGetDcNameEx(NDRCALL):
    opnum = 27
    structure = (('ComputerName', PLOGONSRV_HANDLE), ('DomainName', LPWSTR), ('DomainGuid', PGUID), ('SiteName', LPWSTR), ('Flags', ULONG))

class DsrGetDcNameExResponse(NDRCALL):
    structure = (('DomainControllerInfo', PDOMAIN_CONTROLLER_INFOW), ('ErrorCode', NET_API_STATUS))

class DsrGetDcName(NDRCALL):
    opnum = 20
    structure = (('ComputerName', PLOGONSRV_HANDLE), ('DomainName', LPWSTR), ('DomainGuid', PGUID), ('SiteGuid', PGUID), ('Flags', ULONG))

class DsrGetDcNameResponse(NDRCALL):
    structure = (('DomainControllerInfo', PDOMAIN_CONTROLLER_INFOW), ('ErrorCode', NET_API_STATUS))

class NetrGetDCName(NDRCALL):
    opnum = 11
    structure = (('ServerName', LOGONSRV_HANDLE), ('DomainName', LPWSTR))

class NetrGetDCNameResponse(NDRCALL):
    structure = (('Buffer', LPWSTR), ('ErrorCode', NET_API_STATUS))

class NetrGetAnyDCName(NDRCALL):
    opnum = 13
    structure = (('ServerName', PLOGONSRV_HANDLE), ('DomainName', LPWSTR))

class NetrGetAnyDCNameResponse(NDRCALL):
    structure = (('Buffer', LPWSTR), ('ErrorCode', NET_API_STATUS))

class DsrGetSiteName(NDRCALL):
    opnum = 28
    structure = (('ComputerName', PLOGONSRV_HANDLE),)

class DsrGetSiteNameResponse(NDRCALL):
    structure = (('SiteName', LPWSTR), ('ErrorCode', NET_API_STATUS))

class DsrGetDcSiteCoverageW(NDRCALL):
    opnum = 38
    structure = (('ServerName', PLOGONSRV_HANDLE),)

class DsrGetDcSiteCoverageWResponse(NDRCALL):
    structure = (('SiteNames', PNL_SITE_NAME_ARRAY), ('ErrorCode', NET_API_STATUS))

class DsrAddressToSiteNamesW(NDRCALL):
    opnum = 33
    structure = (('ComputerName', PLOGONSRV_HANDLE), ('EntryCount', ULONG), ('SocketAddresses', NL_SOCKET_ADDRESS_ARRAY))

class DsrAddressToSiteNamesWResponse(NDRCALL):
    structure = (('SiteNames', PNL_SITE_NAME_ARRAY), ('ErrorCode', NET_API_STATUS))

class DsrAddressToSiteNamesExW(NDRCALL):
    opnum = 37
    structure = (('ComputerName', PLOGONSRV_HANDLE), ('EntryCount', ULONG), ('SocketAddresses', NL_SOCKET_ADDRESS_ARRAY))

class DsrAddressToSiteNamesExWResponse(NDRCALL):
    structure = (('SiteNames', PNL_SITE_NAME_EX_ARRAY), ('ErrorCode', NET_API_STATUS))

class DsrDeregisterDnsHostRecords(NDRCALL):
    opnum = 41
    structure = (('ServerName', PLOGONSRV_HANDLE), ('DnsDomainName', LPWSTR), ('DomainGuid', PGUID), ('DsaGuid', PGUID), ('DnsHostName', WSTR))

class DsrDeregisterDnsHostRecordsResponse(NDRCALL):
    structure = (('ErrorCode', NET_API_STATUS),)

class DSRUpdateReadOnlyServerDnsRecords(NDRCALL):
    opnum = 48
    structure = (('ServerName', PLOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('SiteName', LPWSTR), ('DnsTtl', ULONG), ('DnsNames', NL_DNS_NAME_INFO_ARRAY))

class DSRUpdateReadOnlyServerDnsRecordsResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DnsNames', NL_DNS_NAME_INFO_ARRAY), ('ErrorCode', NTSTATUS))

class NetrServerReqChallenge(NDRCALL):
    opnum = 4
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('ComputerName', WSTR), ('ClientChallenge', NETLOGON_CREDENTIAL))

class NetrServerReqChallengeResponse(NDRCALL):
    structure = (('ServerChallenge', NETLOGON_CREDENTIAL), ('ErrorCode', NTSTATUS))

class NetrServerAuthenticate3(NDRCALL):
    opnum = 26
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('ClientCredential', NETLOGON_CREDENTIAL), ('NegotiateFlags', ULONG))

class NetrServerAuthenticate3Response(NDRCALL):
    structure = (('ServerCredential', NETLOGON_CREDENTIAL), ('NegotiateFlags', ULONG), ('AccountRid', ULONG), ('ErrorCode', NTSTATUS))

class NetrServerAuthenticate2(NDRCALL):
    opnum = 15
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('ClientCredential', NETLOGON_CREDENTIAL), ('NegotiateFlags', ULONG))

class NetrServerAuthenticate2Response(NDRCALL):
    structure = (('ServerCredential', NETLOGON_CREDENTIAL), ('NegotiateFlags', ULONG), ('ErrorCode', NTSTATUS))

class NetrServerAuthenticate(NDRCALL):
    opnum = 5
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('ClientCredential', NETLOGON_CREDENTIAL))

class NetrServerAuthenticateResponse(NDRCALL):
    structure = (('ServerCredential', NETLOGON_CREDENTIAL), ('ErrorCode', NTSTATUS))

class NetrServerPasswordSet2(NDRCALL):
    opnum = 30
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ClearNewPassword', NL_TRUST_PASSWORD_FIXED_ARRAY))

class NetrServerPasswordSet2Response(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('ErrorCode', NTSTATUS))

class NetrServerPasswordGet(NDRCALL):
    opnum = 31
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('AccountType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR))

class NetrServerPasswordGetResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('EncryptedNtOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('ErrorCode', NTSTATUS))

class NetrServerTrustPasswordsGet(NDRCALL):
    opnum = 42
    structure = (('TrustedDcName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR))

class NetrServerTrustPasswordsGetResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('EncryptedNewOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('EncryptedOldOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('ErrorCode', NTSTATUS))

class NetrLogonGetDomainInfo(NDRCALL):
    opnum = 29
    structure = (('ServerName', LOGONSRV_HANDLE), ('ComputerName', LPWSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('Level', DWORD), ('WkstaBuffer', NETLOGON_WORKSTATION_INFORMATION))

class NetrLogonGetDomainInfoResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DomBuffer', NETLOGON_DOMAIN_INFORMATION), ('ErrorCode', NTSTATUS))

class NetrLogonGetCapabilities(NDRCALL):
    opnum = 21
    structure = (('ServerName', LOGONSRV_HANDLE), ('ComputerName', LPWSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('QueryLevel', DWORD))

class NetrLogonGetCapabilitiesResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('ServerCapabilities', NETLOGON_CAPABILITIES), ('ErrorCode', NTSTATUS))

class NetrLogonSamLogonEx(NDRCALL):
    opnum = 39
    structure = (('LogonServer', LPWSTR), ('ComputerName', LPWSTR), ('LogonLevel', NETLOGON_LOGON_INFO_CLASS), ('LogonInformation', NETLOGON_LEVEL), ('ValidationLevel', NETLOGON_VALIDATION_INFO_CLASS), ('ExtraFlags', ULONG))

class NetrLogonSamLogonExResponse(NDRCALL):
    structure = (('ValidationInformation', NETLOGON_VALIDATION), ('Authoritative', UCHAR), ('ExtraFlags', ULONG), ('ErrorCode', NTSTATUS))

class NetrLogonSamLogonWithFlags(NDRCALL):
    opnum = 45
    structure = (('LogonServer', LPWSTR), ('ComputerName', LPWSTR), ('Authenticator', PNETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('LogonLevel', NETLOGON_LOGON_INFO_CLASS), ('LogonInformation', NETLOGON_LEVEL), ('ValidationLevel', NETLOGON_VALIDATION_INFO_CLASS), ('ExtraFlags', ULONG))

class NetrLogonSamLogonWithFlagsResponse(NDRCALL):
    structure = (('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('ValidationInformation', NETLOGON_VALIDATION), ('Authoritative', UCHAR), ('ExtraFlags', ULONG), ('ErrorCode', NTSTATUS))

class NetrLogonSamLogon(NDRCALL):
    opnum = 2
    structure = (('LogonServer', LPWSTR), ('ComputerName', LPWSTR), ('Authenticator', PNETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('LogonLevel', NETLOGON_LOGON_INFO_CLASS), ('LogonInformation', NETLOGON_LEVEL), ('ValidationLevel', NETLOGON_VALIDATION_INFO_CLASS))

class NetrLogonSamLogonResponse(NDRCALL):
    structure = (('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('ValidationInformation', NETLOGON_VALIDATION), ('Authoritative', UCHAR), ('ErrorCode', NTSTATUS))

class NetrLogonSamLogoff(NDRCALL):
    opnum = 3
    structure = (('LogonServer', LPWSTR), ('ComputerName', LPWSTR), ('Authenticator', PNETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('LogonLevel', NETLOGON_LOGON_INFO_CLASS), ('LogonInformation', NETLOGON_LEVEL))

class NetrLogonSamLogoffResponse(NDRCALL):
    structure = (('ReturnAuthenticator', PNETLOGON_AUTHENTICATOR), ('ErrorCode', NTSTATUS))

class NetrDatabaseDeltas(NDRCALL):
    opnum = 7
    structure = (('PrimaryName', LOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DatabaseID', DWORD), ('DomainModifiedCount', NLPR_MODIFIED_COUNT), ('PreferredMaximumLength', DWORD))

class NetrDatabaseDeltasResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DomainModifiedCount', NLPR_MODIFIED_COUNT), ('DeltaArray', PNETLOGON_DELTA_ENUM_ARRAY), ('ErrorCode', NTSTATUS))

class NetrDatabaseSync2(NDRCALL):
    opnum = 16
    structure = (('PrimaryName', LOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DatabaseID', DWORD), ('RestartState', SYNC_STATE), ('SyncContext', ULONG), ('PreferredMaximumLength', DWORD))

class NetrDatabaseSync2Response(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('SyncContext', ULONG), ('DeltaArray', PNETLOGON_DELTA_ENUM_ARRAY), ('ErrorCode', NTSTATUS))

class NetrDatabaseSync(NDRCALL):
    opnum = 8
    structure = (('PrimaryName', LOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DatabaseID', DWORD), ('SyncContext', ULONG), ('PreferredMaximumLength', DWORD))

class NetrDatabaseSyncResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('SyncContext', ULONG), ('DeltaArray', PNETLOGON_DELTA_ENUM_ARRAY), ('ErrorCode', NTSTATUS))

class NetrDatabaseRedo(NDRCALL):
    opnum = 17
    structure = (('PrimaryName', LOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('ChangeLogEntry', PUCHAR_ARRAY), ('ChangeLogEntrySize', DWORD))

class NetrDatabaseRedoResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('DeltaArray', PNETLOGON_DELTA_ENUM_ARRAY), ('ErrorCode', NTSTATUS))

class DsrEnumerateDomainTrusts(NDRCALL):
    opnum = 40
    structure = (('ServerName', PLOGONSRV_HANDLE), ('Flags', ULONG))

class DsrEnumerateDomainTrustsResponse(NDRCALL):
    structure = (('Domains', NETLOGON_TRUSTED_DOMAIN_ARRAY), ('ErrorCode', NTSTATUS))

class NetrEnumerateTrustedDomainsEx(NDRCALL):
    opnum = 36
    structure = (('ServerName', PLOGONSRV_HANDLE),)

class NetrEnumerateTrustedDomainsExResponse(NDRCALL):
    structure = (('Domains', NETLOGON_TRUSTED_DOMAIN_ARRAY), ('ErrorCode', NTSTATUS))

class NetrEnumerateTrustedDomains(NDRCALL):
    opnum = 19
    structure = (('ServerName', PLOGONSRV_HANDLE),)

class NetrEnumerateTrustedDomainsResponse(NDRCALL):
    structure = (('DomainNameBuffer', DOMAIN_NAME_BUFFER), ('ErrorCode', NTSTATUS))

class NetrGetForestTrustInformation(NDRCALL):
    opnum = 44
    structure = (('ServerName', PLOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('Flags', DWORD))

class NetrGetForestTrustInformationResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('ForestTrustInfo', PLSA_FOREST_TRUST_INFORMATION), ('ErrorCode', NTSTATUS))

class DsrGetForestTrustInformation(NDRCALL):
    opnum = 43
    structure = (('ServerName', PLOGONSRV_HANDLE), ('TrustedDomainName', LPWSTR), ('Flags', DWORD))

class DsrGetForestTrustInformationResponse(NDRCALL):
    structure = (('ForestTrustInfo', PLSA_FOREST_TRUST_INFORMATION), ('ErrorCode', NTSTATUS))

class NetrServerGetTrustInfo(NDRCALL):
    opnum = 46
    structure = (('TrustedDcName', PLOGONSRV_HANDLE), ('AccountName', WSTR), ('SecureChannelType', NETLOGON_SECURE_CHANNEL_TYPE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR))

class NetrServerGetTrustInfoResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('EncryptedNewOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('EncryptedOldOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('TrustInfo', PNL_GENERIC_RPC_DATA), ('ErrorCode', NTSTATUS))

class NetrLogonGetTrustRid(NDRCALL):
    opnum = 23
    structure = (('ServerName', PLOGONSRV_HANDLE), ('DomainName', LPWSTR))

class NetrLogonGetTrustRidResponse(NDRCALL):
    structure = (('Rid', ULONG), ('ErrorCode', NTSTATUS))

class NetrLogonComputeServerDigest(NDRCALL):
    opnum = 24
    structure = (('ServerName', PLOGONSRV_HANDLE), ('Rid', ULONG), ('Message', UCHAR_ARRAY), ('MessageSize', ULONG))

class NetrLogonComputeServerDigestResponse(NDRCALL):
    structure = (('NewMessageDigest', CHAR_FIXED_16_ARRAY), ('OldMessageDigest', CHAR_FIXED_16_ARRAY), ('ErrorCode', NTSTATUS))

class NetrLogonComputeClientDigest(NDRCALL):
    opnum = 25
    structure = (('ServerName', PLOGONSRV_HANDLE), ('DomainName', LPWSTR), ('Message', UCHAR_ARRAY), ('MessageSize', ULONG))

class NetrLogonComputeClientDigestResponse(NDRCALL):
    structure = (('NewMessageDigest', CHAR_FIXED_16_ARRAY), ('OldMessageDigest', CHAR_FIXED_16_ARRAY), ('ErrorCode', NTSTATUS))

class NetrLogonSendToSam(NDRCALL):
    opnum = 32
    structure = (('PrimaryName', PLOGONSRV_HANDLE), ('ComputerName', WSTR), ('Authenticator', NETLOGON_AUTHENTICATOR), ('OpaqueBuffer', UCHAR_ARRAY), ('OpaqueBufferSize', ULONG))

class NetrLogonSendToSamResponse(NDRCALL):
    structure = (('ReturnAuthenticator', NETLOGON_AUTHENTICATOR), ('ErrorCode', NTSTATUS))

class NetrLogonSetServiceBits(NDRCALL):
    opnum = 22
    structure = (('ServerName', PLOGONSRV_HANDLE), ('ServiceBitsOfInterest', DWORD), ('ServiceBits', DWORD))

class NetrLogonSetServiceBitsResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class NetrLogonGetTimeServiceParentDomain(NDRCALL):
    opnum = 35
    structure = (('ServerName', PLOGONSRV_HANDLE),)

class NetrLogonGetTimeServiceParentDomainResponse(NDRCALL):
    structure = (('DomainName', LPWSTR), ('PdcSameSite', LONG), ('ErrorCode', NET_API_STATUS))

class NetrLogonControl2Ex(NDRCALL):
    opnum = 18
    structure = (('ServerName', PLOGONSRV_HANDLE), ('FunctionCode', DWORD), ('QueryLevel', DWORD), ('Data', NETLOGON_CONTROL_DATA_INFORMATION))

class NetrLogonControl2ExResponse(NDRCALL):
    structure = (('Buffer', NETLOGON_CONTROL_QUERY_INFORMATION), ('ErrorCode', NET_API_STATUS))

class NetrLogonControl2(NDRCALL):
    opnum = 14
    structure = (('ServerName', PLOGONSRV_HANDLE), ('FunctionCode', DWORD), ('QueryLevel', DWORD), ('Data', NETLOGON_CONTROL_DATA_INFORMATION))

class NetrLogonControl2Response(NDRCALL):
    structure = (('Buffer', NETLOGON_CONTROL_QUERY_INFORMATION), ('ErrorCode', NET_API_STATUS))

class NetrLogonControl(NDRCALL):
    opnum = 12
    structure = (('ServerName', PLOGONSRV_HANDLE), ('FunctionCode', DWORD), ('QueryLevel', DWORD), ('Data', NETLOGON_CONTROL_DATA_INFORMATION))

class NetrLogonControlResponse(NDRCALL):
    structure = (('Buffer', NETLOGON_CONTROL_DATA_INFORMATION), ('ErrorCode', NET_API_STATUS))

class NetrLogonUasLogon(NDRCALL):
    opnum = 0
    structure = (('ServerName', PLOGONSRV_HANDLE), ('UserName', WSTR), ('Workstation', WSTR))

class NetrLogonUasLogonResponse(NDRCALL):
    structure = (('ValidationInformation', PNETLOGON_VALIDATION_UAS_INFO), ('ErrorCode', NET_API_STATUS))

class NetrLogonUasLogoff(NDRCALL):
    opnum = 1
    structure = (('ServerName', PLOGONSRV_HANDLE), ('UserName', WSTR), ('Workstation', WSTR))

class NetrLogonUasLogoffResponse(NDRCALL):
    structure = (('LogoffInformation', NETLOGON_LOGOFF_UAS_INFO), ('ErrorCode', NET_API_STATUS))
OPNUMS = {0: (NetrLogonUasLogon, NetrLogonUasLogonResponse), 1: (NetrLogonUasLogoff, NetrLogonUasLogoffResponse), 2: (NetrLogonSamLogon, NetrLogonSamLogonResponse), 3: (NetrLogonSamLogoff, NetrLogonSamLogoffResponse), 4: (NetrServerReqChallenge, NetrServerReqChallengeResponse), 5: (NetrServerAuthenticate, NetrServerAuthenticateResponse), 7: (NetrDatabaseDeltas, NetrDatabaseDeltasResponse), 8: (NetrDatabaseSync, NetrDatabaseSyncResponse), 11: (NetrGetDCName, NetrGetDCNameResponse), 12: (NetrLogonControl, NetrLogonControlResponse), 13: (NetrGetAnyDCName, NetrGetAnyDCNameResponse), 14: (NetrLogonControl2, NetrLogonControl2Response), 15: (NetrServerAuthenticate2, NetrServerAuthenticate2Response), 16: (NetrDatabaseSync2, NetrDatabaseSync2Response), 17: (NetrDatabaseRedo, NetrDatabaseRedoResponse), 18: (NetrLogonControl2Ex, NetrLogonControl2ExResponse), 19: (NetrEnumerateTrustedDomains, NetrEnumerateTrustedDomainsResponse), 20: (DsrGetDcName, DsrGetDcNameResponse), 21: (NetrLogonGetCapabilities, NetrLogonGetCapabilitiesResponse), 22: (NetrLogonSetServiceBits, NetrLogonSetServiceBitsResponse), 23: (NetrLogonGetTrustRid, NetrLogonGetTrustRidResponse), 24: (NetrLogonComputeServerDigest, NetrLogonComputeServerDigestResponse), 25: (NetrLogonComputeClientDigest, NetrLogonComputeClientDigestResponse), 26: (NetrServerAuthenticate3, NetrServerAuthenticate3Response), 27: (DsrGetDcNameEx, DsrGetDcNameExResponse), 28: (DsrGetSiteName, DsrGetSiteNameResponse), 29: (NetrLogonGetDomainInfo, NetrLogonGetDomainInfoResponse), 30: (NetrServerPasswordSet2, NetrServerPasswordSet2Response), 31: (NetrServerPasswordGet, NetrServerPasswordGetResponse), 32: (NetrLogonSendToSam, NetrLogonSendToSamResponse), 33: (DsrAddressToSiteNamesW, DsrAddressToSiteNamesWResponse), 34: (DsrGetDcNameEx2, DsrGetDcNameEx2Response), 35: (NetrLogonGetTimeServiceParentDomain, NetrLogonGetTimeServiceParentDomainResponse), 36: (NetrEnumerateTrustedDomainsEx, NetrEnumerateTrustedDomainsExResponse), 37: (DsrAddressToSiteNamesExW, DsrAddressToSiteNamesExWResponse), 38: (DsrGetDcSiteCoverageW, DsrGetDcSiteCoverageWResponse), 39: (NetrLogonSamLogonEx, NetrLogonSamLogonExResponse), 40: (DsrEnumerateDomainTrusts, DsrEnumerateDomainTrustsResponse), 41: (DsrDeregisterDnsHostRecords, DsrDeregisterDnsHostRecordsResponse), 42: (NetrServerTrustPasswordsGet, NetrServerTrustPasswordsGetResponse), 43: (DsrGetForestTrustInformation, DsrGetForestTrustInformationResponse), 44: (NetrGetForestTrustInformation, NetrGetForestTrustInformationResponse), 45: (NetrLogonSamLogonWithFlags, NetrLogonSamLogonWithFlagsResponse), 46: (NetrServerGetTrustInfo, NetrServerGetTrustInfoResponse)}

def checkNullString(string):
    if False:
        for i in range(10):
            print('nop')
    if string == NULL:
        return string
    if string[-1:] != '\x00':
        return string + '\x00'
    else:
        return string

def hNetrServerReqChallenge(dce, primaryName, computerName, clientChallenge):
    if False:
        return 10
    request = NetrServerReqChallenge()
    request['PrimaryName'] = checkNullString(primaryName)
    request['ComputerName'] = checkNullString(computerName)
    request['ClientChallenge'] = clientChallenge
    return dce.request(request)

def hNetrServerAuthenticate3(dce, primaryName, accountName, secureChannelType, computerName, clientCredential, negotiateFlags):
    if False:
        print('Hello World!')
    request = NetrServerAuthenticate3()
    request['PrimaryName'] = checkNullString(primaryName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ClientCredential'] = clientCredential
    request['ComputerName'] = checkNullString(computerName)
    request['NegotiateFlags'] = negotiateFlags
    return dce.request(request)

def hDsrGetDcNameEx2(dce, computerName, accountName, allowableAccountControlBits, domainName, domainGuid, siteName, flags):
    if False:
        i = 10
        return i + 15
    request = DsrGetDcNameEx2()
    request['ComputerName'] = checkNullString(computerName)
    request['AccountName'] = checkNullString(accountName)
    request['AllowableAccountControlBits'] = allowableAccountControlBits
    request['DomainName'] = checkNullString(domainName)
    request['DomainGuid'] = domainGuid
    request['SiteName'] = checkNullString(siteName)
    request['Flags'] = flags
    return dce.request(request)

def hDsrGetDcNameEx(dce, computerName, domainName, domainGuid, siteName, flags):
    if False:
        for i in range(10):
            print('nop')
    request = DsrGetDcNameEx()
    request['ComputerName'] = checkNullString(computerName)
    request['DomainName'] = checkNullString(domainName)
    request['DomainGuid'] = domainGuid
    request['SiteName'] = siteName
    request['Flags'] = flags
    return dce.request(request)

def hDsrGetDcName(dce, computerName, domainName, domainGuid, siteGuid, flags):
    if False:
        return 10
    request = DsrGetDcName()
    request['ComputerName'] = checkNullString(computerName)
    request['DomainName'] = checkNullString(domainName)
    request['DomainGuid'] = domainGuid
    request['SiteGuid'] = siteGuid
    request['Flags'] = flags
    return dce.request(request)

def hNetrGetAnyDCName(dce, serverName, domainName):
    if False:
        for i in range(10):
            print('nop')
    request = NetrGetAnyDCName()
    request['ServerName'] = checkNullString(serverName)
    request['DomainName'] = checkNullString(domainName)
    return dce.request(request)

def hNetrGetDCName(dce, serverName, domainName):
    if False:
        print('Hello World!')
    request = NetrGetDCName()
    request['ServerName'] = checkNullString(serverName)
    request['DomainName'] = checkNullString(domainName)
    return dce.request(request)

def hDsrGetSiteName(dce, computerName):
    if False:
        i = 10
        return i + 15
    request = DsrGetSiteName()
    request['ComputerName'] = checkNullString(computerName)
    return dce.request(request)

def hDsrGetDcSiteCoverageW(dce, serverName):
    if False:
        while True:
            i = 10
    request = DsrGetDcSiteCoverageW()
    request['ServerName'] = checkNullString(serverName)
    return dce.request(request)

def hNetrServerAuthenticate2(dce, primaryName, accountName, secureChannelType, computerName, clientCredential, negotiateFlags):
    if False:
        for i in range(10):
            print('nop')
    request = NetrServerAuthenticate2()
    request['PrimaryName'] = checkNullString(primaryName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ClientCredential'] = clientCredential
    request['ComputerName'] = checkNullString(computerName)
    request['NegotiateFlags'] = negotiateFlags
    return dce.request(request)

def hNetrServerAuthenticate(dce, primaryName, accountName, secureChannelType, computerName, clientCredential):
    if False:
        return 10
    request = NetrServerAuthenticate()
    request['PrimaryName'] = checkNullString(primaryName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ClientCredential'] = clientCredential
    request['ComputerName'] = checkNullString(computerName)
    return dce.request(request)

def hNetrServerPasswordGet(dce, primaryName, accountName, accountType, computerName, authenticator):
    if False:
        return 10
    request = NetrServerPasswordGet()
    request['PrimaryName'] = checkNullString(primaryName)
    request['AccountName'] = checkNullString(accountName)
    request['AccountType'] = accountType
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    return dce.request(request)

def hNetrServerTrustPasswordsGet(dce, trustedDcName, accountName, secureChannelType, computerName, authenticator):
    if False:
        i = 10
        return i + 15
    request = NetrServerTrustPasswordsGet()
    request['TrustedDcName'] = checkNullString(trustedDcName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    return dce.request(request)

def hNetrServerPasswordSet2(dce, primaryName, accountName, secureChannelType, computerName, authenticator, clearNewPasswordBlob):
    if False:
        print('Hello World!')
    request = NetrServerPasswordSet2()
    request['PrimaryName'] = checkNullString(primaryName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    request['ClearNewPassword'] = clearNewPasswordBlob
    return dce.request(request)

def hNetrLogonGetDomainInfo(dce, serverName, computerName, authenticator, returnAuthenticator=0, level=1):
    if False:
        while True:
            i = 10
    request = NetrLogonGetDomainInfo()
    request['ServerName'] = checkNullString(serverName)
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    if returnAuthenticator == 0:
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
    else:
        request['ReturnAuthenticator'] = returnAuthenticator
    request['Level'] = 1
    if level == 1:
        request['WkstaBuffer']['tag'] = 1
        request['WkstaBuffer']['WorkstationInfo']['DnsHostName'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['SiteName'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['OsName'] = ''
        request['WkstaBuffer']['WorkstationInfo']['Dummy1'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy2'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy3'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy4'] = NULL
    else:
        request['WkstaBuffer']['tag'] = 2
        request['WkstaBuffer']['LsaPolicyInfo']['LsaPolicy'] = NULL
    return dce.request(request)

def hNetrLogonGetCapabilities(dce, serverName, computerName, authenticator, returnAuthenticator=0, queryLevel=1):
    if False:
        i = 10
        return i + 15
    request = NetrLogonGetCapabilities()
    request['ServerName'] = checkNullString(serverName)
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    if returnAuthenticator == 0:
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
    else:
        request['ReturnAuthenticator'] = returnAuthenticator
    request['QueryLevel'] = queryLevel
    return dce.request(request)

def hNetrServerGetTrustInfo(dce, trustedDcName, accountName, secureChannelType, computerName, authenticator):
    if False:
        print('Hello World!')
    request = NetrServerGetTrustInfo()
    request['TrustedDcName'] = checkNullString(trustedDcName)
    request['AccountName'] = checkNullString(accountName)
    request['SecureChannelType'] = secureChannelType
    request['ComputerName'] = checkNullString(computerName)
    request['Authenticator'] = authenticator
    return dce.request(request)