from __future__ import division
from __future__ import print_function
from impacket.dcerpc.v5.ndr import NDRCALL, NDRENUM, NDRUNION, NDRUniConformantVaryingArray, NDRPOINTER, NDR, NDRSTRUCT, NDRUniConformantArray
from impacket.dcerpc.v5.dtypes import DWORD, LPWSTR, STR, LUID, LONG, ULONG, RPC_UNICODE_STRING, PRPC_SID, LPBYTE, LARGE_INTEGER, NTSTATUS, RPC_SID, ACCESS_MASK, UCHAR, PRPC_UNICODE_STRING, PLARGE_INTEGER, USHORT, SECURITY_INFORMATION, NULL, MAXIMUM_ALLOWED, GUID, SECURITY_DESCRIPTOR, OWNER_SECURITY_INFORMATION
from impacket import nt_errors
from impacket.uuid import uuidtup_to_bin
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.rpcrt import DCERPCException
MSRPC_UUID_LSAD = uuidtup_to_bin(('12345778-1234-ABCD-EF00-0123456789AB', '0.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            while True:
                i = 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        key = self.error_code
        if key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'LSAD SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'LSAD SessionError: unknown error code: 0x%x' % self.error_code
POLICY_VIEW_LOCAL_INFORMATION = 1
POLICY_VIEW_AUDIT_INFORMATION = 2
POLICY_GET_PRIVATE_INFORMATION = 4
POLICY_TRUST_ADMIN = 8
POLICY_CREATE_ACCOUNT = 16
POLICY_CREATE_SECRET = 32
POLICY_CREATE_PRIVILEGE = 64
POLICY_SET_DEFAULT_QUOTA_LIMITS = 128
POLICY_SET_AUDIT_REQUIREMENTS = 256
POLICY_AUDIT_LOG_ADMIN = 512
POLICY_SERVER_ADMIN = 1024
POLICY_LOOKUP_NAMES = 2048
POLICY_NOTIFICATION = 4096
ACCOUNT_VIEW = 1
ACCOUNT_ADJUST_PRIVILEGES = 2
ACCOUNT_ADJUST_QUOTAS = 4
ACCOUNT_ADJUST_SYSTEM_ACCESS = 8
SECRET_SET_VALUE = 1
SECRET_QUERY_VALUE = 2
TRUSTED_QUERY_DOMAIN_NAME = 1
TRUSTED_QUERY_CONTROLLERS = 2
TRUSTED_SET_CONTROLLERS = 4
TRUSTED_QUERY_POSIX = 8
TRUSTED_SET_POSIX = 16
TRUSTED_SET_AUTH = 32
TRUSTED_QUERY_AUTH = 64
POLICY_MODE_INTERACTIVE = 1
POLICY_MODE_NETWORK = 2
POLICY_MODE_BATCH = 4
POLICY_MODE_SERVICE = 16
POLICY_MODE_DENY_INTERACTIVE = 64
POLICY_MODE_DENY_NETWORK = 128
POLICY_MODE_DENY_BATCH = 256
POLICY_MODE_DENY_SERVICE = 512
POLICY_MODE_REMOTE_INTERACTIVE = 1024
POLICY_MODE_DENY_REMOTE_INTERACTIVE = 2048
POLICY_MODE_ALL = 4087
POLICY_MODE_ALL_NT4 = 55
POLICY_AUDIT_EVENT_UNCHANGED = 0
POLICY_AUDIT_EVENT_NONE = 4
POLICY_AUDIT_EVENT_SUCCESS = 1
POLICY_AUDIT_EVENT_FAILURE = 2
POLICY_KERBEROS_VALIDATE_CLIENT = 128
LSA_TLN_DISABLED_NEW = 1
LSA_TLN_DISABLED_ADMIN = 2
LSA_TLN_DISABLED_CONFLICT = 4
LSA_SID_DISABLED_ADMIN = 1
LSA_SID_DISABLED_CONFLICT = 2
LSA_NB_DISABLED_ADMIN = 4
LSA_NB_DISABLED_CONFLICT = 8
LSA_FTRECORD_DISABLED_REASONS = 65535

class LSAPR_HANDLE(NDRSTRUCT):
    align = 1
    structure = (('Data', '20s=""'),)
LSA_UNICODE_STRING = RPC_UNICODE_STRING

class STRING(NDRSTRUCT):
    commonHdr = (('MaximumLength', '<H=len(Data)-12'), ('Length', '<H=len(Data)-12'), ('ReferentID', '<L=0xff'))
    commonHdr64 = (('MaximumLength', '<H=len(Data)-24'), ('Length', '<H=len(Data)-24'), ('ReferentID', '<Q=0xff'))
    referent = (('Data', STR),)

    def dump(self, msg=None, indent=0):
        if False:
            while True:
                i = 10
        if msg is None:
            msg = self.__class__.__name__
        if msg != '':
            print('%s' % msg, end=' ')
        print(' %r' % self['Data'], end=' ')

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if key == 'Data':
            self.fields['MaximumLength'] = None
            self.fields['Length'] = None
            self.data = None
        return NDR.__setitem__(self, key, value)

class LSAPR_ACL(NDRSTRUCT):
    structure = (('AclRevision', UCHAR), ('Sbz1', UCHAR), ('AclSize', USHORT), ('Dummy1', NDRUniConformantArray))
LSAPR_SECURITY_DESCRIPTOR = SECURITY_DESCRIPTOR

class PLSAPR_SECURITY_DESCRIPTOR(NDRPOINTER):
    referent = (('Data', LSAPR_SECURITY_DESCRIPTOR),)

class SECURITY_IMPERSONATION_LEVEL(NDRENUM):

    class enumItems(Enum):
        SecurityAnonymous = 0
        SecurityIdentification = 1
        SecurityImpersonation = 2
        SecurityDelegation = 3
SECURITY_CONTEXT_TRACKING_MODE = UCHAR

class SECURITY_QUALITY_OF_SERVICE(NDRSTRUCT):
    structure = (('Length', DWORD), ('ImpersonationLevel', SECURITY_IMPERSONATION_LEVEL), ('ContextTrackingMode', SECURITY_CONTEXT_TRACKING_MODE), ('EffectiveOnly', UCHAR))

class PSECURITY_QUALITY_OF_SERVICE(NDRPOINTER):
    referent = (('Data', SECURITY_QUALITY_OF_SERVICE),)

class LSAPR_OBJECT_ATTRIBUTES(NDRSTRUCT):
    structure = (('Length', DWORD), ('RootDirectory', LPWSTR), ('ObjectName', LPWSTR), ('Attributes', DWORD), ('SecurityDescriptor', PLSAPR_SECURITY_DESCRIPTOR), ('SecurityQualityOfService', PSECURITY_QUALITY_OF_SERVICE))

class LSAPR_SR_SECURITY_DESCRIPTOR(NDRSTRUCT):
    structure = (('Length', DWORD), ('SecurityDescriptor', LPBYTE))

class PLSAPR_SR_SECURITY_DESCRIPTOR(NDRPOINTER):
    referent = (('Data', LSAPR_SR_SECURITY_DESCRIPTOR),)
SECURITY_DESCRIPTOR_CONTROL = ULONG

class POLICY_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        PolicyAuditLogInformation = 1
        PolicyAuditEventsInformation = 2
        PolicyPrimaryDomainInformation = 3
        PolicyPdAccountInformation = 4
        PolicyAccountDomainInformation = 5
        PolicyLsaServerRoleInformation = 6
        PolicyReplicaSourceInformation = 7
        PolicyInformationNotUsedOnWire = 8
        PolicyModificationInformation = 9
        PolicyAuditFullSetInformation = 10
        PolicyAuditFullQueryInformation = 11
        PolicyDnsDomainInformation = 12
        PolicyDnsDomainInformationInt = 13
        PolicyLocalAccountDomainInformation = 14
        PolicyLastEntry = 15

class POLICY_AUDIT_LOG_INFO(NDRSTRUCT):
    structure = (('AuditLogPercentFull', DWORD), ('MaximumLogSize', DWORD), ('AuditRetentionPeriod', LARGE_INTEGER), ('AuditLogFullShutdownInProgress', UCHAR), ('TimeToShutdown', LARGE_INTEGER), ('NextAuditRecordId', DWORD))

class DWORD_ARRAY(NDRUniConformantArray):
    item = DWORD

class PDWORD_ARRAY(NDRPOINTER):
    referent = (('Data', DWORD_ARRAY),)

class LSAPR_POLICY_AUDIT_EVENTS_INFO(NDRSTRUCT):
    structure = (('AuditingMode', UCHAR), ('EventAuditingOptions', PDWORD_ARRAY), ('MaximumAuditEventCount', DWORD))

class LSAPR_POLICY_PRIMARY_DOM_INFO(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('Sid', PRPC_SID))

class LSAPR_POLICY_ACCOUNT_DOM_INFO(NDRSTRUCT):
    structure = (('DomainName', RPC_UNICODE_STRING), ('DomainSid', PRPC_SID))

class LSAPR_POLICY_PD_ACCOUNT_INFO(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING),)

class POLICY_LSA_SERVER_ROLE(NDRENUM):

    class enumItems(Enum):
        PolicyServerRoleBackup = 2
        PolicyServerRolePrimary = 3

class POLICY_LSA_SERVER_ROLE_INFO(NDRSTRUCT):
    structure = (('LsaServerRole', POLICY_LSA_SERVER_ROLE),)

class LSAPR_POLICY_REPLICA_SRCE_INFO(NDRSTRUCT):
    structure = (('ReplicaSource', RPC_UNICODE_STRING), ('ReplicaAccountName', RPC_UNICODE_STRING))

class POLICY_MODIFICATION_INFO(NDRSTRUCT):
    structure = (('ModifiedId', LARGE_INTEGER), ('DatabaseCreationTime', LARGE_INTEGER))

class POLICY_AUDIT_FULL_SET_INFO(NDRSTRUCT):
    structure = (('ShutDownOnFull', UCHAR),)

class POLICY_AUDIT_FULL_QUERY_INFO(NDRSTRUCT):
    structure = (('ShutDownOnFull', UCHAR), ('LogIsFull', UCHAR))

class LSAPR_POLICY_DNS_DOMAIN_INFO(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('DnsDomainName', RPC_UNICODE_STRING), ('DnsForestName', RPC_UNICODE_STRING), ('DomainGuid', GUID), ('Sid', PRPC_SID))

class LSAPR_POLICY_INFORMATION(NDRUNION):
    union = {POLICY_INFORMATION_CLASS.PolicyAuditLogInformation: ('PolicyAuditLogInfo', POLICY_AUDIT_LOG_INFO), POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation: ('PolicyAuditEventsInfo', LSAPR_POLICY_AUDIT_EVENTS_INFO), POLICY_INFORMATION_CLASS.PolicyPrimaryDomainInformation: ('PolicyPrimaryDomainInfo', LSAPR_POLICY_PRIMARY_DOM_INFO), POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation: ('PolicyAccountDomainInfo', LSAPR_POLICY_ACCOUNT_DOM_INFO), POLICY_INFORMATION_CLASS.PolicyPdAccountInformation: ('PolicyPdAccountInfo', LSAPR_POLICY_PD_ACCOUNT_INFO), POLICY_INFORMATION_CLASS.PolicyLsaServerRoleInformation: ('PolicyServerRoleInfo', POLICY_LSA_SERVER_ROLE_INFO), POLICY_INFORMATION_CLASS.PolicyReplicaSourceInformation: ('PolicyReplicaSourceInfo', LSAPR_POLICY_REPLICA_SRCE_INFO), POLICY_INFORMATION_CLASS.PolicyModificationInformation: ('PolicyModificationInfo', POLICY_MODIFICATION_INFO), POLICY_INFORMATION_CLASS.PolicyAuditFullSetInformation: ('PolicyAuditFullSetInfo', POLICY_AUDIT_FULL_SET_INFO), POLICY_INFORMATION_CLASS.PolicyAuditFullQueryInformation: ('PolicyAuditFullQueryInfo', POLICY_AUDIT_FULL_QUERY_INFO), POLICY_INFORMATION_CLASS.PolicyDnsDomainInformation: ('PolicyDnsDomainInfo', LSAPR_POLICY_DNS_DOMAIN_INFO), POLICY_INFORMATION_CLASS.PolicyDnsDomainInformationInt: ('PolicyDnsDomainInfoInt', LSAPR_POLICY_DNS_DOMAIN_INFO), POLICY_INFORMATION_CLASS.PolicyLocalAccountDomainInformation: ('PolicyLocalAccountDomainInfo', LSAPR_POLICY_ACCOUNT_DOM_INFO)}

class PLSAPR_POLICY_INFORMATION(NDRPOINTER):
    referent = (('Data', LSAPR_POLICY_INFORMATION),)

class POLICY_DOMAIN_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        PolicyDomainQualityOfServiceInformation = 1
        PolicyDomainEfsInformation = 2
        PolicyDomainKerberosTicketInformation = 3

class POLICY_DOMAIN_QUALITY_OF_SERVICE_INFO(NDRSTRUCT):
    structure = (('QualityOfService', DWORD),)

class LSAPR_POLICY_DOMAIN_EFS_INFO(NDRSTRUCT):
    structure = (('InfoLength', DWORD), ('EfsBlob', LPBYTE))

class POLICY_DOMAIN_KERBEROS_TICKET_INFO(NDRSTRUCT):
    structure = (('AuthenticationOptions', DWORD), ('MaxServiceTicketAge', LARGE_INTEGER), ('MaxTicketAge', LARGE_INTEGER), ('MaxRenewAge', LARGE_INTEGER), ('MaxClockSkew', LARGE_INTEGER), ('Reserved', LARGE_INTEGER))

class LSAPR_POLICY_DOMAIN_INFORMATION(NDRUNION):
    union = {POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainQualityOfServiceInformation: ('PolicyDomainQualityOfServiceInfo', POLICY_DOMAIN_QUALITY_OF_SERVICE_INFO), POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainEfsInformation: ('PolicyDomainEfsInfo', LSAPR_POLICY_DOMAIN_EFS_INFO), POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainKerberosTicketInformation: ('PolicyDomainKerbTicketInfo', POLICY_DOMAIN_KERBEROS_TICKET_INFO)}

class PLSAPR_POLICY_DOMAIN_INFORMATION(NDRPOINTER):
    referent = (('Data', LSAPR_POLICY_DOMAIN_INFORMATION),)

class POLICY_AUDIT_EVENT_TYPE(NDRENUM):

    class enumItems(Enum):
        AuditCategorySystem = 0
        AuditCategoryLogon = 1
        AuditCategoryObjectAccess = 2
        AuditCategoryPrivilegeUse = 3
        AuditCategoryDetailedTracking = 4
        AuditCategoryPolicyChange = 5
        AuditCategoryAccountManagement = 6
        AuditCategoryDirectoryServiceAccess = 7
        AuditCategoryAccountLogon = 8

class LSAPR_ACCOUNT_INFORMATION(NDRSTRUCT):
    structure = (('Sid', PRPC_SID),)

class LSAPR_ACCOUNT_INFORMATION_ARRAY(NDRUniConformantArray):
    item = LSAPR_ACCOUNT_INFORMATION

class PLSAPR_ACCOUNT_INFORMATION_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_ACCOUNT_INFORMATION_ARRAY),)

class LSAPR_ACCOUNT_ENUM_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Information', PLSAPR_ACCOUNT_INFORMATION_ARRAY))

class RPC_UNICODE_STRING_ARRAY(NDRUniConformantArray):
    item = RPC_UNICODE_STRING

class PRPC_UNICODE_STRING_ARRAY(NDRPOINTER):
    referent = (('Data', RPC_UNICODE_STRING_ARRAY),)

class LSAPR_USER_RIGHT_SET(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('UserRights', PRPC_UNICODE_STRING_ARRAY))

class LSAPR_LUID_AND_ATTRIBUTES(NDRSTRUCT):
    structure = (('Luid', LUID), ('Attributes', ULONG))

class LSAPR_LUID_AND_ATTRIBUTES_ARRAY(NDRUniConformantArray):
    item = LSAPR_LUID_AND_ATTRIBUTES

class LSAPR_PRIVILEGE_SET(NDRSTRUCT):
    structure = (('PrivilegeCount', ULONG), ('Control', ULONG), ('Privilege', LSAPR_LUID_AND_ATTRIBUTES_ARRAY))

class PLSAPR_PRIVILEGE_SET(NDRPOINTER):
    referent = (('Data', LSAPR_PRIVILEGE_SET),)

class PCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', NDRUniConformantVaryingArray),)

class LSAPR_CR_CIPHER_VALUE(NDRSTRUCT):
    structure = (('Length', LONG), ('MaximumLength', LONG), ('Buffer', PCHAR_ARRAY))

class PLSAPR_CR_CIPHER_VALUE(NDRPOINTER):
    referent = (('Data', LSAPR_CR_CIPHER_VALUE),)

class PPLSAPR_CR_CIPHER_VALUE(NDRPOINTER):
    referent = (('Data', PLSAPR_CR_CIPHER_VALUE),)

class LSAPR_TRUST_INFORMATION(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('Sid', PRPC_SID))

class TRUSTED_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        TrustedDomainNameInformation = 1
        TrustedControllersInformation = 2
        TrustedPosixOffsetInformation = 3
        TrustedPasswordInformation = 4
        TrustedDomainInformationBasic = 5
        TrustedDomainInformationEx = 6
        TrustedDomainAuthInformation = 7
        TrustedDomainFullInformation = 8
        TrustedDomainAuthInformationInternal = 9
        TrustedDomainFullInformationInternal = 10
        TrustedDomainInformationEx2Internal = 11
        TrustedDomainFullInformation2Internal = 12
        TrustedDomainSupportedEncryptionTypes = 13

class LSAPR_TRUSTED_DOMAIN_NAME_INFO(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING),)

class LSAPR_TRUSTED_CONTROLLERS_INFO(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Names', PRPC_UNICODE_STRING_ARRAY))

class TRUSTED_POSIX_OFFSET_INFO(NDRSTRUCT):
    structure = (('Offset', ULONG),)

class LSAPR_TRUSTED_PASSWORD_INFO(NDRSTRUCT):
    structure = (('Password', PLSAPR_CR_CIPHER_VALUE), ('OldPassword', PLSAPR_CR_CIPHER_VALUE))
LSAPR_TRUSTED_DOMAIN_INFORMATION_BASIC = LSAPR_TRUST_INFORMATION

class LSAPR_TRUSTED_DOMAIN_INFORMATION_EX(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('FlatName', RPC_UNICODE_STRING), ('Sid', PRPC_SID), ('TrustDirection', ULONG), ('TrustType', ULONG), ('TrustAttributes', ULONG))

class LSAPR_TRUSTED_DOMAIN_INFORMATION_EX2(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('FlatName', RPC_UNICODE_STRING), ('Sid', PRPC_SID), ('TrustDirection', ULONG), ('TrustType', ULONG), ('TrustAttributes', ULONG), ('ForestTrustLength', ULONG), ('ForestTrustInfo', LPBYTE))

class LSAPR_AUTH_INFORMATION(NDRSTRUCT):
    structure = (('LastUpdateTime', LARGE_INTEGER), ('AuthType', ULONG), ('AuthInfoLength', ULONG), ('AuthInfo', LPBYTE))

class PLSAPR_AUTH_INFORMATION(NDRPOINTER):
    referent = (('Data', LSAPR_AUTH_INFORMATION),)

class LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION(NDRSTRUCT):
    structure = (('IncomingAuthInfos', ULONG), ('IncomingAuthenticationInformation', PLSAPR_AUTH_INFORMATION), ('IncomingPreviousAuthenticationInformation', PLSAPR_AUTH_INFORMATION), ('OutgoingAuthInfos', ULONG), ('OutgoingAuthenticationInformation', PLSAPR_AUTH_INFORMATION), ('OutgoingPreviousAuthenticationInformation', PLSAPR_AUTH_INFORMATION))

class LSAPR_TRUSTED_DOMAIN_AUTH_BLOB(NDRSTRUCT):
    structure = (('AuthSize', ULONG), ('AuthBlob', LPBYTE))

class LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION_INTERNAL(NDRSTRUCT):
    structure = (('AuthBlob', LSAPR_TRUSTED_DOMAIN_AUTH_BLOB),)

class LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION(NDRSTRUCT):
    structure = (('Information', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX), ('PosixOffset', TRUSTED_POSIX_OFFSET_INFO), ('AuthInformation', LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION))

class LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION_INTERNAL(NDRSTRUCT):
    structure = (('Information', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX), ('PosixOffset', TRUSTED_POSIX_OFFSET_INFO), ('AuthInformation', LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION_INTERNAL))

class LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION2(NDRSTRUCT):
    structure = (('Information', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX), ('PosixOffset', TRUSTED_POSIX_OFFSET_INFO), ('AuthInformation', LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION))

class TRUSTED_DOMAIN_SUPPORTED_ENCRYPTION_TYPES(NDRSTRUCT):
    structure = (('SupportedEncryptionTypes', ULONG),)

class LSAPR_TRUSTED_DOMAIN_INFO(NDRUNION):
    union = {TRUSTED_INFORMATION_CLASS.TrustedDomainNameInformation: ('TrustedDomainNameInfo', LSAPR_TRUSTED_DOMAIN_NAME_INFO), TRUSTED_INFORMATION_CLASS.TrustedControllersInformation: ('TrustedControllersInfo', LSAPR_TRUSTED_CONTROLLERS_INFO), TRUSTED_INFORMATION_CLASS.TrustedPosixOffsetInformation: ('TrustedPosixOffsetInfo', TRUSTED_POSIX_OFFSET_INFO), TRUSTED_INFORMATION_CLASS.TrustedPasswordInformation: ('TrustedPasswordInfo', LSAPR_TRUSTED_PASSWORD_INFO), TRUSTED_INFORMATION_CLASS.TrustedDomainInformationBasic: ('TrustedDomainInfoBasic', LSAPR_TRUSTED_DOMAIN_INFORMATION_BASIC), TRUSTED_INFORMATION_CLASS.TrustedDomainInformationEx: ('TrustedDomainInfoEx', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX), TRUSTED_INFORMATION_CLASS.TrustedDomainAuthInformation: ('TrustedAuthInfo', LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION), TRUSTED_INFORMATION_CLASS.TrustedDomainFullInformation: ('TrustedFullInfo', LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION), TRUSTED_INFORMATION_CLASS.TrustedDomainAuthInformationInternal: ('TrustedAuthInfoInternal', LSAPR_TRUSTED_DOMAIN_AUTH_INFORMATION_INTERNAL), TRUSTED_INFORMATION_CLASS.TrustedDomainFullInformationInternal: ('TrustedFullInfoInternal', LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION_INTERNAL), TRUSTED_INFORMATION_CLASS.TrustedDomainInformationEx2Internal: ('TrustedDomainInfoEx2', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX2), TRUSTED_INFORMATION_CLASS.TrustedDomainFullInformation2Internal: ('TrustedFullInfo2', LSAPR_TRUSTED_DOMAIN_FULL_INFORMATION2), TRUSTED_INFORMATION_CLASS.TrustedDomainSupportedEncryptionTypes: ('TrustedDomainSETs', TRUSTED_DOMAIN_SUPPORTED_ENCRYPTION_TYPES)}

class LSAPR_TRUST_INFORMATION_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRUST_INFORMATION

class PLSAPR_TRUST_INFORMATION_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRUST_INFORMATION_ARRAY),)

class LSAPR_TRUSTED_ENUM_BUFFER(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Information', PLSAPR_TRUST_INFORMATION_ARRAY))

class LSAPR_TRUSTED_DOMAIN_INFORMATION_EX_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRUSTED_DOMAIN_INFORMATION_EX

class PLSAPR_TRUSTED_DOMAIN_INFORMATION_EX_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRUSTED_DOMAIN_INFORMATION_EX_ARRAY),)

class LSAPR_TRUSTED_ENUM_BUFFER_EX(NDRSTRUCT):
    structure = (('Entries', ULONG), ('EnumerationBuffer', PLSAPR_TRUSTED_DOMAIN_INFORMATION_EX_ARRAY))

class LSA_FOREST_TRUST_RECORD_TYPE(NDRENUM):

    class enumItems(Enum):
        ForestTrustTopLevelName = 0
        ForestTrustTopLevelNameEx = 1
        ForestTrustDomainInfo = 2

class LSA_FOREST_TRUST_DOMAIN_INFO(NDRSTRUCT):
    structure = (('Sid', PRPC_SID), ('DnsName', LSA_UNICODE_STRING), ('NetbiosName', LSA_UNICODE_STRING))

class LSA_FOREST_TRUST_DATA_UNION(NDRUNION):
    union = {LSA_FOREST_TRUST_RECORD_TYPE.ForestTrustTopLevelName: ('TopLevelName', LSA_UNICODE_STRING), LSA_FOREST_TRUST_RECORD_TYPE.ForestTrustTopLevelNameEx: ('TopLevelName', LSA_UNICODE_STRING), LSA_FOREST_TRUST_RECORD_TYPE.ForestTrustDomainInfo: ('DomainInfo', LSA_FOREST_TRUST_DOMAIN_INFO)}

class LSA_FOREST_TRUST_RECORD(NDRSTRUCT):
    structure = (('Flags', ULONG), ('ForestTrustType', LSA_FOREST_TRUST_RECORD_TYPE), ('Time', LARGE_INTEGER), ('ForestTrustData', LSA_FOREST_TRUST_DATA_UNION))

class PLSA_FOREST_TRUST_RECORD(NDRPOINTER):
    referent = (('Data', LSA_FOREST_TRUST_RECORD),)

class LSA_FOREST_TRUST_BINARY_DATA(NDRSTRUCT):
    structure = (('Length', ULONG), ('Buffer', LPBYTE))

class LSA_FOREST_TRUST_RECORD_ARRAY(NDRUniConformantArray):
    item = PLSA_FOREST_TRUST_RECORD

class PLSA_FOREST_TRUST_RECORD_ARRAY(NDRPOINTER):
    referent = (('Data', LSA_FOREST_TRUST_RECORD_ARRAY),)

class LSA_FOREST_TRUST_INFORMATION(NDRSTRUCT):
    structure = (('RecordCount', ULONG), ('Entries', PLSA_FOREST_TRUST_RECORD_ARRAY))

class PLSA_FOREST_TRUST_INFORMATION(NDRPOINTER):
    referent = (('Data', LSA_FOREST_TRUST_INFORMATION),)

class LSA_FOREST_TRUST_COLLISION_RECORD_TYPE(NDRENUM):

    class enumItems(Enum):
        CollisionTdo = 0
        CollisionXref = 1
        CollisionOther = 2

class LSA_FOREST_TRUST_COLLISION_RECORD(NDRSTRUCT):
    structure = (('Index', ULONG), ('Type', LSA_FOREST_TRUST_COLLISION_RECORD_TYPE), ('Flags', ULONG), ('Name', LSA_UNICODE_STRING))

class LSAPR_POLICY_PRIVILEGE_DEF(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('LocalValue', LUID))

class LSAPR_POLICY_PRIVILEGE_DEF_ARRAY(NDRUniConformantArray):
    item = LSAPR_POLICY_PRIVILEGE_DEF

class PLSAPR_POLICY_PRIVILEGE_DEF_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_POLICY_PRIVILEGE_DEF_ARRAY),)

class LSAPR_PRIVILEGE_ENUM_BUFFER(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Privileges', PLSAPR_POLICY_PRIVILEGE_DEF_ARRAY))

class LsarOpenPolicy2(NDRCALL):
    opnum = 44
    structure = (('SystemName', LPWSTR), ('ObjectAttributes', LSAPR_OBJECT_ATTRIBUTES), ('DesiredAccess', ACCESS_MASK))

class LsarOpenPolicy2Response(NDRCALL):
    structure = (('PolicyHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarOpenPolicy(NDRCALL):
    opnum = 6
    structure = (('SystemName', LPWSTR), ('ObjectAttributes', LSAPR_OBJECT_ATTRIBUTES), ('DesiredAccess', ACCESS_MASK))

class LsarOpenPolicyResponse(NDRCALL):
    structure = (('PolicyHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarQueryInformationPolicy2(NDRCALL):
    opnum = 46
    structure = (('PolicyHandle', LSAPR_HANDLE), ('InformationClass', POLICY_INFORMATION_CLASS))

class LsarQueryInformationPolicy2Response(NDRCALL):
    structure = (('PolicyInformation', PLSAPR_POLICY_INFORMATION), ('ErrorCode', NTSTATUS))

class LsarQueryInformationPolicy(NDRCALL):
    opnum = 7
    structure = (('PolicyHandle', LSAPR_HANDLE), ('InformationClass', POLICY_INFORMATION_CLASS))

class LsarQueryInformationPolicyResponse(NDRCALL):
    structure = (('PolicyInformation', PLSAPR_POLICY_INFORMATION), ('ErrorCode', NTSTATUS))

class LsarSetInformationPolicy2(NDRCALL):
    opnum = 47
    structure = (('PolicyHandle', LSAPR_HANDLE), ('InformationClass', POLICY_INFORMATION_CLASS), ('PolicyInformation', LSAPR_POLICY_INFORMATION))

class LsarSetInformationPolicy2Response(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarSetInformationPolicy(NDRCALL):
    opnum = 8
    structure = (('PolicyHandle', LSAPR_HANDLE), ('InformationClass', POLICY_INFORMATION_CLASS), ('PolicyInformation', LSAPR_POLICY_INFORMATION))

class LsarSetInformationPolicyResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarQueryDomainInformationPolicy(NDRCALL):
    opnum = 53
    structure = (('PolicyHandle', LSAPR_HANDLE), ('InformationClass', POLICY_DOMAIN_INFORMATION_CLASS))

class LsarQueryDomainInformationPolicyResponse(NDRCALL):
    structure = (('PolicyDomainInformation', PLSAPR_POLICY_DOMAIN_INFORMATION), ('ErrorCode', NTSTATUS))

class LsarCreateAccount(NDRCALL):
    opnum = 10
    structure = (('PolicyHandle', LSAPR_HANDLE), ('AccountSid', RPC_SID), ('DesiredAccess', ACCESS_MASK))

class LsarCreateAccountResponse(NDRCALL):
    structure = (('AccountHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarEnumerateAccounts(NDRCALL):
    opnum = 11
    structure = (('PolicyHandle', LSAPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class LsarEnumerateAccountsResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('EnumerationBuffer', LSAPR_ACCOUNT_ENUM_BUFFER), ('ErrorCode', NTSTATUS))

class LsarOpenAccount(NDRCALL):
    opnum = 17
    structure = (('PolicyHandle', LSAPR_HANDLE), ('AccountSid', RPC_SID), ('DesiredAccess', ACCESS_MASK))

class LsarOpenAccountResponse(NDRCALL):
    structure = (('AccountHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarEnumeratePrivilegesAccount(NDRCALL):
    opnum = 18
    structure = (('AccountHandle', LSAPR_HANDLE),)

class LsarEnumeratePrivilegesAccountResponse(NDRCALL):
    structure = (('Privileges', PLSAPR_PRIVILEGE_SET), ('ErrorCode', NTSTATUS))

class LsarAddPrivilegesToAccount(NDRCALL):
    opnum = 19
    structure = (('AccountHandle', LSAPR_HANDLE), ('Privileges', LSAPR_PRIVILEGE_SET))

class LsarAddPrivilegesToAccountResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarRemovePrivilegesFromAccount(NDRCALL):
    opnum = 20
    structure = (('AccountHandle', LSAPR_HANDLE), ('AllPrivileges', UCHAR), ('Privileges', PLSAPR_PRIVILEGE_SET))

class LsarRemovePrivilegesFromAccountResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarGetSystemAccessAccount(NDRCALL):
    opnum = 23
    structure = (('AccountHandle', LSAPR_HANDLE),)

class LsarGetSystemAccessAccountResponse(NDRCALL):
    structure = (('SystemAccess', ULONG), ('ErrorCode', NTSTATUS))

class LsarSetSystemAccessAccount(NDRCALL):
    opnum = 24
    structure = (('AccountHandle', LSAPR_HANDLE), ('SystemAccess', ULONG))

class LsarSetSystemAccessAccountResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarEnumerateAccountsWithUserRight(NDRCALL):
    opnum = 35
    structure = (('PolicyHandle', LSAPR_HANDLE), ('UserRight', PRPC_UNICODE_STRING))

class LsarEnumerateAccountsWithUserRightResponse(NDRCALL):
    structure = (('EnumerationBuffer', LSAPR_ACCOUNT_ENUM_BUFFER), ('ErrorCode', NTSTATUS))

class LsarEnumerateAccountRights(NDRCALL):
    opnum = 36
    structure = (('PolicyHandle', LSAPR_HANDLE), ('AccountSid', RPC_SID))

class LsarEnumerateAccountRightsResponse(NDRCALL):
    structure = (('UserRights', LSAPR_USER_RIGHT_SET), ('ErrorCode', NTSTATUS))

class LsarAddAccountRights(NDRCALL):
    opnum = 37
    structure = (('PolicyHandle', LSAPR_HANDLE), ('AccountSid', RPC_SID), ('UserRights', LSAPR_USER_RIGHT_SET))

class LsarAddAccountRightsResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarRemoveAccountRights(NDRCALL):
    opnum = 38
    structure = (('PolicyHandle', LSAPR_HANDLE), ('AccountSid', RPC_SID), ('AllRights', UCHAR), ('UserRights', LSAPR_USER_RIGHT_SET))

class LsarRemoveAccountRightsResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarCreateSecret(NDRCALL):
    opnum = 16
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SecretName', RPC_UNICODE_STRING), ('DesiredAccess', ACCESS_MASK))

class LsarCreateSecretResponse(NDRCALL):
    structure = (('SecretHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarOpenSecret(NDRCALL):
    opnum = 28
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SecretName', RPC_UNICODE_STRING), ('DesiredAccess', ACCESS_MASK))

class LsarOpenSecretResponse(NDRCALL):
    structure = (('SecretHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarSetSecret(NDRCALL):
    opnum = 29
    structure = (('SecretHandle', LSAPR_HANDLE), ('EncryptedCurrentValue', PLSAPR_CR_CIPHER_VALUE), ('EncryptedOldValue', PLSAPR_CR_CIPHER_VALUE))

class LsarSetSecretResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarQuerySecret(NDRCALL):
    opnum = 30
    structure = (('SecretHandle', LSAPR_HANDLE), ('EncryptedCurrentValue', PPLSAPR_CR_CIPHER_VALUE), ('CurrentValueSetTime', PLARGE_INTEGER), ('EncryptedOldValue', PPLSAPR_CR_CIPHER_VALUE), ('OldValueSetTime', PLARGE_INTEGER))

class LsarQuerySecretResponse(NDRCALL):
    structure = (('EncryptedCurrentValue', PPLSAPR_CR_CIPHER_VALUE), ('CurrentValueSetTime', PLARGE_INTEGER), ('EncryptedOldValue', PPLSAPR_CR_CIPHER_VALUE), ('OldValueSetTime', PLARGE_INTEGER), ('ErrorCode', NTSTATUS))

class LsarStorePrivateData(NDRCALL):
    opnum = 42
    structure = (('PolicyHandle', LSAPR_HANDLE), ('KeyName', RPC_UNICODE_STRING), ('EncryptedData', PLSAPR_CR_CIPHER_VALUE))

class LsarStorePrivateDataResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarRetrievePrivateData(NDRCALL):
    opnum = 43
    structure = (('PolicyHandle', LSAPR_HANDLE), ('KeyName', RPC_UNICODE_STRING), ('EncryptedData', PLSAPR_CR_CIPHER_VALUE))

class LsarRetrievePrivateDataResponse(NDRCALL):
    structure = (('EncryptedData', PLSAPR_CR_CIPHER_VALUE), ('ErrorCode', NTSTATUS))

class LsarEnumerateTrustedDomainsEx(NDRCALL):
    opnum = 50
    structure = (('PolicyHandle', LSAPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class LsarEnumerateTrustedDomainsExResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('EnumerationBuffer', LSAPR_TRUSTED_ENUM_BUFFER_EX), ('ErrorCode', NTSTATUS))

class LsarEnumerateTrustedDomains(NDRCALL):
    opnum = 13
    structure = (('PolicyHandle', LSAPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class LsarEnumerateTrustedDomainsResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('EnumerationBuffer', LSAPR_TRUSTED_ENUM_BUFFER), ('ErrorCode', NTSTATUS))

class LsarQueryForestTrustInformation(NDRCALL):
    opnum = 73
    structure = (('PolicyHandle', LSAPR_HANDLE), ('TrustedDomainName', LSA_UNICODE_STRING), ('HighestRecordType', LSA_FOREST_TRUST_RECORD_TYPE))

class LsarQueryForestTrustInformationResponse(NDRCALL):
    structure = (('ForestTrustInfo', PLSA_FOREST_TRUST_INFORMATION), ('ErrorCode', NTSTATUS))

class LsarEnumeratePrivileges(NDRCALL):
    opnum = 2
    structure = (('PolicyHandle', LSAPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class LsarEnumeratePrivilegesResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('EnumerationBuffer', LSAPR_PRIVILEGE_ENUM_BUFFER), ('ErrorCode', NTSTATUS))

class LsarLookupPrivilegeValue(NDRCALL):
    opnum = 31
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Name', RPC_UNICODE_STRING))

class LsarLookupPrivilegeValueResponse(NDRCALL):
    structure = (('Value', LUID), ('ErrorCode', NTSTATUS))

class LsarLookupPrivilegeName(NDRCALL):
    opnum = 32
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Value', LUID))

class LsarLookupPrivilegeNameResponse(NDRCALL):
    structure = (('Name', PRPC_UNICODE_STRING), ('ErrorCode', NTSTATUS))

class LsarLookupPrivilegeDisplayName(NDRCALL):
    opnum = 33
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Name', RPC_UNICODE_STRING), ('ClientLanguage', USHORT), ('ClientSystemDefaultLanguage', USHORT))

class LsarLookupPrivilegeDisplayNameResponse(NDRCALL):
    structure = (('Name', PRPC_UNICODE_STRING), ('LanguageReturned', UCHAR), ('ErrorCode', NTSTATUS))

class LsarQuerySecurityObject(NDRCALL):
    opnum = 3
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SecurityInformation', SECURITY_INFORMATION))

class LsarQuerySecurityObjectResponse(NDRCALL):
    structure = (('SecurityDescriptor', PLSAPR_SR_SECURITY_DESCRIPTOR), ('ErrorCode', NTSTATUS))

class LsarSetSecurityObject(NDRCALL):
    opnum = 4
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SecurityInformation', SECURITY_INFORMATION), ('SecurityDescriptor', LSAPR_SR_SECURITY_DESCRIPTOR))

class LsarSetSecurityObjectResponse(NDRCALL):
    structure = (('ErrorCode', NTSTATUS),)

class LsarDeleteObject(NDRCALL):
    opnum = 34
    structure = (('ObjectHandle', LSAPR_HANDLE),)

class LsarDeleteObjectResponse(NDRCALL):
    structure = (('ObjectHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))

class LsarClose(NDRCALL):
    opnum = 0
    structure = (('ObjectHandle', LSAPR_HANDLE),)

class LsarCloseResponse(NDRCALL):
    structure = (('ObjectHandle', LSAPR_HANDLE), ('ErrorCode', NTSTATUS))
OPNUMS = {0: (LsarClose, LsarCloseResponse), 2: (LsarEnumeratePrivileges, LsarEnumeratePrivilegesResponse), 3: (LsarQuerySecurityObject, LsarQuerySecurityObjectResponse), 4: (LsarSetSecurityObject, LsarSetSecurityObjectResponse), 6: (LsarOpenPolicy, LsarOpenPolicyResponse), 7: (LsarQueryInformationPolicy, LsarQueryInformationPolicyResponse), 8: (LsarSetInformationPolicy, LsarSetInformationPolicyResponse), 10: (LsarCreateAccount, LsarCreateAccountResponse), 11: (LsarEnumerateAccounts, LsarEnumerateAccountsResponse), 13: (LsarEnumerateTrustedDomains, LsarEnumerateTrustedDomainsResponse), 16: (LsarCreateSecret, LsarCreateSecretResponse), 17: (LsarOpenAccount, LsarOpenAccountResponse), 18: (LsarEnumeratePrivilegesAccount, LsarEnumeratePrivilegesAccountResponse), 19: (LsarAddPrivilegesToAccount, LsarAddPrivilegesToAccountResponse), 20: (LsarRemovePrivilegesFromAccount, LsarRemovePrivilegesFromAccountResponse), 23: (LsarGetSystemAccessAccount, LsarGetSystemAccessAccountResponse), 24: (LsarSetSystemAccessAccount, LsarSetSystemAccessAccountResponse), 28: (LsarOpenSecret, LsarOpenSecretResponse), 29: (LsarSetSecret, LsarSetSecretResponse), 30: (LsarQuerySecret, LsarQuerySecretResponse), 31: (LsarLookupPrivilegeValue, LsarLookupPrivilegeValueResponse), 32: (LsarLookupPrivilegeName, LsarLookupPrivilegeNameResponse), 33: (LsarLookupPrivilegeDisplayName, LsarLookupPrivilegeDisplayNameResponse), 34: (LsarDeleteObject, LsarDeleteObjectResponse), 35: (LsarEnumerateAccountsWithUserRight, LsarEnumerateAccountsWithUserRightResponse), 36: (LsarEnumerateAccountRights, LsarEnumerateAccountRightsResponse), 37: (LsarAddAccountRights, LsarAddAccountRightsResponse), 38: (LsarRemoveAccountRights, LsarRemoveAccountRightsResponse), 42: (LsarStorePrivateData, LsarStorePrivateDataResponse), 43: (LsarRetrievePrivateData, LsarRetrievePrivateDataResponse), 44: (LsarOpenPolicy2, LsarOpenPolicy2Response), 46: (LsarQueryInformationPolicy2, LsarQueryInformationPolicy2Response), 47: (LsarSetInformationPolicy2, LsarSetInformationPolicy2Response), 50: (LsarEnumerateTrustedDomainsEx, LsarEnumerateTrustedDomainsExResponse), 53: (LsarQueryDomainInformationPolicy, LsarQueryDomainInformationPolicyResponse)}

def hLsarOpenPolicy2(dce, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        for i in range(10):
            print('nop')
    request = LsarOpenPolicy2()
    request['SystemName'] = NULL
    request['ObjectAttributes']['RootDirectory'] = NULL
    request['ObjectAttributes']['ObjectName'] = NULL
    request['ObjectAttributes']['SecurityDescriptor'] = NULL
    request['ObjectAttributes']['SecurityQualityOfService'] = NULL
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarOpenPolicy(dce, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        print('Hello World!')
    request = LsarOpenPolicy()
    request['SystemName'] = NULL
    request['ObjectAttributes']['RootDirectory'] = NULL
    request['ObjectAttributes']['ObjectName'] = NULL
    request['ObjectAttributes']['SecurityDescriptor'] = NULL
    request['ObjectAttributes']['SecurityQualityOfService'] = NULL
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarQueryInformationPolicy2(dce, policyHandle, informationClass):
    if False:
        return 10
    request = LsarQueryInformationPolicy2()
    request['PolicyHandle'] = policyHandle
    request['InformationClass'] = informationClass
    return dce.request(request)

def hLsarQueryInformationPolicy(dce, policyHandle, informationClass):
    if False:
        for i in range(10):
            print('nop')
    request = LsarQueryInformationPolicy()
    request['PolicyHandle'] = policyHandle
    request['InformationClass'] = informationClass
    return dce.request(request)

def hLsarQueryDomainInformationPolicy(dce, policyHandle, informationClass):
    if False:
        for i in range(10):
            print('nop')
    request = LsarQueryInformationPolicy()
    request['PolicyHandle'] = policyHandle
    request['InformationClass'] = informationClass
    return dce.request(request)

def hLsarEnumerateAccounts(dce, policyHandle, preferedMaximumLength=4294967295):
    if False:
        return 10
    request = LsarEnumerateAccounts()
    request['PolicyHandle'] = policyHandle
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hLsarEnumerateAccountsWithUserRight(dce, policyHandle, UserRight):
    if False:
        return 10
    request = LsarEnumerateAccountsWithUserRight()
    request['PolicyHandle'] = policyHandle
    request['UserRight'] = UserRight
    return dce.request(request)

def hLsarEnumerateTrustedDomainsEx(dce, policyHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        i = 10
        return i + 15
    request = LsarEnumerateTrustedDomainsEx()
    request['PolicyHandle'] = policyHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hLsarEnumerateTrustedDomains(dce, policyHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        return 10
    request = LsarEnumerateTrustedDomains()
    request['PolicyHandle'] = policyHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hLsarOpenAccount(dce, policyHandle, accountSid, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        print('Hello World!')
    request = LsarOpenAccount()
    request['PolicyHandle'] = policyHandle
    request['AccountSid'].fromCanonical(accountSid)
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarClose(dce, objectHandle):
    if False:
        while True:
            i = 10
    request = LsarClose()
    request['ObjectHandle'] = objectHandle
    return dce.request(request)

def hLsarCreateAccount(dce, policyHandle, accountSid, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        print('Hello World!')
    request = LsarCreateAccount()
    request['PolicyHandle'] = policyHandle
    request['AccountSid'].fromCanonical(accountSid)
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarDeleteObject(dce, objectHandle):
    if False:
        return 10
    request = LsarDeleteObject()
    request['ObjectHandle'] = objectHandle
    return dce.request(request)

def hLsarEnumeratePrivilegesAccount(dce, accountHandle):
    if False:
        print('Hello World!')
    request = LsarEnumeratePrivilegesAccount()
    request['AccountHandle'] = accountHandle
    return dce.request(request)

def hLsarGetSystemAccessAccount(dce, accountHandle):
    if False:
        i = 10
        return i + 15
    request = LsarGetSystemAccessAccount()
    request['AccountHandle'] = accountHandle
    return dce.request(request)

def hLsarSetSystemAccessAccount(dce, accountHandle, systemAccess):
    if False:
        print('Hello World!')
    request = LsarSetSystemAccessAccount()
    request['AccountHandle'] = accountHandle
    request['SystemAccess'] = systemAccess
    return dce.request(request)

def hLsarAddPrivilegesToAccount(dce, accountHandle, privileges):
    if False:
        for i in range(10):
            print('nop')
    request = LsarAddPrivilegesToAccount()
    request['AccountHandle'] = accountHandle
    request['Privileges']['PrivilegeCount'] = len(privileges)
    request['Privileges']['Control'] = 0
    for priv in privileges:
        request['Privileges']['Privilege'].append(priv)
    return dce.request(request)

def hLsarRemovePrivilegesFromAccount(dce, accountHandle, privileges, allPrivileges=False):
    if False:
        while True:
            i = 10
    request = LsarRemovePrivilegesFromAccount()
    request['AccountHandle'] = accountHandle
    request['Privileges']['Control'] = 0
    if privileges != NULL:
        request['Privileges']['PrivilegeCount'] = len(privileges)
        for priv in privileges:
            request['Privileges']['Privilege'].append(priv)
    else:
        request['Privileges']['PrivilegeCount'] = NULL
    request['AllPrivileges'] = allPrivileges
    return dce.request(request)

def hLsarEnumerateAccountRights(dce, policyHandle, accountSid):
    if False:
        i = 10
        return i + 15
    request = LsarEnumerateAccountRights()
    request['PolicyHandle'] = policyHandle
    request['AccountSid'].fromCanonical(accountSid)
    return dce.request(request)

def hLsarAddAccountRights(dce, policyHandle, accountSid, userRights):
    if False:
        for i in range(10):
            print('nop')
    request = LsarAddAccountRights()
    request['PolicyHandle'] = policyHandle
    request['AccountSid'].fromCanonical(accountSid)
    request['UserRights']['EntriesRead'] = len(userRights)
    for userRight in userRights:
        right = RPC_UNICODE_STRING()
        right['Data'] = userRight
        request['UserRights']['UserRights'].append(right)
    return dce.request(request)

def hLsarRemoveAccountRights(dce, policyHandle, accountSid, userRights):
    if False:
        i = 10
        return i + 15
    request = LsarRemoveAccountRights()
    request['PolicyHandle'] = policyHandle
    request['AccountSid'].fromCanonical(accountSid)
    request['UserRights']['EntriesRead'] = len(userRights)
    for userRight in userRights:
        right = RPC_UNICODE_STRING()
        right['Data'] = userRight
        request['UserRights']['UserRights'].append(right)
    return dce.request(request)

def hLsarCreateSecret(dce, policyHandle, secretName, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        while True:
            i = 10
    request = LsarCreateSecret()
    request['PolicyHandle'] = policyHandle
    request['SecretName'] = secretName
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarOpenSecret(dce, policyHandle, secretName, desiredAccess=MAXIMUM_ALLOWED):
    if False:
        for i in range(10):
            print('nop')
    request = LsarOpenSecret()
    request['PolicyHandle'] = policyHandle
    request['SecretName'] = secretName
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hLsarSetSecret(dce, secretHandle, encryptedCurrentValue, encryptedOldValue):
    if False:
        while True:
            i = 10
    request = LsarOpenSecret()
    request['SecretHandle'] = secretHandle
    if encryptedCurrentValue != NULL:
        request['EncryptedCurrentValue']['Length'] = len(encryptedCurrentValue)
        request['EncryptedCurrentValue']['MaximumLength'] = len(encryptedCurrentValue)
        request['EncryptedCurrentValue']['Buffer'] = list(encryptedCurrentValue)
    if encryptedOldValue != NULL:
        request['EncryptedOldValue']['Length'] = len(encryptedOldValue)
        request['EncryptedOldValue']['MaximumLength'] = len(encryptedOldValue)
        request['EncryptedOldValue']['Buffer'] = list(encryptedOldValue)
    return dce.request(request)

def hLsarQuerySecret(dce, secretHandle):
    if False:
        return 10
    request = LsarQuerySecret()
    request['SecretHandle'] = secretHandle
    request['EncryptedCurrentValue']['Buffer'] = NULL
    request['EncryptedOldValue']['Buffer'] = NULL
    request['OldValueSetTime'] = NULL
    return dce.request(request)

def hLsarRetrievePrivateData(dce, policyHandle, keyName):
    if False:
        for i in range(10):
            print('nop')
    request = LsarRetrievePrivateData()
    request['PolicyHandle'] = policyHandle
    request['KeyName'] = keyName
    retVal = dce.request(request)
    return b''.join(retVal['EncryptedData']['Buffer'])

def hLsarStorePrivateData(dce, policyHandle, keyName, encryptedData):
    if False:
        while True:
            i = 10
    request = LsarStorePrivateData()
    request['PolicyHandle'] = policyHandle
    request['KeyName'] = keyName
    if encryptedData != NULL:
        request['EncryptedData']['Length'] = len(encryptedData)
        request['EncryptedData']['MaximumLength'] = len(encryptedData)
        request['EncryptedData']['Buffer'] = list(encryptedData)
    else:
        request['EncryptedData'] = NULL
    return dce.request(request)

def hLsarEnumeratePrivileges(dce, policyHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        i = 10
        return i + 15
    request = LsarEnumeratePrivileges()
    request['PolicyHandle'] = policyHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hLsarLookupPrivilegeValue(dce, policyHandle, name):
    if False:
        while True:
            i = 10
    request = LsarLookupPrivilegeValue()
    request['PolicyHandle'] = policyHandle
    request['Name'] = name
    return dce.request(request)

def hLsarLookupPrivilegeName(dce, policyHandle, luid):
    if False:
        i = 10
        return i + 15
    request = LsarLookupPrivilegeName()
    request['PolicyHandle'] = policyHandle
    request['Value'] = luid
    return dce.request(request)

def hLsarQuerySecurityObject(dce, policyHandle, securityInformation=OWNER_SECURITY_INFORMATION):
    if False:
        return 10
    request = LsarQuerySecurityObject()
    request['PolicyHandle'] = policyHandle
    request['SecurityInformation'] = securityInformation
    retVal = dce.request(request)
    return b''.join(retVal['SecurityDescriptor']['SecurityDescriptor'])

def hLsarSetSecurityObject(dce, policyHandle, securityInformation, securityDescriptor):
    if False:
        for i in range(10):
            print('nop')
    request = LsarSetSecurityObject()
    request['PolicyHandle'] = policyHandle
    request['SecurityInformation'] = securityInformation
    request['SecurityDescriptor']['Length'] = len(securityDescriptor)
    request['SecurityDescriptor']['SecurityDescriptor'] = list(securityDescriptor)
    return dce.request(request)

def hLsarSetInformationPolicy2(dce, policyHandle, informationClass, policyInformation):
    if False:
        print('Hello World!')
    request = LsarSetInformationPolicy2()
    request['PolicyHandle'] = policyHandle
    request['InformationClass'] = informationClass
    request['PolicyInformation'] = policyInformation
    return dce.request(request)

def hLsarSetInformationPolicy(dce, policyHandle, informationClass, policyInformation):
    if False:
        print('Hello World!')
    request = LsarSetInformationPolicy()
    request['PolicyHandle'] = policyHandle
    request['InformationClass'] = informationClass
    request['PolicyInformation'] = policyInformation
    return dce.request(request)