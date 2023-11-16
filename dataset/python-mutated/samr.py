from __future__ import division
from __future__ import print_function
from binascii import unhexlify
from impacket.dcerpc.v5.ndr import NDRCALL, NDR, NDRSTRUCT, NDRUNION, NDRPOINTER, NDRUniConformantArray, NDRUniConformantVaryingArray, NDRENUM
from impacket.dcerpc.v5.dtypes import NULL, RPC_UNICODE_STRING, ULONG, USHORT, UCHAR, LARGE_INTEGER, RPC_SID, LONG, STR, LPBYTE, SECURITY_INFORMATION, PRPC_SID, PRPC_UNICODE_STRING, LPWSTR
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket import nt_errors, LOG
from impacket.uuid import uuidtup_to_bin
from impacket.dcerpc.v5.enum import Enum
from impacket.structure import Structure
import struct
import os
from hashlib import md5
from Cryptodome.Cipher import ARC4
MSRPC_UUID_SAMR = uuidtup_to_bin(('12345778-1234-ABCD-EF00-0123456789AC', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            return 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            while True:
                i = 10
        key = self.error_code
        if key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'SAMR SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'SAMR SessionError: unknown error code: 0x%x' % self.error_code
PSAMPR_SERVER_NAME = LPWSTR
DELETE = 65536
READ_CONTROL = 131072
WRITE_DAC = 262144
WRITE_OWNER = 524288
ACCESS_SYSTEM_SECURITY = 16777216
MAXIMUM_ALLOWED = 33554432
GENERIC_READ = 2147483648
GENERIC_WRITE = 1073741824
GENERIC_EXECUTE = 536870912
GENERIC_ALL = 268435456
SAM_SERVER_CONNECT = 1
SAM_SERVER_SHUTDOWN = 2
SAM_SERVER_INITIALIZE = 4
SAM_SERVER_CREATE_DOMAIN = 8
SAM_SERVER_ENUMERATE_DOMAINS = 16
SAM_SERVER_LOOKUP_DOMAIN = 32
SAM_SERVER_ALL_ACCESS = 983103
SAM_SERVER_READ = 131088
SAM_SERVER_WRITE = 131086
SAM_SERVER_EXECUTE = 131105
DOMAIN_READ_PASSWORD_PARAMETERS = 1
DOMAIN_WRITE_PASSWORD_PARAMS = 2
DOMAIN_READ_OTHER_PARAMETERS = 4
DOMAIN_WRITE_OTHER_PARAMETERS = 8
DOMAIN_CREATE_USER = 16
DOMAIN_CREATE_GROUP = 32
DOMAIN_CREATE_ALIAS = 64
DOMAIN_GET_ALIAS_MEMBERSHIP = 128
DOMAIN_LIST_ACCOUNTS = 256
DOMAIN_LOOKUP = 512
DOMAIN_ADMINISTER_SERVER = 1024
DOMAIN_ALL_ACCESS = 985087
DOMAIN_READ = 131204
DOMAIN_WRITE = 132218
DOMAIN_EXECUTE = 131841
GROUP_READ_INFORMATION = 1
GROUP_WRITE_ACCOUNT = 2
GROUP_ADD_MEMBER = 4
GROUP_REMOVE_MEMBER = 8
GROUP_LIST_MEMBERS = 16
GROUP_ALL_ACCESS = 983071
GROUP_READ = 131088
GROUP_WRITE = 131086
GROUP_EXECUTE = 131073
ALIAS_ADD_MEMBER = 1
ALIAS_REMOVE_MEMBER = 2
ALIAS_LIST_MEMBERS = 4
ALIAS_READ_INFORMATION = 8
ALIAS_WRITE_ACCOUNT = 16
ALIAS_ALL_ACCESS = 983071
ALIAS_READ = 131076
ALIAS_WRITE = 131091
ALIAS_EXECUTE = 131080
USER_READ_GENERAL = 1
USER_READ_PREFERENCES = 2
USER_WRITE_PREFERENCES = 4
USER_READ_LOGON = 8
USER_READ_ACCOUNT = 16
USER_WRITE_ACCOUNT = 32
USER_CHANGE_PASSWORD = 64
USER_FORCE_PASSWORD_CHANGE = 128
USER_LIST_GROUPS = 256
USER_READ_GROUP_INFORMATION = 512
USER_WRITE_GROUP_INFORMATION = 1024
USER_ALL_ACCESS = 985087
USER_READ = 131866
USER_WRITE = 131140
USER_EXECUTE = 131137
USER_ALL_USERNAME = 1
USER_ALL_FULLNAME = 2
USER_ALL_USERID = 4
USER_ALL_PRIMARYGROUPID = 8
USER_ALL_ADMINCOMMENT = 16
USER_ALL_USERCOMMENT = 32
USER_ALL_HOMEDIRECTORY = 64
USER_ALL_HOMEDIRECTORYDRIVE = 128
USER_ALL_SCRIPTPATH = 256
USER_ALL_PROFILEPATH = 512
USER_ALL_WORKSTATIONS = 1024
USER_ALL_LASTLOGON = 2048
USER_ALL_LASTLOGOFF = 4096
USER_ALL_LOGONHOURS = 8192
USER_ALL_BADPASSWORDCOUNT = 16384
USER_ALL_LOGONCOUNT = 32768
USER_ALL_PASSWORDCANCHANGE = 65536
USER_ALL_PASSWORDMUSTCHANGE = 131072
USER_ALL_PASSWORDLASTSET = 262144
USER_ALL_ACCOUNTEXPIRES = 524288
USER_ALL_USERACCOUNTCONTROL = 1048576
USER_ALL_PARAMETERS = 2097152
USER_ALL_COUNTRYCODE = 4194304
USER_ALL_CODEPAGE = 8388608
USER_ALL_NTPASSWORDPRESENT = 16777216
USER_ALL_LMPASSWORDPRESENT = 33554432
USER_ALL_PRIVATEDATA = 67108864
USER_ALL_PASSWORDEXPIRED = 134217728
USER_ALL_SECURITYDESCRIPTOR = 268435456
USER_ALL_UNDEFINED_MASK = 3221225472
SAM_DOMAIN_OBJECT = 0
SAM_GROUP_OBJECT = 268435456
SAM_NON_SECURITY_GROUP_OBJECT = 268435457
SAM_ALIAS_OBJECT = 536870912
SAM_NON_SECURITY_ALIAS_OBJECT = 536870913
SAM_USER_OBJECT = 805306368
SAM_MACHINE_ACCOUNT = 805306369
SAM_TRUST_ACCOUNT = 805306370
SAM_APP_BASIC_GROUP = 1073741824
SAM_APP_QUERY_GROUP = 1073741825
SE_GROUP_MANDATORY = 1
SE_GROUP_ENABLED_BY_DEFAULT = 2
SE_GROUP_ENABLED = 4
GROUP_TYPE_ACCOUNT_GROUP = 2
GROUP_TYPE_RESOURCE_GROUP = 4
GROUP_TYPE_UNIVERSAL_GROUP = 8
GROUP_TYPE_SECURITY_ENABLED = 2147483648
GROUP_TYPE_SECURITY_ACCOUNT = 2147483650
GROUP_TYPE_SECURITY_RESOURCE = 2147483652
GROUP_TYPE_SECURITY_UNIVERSAL = 2147483656
USER_ACCOUNT_DISABLED = 1
USER_HOME_DIRECTORY_REQUIRED = 2
USER_PASSWORD_NOT_REQUIRED = 4
USER_TEMP_DUPLICATE_ACCOUNT = 8
USER_NORMAL_ACCOUNT = 16
USER_MNS_LOGON_ACCOUNT = 32
USER_INTERDOMAIN_TRUST_ACCOUNT = 64
USER_WORKSTATION_TRUST_ACCOUNT = 128
USER_SERVER_TRUST_ACCOUNT = 256
USER_DONT_EXPIRE_PASSWORD = 512
USER_ACCOUNT_AUTO_LOCKED = 1024
USER_ENCRYPTED_TEXT_PASSWORD_ALLOWED = 2048
USER_SMARTCARD_REQUIRED = 4096
USER_TRUSTED_FOR_DELEGATION = 8192
USER_NOT_DELEGATED = 16384
USER_USE_DES_KEY_ONLY = 32768
USER_DONT_REQUIRE_PREAUTH = 65536
USER_PASSWORD_EXPIRED = 131072
USER_TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION = 262144
USER_NO_AUTH_DATA_REQUIRED = 524288
USER_PARTIAL_SECRETS_ACCOUNT = 1048576
USER_USE_AES_KEYS = 2097152
UF_SCRIPT = 1
UF_ACCOUNTDISABLE = 2
UF_HOMEDIR_REQUIRED = 8
UF_LOCKOUT = 16
UF_PASSWD_NOTREQD = 32
UF_PASSWD_CANT_CHANGE = 64
UF_ENCRYPTED_TEXT_PASSWORD_ALLOWED = 128
UF_TEMP_DUPLICATE_ACCOUNT = 256
UF_NORMAL_ACCOUNT = 512
UF_INTERDOMAIN_TRUST_ACCOUNT = 2048
UF_WORKSTATION_TRUST_ACCOUNT = 4096
UF_SERVER_TRUST_ACCOUNT = 8192
UF_DONT_EXPIRE_PASSWD = 65536
UF_MNS_LOGON_ACCOUNT = 131072
UF_SMARTCARD_REQUIRED = 262144
UF_TRUSTED_FOR_DELEGATION = 524288
UF_NOT_DELEGATED = 1048576
UF_USE_DES_KEY_ONLY = 2097152
UF_DONT_REQUIRE_PREAUTH = 4194304
UF_PASSWORD_EXPIRED = 8388608
UF_TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION = 16777216
UF_NO_AUTH_DATA_REQUIRED = 33554432
UF_PARTIAL_SECRETS_ACCOUNT = 67108864
UF_USE_AES_KEYS = 134217728
DOMAIN_USER_RID_ADMIN = 500
DOMAIN_USER_RID_GUEST = 501
DOMAIN_USER_RID_KRBTGT = 502
DOMAIN_GROUP_RID_ADMINS = 512
DOMAIN_GROUP_RID_USERS = 513
DOMAIN_GROUP_RID_COMPUTERS = 515
DOMAIN_GROUP_RID_CONTROLLERS = 516
DOMAIN_ALIAS_RID_ADMINS = 544
DOMAIN_GROUP_RID_READONLY_CONTROLLERS = 521
DOMAIN_PASSWORD_COMPLEX = 1
DOMAIN_PASSWORD_NO_ANON_CHANGE = 2
DOMAIN_PASSWORD_NO_CLEAR_CHANGE = 4
DOMAIN_LOCKOUT_ADMINS = 8
DOMAIN_PASSWORD_STORE_CLEARTEXT = 16
DOMAIN_REFUSE_PASSWORD_CHANGE = 32
SAM_VALIDATE_PASSWORD_LAST_SET = 1
SAM_VALIDATE_BAD_PASSWORD_TIME = 2
SAM_VALIDATE_LOCKOUT_TIME = 4
SAM_VALIDATE_BAD_PASSWORD_COUNT = 8
SAM_VALIDATE_PASSWORD_HISTORY_LENGTH = 16
SAM_VALIDATE_PASSWORD_HISTORY = 32

class RPC_UNICODE_STRING_ARRAY(NDRUniConformantVaryingArray):
    item = RPC_UNICODE_STRING

class RPC_UNICODE_STRING_ARRAY_C(NDRUniConformantArray):
    item = RPC_UNICODE_STRING

class PRPC_UNICODE_STRING_ARRAY(NDRPOINTER):
    referent = (('Data', RPC_UNICODE_STRING_ARRAY_C),)

class RPC_STRING(NDRSTRUCT):
    commonHdr = (('MaximumLength', '<H=len(Data)-12'), ('Length', '<H=len(Data)-12'), ('ReferentID', '<L=0xff'))
    commonHdr64 = (('MaximumLength', '<H=len(Data)-24'), ('Length', '<H=len(Data)-24'), ('ReferentID', '<Q=0xff'))
    referent = (('Data', STR),)

    def dump(self, msg=None, indent=0):
        if False:
            for i in range(10):
                print('nop')
        if msg is None:
            msg = self.__class__.__name__
        if msg != '':
            print('%s' % msg, end=' ')
        print(' %r' % self['Data'], end=' ')

class PRPC_STRING(NDRPOINTER):
    referent = (('Data', RPC_STRING),)

class OLD_LARGE_INTEGER(NDRSTRUCT):
    structure = (('LowPart', ULONG), ('HighPart', LONG))

class SID_NAME_USE(NDRENUM):

    class enumItems(Enum):
        SidTypeUser = 1
        SidTypeGroup = 2
        SidTypeDomain = 3
        SidTypeAlias = 4
        SidTypeWellKnownGroup = 5
        SidTypeDeletedAccount = 6
        SidTypeInvalid = 7
        SidTypeUnknown = 8
        SidTypeComputer = 9
        SidTypeLabel = 10

class USHORT_ARRAY(NDRUniConformantVaryingArray):
    item = '<H'
    pass

class PUSHORT_ARRAY(NDRPOINTER):
    referent = (('Data', USHORT_ARRAY),)

class RPC_SHORT_BLOB(NDRSTRUCT):
    structure = (('Length', USHORT), ('MaximumLength', USHORT), ('Buffer', PUSHORT_ARRAY))

class SAMPR_HANDLE(NDRSTRUCT):
    structure = (('Data', '20s=b""'),)

    def getAlignment(self):
        if False:
            print('Hello World!')
        if self._isNDR64 is True:
            return 8
        else:
            return 4

class ENCRYPTED_LM_OWF_PASSWORD(NDRSTRUCT):
    structure = (('Data', '16s=b""'),)

    def getAlignment(self):
        if False:
            while True:
                i = 10
        return 1
ENCRYPTED_NT_OWF_PASSWORD = ENCRYPTED_LM_OWF_PASSWORD

class PENCRYPTED_LM_OWF_PASSWORD(NDRPOINTER):
    referent = (('Data', ENCRYPTED_LM_OWF_PASSWORD),)
PENCRYPTED_NT_OWF_PASSWORD = PENCRYPTED_LM_OWF_PASSWORD

class ULONG_ARRAY(NDRUniConformantArray):
    item = ULONG

class PULONG_ARRAY(NDRPOINTER):
    referent = (('Data', ULONG_ARRAY),)

class ULONG_ARRAY_CV(NDRUniConformantVaryingArray):
    item = ULONG

class SAMPR_ULONG_ARRAY(NDRSTRUCT):
    structure = (('Count', ULONG), ('Element', PULONG_ARRAY))

class SAMPR_SID_INFORMATION(NDRSTRUCT):
    structure = (('SidPointer', RPC_SID),)

class PSAMPR_SID_INFORMATION(NDRPOINTER):
    referent = (('Data', SAMPR_SID_INFORMATION),)

class SAMPR_SID_INFORMATION_ARRAY(NDRUniConformantArray):
    item = PSAMPR_SID_INFORMATION

class PSAMPR_SID_INFORMATION_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_SID_INFORMATION_ARRAY),)

class SAMPR_PSID_ARRAY(NDRSTRUCT):
    structure = (('Count', ULONG), ('Sids', PSAMPR_SID_INFORMATION_ARRAY))

class SAMPR_PSID_ARRAY_OUT(NDRSTRUCT):
    structure = (('Count', ULONG), ('Sids', PSAMPR_SID_INFORMATION_ARRAY))

class SAMPR_RETURNED_USTRING_ARRAY(NDRSTRUCT):
    structure = (('Count', ULONG), ('Element', PRPC_UNICODE_STRING_ARRAY))

class SAMPR_RID_ENUMERATION(NDRSTRUCT):
    structure = (('RelativeId', ULONG), ('Name', RPC_UNICODE_STRING))

class SAMPR_RID_ENUMERATION_ARRAY(NDRUniConformantArray):
    item = SAMPR_RID_ENUMERATION

class PSAMPR_RID_ENUMERATION_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_RID_ENUMERATION_ARRAY),)

class SAMPR_ENUMERATION_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_RID_ENUMERATION_ARRAY))

class PSAMPR_ENUMERATION_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_ENUMERATION_BUFFER),)

class CHAR_ARRAY(NDRUniConformantArray):
    pass

class PCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', CHAR_ARRAY),)

class SAMPR_SR_SECURITY_DESCRIPTOR(NDRSTRUCT):
    structure = (('Length', ULONG), ('SecurityDescriptor', PCHAR_ARRAY))

class PSAMPR_SR_SECURITY_DESCRIPTOR(NDRPOINTER):
    referent = (('Data', SAMPR_SR_SECURITY_DESCRIPTOR),)

class GROUP_MEMBERSHIP(NDRSTRUCT):
    structure = (('RelativeId', ULONG), ('Attributes', ULONG))

class GROUP_MEMBERSHIP_ARRAY(NDRUniConformantArray):
    item = GROUP_MEMBERSHIP

class PGROUP_MEMBERSHIP_ARRAY(NDRPOINTER):
    referent = (('Data', GROUP_MEMBERSHIP_ARRAY),)

class SAMPR_GET_GROUPS_BUFFER(NDRSTRUCT):
    structure = (('MembershipCount', ULONG), ('Groups', PGROUP_MEMBERSHIP_ARRAY))

class PSAMPR_GET_GROUPS_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_GET_GROUPS_BUFFER),)

class SAMPR_GET_MEMBERS_BUFFER(NDRSTRUCT):
    structure = (('MemberCount', ULONG), ('Members', PULONG_ARRAY), ('Attributes', PULONG_ARRAY))

class PSAMPR_GET_MEMBERS_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_GET_MEMBERS_BUFFER),)

class SAMPR_REVISION_INFO_V1(NDRSTRUCT):
    structure = (('Revision', ULONG), ('SupportedFeatures', ULONG))

class SAMPR_REVISION_INFO(NDRUNION):
    commonHdr = (('tag', ULONG),)
    union = {1: ('V1', SAMPR_REVISION_INFO_V1)}

class USER_DOMAIN_PASSWORD_INFORMATION(NDRSTRUCT):
    structure = (('MinPasswordLength', USHORT), ('PasswordProperties', ULONG))

class DOMAIN_SERVER_ENABLE_STATE(NDRENUM):

    class enumItems(Enum):
        DomainServerEnabled = 1
        DomainServerDisabled = 2

class DOMAIN_STATE_INFORMATION(NDRSTRUCT):
    structure = (('DomainServerState', DOMAIN_SERVER_ENABLE_STATE),)

class DOMAIN_SERVER_ROLE(NDRENUM):

    class enumItems(Enum):
        DomainServerRoleBackup = 2
        DomainServerRolePrimary = 3

class DOMAIN_PASSWORD_INFORMATION(NDRSTRUCT):
    structure = (('MinPasswordLength', USHORT), ('PasswordHistoryLength', USHORT), ('PasswordProperties', ULONG), ('MaxPasswordAge', OLD_LARGE_INTEGER), ('MinPasswordAge', OLD_LARGE_INTEGER))

class DOMAIN_LOGOFF_INFORMATION(NDRSTRUCT):
    structure = (('ForceLogoff', OLD_LARGE_INTEGER),)

class DOMAIN_SERVER_ROLE_INFORMATION(NDRSTRUCT):
    structure = (('DomainServerRole', DOMAIN_SERVER_ROLE),)

class DOMAIN_MODIFIED_INFORMATION(NDRSTRUCT):
    structure = (('DomainModifiedCount', OLD_LARGE_INTEGER), ('CreationTime', OLD_LARGE_INTEGER))

class DOMAIN_MODIFIED_INFORMATION2(NDRSTRUCT):
    structure = (('DomainModifiedCount', OLD_LARGE_INTEGER), ('CreationTime', OLD_LARGE_INTEGER), ('ModifiedCountAtLastPromotion', OLD_LARGE_INTEGER))

class SAMPR_DOMAIN_GENERAL_INFORMATION(NDRSTRUCT):
    structure = (('ForceLogoff', OLD_LARGE_INTEGER), ('OemInformation', RPC_UNICODE_STRING), ('DomainName', RPC_UNICODE_STRING), ('ReplicaSourceNodeName', RPC_UNICODE_STRING), ('DomainModifiedCount', OLD_LARGE_INTEGER), ('DomainServerState', ULONG), ('DomainServerRole', ULONG), ('UasCompatibilityRequired', UCHAR), ('UserCount', ULONG), ('GroupCount', ULONG), ('AliasCount', ULONG))

class SAMPR_DOMAIN_GENERAL_INFORMATION2(NDRSTRUCT):
    structure = (('I1', SAMPR_DOMAIN_GENERAL_INFORMATION), ('LockoutDuration', LARGE_INTEGER), ('LockoutObservationWindow', LARGE_INTEGER), ('LockoutThreshold', USHORT))

class SAMPR_DOMAIN_OEM_INFORMATION(NDRSTRUCT):
    structure = (('OemInformation', RPC_UNICODE_STRING),)

class SAMPR_DOMAIN_NAME_INFORMATION(NDRSTRUCT):
    structure = (('DomainName', RPC_UNICODE_STRING),)

class SAMPR_DOMAIN_REPLICATION_INFORMATION(NDRSTRUCT):
    structure = (('ReplicaSourceNodeName', RPC_UNICODE_STRING),)

class SAMPR_DOMAIN_LOCKOUT_INFORMATION(NDRSTRUCT):
    structure = (('LockoutDuration', LARGE_INTEGER), ('LockoutObservationWindow', LARGE_INTEGER), ('LockoutThreshold', USHORT))

class DOMAIN_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        DomainPasswordInformation = 1
        DomainGeneralInformation = 2
        DomainLogoffInformation = 3
        DomainOemInformation = 4
        DomainNameInformation = 5
        DomainReplicationInformation = 6
        DomainServerRoleInformation = 7
        DomainModifiedInformation = 8
        DomainStateInformation = 9
        DomainGeneralInformation2 = 11
        DomainLockoutInformation = 12
        DomainModifiedInformation2 = 13

class SAMPR_DOMAIN_INFO_BUFFER(NDRUNION):
    union = {DOMAIN_INFORMATION_CLASS.DomainPasswordInformation: ('Password', DOMAIN_PASSWORD_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainGeneralInformation: ('General', SAMPR_DOMAIN_GENERAL_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainLogoffInformation: ('Logoff', DOMAIN_LOGOFF_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainOemInformation: ('Oem', SAMPR_DOMAIN_OEM_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainNameInformation: ('Name', SAMPR_DOMAIN_NAME_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainServerRoleInformation: ('Role', DOMAIN_SERVER_ROLE_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainReplicationInformation: ('Replication', SAMPR_DOMAIN_REPLICATION_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainModifiedInformation: ('Modified', DOMAIN_MODIFIED_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainStateInformation: ('State', DOMAIN_STATE_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainGeneralInformation2: ('General2', SAMPR_DOMAIN_GENERAL_INFORMATION2), DOMAIN_INFORMATION_CLASS.DomainLockoutInformation: ('Lockout', SAMPR_DOMAIN_LOCKOUT_INFORMATION), DOMAIN_INFORMATION_CLASS.DomainModifiedInformation2: ('Modified2', DOMAIN_MODIFIED_INFORMATION2)}

class PSAMPR_DOMAIN_INFO_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_INFO_BUFFER),)

class GROUP_ATTRIBUTE_INFORMATION(NDRSTRUCT):
    structure = (('Attributes', ULONG),)

class SAMPR_GROUP_GENERAL_INFORMATION(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('Attributes', ULONG), ('MemberCount', ULONG), ('AdminComment', RPC_UNICODE_STRING))

class SAMPR_GROUP_NAME_INFORMATION(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING),)

class SAMPR_GROUP_ADM_COMMENT_INFORMATION(NDRSTRUCT):
    structure = (('AdminComment', RPC_UNICODE_STRING),)

class GROUP_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        GroupGeneralInformation = 1
        GroupNameInformation = 2
        GroupAttributeInformation = 3
        GroupAdminCommentInformation = 4
        GroupReplicationInformation = 5

class SAMPR_GROUP_INFO_BUFFER(NDRUNION):
    union = {GROUP_INFORMATION_CLASS.GroupGeneralInformation: ('General', SAMPR_GROUP_GENERAL_INFORMATION), GROUP_INFORMATION_CLASS.GroupNameInformation: ('Name', SAMPR_GROUP_NAME_INFORMATION), GROUP_INFORMATION_CLASS.GroupAttributeInformation: ('Attribute', GROUP_ATTRIBUTE_INFORMATION), GROUP_INFORMATION_CLASS.GroupAdminCommentInformation: ('AdminComment', SAMPR_GROUP_ADM_COMMENT_INFORMATION), GROUP_INFORMATION_CLASS.GroupReplicationInformation: ('DoNotUse', SAMPR_GROUP_GENERAL_INFORMATION)}

class PSAMPR_GROUP_INFO_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_GROUP_INFO_BUFFER),)

class SAMPR_ALIAS_GENERAL_INFORMATION(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING), ('MemberCount', ULONG), ('AdminComment', RPC_UNICODE_STRING))

class SAMPR_ALIAS_NAME_INFORMATION(NDRSTRUCT):
    structure = (('Name', RPC_UNICODE_STRING),)

class SAMPR_ALIAS_ADM_COMMENT_INFORMATION(NDRSTRUCT):
    structure = (('AdminComment', RPC_UNICODE_STRING),)

class ALIAS_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        AliasGeneralInformation = 1
        AliasNameInformation = 2
        AliasAdminCommentInformation = 3

class SAMPR_ALIAS_INFO_BUFFER(NDRUNION):
    union = {ALIAS_INFORMATION_CLASS.AliasGeneralInformation: ('General', SAMPR_ALIAS_GENERAL_INFORMATION), ALIAS_INFORMATION_CLASS.AliasNameInformation: ('Name', SAMPR_ALIAS_NAME_INFORMATION), ALIAS_INFORMATION_CLASS.AliasAdminCommentInformation: ('AdminComment', SAMPR_ALIAS_ADM_COMMENT_INFORMATION)}

class PSAMPR_ALIAS_INFO_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_ALIAS_INFO_BUFFER),)

class USER_PRIMARY_GROUP_INFORMATION(NDRSTRUCT):
    structure = (('PrimaryGroupId', ULONG),)

class USER_CONTROL_INFORMATION(NDRSTRUCT):
    structure = (('UserAccountControl', ULONG),)

class USER_EXPIRES_INFORMATION(NDRSTRUCT):
    structure = (('AccountExpires', OLD_LARGE_INTEGER),)

class LOGON_HOURS_ARRAY(NDRUniConformantVaryingArray):
    pass

class PLOGON_HOURS_ARRAY(NDRPOINTER):
    referent = (('Data', LOGON_HOURS_ARRAY),)

class SAMPR_LOGON_HOURS(NDRSTRUCT):
    structure = (('UnitsPerWeek', ULONG), ('LogonHours', PLOGON_HOURS_ARRAY))

    def getData(self, soFar=0):
        if False:
            while True:
                i = 10
        if self['LogonHours'] != 0:
            self['UnitsPerWeek'] = len(self['LogonHours']) * 8
        return NDR.getData(self, soFar)

class SAMPR_USER_ALL_INFORMATION(NDRSTRUCT):
    structure = (('LastLogon', OLD_LARGE_INTEGER), ('LastLogoff', OLD_LARGE_INTEGER), ('PasswordLastSet', OLD_LARGE_INTEGER), ('AccountExpires', OLD_LARGE_INTEGER), ('PasswordCanChange', OLD_LARGE_INTEGER), ('PasswordMustChange', OLD_LARGE_INTEGER), ('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('ScriptPath', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING), ('WorkStations', RPC_UNICODE_STRING), ('UserComment', RPC_UNICODE_STRING), ('Parameters', RPC_UNICODE_STRING), ('LmOwfPassword', RPC_SHORT_BLOB), ('NtOwfPassword', RPC_SHORT_BLOB), ('PrivateData', RPC_UNICODE_STRING), ('SecurityDescriptor', SAMPR_SR_SECURITY_DESCRIPTOR), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('UserAccountControl', ULONG), ('WhichFields', ULONG), ('LogonHours', SAMPR_LOGON_HOURS), ('BadPasswordCount', USHORT), ('LogonCount', USHORT), ('CountryCode', USHORT), ('CodePage', USHORT), ('LmPasswordPresent', UCHAR), ('NtPasswordPresent', UCHAR), ('PasswordExpired', UCHAR), ('PrivateDataSensitive', UCHAR))

class SAMPR_USER_GENERAL_INFORMATION(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('PrimaryGroupId', ULONG), ('AdminComment', RPC_UNICODE_STRING), ('UserComment', RPC_UNICODE_STRING))

class SAMPR_USER_PREFERENCES_INFORMATION(NDRSTRUCT):
    structure = (('UserComment', RPC_UNICODE_STRING), ('Reserved1', RPC_UNICODE_STRING), ('CountryCode', USHORT), ('CodePage', USHORT))

class SAMPR_USER_PARAMETERS_INFORMATION(NDRSTRUCT):
    structure = (('Parameters', RPC_UNICODE_STRING),)

class SAMPR_USER_LOGON_INFORMATION(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('ScriptPath', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('WorkStations', RPC_UNICODE_STRING), ('LastLogon', OLD_LARGE_INTEGER), ('LastLogoff', OLD_LARGE_INTEGER), ('PasswordLastSet', OLD_LARGE_INTEGER), ('PasswordCanChange', OLD_LARGE_INTEGER), ('PasswordMustChange', OLD_LARGE_INTEGER), ('LogonHours', SAMPR_LOGON_HOURS), ('BadPasswordCount', USHORT), ('LogonCount', USHORT), ('UserAccountControl', ULONG))

class SAMPR_USER_ACCOUNT_INFORMATION(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING), ('UserId', ULONG), ('PrimaryGroupId', ULONG), ('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING), ('ScriptPath', RPC_UNICODE_STRING), ('ProfilePath', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING), ('WorkStations', RPC_UNICODE_STRING), ('LastLogon', OLD_LARGE_INTEGER), ('LastLogoff', OLD_LARGE_INTEGER), ('LogonHours', SAMPR_LOGON_HOURS), ('BadPasswordCount', USHORT), ('LogonCount', USHORT), ('PasswordLastSet', OLD_LARGE_INTEGER), ('AccountExpires', OLD_LARGE_INTEGER), ('UserAccountControl', ULONG))

class SAMPR_USER_A_NAME_INFORMATION(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING),)

class SAMPR_USER_F_NAME_INFORMATION(NDRSTRUCT):
    structure = (('FullName', RPC_UNICODE_STRING),)

class SAMPR_USER_NAME_INFORMATION(NDRSTRUCT):
    structure = (('UserName', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING))

class SAMPR_USER_HOME_INFORMATION(NDRSTRUCT):
    structure = (('HomeDirectory', RPC_UNICODE_STRING), ('HomeDirectoryDrive', RPC_UNICODE_STRING))

class SAMPR_USER_SCRIPT_INFORMATION(NDRSTRUCT):
    structure = (('ScriptPath', RPC_UNICODE_STRING),)

class SAMPR_USER_PROFILE_INFORMATION(NDRSTRUCT):
    structure = (('ProfilePath', RPC_UNICODE_STRING),)

class SAMPR_USER_ADMIN_COMMENT_INFORMATION(NDRSTRUCT):
    structure = (('AdminComment', RPC_UNICODE_STRING),)

class SAMPR_USER_WORKSTATIONS_INFORMATION(NDRSTRUCT):
    structure = (('WorkStations', RPC_UNICODE_STRING),)

class SAMPR_USER_LOGON_HOURS_INFORMATION(NDRSTRUCT):
    structure = (('LogonHours', SAMPR_LOGON_HOURS),)

class SAMPR_USER_PASSWORD(NDRSTRUCT):
    structure = (('Buffer', '512s=b""'), ('Length', ULONG))

    def getAlignment(self):
        if False:
            return 10
        return 4

class SAMPR_ENCRYPTED_USER_PASSWORD(NDRSTRUCT):
    structure = (('Buffer', '516s=b""'),)

    def getAlignment(self):
        if False:
            print('Hello World!')
        return 1

class PSAMPR_ENCRYPTED_USER_PASSWORD(NDRPOINTER):
    referent = (('Data', SAMPR_ENCRYPTED_USER_PASSWORD),)

class SAMPR_ENCRYPTED_USER_PASSWORD_NEW(NDRSTRUCT):
    structure = (('Buffer', '532s=b""'),)

    def getAlignment(self):
        if False:
            print('Hello World!')
        return 1

class SAMPR_USER_INTERNAL1_INFORMATION(NDRSTRUCT):
    structure = (('EncryptedNtOwfPassword', ENCRYPTED_NT_OWF_PASSWORD), ('EncryptedLmOwfPassword', ENCRYPTED_LM_OWF_PASSWORD), ('NtPasswordPresent', UCHAR), ('LmPasswordPresent', UCHAR), ('PasswordExpired', UCHAR))

class SAMPR_USER_INTERNAL4_INFORMATION(NDRSTRUCT):
    structure = (('I1', SAMPR_USER_ALL_INFORMATION), ('UserPassword', SAMPR_ENCRYPTED_USER_PASSWORD))

class SAMPR_USER_INTERNAL4_INFORMATION_NEW(NDRSTRUCT):
    structure = (('I1', SAMPR_USER_ALL_INFORMATION), ('UserPassword', SAMPR_ENCRYPTED_USER_PASSWORD_NEW))

class SAMPR_USER_INTERNAL5_INFORMATION(NDRSTRUCT):
    structure = (('UserPassword', SAMPR_ENCRYPTED_USER_PASSWORD), ('PasswordExpired', UCHAR))

class SAMPR_USER_INTERNAL5_INFORMATION_NEW(NDRSTRUCT):
    structure = (('UserPassword', SAMPR_ENCRYPTED_USER_PASSWORD_NEW), ('PasswordExpired', UCHAR))

class USER_INFORMATION_CLASS(NDRENUM):

    class enumItems(Enum):
        UserGeneralInformation = 1
        UserPreferencesInformation = 2
        UserLogonInformation = 3
        UserLogonHoursInformation = 4
        UserAccountInformation = 5
        UserNameInformation = 6
        UserAccountNameInformation = 7
        UserFullNameInformation = 8
        UserPrimaryGroupInformation = 9
        UserHomeInformation = 10
        UserScriptInformation = 11
        UserProfileInformation = 12
        UserAdminCommentInformation = 13
        UserWorkStationsInformation = 14
        UserControlInformation = 16
        UserExpiresInformation = 17
        UserInternal1Information = 18
        UserParametersInformation = 20
        UserAllInformation = 21
        UserInternal4Information = 23
        UserInternal5Information = 24
        UserInternal4InformationNew = 25
        UserInternal5InformationNew = 26

class SAMPR_USER_INFO_BUFFER(NDRUNION):
    union = {USER_INFORMATION_CLASS.UserGeneralInformation: ('General', SAMPR_USER_GENERAL_INFORMATION), USER_INFORMATION_CLASS.UserPreferencesInformation: ('Preferences', SAMPR_USER_PREFERENCES_INFORMATION), USER_INFORMATION_CLASS.UserLogonInformation: ('Logon', SAMPR_USER_LOGON_INFORMATION), USER_INFORMATION_CLASS.UserLogonHoursInformation: ('LogonHours', SAMPR_USER_LOGON_HOURS_INFORMATION), USER_INFORMATION_CLASS.UserAccountInformation: ('Account', SAMPR_USER_ACCOUNT_INFORMATION), USER_INFORMATION_CLASS.UserNameInformation: ('Name', SAMPR_USER_NAME_INFORMATION), USER_INFORMATION_CLASS.UserAccountNameInformation: ('AccountName', SAMPR_USER_A_NAME_INFORMATION), USER_INFORMATION_CLASS.UserFullNameInformation: ('FullName', SAMPR_USER_F_NAME_INFORMATION), USER_INFORMATION_CLASS.UserPrimaryGroupInformation: ('PrimaryGroup', USER_PRIMARY_GROUP_INFORMATION), USER_INFORMATION_CLASS.UserHomeInformation: ('Home', SAMPR_USER_HOME_INFORMATION), USER_INFORMATION_CLASS.UserScriptInformation: ('Script', SAMPR_USER_SCRIPT_INFORMATION), USER_INFORMATION_CLASS.UserProfileInformation: ('Profile', SAMPR_USER_PROFILE_INFORMATION), USER_INFORMATION_CLASS.UserAdminCommentInformation: ('AdminComment', SAMPR_USER_ADMIN_COMMENT_INFORMATION), USER_INFORMATION_CLASS.UserWorkStationsInformation: ('WorkStations', SAMPR_USER_WORKSTATIONS_INFORMATION), USER_INFORMATION_CLASS.UserControlInformation: ('Control', USER_CONTROL_INFORMATION), USER_INFORMATION_CLASS.UserExpiresInformation: ('Expires', USER_EXPIRES_INFORMATION), USER_INFORMATION_CLASS.UserInternal1Information: ('Internal1', SAMPR_USER_INTERNAL1_INFORMATION), USER_INFORMATION_CLASS.UserParametersInformation: ('Parameters', SAMPR_USER_PARAMETERS_INFORMATION), USER_INFORMATION_CLASS.UserAllInformation: ('All', SAMPR_USER_ALL_INFORMATION), USER_INFORMATION_CLASS.UserInternal4Information: ('Internal4', SAMPR_USER_INTERNAL4_INFORMATION), USER_INFORMATION_CLASS.UserInternal5Information: ('Internal5', SAMPR_USER_INTERNAL5_INFORMATION), USER_INFORMATION_CLASS.UserInternal4InformationNew: ('Internal4New', SAMPR_USER_INTERNAL4_INFORMATION_NEW), USER_INFORMATION_CLASS.UserInternal5InformationNew: ('Internal5New', SAMPR_USER_INTERNAL5_INFORMATION_NEW)}

class PSAMPR_USER_INFO_BUFFER(NDRPOINTER):
    referent = (('Data', SAMPR_USER_INFO_BUFFER),)

class PSAMPR_SERVER_NAME2(NDRPOINTER):
    referent = (('Data', '4s=b""'),)

class SAMPR_DOMAIN_DISPLAY_USER(NDRSTRUCT):
    structure = (('Index', ULONG), ('Rid', ULONG), ('AccountControl', ULONG), ('AccountName', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING), ('FullName', RPC_UNICODE_STRING))

class SAMPR_DOMAIN_DISPLAY_USER_ARRAY(NDRUniConformantArray):
    item = SAMPR_DOMAIN_DISPLAY_USER

class PSAMPR_DOMAIN_DISPLAY_USER_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_DISPLAY_USER_ARRAY),)

class SAMPR_DOMAIN_DISPLAY_MACHINE(NDRSTRUCT):
    structure = (('Index', ULONG), ('Rid', ULONG), ('AccountControl', ULONG), ('AccountName', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING))

class SAMPR_DOMAIN_DISPLAY_MACHINE_ARRAY(NDRUniConformantArray):
    item = SAMPR_DOMAIN_DISPLAY_MACHINE

class PSAMPR_DOMAIN_DISPLAY_MACHINE_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_DISPLAY_MACHINE_ARRAY),)

class SAMPR_DOMAIN_DISPLAY_GROUP(NDRSTRUCT):
    structure = (('Index', ULONG), ('Rid', ULONG), ('AccountControl', ULONG), ('AccountName', RPC_UNICODE_STRING), ('AdminComment', RPC_UNICODE_STRING))

class SAMPR_DOMAIN_DISPLAY_GROUP_ARRAY(NDRUniConformantArray):
    item = SAMPR_DOMAIN_DISPLAY_GROUP

class PSAMPR_DOMAIN_DISPLAY_GROUP_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_DISPLAY_GROUP_ARRAY),)

class SAMPR_DOMAIN_DISPLAY_OEM_USER(NDRSTRUCT):
    structure = (('Index', ULONG), ('OemAccountName', RPC_STRING))

class SAMPR_DOMAIN_DISPLAY_OEM_USER_ARRAY(NDRUniConformantArray):
    item = SAMPR_DOMAIN_DISPLAY_OEM_USER

class PSAMPR_DOMAIN_DISPLAY_OEM_USER_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_DISPLAY_OEM_USER_ARRAY),)

class SAMPR_DOMAIN_DISPLAY_OEM_GROUP(NDRSTRUCT):
    structure = (('Index', ULONG), ('OemAccountName', RPC_STRING))

class SAMPR_DOMAIN_DISPLAY_OEM_GROUP_ARRAY(NDRUniConformantArray):
    item = SAMPR_DOMAIN_DISPLAY_OEM_GROUP

class PSAMPR_DOMAIN_DISPLAY_OEM_GROUP_ARRAY(NDRPOINTER):
    referent = (('Data', SAMPR_DOMAIN_DISPLAY_OEM_GROUP_ARRAY),)

class SAMPR_DOMAIN_DISPLAY_USER_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_DOMAIN_DISPLAY_USER_ARRAY))

class SAMPR_DOMAIN_DISPLAY_MACHINE_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_DOMAIN_DISPLAY_MACHINE_ARRAY))

class SAMPR_DOMAIN_DISPLAY_GROUP_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_DOMAIN_DISPLAY_GROUP_ARRAY))

class SAMPR_DOMAIN_DISPLAY_OEM_USER_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_DOMAIN_DISPLAY_OEM_USER_ARRAY))

class SAMPR_DOMAIN_DISPLAY_OEM_GROUP_BUFFER(NDRSTRUCT):
    structure = (('EntriesRead', ULONG), ('Buffer', PSAMPR_DOMAIN_DISPLAY_OEM_GROUP_ARRAY))

class DOMAIN_DISPLAY_INFORMATION(NDRENUM):

    class enumItems(Enum):
        DomainDisplayUser = 1
        DomainDisplayMachine = 2
        DomainDisplayGroup = 3
        DomainDisplayOemUser = 4
        DomainDisplayOemGroup = 5

class SAMPR_DISPLAY_INFO_BUFFER(NDRUNION):
    union = {DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser: ('UserInformation', SAMPR_DOMAIN_DISPLAY_USER_BUFFER), DOMAIN_DISPLAY_INFORMATION.DomainDisplayMachine: ('MachineInformation', SAMPR_DOMAIN_DISPLAY_MACHINE_BUFFER), DOMAIN_DISPLAY_INFORMATION.DomainDisplayGroup: ('GroupInformation', SAMPR_DOMAIN_DISPLAY_GROUP_BUFFER), DOMAIN_DISPLAY_INFORMATION.DomainDisplayOemUser: ('OemUserInformation', SAMPR_DOMAIN_DISPLAY_OEM_USER_BUFFER), DOMAIN_DISPLAY_INFORMATION.DomainDisplayOemGroup: ('OemGroupInformation', SAMPR_DOMAIN_DISPLAY_OEM_GROUP_BUFFER)}

class SAM_VALIDATE_PASSWORD_HASH(NDRSTRUCT):
    structure = (('Length', ULONG), ('Hash', LPBYTE))

class PSAM_VALIDATE_PASSWORD_HASH(NDRPOINTER):
    referent = (('Data', SAM_VALIDATE_PASSWORD_HASH),)

class SAM_VALIDATE_PERSISTED_FIELDS(NDRSTRUCT):
    structure = (('PresentFields', ULONG), ('PasswordLastSet', LARGE_INTEGER), ('BadPasswordTime', LARGE_INTEGER), ('LockoutTime', LARGE_INTEGER), ('BadPasswordCount', ULONG), ('PasswordHistoryLength', ULONG), ('PasswordHistory', PSAM_VALIDATE_PASSWORD_HASH))

class SAM_VALIDATE_VALIDATION_STATUS(NDRENUM):

    class enumItems(Enum):
        SamValidateSuccess = 0
        SamValidatePasswordMustChange = 1
        SamValidateAccountLockedOut = 2
        SamValidatePasswordExpired = 3
        SamValidatePasswordIncorrect = 4
        SamValidatePasswordIsInHistory = 5
        SamValidatePasswordTooShort = 6
        SamValidatePasswordTooLong = 7
        SamValidatePasswordNotComplexEnough = 8
        SamValidatePasswordTooRecent = 9
        SamValidatePasswordFilterError = 10

class SAM_VALIDATE_STANDARD_OUTPUT_ARG(NDRSTRUCT):
    structure = (('ChangedPersistedFields', SAM_VALIDATE_PERSISTED_FIELDS), ('ValidationStatus', SAM_VALIDATE_VALIDATION_STATUS))

class PSAM_VALIDATE_STANDARD_OUTPUT_ARG(NDRPOINTER):
    referent = (('Data', SAM_VALIDATE_STANDARD_OUTPUT_ARG),)

class SAM_VALIDATE_AUTHENTICATION_INPUT_ARG(NDRSTRUCT):
    structure = (('InputPersistedFields', SAM_VALIDATE_PERSISTED_FIELDS), ('PasswordMatched', UCHAR))

class SAM_VALIDATE_PASSWORD_CHANGE_INPUT_ARG(NDRSTRUCT):
    structure = (('InputPersistedFields', SAM_VALIDATE_PERSISTED_FIELDS), ('ClearPassword', RPC_UNICODE_STRING), ('UserAccountName', RPC_UNICODE_STRING), ('HashedPassword', SAM_VALIDATE_PASSWORD_HASH), ('PasswordMatch', UCHAR))

class SAM_VALIDATE_PASSWORD_RESET_INPUT_ARG(NDRSTRUCT):
    structure = (('InputPersistedFields', SAM_VALIDATE_PERSISTED_FIELDS), ('ClearPassword', RPC_UNICODE_STRING), ('UserAccountName', RPC_UNICODE_STRING), ('HashedPassword', SAM_VALIDATE_PASSWORD_HASH), ('PasswordMustChangeAtNextLogon', UCHAR), ('ClearLockout', UCHAR))

class PASSWORD_POLICY_VALIDATION_TYPE(NDRENUM):

    class enumItems(Enum):
        SamValidateAuthentication = 1
        SamValidatePasswordChange = 2
        SamValidatePasswordReset = 3

class SAM_VALIDATE_INPUT_ARG(NDRUNION):
    union = {PASSWORD_POLICY_VALIDATION_TYPE.SamValidateAuthentication: ('ValidateAuthenticationInput', SAM_VALIDATE_AUTHENTICATION_INPUT_ARG), PASSWORD_POLICY_VALIDATION_TYPE.SamValidatePasswordChange: ('ValidatePasswordChangeInput', SAM_VALIDATE_PASSWORD_CHANGE_INPUT_ARG), PASSWORD_POLICY_VALIDATION_TYPE.SamValidatePasswordReset: ('ValidatePasswordResetInput', SAM_VALIDATE_PASSWORD_RESET_INPUT_ARG)}

class SAM_VALIDATE_OUTPUT_ARG(NDRUNION):
    union = {PASSWORD_POLICY_VALIDATION_TYPE.SamValidateAuthentication: ('ValidateAuthenticationOutput', SAM_VALIDATE_STANDARD_OUTPUT_ARG), PASSWORD_POLICY_VALIDATION_TYPE.SamValidatePasswordChange: ('ValidatePasswordChangeOutput', SAM_VALIDATE_STANDARD_OUTPUT_ARG), PASSWORD_POLICY_VALIDATION_TYPE.SamValidatePasswordReset: ('ValidatePasswordResetOutput', SAM_VALIDATE_STANDARD_OUTPUT_ARG)}

class PSAM_VALIDATE_OUTPUT_ARG(NDRPOINTER):
    referent = (('Data', SAM_VALIDATE_OUTPUT_ARG),)

class USER_PROPERTIES(Structure):
    structure = (('Reserved1', '<L=0'), ('Length', '<L=0'), ('Reserved2', '<H=0'), ('Reserved3', '<H=0'), ('Reserved4', '96s=""'), ('PropertySignature', '<H=0x50'), ('PropertyCount', '<H=0'), ('UserProperties', ':'))

class USER_PROPERTY(Structure):
    structure = (('NameLength', '<H=0'), ('ValueLength', '<H=0'), ('Reserved', '<H=0'), ('_PropertyName', '_-PropertyName', "self['NameLength']"), ('PropertyName', ':'), ('_PropertyValue', '_-PropertyValue', "self['ValueLength']"), ('PropertyValue', ':'))

class WDIGEST_CREDENTIALS(Structure):
    structure = (('Reserved1', 'B=0'), ('Reserved2', 'B=0'), ('Version', 'B=1'), ('NumberOfHashes', 'B=29'), ('Reserved3', '12s=""'), ('Hash1', '16s=""'), ('Hash2', '16s=""'), ('Hash3', '16s=""'), ('Hash4', '16s=""'), ('Hash5', '16s=""'), ('Hash6', '16s=""'), ('Hash7', '16s=""'), ('Hash8', '16s=""'), ('Hash9', '16s=""'), ('Hash10', '16s=""'), ('Hash11', '16s=""'), ('Hash12', '16s=""'), ('Hash13', '16s=""'), ('Hash14', '16s=""'), ('Hash15', '16s=""'), ('Hash16', '16s=""'), ('Hash17', '16s=""'), ('Hash18', '16s=""'), ('Hash19', '16s=""'), ('Hash20', '16s=""'), ('Hash21', '16s=""'), ('Hash22', '16s=""'), ('Hash23', '16s=""'), ('Hash24', '16s=""'), ('Hash25', '16s=""'), ('Hash26', '16s=""'), ('Hash27', '16s=""'), ('Hash28', '16s=""'), ('Hash29', '16s=""'))

class KERB_KEY_DATA(Structure):
    structure = (('Reserved1', '<H=0'), ('Reserved2', '<H=0'), ('Reserved3', '<H=0'), ('KeyType', '<L=0'), ('KeyLength', '<L=0'), ('KeyOffset', '<L=0'))

class KERB_STORED_CREDENTIAL(Structure):
    structure = (('Revision', '<H=3'), ('Flags', '<H=0'), ('CredentialCount', '<H=0'), ('OldCredentialCount', '<H=0'), ('DefaultSaltLength', '<H=0'), ('DefaultSaltMaximumLength', '<H=0'), ('DefaultSaltOffset', '<L=0'), ('Buffer', ':'))

class KERB_KEY_DATA_NEW(Structure):
    structure = (('Reserved1', '<H=0'), ('Reserved2', '<H=0'), ('Reserved3', '<L=0'), ('IterationCount', '<L=0'), ('KeyType', '<L=0'), ('KeyLength', '<L=0'), ('KeyOffset', '<L=0'))

class KERB_STORED_CREDENTIAL_NEW(Structure):
    structure = (('Revision', '<H=4'), ('Flags', '<H=0'), ('CredentialCount', '<H=0'), ('ServiceCredentialCount', '<H=0'), ('OldCredentialCount', '<H=0'), ('OlderCredentialCount', '<H=0'), ('DefaultSaltLength', '<H=0'), ('DefaultSaltMaximumLength', '<H=0'), ('DefaultSaltOffset', '<L=0'), ('DefaultIterationCount', '<L=0'), ('Buffer', ':'))

class SamrConnect(NDRCALL):
    opnum = 0
    structure = (('ServerName', PSAMPR_SERVER_NAME2), ('DesiredAccess', ULONG))

class SamrConnectResponse(NDRCALL):
    structure = (('ServerHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrCloseHandle(NDRCALL):
    opnum = 1
    structure = (('SamHandle', SAMPR_HANDLE), ('DesiredAccess', LONG))

class SamrCloseHandleResponse(NDRCALL):
    structure = (('SamHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrSetSecurityObject(NDRCALL):
    opnum = 2
    structure = (('ObjectHandle', SAMPR_HANDLE), ('SecurityInformation', SECURITY_INFORMATION), ('SecurityDescriptor', SAMPR_SR_SECURITY_DESCRIPTOR))

class SamrSetSecurityObjectResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrQuerySecurityObject(NDRCALL):
    opnum = 3
    structure = (('ObjectHandle', SAMPR_HANDLE), ('SecurityInformation', SECURITY_INFORMATION))

class SamrQuerySecurityObjectResponse(NDRCALL):
    structure = (('SecurityDescriptor', PSAMPR_SR_SECURITY_DESCRIPTOR), ('ErrorCode', ULONG))

class SamrLookupDomainInSamServer(NDRCALL):
    opnum = 5
    structure = (('ServerHandle', SAMPR_HANDLE), ('Name', RPC_UNICODE_STRING))

class SamrLookupDomainInSamServerResponse(NDRCALL):
    structure = (('DomainId', PRPC_SID), ('ErrorCode', ULONG))

class SamrEnumerateDomainsInSamServer(NDRCALL):
    opnum = 6
    structure = (('ServerHandle', SAMPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class SamrEnumerateDomainsInSamServerResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('Buffer', PSAMPR_ENUMERATION_BUFFER), ('CountReturned', ULONG), ('ErrorCode', ULONG))

class SamrOpenDomain(NDRCALL):
    opnum = 7
    structure = (('ServerHandle', SAMPR_HANDLE), ('DesiredAccess', ULONG), ('DomainId', RPC_SID))

class SamrOpenDomainResponse(NDRCALL):
    structure = (('DomainHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrQueryInformationDomain(NDRCALL):
    opnum = 8
    structure = (('DomainHandle', SAMPR_HANDLE), ('DomainInformationClass', DOMAIN_INFORMATION_CLASS))

class SamrQueryInformationDomainResponse(NDRCALL):
    structure = (('Buffer', PSAMPR_DOMAIN_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrSetInformationDomain(NDRCALL):
    opnum = 9
    structure = (('DomainHandle', SAMPR_HANDLE), ('DomainInformationClass', DOMAIN_INFORMATION_CLASS), ('DomainInformation', SAMPR_DOMAIN_INFO_BUFFER))

class SamrSetInformationDomainResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrCreateGroupInDomain(NDRCALL):
    opnum = 10
    structure = (('DomainHandle', SAMPR_HANDLE), ('Name', RPC_UNICODE_STRING), ('DesiredAccess', ULONG))

class SamrCreateGroupInDomainResponse(NDRCALL):
    structure = (('GroupHandle', SAMPR_HANDLE), ('RelativeId', ULONG), ('ErrorCode', ULONG))

class SamrEnumerateGroupsInDomain(NDRCALL):
    opnum = 11
    structure = (('DomainHandle', SAMPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class SamrCreateUserInDomain(NDRCALL):
    opnum = 12
    structure = (('DomainHandle', SAMPR_HANDLE), ('Name', RPC_UNICODE_STRING), ('DesiredAccess', ULONG))

class SamrCreateUserInDomainResponse(NDRCALL):
    structure = (('UserHandle', SAMPR_HANDLE), ('RelativeId', ULONG), ('ErrorCode', ULONG))

class SamrEnumerateGroupsInDomainResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('Buffer', PSAMPR_ENUMERATION_BUFFER), ('CountReturned', ULONG), ('ErrorCode', ULONG))

class SamrEnumerateUsersInDomain(NDRCALL):
    opnum = 13
    structure = (('DomainHandle', SAMPR_HANDLE), ('EnumerationContext', ULONG), ('UserAccountControl', ULONG), ('PreferedMaximumLength', ULONG))

class SamrEnumerateUsersInDomainResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('Buffer', PSAMPR_ENUMERATION_BUFFER), ('CountReturned', ULONG), ('ErrorCode', ULONG))

class SamrCreateAliasInDomain(NDRCALL):
    opnum = 14
    structure = (('DomainHandle', SAMPR_HANDLE), ('AccountName', RPC_UNICODE_STRING), ('DesiredAccess', ULONG))

class SamrCreateAliasInDomainResponse(NDRCALL):
    structure = (('AliasHandle', SAMPR_HANDLE), ('RelativeId', ULONG), ('ErrorCode', ULONG))

class SamrEnumerateAliasesInDomain(NDRCALL):
    opnum = 15
    structure = (('DomainHandle', SAMPR_HANDLE), ('EnumerationContext', ULONG), ('PreferedMaximumLength', ULONG))

class SamrEnumerateAliasesInDomainResponse(NDRCALL):
    structure = (('EnumerationContext', ULONG), ('Buffer', PSAMPR_ENUMERATION_BUFFER), ('CountReturned', ULONG), ('ErrorCode', ULONG))

class SamrGetAliasMembership(NDRCALL):
    opnum = 16
    structure = (('DomainHandle', SAMPR_HANDLE), ('SidArray', SAMPR_PSID_ARRAY))

class SamrGetAliasMembershipResponse(NDRCALL):
    structure = (('Membership', SAMPR_ULONG_ARRAY), ('ErrorCode', ULONG))

class SamrLookupNamesInDomain(NDRCALL):
    opnum = 17
    structure = (('DomainHandle', SAMPR_HANDLE), ('Count', ULONG), ('Names', RPC_UNICODE_STRING_ARRAY))

class SamrLookupNamesInDomainResponse(NDRCALL):
    structure = (('RelativeIds', SAMPR_ULONG_ARRAY), ('Use', SAMPR_ULONG_ARRAY), ('ErrorCode', ULONG))

class SamrLookupIdsInDomain(NDRCALL):
    opnum = 18
    structure = (('DomainHandle', SAMPR_HANDLE), ('Count', ULONG), ('RelativeIds', ULONG_ARRAY_CV))

class SamrLookupIdsInDomainResponse(NDRCALL):
    structure = (('Names', SAMPR_RETURNED_USTRING_ARRAY), ('Use', SAMPR_ULONG_ARRAY), ('ErrorCode', ULONG))

class SamrOpenGroup(NDRCALL):
    opnum = 19
    structure = (('DomainHandle', SAMPR_HANDLE), ('DesiredAccess', ULONG), ('GroupId', ULONG))

class SamrOpenGroupResponse(NDRCALL):
    structure = (('GroupHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrQueryInformationGroup(NDRCALL):
    opnum = 20
    structure = (('GroupHandle', SAMPR_HANDLE), ('GroupInformationClass', GROUP_INFORMATION_CLASS))

class SamrQueryInformationGroupResponse(NDRCALL):
    structure = (('Buffer', PSAMPR_GROUP_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrSetInformationGroup(NDRCALL):
    opnum = 21
    structure = (('GroupHandle', SAMPR_HANDLE), ('GroupInformationClass', GROUP_INFORMATION_CLASS), ('Buffer', SAMPR_GROUP_INFO_BUFFER))

class SamrSetInformationGroupResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrAddMemberToGroup(NDRCALL):
    opnum = 22
    structure = (('GroupHandle', SAMPR_HANDLE), ('MemberId', ULONG), ('Attributes', ULONG))

class SamrAddMemberToGroupResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrDeleteGroup(NDRCALL):
    opnum = 23
    structure = (('GroupHandle', SAMPR_HANDLE),)

class SamrDeleteGroupResponse(NDRCALL):
    structure = (('GroupHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrRemoveMemberFromGroup(NDRCALL):
    opnum = 24
    structure = (('GroupHandle', SAMPR_HANDLE), ('MemberId', ULONG))

class SamrRemoveMemberFromGroupResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrGetMembersInGroup(NDRCALL):
    opnum = 25
    structure = (('GroupHandle', SAMPR_HANDLE),)

class SamrGetMembersInGroupResponse(NDRCALL):
    structure = (('Members', PSAMPR_GET_MEMBERS_BUFFER), ('ErrorCode', ULONG))

class SamrSetMemberAttributesOfGroup(NDRCALL):
    opnum = 26
    structure = (('GroupHandle', SAMPR_HANDLE), ('MemberId', ULONG), ('Attributes', ULONG))

class SamrSetMemberAttributesOfGroupResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrOpenAlias(NDRCALL):
    opnum = 27
    structure = (('DomainHandle', SAMPR_HANDLE), ('DesiredAccess', ULONG), ('AliasId', ULONG))

class SamrOpenAliasResponse(NDRCALL):
    structure = (('AliasHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrQueryInformationAlias(NDRCALL):
    opnum = 28
    structure = (('AliasHandle', SAMPR_HANDLE), ('AliasInformationClass', ALIAS_INFORMATION_CLASS))

class SamrQueryInformationAliasResponse(NDRCALL):
    structure = (('Buffer', PSAMPR_ALIAS_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrSetInformationAlias(NDRCALL):
    opnum = 29
    structure = (('AliasHandle', SAMPR_HANDLE), ('AliasInformationClass', ALIAS_INFORMATION_CLASS), ('Buffer', SAMPR_ALIAS_INFO_BUFFER))

class SamrSetInformationAliasResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrDeleteAlias(NDRCALL):
    opnum = 30
    structure = (('AliasHandle', SAMPR_HANDLE),)

class SamrDeleteAliasResponse(NDRCALL):
    structure = (('AliasHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrAddMemberToAlias(NDRCALL):
    opnum = 31
    structure = (('AliasHandle', SAMPR_HANDLE), ('MemberId', RPC_SID))

class SamrAddMemberToAliasResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrRemoveMemberFromAlias(NDRCALL):
    opnum = 32
    structure = (('AliasHandle', SAMPR_HANDLE), ('MemberId', RPC_SID))

class SamrRemoveMemberFromAliasResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrGetMembersInAlias(NDRCALL):
    opnum = 33
    structure = (('AliasHandle', SAMPR_HANDLE),)

class SamrGetMembersInAliasResponse(NDRCALL):
    structure = (('Members', SAMPR_PSID_ARRAY_OUT), ('ErrorCode', ULONG))

class SamrOpenUser(NDRCALL):
    opnum = 34
    structure = (('DomainHandle', SAMPR_HANDLE), ('DesiredAccess', ULONG), ('UserId', ULONG))

class SamrOpenUserResponse(NDRCALL):
    structure = (('UserHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrDeleteUser(NDRCALL):
    opnum = 35
    structure = (('UserHandle', SAMPR_HANDLE),)

class SamrDeleteUserResponse(NDRCALL):
    structure = (('UserHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrQueryInformationUser(NDRCALL):
    opnum = 36
    structure = (('UserHandle', SAMPR_HANDLE), ('UserInformationClass', USER_INFORMATION_CLASS))

class SamrQueryInformationUserResponse(NDRCALL):
    structure = (('Buffer', PSAMPR_USER_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrSetInformationUser(NDRCALL):
    opnum = 37
    structure = (('UserHandle', SAMPR_HANDLE), ('UserInformationClass', USER_INFORMATION_CLASS), ('Buffer', SAMPR_USER_INFO_BUFFER))

class SamrSetInformationUserResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrChangePasswordUser(NDRCALL):
    opnum = 38
    structure = (('UserHandle', SAMPR_HANDLE), ('LmPresent', UCHAR), ('OldLmEncryptedWithNewLm', PENCRYPTED_LM_OWF_PASSWORD), ('NewLmEncryptedWithOldLm', PENCRYPTED_LM_OWF_PASSWORD), ('NtPresent', UCHAR), ('OldNtEncryptedWithNewNt', PENCRYPTED_NT_OWF_PASSWORD), ('NewNtEncryptedWithOldNt', PENCRYPTED_NT_OWF_PASSWORD), ('NtCrossEncryptionPresent', UCHAR), ('NewNtEncryptedWithNewLm', PENCRYPTED_NT_OWF_PASSWORD), ('LmCrossEncryptionPresent', UCHAR), ('NewLmEncryptedWithNewNt', PENCRYPTED_NT_OWF_PASSWORD))

class SamrChangePasswordUserResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrGetGroupsForUser(NDRCALL):
    opnum = 39
    structure = (('UserHandle', SAMPR_HANDLE),)

class SamrGetGroupsForUserResponse(NDRCALL):
    structure = (('Groups', PSAMPR_GET_GROUPS_BUFFER), ('ErrorCode', ULONG))

class SamrQueryDisplayInformation(NDRCALL):
    opnum = 40
    structure = (('DomainHandle', SAMPR_HANDLE), ('DisplayInformationClass', DOMAIN_DISPLAY_INFORMATION), ('Index', ULONG), ('EntryCount', ULONG), ('PreferredMaximumLength', ULONG))

class SamrQueryDisplayInformationResponse(NDRCALL):
    structure = (('TotalAvailable', ULONG), ('TotalReturned', ULONG), ('Buffer', SAMPR_DISPLAY_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrGetDisplayEnumerationIndex(NDRCALL):
    opnum = 41
    structure = (('DomainHandle', SAMPR_HANDLE), ('DisplayInformationClass', DOMAIN_DISPLAY_INFORMATION), ('Prefix', RPC_UNICODE_STRING))

class SamrGetDisplayEnumerationIndexResponse(NDRCALL):
    structure = (('Index', ULONG), ('ErrorCode', ULONG))

class SamrGetUserDomainPasswordInformation(NDRCALL):
    opnum = 44
    structure = (('UserHandle', SAMPR_HANDLE),)

class SamrGetUserDomainPasswordInformationResponse(NDRCALL):
    structure = (('PasswordInformation', USER_DOMAIN_PASSWORD_INFORMATION), ('ErrorCode', ULONG))

class SamrRemoveMemberFromForeignDomain(NDRCALL):
    opnum = 45
    structure = (('DomainHandle', SAMPR_HANDLE), ('MemberSid', RPC_SID))

class SamrRemoveMemberFromForeignDomainResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrQueryInformationDomain2(NDRCALL):
    opnum = 46
    structure = (('DomainHandle', SAMPR_HANDLE), ('DomainInformationClass', DOMAIN_INFORMATION_CLASS))

class SamrQueryInformationDomain2Response(NDRCALL):
    structure = (('Buffer', PSAMPR_DOMAIN_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrQueryInformationUser2(NDRCALL):
    opnum = 47
    structure = (('UserHandle', SAMPR_HANDLE), ('UserInformationClass', USER_INFORMATION_CLASS))

class SamrQueryInformationUser2Response(NDRCALL):
    structure = (('Buffer', PSAMPR_USER_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrQueryDisplayInformation2(NDRCALL):
    opnum = 48
    structure = (('DomainHandle', SAMPR_HANDLE), ('DisplayInformationClass', DOMAIN_DISPLAY_INFORMATION), ('Index', ULONG), ('EntryCount', ULONG), ('PreferredMaximumLength', ULONG))

class SamrQueryDisplayInformation2Response(NDRCALL):
    structure = (('TotalAvailable', ULONG), ('TotalReturned', ULONG), ('Buffer', SAMPR_DISPLAY_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrGetDisplayEnumerationIndex2(NDRCALL):
    opnum = 49
    structure = (('DomainHandle', SAMPR_HANDLE), ('DisplayInformationClass', DOMAIN_DISPLAY_INFORMATION), ('Prefix', RPC_UNICODE_STRING))

class SamrGetDisplayEnumerationIndex2Response(NDRCALL):
    structure = (('Index', ULONG), ('ErrorCode', ULONG))

class SamrCreateUser2InDomain(NDRCALL):
    opnum = 50
    structure = (('DomainHandle', SAMPR_HANDLE), ('Name', RPC_UNICODE_STRING), ('AccountType', ULONG), ('DesiredAccess', ULONG))

class SamrCreateUser2InDomainResponse(NDRCALL):
    structure = (('UserHandle', SAMPR_HANDLE), ('GrantedAccess', ULONG), ('RelativeId', ULONG), ('ErrorCode', ULONG))

class SamrQueryDisplayInformation3(NDRCALL):
    opnum = 51
    structure = (('DomainHandle', SAMPR_HANDLE), ('DisplayInformationClass', DOMAIN_DISPLAY_INFORMATION), ('Index', ULONG), ('EntryCount', ULONG), ('PreferredMaximumLength', ULONG))

class SamrQueryDisplayInformation3Response(NDRCALL):
    structure = (('TotalAvailable', ULONG), ('TotalReturned', ULONG), ('Buffer', SAMPR_DISPLAY_INFO_BUFFER), ('ErrorCode', ULONG))

class SamrAddMultipleMembersToAlias(NDRCALL):
    opnum = 52
    structure = (('AliasHandle', SAMPR_HANDLE), ('MembersBuffer', SAMPR_PSID_ARRAY))

class SamrAddMultipleMembersToAliasResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrRemoveMultipleMembersFromAlias(NDRCALL):
    opnum = 53
    structure = (('AliasHandle', SAMPR_HANDLE), ('MembersBuffer', SAMPR_PSID_ARRAY))

class SamrRemoveMultipleMembersFromAliasResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrOemChangePasswordUser2(NDRCALL):
    opnum = 54
    structure = (('ServerName', PRPC_STRING), ('UserName', RPC_STRING), ('NewPasswordEncryptedWithOldLm', PSAMPR_ENCRYPTED_USER_PASSWORD), ('OldLmOwfPasswordEncryptedWithNewLm', PENCRYPTED_LM_OWF_PASSWORD))

class SamrOemChangePasswordUser2Response(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrUnicodeChangePasswordUser2(NDRCALL):
    opnum = 55
    structure = (('ServerName', PRPC_UNICODE_STRING), ('UserName', RPC_UNICODE_STRING), ('NewPasswordEncryptedWithOldNt', PSAMPR_ENCRYPTED_USER_PASSWORD), ('OldNtOwfPasswordEncryptedWithNewNt', PENCRYPTED_NT_OWF_PASSWORD), ('LmPresent', UCHAR), ('NewPasswordEncryptedWithOldLm', PSAMPR_ENCRYPTED_USER_PASSWORD), ('OldLmOwfPasswordEncryptedWithNewNt', PENCRYPTED_LM_OWF_PASSWORD))

class SamrUnicodeChangePasswordUser2Response(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrGetDomainPasswordInformation(NDRCALL):
    opnum = 56
    structure = (('Unused', PRPC_UNICODE_STRING),)

class SamrGetDomainPasswordInformationResponse(NDRCALL):
    structure = (('PasswordInformation', USER_DOMAIN_PASSWORD_INFORMATION), ('ErrorCode', ULONG))

class SamrConnect2(NDRCALL):
    opnum = 57
    structure = (('ServerName', PSAMPR_SERVER_NAME), ('DesiredAccess', ULONG))

class SamrConnect2Response(NDRCALL):
    structure = (('ServerHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrSetInformationUser2(NDRCALL):
    opnum = 58
    structure = (('UserHandle', SAMPR_HANDLE), ('UserInformationClass', USER_INFORMATION_CLASS), ('Buffer', SAMPR_USER_INFO_BUFFER))

class SamrSetInformationUser2Response(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrConnect4(NDRCALL):
    opnum = 62
    structure = (('ServerName', PSAMPR_SERVER_NAME), ('ClientRevision', ULONG), ('DesiredAccess', ULONG))

class SamrConnect4Response(NDRCALL):
    structure = (('ServerHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrConnect5(NDRCALL):
    opnum = 64
    structure = (('ServerName', PSAMPR_SERVER_NAME), ('DesiredAccess', ULONG), ('InVersion', ULONG), ('InRevisionInfo', SAMPR_REVISION_INFO))

class SamrConnect5Response(NDRCALL):
    structure = (('OutVersion', ULONG), ('OutRevisionInfo', SAMPR_REVISION_INFO), ('ServerHandle', SAMPR_HANDLE), ('ErrorCode', ULONG))

class SamrRidToSid(NDRCALL):
    opnum = 65
    structure = (('ObjectHandle', SAMPR_HANDLE), ('Rid', ULONG))

class SamrRidToSidResponse(NDRCALL):
    structure = (('Sid', PRPC_SID), ('ErrorCode', ULONG))

class SamrSetDSRMPassword(NDRCALL):
    opnum = 66
    structure = (('Unused', PRPC_UNICODE_STRING), ('UserId', ULONG), ('EncryptedNtOwfPassword', PENCRYPTED_NT_OWF_PASSWORD))

class SamrSetDSRMPasswordResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SamrValidatePassword(NDRCALL):
    opnum = 67
    structure = (('ValidationType', PASSWORD_POLICY_VALIDATION_TYPE), ('InputArg', SAM_VALIDATE_INPUT_ARG))

class SamrValidatePasswordResponse(NDRCALL):
    structure = (('OutputArg', PSAM_VALIDATE_OUTPUT_ARG), ('ErrorCode', ULONG))
OPNUMS = {0: (SamrConnect, SamrConnectResponse), 1: (SamrCloseHandle, SamrCloseHandleResponse), 2: (SamrSetSecurityObject, SamrSetSecurityObjectResponse), 3: (SamrQuerySecurityObject, SamrQuerySecurityObjectResponse), 5: (SamrLookupDomainInSamServer, SamrLookupDomainInSamServerResponse), 6: (SamrEnumerateDomainsInSamServer, SamrEnumerateDomainsInSamServerResponse), 7: (SamrOpenDomain, SamrOpenDomainResponse), 8: (SamrQueryInformationDomain, SamrQueryInformationDomainResponse), 9: (SamrSetInformationDomain, SamrSetInformationDomainResponse), 10: (SamrCreateGroupInDomain, SamrCreateGroupInDomainResponse), 11: (SamrEnumerateGroupsInDomain, SamrEnumerateGroupsInDomainResponse), 12: (SamrCreateUserInDomain, SamrCreateUserInDomainResponse), 13: (SamrEnumerateUsersInDomain, SamrEnumerateUsersInDomainResponse), 14: (SamrCreateAliasInDomain, SamrCreateAliasInDomainResponse), 15: (SamrEnumerateAliasesInDomain, SamrEnumerateAliasesInDomainResponse), 16: (SamrGetAliasMembership, SamrGetAliasMembershipResponse), 17: (SamrLookupNamesInDomain, SamrLookupNamesInDomainResponse), 18: (SamrLookupIdsInDomain, SamrLookupIdsInDomainResponse), 19: (SamrOpenGroup, SamrOpenGroupResponse), 20: (SamrQueryInformationGroup, SamrQueryInformationGroupResponse), 21: (SamrSetInformationGroup, SamrSetInformationGroupResponse), 22: (SamrAddMemberToGroup, SamrAddMemberToGroupResponse), 23: (SamrDeleteGroup, SamrDeleteGroupResponse), 24: (SamrRemoveMemberFromGroup, SamrRemoveMemberFromGroupResponse), 25: (SamrGetMembersInGroup, SamrGetMembersInGroupResponse), 26: (SamrSetMemberAttributesOfGroup, SamrSetMemberAttributesOfGroupResponse), 27: (SamrOpenAlias, SamrOpenAliasResponse), 28: (SamrQueryInformationAlias, SamrQueryInformationAliasResponse), 29: (SamrSetInformationAlias, SamrSetInformationAliasResponse), 30: (SamrDeleteAlias, SamrDeleteAliasResponse), 31: (SamrAddMemberToAlias, SamrAddMemberToAliasResponse), 32: (SamrRemoveMemberFromAlias, SamrRemoveMemberFromAliasResponse), 33: (SamrGetMembersInAlias, SamrGetMembersInAliasResponse), 34: (SamrOpenUser, SamrOpenUserResponse), 35: (SamrDeleteUser, SamrDeleteUserResponse), 36: (SamrQueryInformationUser, SamrQueryInformationUserResponse), 37: (SamrSetInformationUser, SamrSetInformationUserResponse), 38: (SamrChangePasswordUser, SamrChangePasswordUserResponse), 39: (SamrGetGroupsForUser, SamrGetGroupsForUserResponse), 40: (SamrQueryDisplayInformation, SamrQueryDisplayInformationResponse), 41: (SamrGetDisplayEnumerationIndex, SamrGetDisplayEnumerationIndexResponse), 44: (SamrGetUserDomainPasswordInformation, SamrGetUserDomainPasswordInformationResponse), 45: (SamrRemoveMemberFromForeignDomain, SamrRemoveMemberFromForeignDomainResponse), 46: (SamrQueryInformationDomain2, SamrQueryInformationDomain2Response), 47: (SamrQueryInformationUser2, SamrQueryInformationUser2Response), 48: (SamrQueryDisplayInformation2, SamrQueryDisplayInformation2Response), 49: (SamrGetDisplayEnumerationIndex2, SamrGetDisplayEnumerationIndex2Response), 50: (SamrCreateUser2InDomain, SamrCreateUser2InDomainResponse), 51: (SamrQueryDisplayInformation3, SamrQueryDisplayInformation3Response), 52: (SamrAddMultipleMembersToAlias, SamrAddMultipleMembersToAliasResponse), 53: (SamrRemoveMultipleMembersFromAlias, SamrRemoveMultipleMembersFromAliasResponse), 54: (SamrOemChangePasswordUser2, SamrOemChangePasswordUser2Response), 55: (SamrUnicodeChangePasswordUser2, SamrUnicodeChangePasswordUser2Response), 56: (SamrGetDomainPasswordInformation, SamrGetDomainPasswordInformationResponse), 57: (SamrConnect2, SamrConnect2Response), 58: (SamrSetInformationUser2, SamrSetInformationUser2Response), 62: (SamrConnect4, SamrConnect4Response), 64: (SamrConnect5, SamrConnect5Response), 65: (SamrRidToSid, SamrRidToSidResponse), 66: (SamrSetDSRMPassword, SamrSetDSRMPasswordResponse), 67: (SamrValidatePassword, SamrValidatePasswordResponse)}

def hSamrConnect5(dce, serverName='\x00', desiredAccess=MAXIMUM_ALLOWED, inVersion=1, revision=3):
    if False:
        print('Hello World!')
    request = SamrConnect5()
    request['ServerName'] = serverName
    request['DesiredAccess'] = desiredAccess
    request['InVersion'] = inVersion
    request['InRevisionInfo']['tag'] = inVersion
    request['InRevisionInfo']['V1']['Revision'] = revision
    return dce.request(request)

def hSamrConnect4(dce, serverName='\x00', desiredAccess=MAXIMUM_ALLOWED, clientRevision=2):
    if False:
        for i in range(10):
            print('nop')
    request = SamrConnect4()
    request['ServerName'] = serverName
    request['DesiredAccess'] = desiredAccess
    request['ClientRevision'] = clientRevision
    return dce.request(request)

def hSamrConnect2(dce, serverName='\x00', desiredAccess=MAXIMUM_ALLOWED):
    if False:
        return 10
    request = SamrConnect2()
    request['ServerName'] = serverName
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrConnect(dce, serverName='\x00', desiredAccess=MAXIMUM_ALLOWED):
    if False:
        for i in range(10):
            print('nop')
    request = SamrConnect()
    request['ServerName'] = serverName
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrOpenDomain(dce, serverHandle, desiredAccess=MAXIMUM_ALLOWED, domainId=NULL):
    if False:
        while True:
            i = 10
    request = SamrOpenDomain()
    request['ServerHandle'] = serverHandle
    request['DesiredAccess'] = desiredAccess
    request['DomainId'] = domainId
    return dce.request(request)

def hSamrOpenGroup(dce, domainHandle, desiredAccess=MAXIMUM_ALLOWED, groupId=0):
    if False:
        while True:
            i = 10
    request = SamrOpenGroup()
    request['DomainHandle'] = domainHandle
    request['DesiredAccess'] = desiredAccess
    request['GroupId'] = groupId
    return dce.request(request)

def hSamrOpenAlias(dce, domainHandle, desiredAccess=MAXIMUM_ALLOWED, aliasId=0):
    if False:
        return 10
    request = SamrOpenAlias()
    request['DomainHandle'] = domainHandle
    request['DesiredAccess'] = desiredAccess
    request['AliasId'] = aliasId
    return dce.request(request)

def hSamrOpenUser(dce, domainHandle, desiredAccess=MAXIMUM_ALLOWED, userId=0):
    if False:
        i = 10
        return i + 15
    request = SamrOpenUser()
    request['DomainHandle'] = domainHandle
    request['DesiredAccess'] = desiredAccess
    request['UserId'] = userId
    return dce.request(request)

def hSamrEnumerateDomainsInSamServer(dce, serverHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        for i in range(10):
            print('nop')
    request = SamrEnumerateDomainsInSamServer()
    request['ServerHandle'] = serverHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrEnumerateGroupsInDomain(dce, domainHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        while True:
            i = 10
    request = SamrEnumerateGroupsInDomain()
    request['DomainHandle'] = domainHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrEnumerateAliasesInDomain(dce, domainHandle, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        i = 10
        return i + 15
    request = SamrEnumerateAliasesInDomain()
    request['DomainHandle'] = domainHandle
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrEnumerateUsersInDomain(dce, domainHandle, userAccountControl=USER_NORMAL_ACCOUNT, enumerationContext=0, preferedMaximumLength=4294967295):
    if False:
        print('Hello World!')
    request = SamrEnumerateUsersInDomain()
    request['DomainHandle'] = domainHandle
    request['UserAccountControl'] = userAccountControl
    request['EnumerationContext'] = enumerationContext
    request['PreferedMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrQueryDisplayInformation3(dce, domainHandle, displayInformationClass=DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser, index=0, entryCount=4294967295, preferedMaximumLength=4294967295):
    if False:
        for i in range(10):
            print('nop')
    request = SamrQueryDisplayInformation3()
    request['DomainHandle'] = domainHandle
    request['DisplayInformationClass'] = displayInformationClass
    request['Index'] = index
    request['EntryCount'] = entryCount
    request['PreferredMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrQueryDisplayInformation2(dce, domainHandle, displayInformationClass=DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser, index=0, entryCount=4294967295, preferedMaximumLength=4294967295):
    if False:
        print('Hello World!')
    request = SamrQueryDisplayInformation2()
    request['DomainHandle'] = domainHandle
    request['DisplayInformationClass'] = displayInformationClass
    request['Index'] = index
    request['EntryCount'] = entryCount
    request['PreferredMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrQueryDisplayInformation(dce, domainHandle, displayInformationClass=DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser, index=0, entryCount=4294967295, preferedMaximumLength=4294967295):
    if False:
        while True:
            i = 10
    request = SamrQueryDisplayInformation()
    request['DomainHandle'] = domainHandle
    request['DisplayInformationClass'] = displayInformationClass
    request['Index'] = index
    request['EntryCount'] = entryCount
    request['PreferredMaximumLength'] = preferedMaximumLength
    return dce.request(request)

def hSamrGetDisplayEnumerationIndex2(dce, domainHandle, displayInformationClass=DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser, prefix=''):
    if False:
        print('Hello World!')
    request = SamrGetDisplayEnumerationIndex2()
    request['DomainHandle'] = domainHandle
    request['DisplayInformationClass'] = displayInformationClass
    request['Prefix'] = prefix
    return dce.request(request)

def hSamrGetDisplayEnumerationIndex(dce, domainHandle, displayInformationClass=DOMAIN_DISPLAY_INFORMATION.DomainDisplayUser, prefix=''):
    if False:
        print('Hello World!')
    request = SamrGetDisplayEnumerationIndex()
    request['DomainHandle'] = domainHandle
    request['DisplayInformationClass'] = displayInformationClass
    request['Prefix'] = prefix
    return dce.request(request)

def hSamrCreateGroupInDomain(dce, domainHandle, name, desiredAccess=GROUP_ALL_ACCESS):
    if False:
        while True:
            i = 10
    request = SamrCreateGroupInDomain()
    request['DomainHandle'] = domainHandle
    request['Name'] = name
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrCreateAliasInDomain(dce, domainHandle, accountName, desiredAccess=GROUP_ALL_ACCESS):
    if False:
        print('Hello World!')
    request = SamrCreateAliasInDomain()
    request['DomainHandle'] = domainHandle
    request['AccountName'] = accountName
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrCreateUser2InDomain(dce, domainHandle, name, accountType=USER_NORMAL_ACCOUNT, desiredAccess=GROUP_ALL_ACCESS):
    if False:
        i = 10
        return i + 15
    request = SamrCreateUser2InDomain()
    request['DomainHandle'] = domainHandle
    request['Name'] = name
    request['AccountType'] = accountType
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrCreateUserInDomain(dce, domainHandle, name, desiredAccess=GROUP_ALL_ACCESS):
    if False:
        for i in range(10):
            print('nop')
    request = SamrCreateUserInDomain()
    request['DomainHandle'] = domainHandle
    request['Name'] = name
    request['DesiredAccess'] = desiredAccess
    return dce.request(request)

def hSamrQueryInformationDomain(dce, domainHandle, domainInformationClass=DOMAIN_INFORMATION_CLASS.DomainGeneralInformation2):
    if False:
        for i in range(10):
            print('nop')
    request = SamrQueryInformationDomain()
    request['DomainHandle'] = domainHandle
    request['DomainInformationClass'] = domainInformationClass
    return dce.request(request)

def hSamrQueryInformationDomain2(dce, domainHandle, domainInformationClass=DOMAIN_INFORMATION_CLASS.DomainGeneralInformation2):
    if False:
        while True:
            i = 10
    request = SamrQueryInformationDomain2()
    request['DomainHandle'] = domainHandle
    request['DomainInformationClass'] = domainInformationClass
    return dce.request(request)

def hSamrQueryInformationGroup(dce, groupHandle, groupInformationClass=GROUP_INFORMATION_CLASS.GroupGeneralInformation):
    if False:
        print('Hello World!')
    request = SamrQueryInformationGroup()
    request['GroupHandle'] = groupHandle
    request['GroupInformationClass'] = groupInformationClass
    return dce.request(request)

def hSamrQueryInformationAlias(dce, aliasHandle, aliasInformationClass=ALIAS_INFORMATION_CLASS.AliasGeneralInformation):
    if False:
        print('Hello World!')
    request = SamrQueryInformationAlias()
    request['AliasHandle'] = aliasHandle
    request['AliasInformationClass'] = aliasInformationClass
    return dce.request(request)

def hSamrQueryInformationUser2(dce, userHandle, userInformationClass=USER_INFORMATION_CLASS.UserGeneralInformation):
    if False:
        return 10
    request = SamrQueryInformationUser2()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = userInformationClass
    return dce.request(request)

def hSamrQueryInformationUser(dce, userHandle, userInformationClass=USER_INFORMATION_CLASS.UserGeneralInformation):
    if False:
        for i in range(10):
            print('nop')
    request = SamrQueryInformationUser()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = userInformationClass
    return dce.request(request)

def hSamrSetInformationDomain(dce, domainHandle, domainInformation):
    if False:
        return 10
    request = SamrSetInformationDomain()
    request['DomainHandle'] = domainHandle
    request['DomainInformationClass'] = domainInformation['tag']
    request['DomainInformation'] = domainInformation
    return dce.request(request)

def hSamrSetInformationGroup(dce, groupHandle, buffer):
    if False:
        return 10
    request = SamrSetInformationGroup()
    request['GroupHandle'] = groupHandle
    request['GroupInformationClass'] = buffer['tag']
    request['Buffer'] = buffer
    return dce.request(request)

def hSamrSetInformationAlias(dce, aliasHandle, buffer):
    if False:
        for i in range(10):
            print('nop')
    request = SamrSetInformationAlias()
    request['AliasHandle'] = aliasHandle
    request['AliasInformationClass'] = buffer['tag']
    request['Buffer'] = buffer
    return dce.request(request)

def hSamrSetInformationUser2(dce, userHandle, buffer):
    if False:
        i = 10
        return i + 15
    request = SamrSetInformationUser2()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = buffer['tag']
    request['Buffer'] = buffer
    return dce.request(request)

def hSamrSetInformationUser(dce, userHandle, buffer):
    if False:
        return 10
    request = SamrSetInformationUser()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = buffer['tag']
    request['Buffer'] = buffer
    return dce.request(request)

def hSamrDeleteGroup(dce, groupHandle):
    if False:
        return 10
    request = SamrDeleteGroup()
    request['GroupHandle'] = groupHandle
    return dce.request(request)

def hSamrDeleteAlias(dce, aliasHandle):
    if False:
        while True:
            i = 10
    request = SamrDeleteAlias()
    request['AliasHandle'] = aliasHandle
    return dce.request(request)

def hSamrDeleteUser(dce, userHandle):
    if False:
        while True:
            i = 10
    request = SamrDeleteUser()
    request['UserHandle'] = userHandle
    return dce.request(request)

def hSamrAddMemberToGroup(dce, groupHandle, memberId, attributes):
    if False:
        i = 10
        return i + 15
    request = SamrAddMemberToGroup()
    request['GroupHandle'] = groupHandle
    request['MemberId'] = memberId
    request['Attributes'] = attributes
    return dce.request(request)

def hSamrRemoveMemberFromGroup(dce, groupHandle, memberId):
    if False:
        while True:
            i = 10
    request = SamrRemoveMemberFromGroup()
    request['GroupHandle'] = groupHandle
    request['MemberId'] = memberId
    return dce.request(request)

def hSamrGetMembersInGroup(dce, groupHandle):
    if False:
        print('Hello World!')
    request = SamrGetMembersInGroup()
    request['GroupHandle'] = groupHandle
    return dce.request(request)

def hSamrAddMemberToAlias(dce, aliasHandle, memberId):
    if False:
        for i in range(10):
            print('nop')
    request = SamrAddMemberToAlias()
    request['AliasHandle'] = aliasHandle
    request['MemberId'] = memberId
    return dce.request(request)

def hSamrRemoveMemberFromAlias(dce, aliasHandle, memberId):
    if False:
        while True:
            i = 10
    request = SamrRemoveMemberFromAlias()
    request['AliasHandle'] = aliasHandle
    request['MemberId'] = memberId
    return dce.request(request)

def hSamrGetMembersInAlias(dce, aliasHandle):
    if False:
        for i in range(10):
            print('nop')
    request = SamrGetMembersInAlias()
    request['AliasHandle'] = aliasHandle
    return dce.request(request)

def hSamrRemoveMemberFromForeignDomain(dce, domainHandle, memberSid):
    if False:
        print('Hello World!')
    request = SamrRemoveMemberFromForeignDomain()
    request['DomainHandle'] = domainHandle
    request['MemberSid'] = memberSid
    return dce.request(request)

def hSamrAddMultipleMembersToAlias(dce, aliasHandle, membersBuffer):
    if False:
        return 10
    request = SamrAddMultipleMembersToAlias()
    request['AliasHandle'] = aliasHandle
    request['MembersBuffer'] = membersBuffer
    request['MembersBuffer']['Count'] = len(membersBuffer['Sids'])
    return dce.request(request)

def hSamrRemoveMultipleMembersFromAlias(dce, aliasHandle, membersBuffer):
    if False:
        print('Hello World!')
    request = SamrRemoveMultipleMembersFromAlias()
    request['AliasHandle'] = aliasHandle
    request['MembersBuffer'] = membersBuffer
    request['MembersBuffer']['Count'] = len(membersBuffer['Sids'])
    return dce.request(request)

def hSamrGetGroupsForUser(dce, userHandle):
    if False:
        print('Hello World!')
    request = SamrGetGroupsForUser()
    request['UserHandle'] = userHandle
    return dce.request(request)

def hSamrGetAliasMembership(dce, domainHandle, sidArray):
    if False:
        for i in range(10):
            print('nop')
    request = SamrGetAliasMembership()
    request['DomainHandle'] = domainHandle
    request['SidArray'] = sidArray
    request['SidArray']['Count'] = len(sidArray['Sids'])
    return dce.request(request)

def hSamrChangePasswordUser(dce, userHandle, oldPassword, newPassword, oldPwdHashNT='', newPwdHashLM='', newPwdHashNT=''):
    if False:
        print('Hello World!')
    request = SamrChangePasswordUser()
    request['UserHandle'] = userHandle
    from impacket import crypto, ntlm
    if oldPwdHashNT == '':
        oldPwdHashNT = ntlm.NTOWFv1(oldPassword)
    else:
        try:
            oldPwdHashNT = unhexlify(oldPwdHashNT)
        except:
            pass
    if newPwdHashLM == '':
        newPwdHashLM = ntlm.LMOWFv1(newPassword)
    else:
        try:
            newPwdHashLM = unhexlify(newPwdHashLM)
        except:
            pass
    if newPwdHashNT == '':
        newPwdHashNT = ntlm.NTOWFv1(newPassword)
    else:
        try:
            newPwdHashNT = unhexlify(newPwdHashNT)
        except:
            pass
    request['LmPresent'] = 0
    request['OldLmEncryptedWithNewLm'] = NULL
    request['NewLmEncryptedWithOldLm'] = NULL
    request['NtPresent'] = 1
    request['OldNtEncryptedWithNewNt'] = crypto.SamEncryptNTLMHash(oldPwdHashNT, newPwdHashNT)
    request['NewNtEncryptedWithOldNt'] = crypto.SamEncryptNTLMHash(newPwdHashNT, oldPwdHashNT)
    request['NtCrossEncryptionPresent'] = 0
    request['NewNtEncryptedWithNewLm'] = NULL
    request['LmCrossEncryptionPresent'] = 1
    request['NewLmEncryptedWithNewNt'] = crypto.SamEncryptNTLMHash(newPwdHashLM, newPwdHashNT)
    return dce.request(request)

def hSamrUnicodeChangePasswordUser2(dce, serverName='\x00', userName='', oldPassword='', newPassword='', oldPwdHashLM='', oldPwdHashNT=''):
    if False:
        for i in range(10):
            print('nop')
    request = SamrUnicodeChangePasswordUser2()
    request['ServerName'] = serverName
    request['UserName'] = userName
    try:
        from Cryptodome.Cipher import ARC4
    except Exception:
        LOG.critical("Warning: You don't have any crypto installed. You need pycryptodomex")
        LOG.critical('See https://pypi.org/project/pycryptodomex/')
    from impacket import crypto, ntlm
    if oldPwdHashLM == '' and oldPwdHashNT == '':
        oldPwdHashLM = ntlm.LMOWFv1(oldPassword)
        oldPwdHashNT = ntlm.NTOWFv1(oldPassword)
    else:
        try:
            oldPwdHashLM = unhexlify(oldPwdHashLM)
        except:
            pass
        try:
            oldPwdHashNT = unhexlify(oldPwdHashNT)
        except:
            pass
    newPwdHashNT = ntlm.NTOWFv1(newPassword)
    samUser = SAMPR_USER_PASSWORD()
    try:
        samUser['Buffer'] = b'A' * (512 - len(newPassword) * 2) + newPassword.encode('utf-16le')
    except UnicodeDecodeError:
        import sys
        samUser['Buffer'] = b'A' * (512 - len(newPassword) * 2) + newPassword.decode(sys.getfilesystemencoding()).encode('utf-16le')
    samUser['Length'] = len(newPassword) * 2
    pwdBuff = samUser.getData()
    rc4 = ARC4.new(oldPwdHashNT)
    encBuf = rc4.encrypt(pwdBuff)
    request['NewPasswordEncryptedWithOldNt']['Buffer'] = encBuf
    request['OldNtOwfPasswordEncryptedWithNewNt'] = crypto.SamEncryptNTLMHash(oldPwdHashNT, newPwdHashNT)
    request['LmPresent'] = 0
    request['NewPasswordEncryptedWithOldLm'] = NULL
    request['OldLmOwfPasswordEncryptedWithNewNt'] = NULL
    return dce.request(request)

def hSamrLookupDomainInSamServer(dce, serverHandle, name):
    if False:
        return 10
    request = SamrLookupDomainInSamServer()
    request['ServerHandle'] = serverHandle
    request['Name'] = name
    return dce.request(request)

def hSamrSetSecurityObject(dce, objectHandle, securityInformation, securityDescriptor):
    if False:
        return 10
    request = SamrSetSecurityObject()
    request['ObjectHandle'] = objectHandle
    request['SecurityInformation'] = securityInformation
    request['SecurityDescriptor'] = securityDescriptor
    return dce.request(request)

def hSamrQuerySecurityObject(dce, objectHandle, securityInformation):
    if False:
        while True:
            i = 10
    request = SamrQuerySecurityObject()
    request['ObjectHandle'] = objectHandle
    request['SecurityInformation'] = securityInformation
    return dce.request(request)

def hSamrCloseHandle(dce, samHandle):
    if False:
        while True:
            i = 10
    request = SamrCloseHandle()
    request['SamHandle'] = samHandle
    return dce.request(request)

def hSamrSetMemberAttributesOfGroup(dce, groupHandle, memberId, attributes):
    if False:
        while True:
            i = 10
    request = SamrSetMemberAttributesOfGroup()
    request['GroupHandle'] = groupHandle
    request['MemberId'] = memberId
    request['Attributes'] = attributes
    return dce.request(request)

def hSamrGetUserDomainPasswordInformation(dce, userHandle):
    if False:
        while True:
            i = 10
    request = SamrGetUserDomainPasswordInformation()
    request['UserHandle'] = userHandle
    return dce.request(request)

def hSamrGetDomainPasswordInformation(dce):
    if False:
        print('Hello World!')
    request = SamrGetDomainPasswordInformation()
    request['Unused'] = NULL
    return dce.request(request)

def hSamrRidToSid(dce, objectHandle, rid):
    if False:
        i = 10
        return i + 15
    request = SamrRidToSid()
    request['ObjectHandle'] = objectHandle
    request['Rid'] = rid
    return dce.request(request)

def hSamrValidatePassword(dce, inputArg):
    if False:
        print('Hello World!')
    request = SamrValidatePassword()
    request['ValidationType'] = inputArg['tag']
    request['InputArg'] = inputArg
    return dce.request(request)

def hSamrLookupNamesInDomain(dce, domainHandle, names):
    if False:
        for i in range(10):
            print('nop')
    request = SamrLookupNamesInDomain()
    request['DomainHandle'] = domainHandle
    request['Count'] = len(names)
    for name in names:
        entry = RPC_UNICODE_STRING()
        entry['Data'] = name
        request['Names'].append(entry)
    request.fields['Names'].fields['MaximumCount'] = 1000
    return dce.request(request)

def hSamrLookupIdsInDomain(dce, domainHandle, ids):
    if False:
        for i in range(10):
            print('nop')
    request = SamrLookupIdsInDomain()
    request['DomainHandle'] = domainHandle
    request['Count'] = len(ids)
    for dId in ids:
        entry = ULONG()
        entry['Data'] = dId
        request['RelativeIds'].append(entry)
    request.fields['RelativeIds'].fields['MaximumCount'] = 1000
    return dce.request(request)

def hSamrSetPasswordInternal4New(dce, userHandle, password):
    if False:
        while True:
            i = 10
    request = SamrSetInformationUser2()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = USER_INFORMATION_CLASS.UserInternal4InformationNew
    request['Buffer']['tag'] = USER_INFORMATION_CLASS.UserInternal4InformationNew
    request['Buffer']['Internal4New']['I1']['WhichFields'] = 16777216 | 134217728
    request['Buffer']['Internal4New']['I1']['UserName'] = NULL
    request['Buffer']['Internal4New']['I1']['FullName'] = NULL
    request['Buffer']['Internal4New']['I1']['HomeDirectory'] = NULL
    request['Buffer']['Internal4New']['I1']['HomeDirectoryDrive'] = NULL
    request['Buffer']['Internal4New']['I1']['ScriptPath'] = NULL
    request['Buffer']['Internal4New']['I1']['ProfilePath'] = NULL
    request['Buffer']['Internal4New']['I1']['AdminComment'] = NULL
    request['Buffer']['Internal4New']['I1']['WorkStations'] = NULL
    request['Buffer']['Internal4New']['I1']['UserComment'] = NULL
    request['Buffer']['Internal4New']['I1']['Parameters'] = NULL
    request['Buffer']['Internal4New']['I1']['LmOwfPassword']['Buffer'] = NULL
    request['Buffer']['Internal4New']['I1']['NtOwfPassword']['Buffer'] = NULL
    request['Buffer']['Internal4New']['I1']['PrivateData'] = NULL
    request['Buffer']['Internal4New']['I1']['SecurityDescriptor']['SecurityDescriptor'] = NULL
    request['Buffer']['Internal4New']['I1']['LogonHours']['LogonHours'] = NULL
    request['Buffer']['Internal4New']['I1']['PasswordExpired'] = 1
    pwdbuff = password.encode('utf-16le')
    bufflen = len(pwdbuff)
    pwdbuff = pwdbuff.rjust(512, b'\x00')
    pwdbuff += struct.pack('<I', bufflen)
    salt = os.urandom(16)
    session_key = dce.get_rpc_transport().get_smb_connection().getSessionKey()
    keymd = md5()
    keymd.update(salt)
    keymd.update(session_key)
    key = keymd.digest()
    cipher = ARC4.new(key)
    buffercrypt = cipher.encrypt(pwdbuff) + salt
    request['Buffer']['Internal4New']['UserPassword']['Buffer'] = buffercrypt
    return dce.request(request)

def hSamrSetNTInternal1(dce, userHandle, password, hashNT=''):
    if False:
        return 10
    request = SamrSetInformationUser()
    request['UserHandle'] = userHandle
    request['UserInformationClass'] = USER_INFORMATION_CLASS.UserInternal1Information
    request['Buffer']['tag'] = USER_INFORMATION_CLASS.UserInternal1Information
    from impacket import crypto, ntlm
    if hashNT == '':
        hashNT = ntlm.NTOWFv1(password)
    else:
        try:
            hashNT = unhexlify(hashNT)
        except:
            pass
    session_key = dce.get_rpc_transport().get_smb_connection().getSessionKey()
    request['Buffer']['Internal1']['EncryptedNtOwfPassword'] = crypto.SamEncryptNTLMHash(hashNT, session_key)
    request['Buffer']['Internal1']['EncryptedLmOwfPassword'] = NULL
    request['Buffer']['Internal1']['NtPasswordPresent'] = 1
    request['Buffer']['Internal1']['LmPasswordPresent'] = 0
    return dce.request(request)