from __future__ import division
from __future__ import print_function
from impacket import system_errors
from impacket.dcerpc.v5.dtypes import LPWSTR, ULONG, NULL, DWORD, BOOL, BYTE, LPDWORD, WORD
from impacket.dcerpc.v5.ndr import NDRCALL, NDRUniConformantArray, NDRPOINTER, NDRSTRUCT, NDRENUM, NDRUNION
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.enum import Enum
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_DHCPSRV = uuidtup_to_bin(('6BFFD098-A112-3610-9833-46C3F874532D', '1.0'))
MSRPC_UUID_DHCPSRV2 = uuidtup_to_bin(('5B821720-F63B-11D0-AAD2-00C04FC324DB', '1.0'))
DHCP_SRV_HANDLE = LPWSTR
DHCP_IP_ADDRESS = DWORD
DHCP_IP_MASK = DWORD
DHCP_OPTION_ID = DWORD
DHCP_FLAGS_OPTION_DEFAULT = 0
DHCP_FLAGS_OPTION_IS_VENDOR = 3
ERROR_DHCP_REGISTRY_INIT_FAILED = 20000
ERROR_DHCP_DATABASE_INIT_FAILED = 20001
ERROR_DHCP_RPC_INIT_FAILED = 20002
ERROR_DHCP_NETWORK_INIT_FAILED = 20003
ERROR_DHCP_SUBNET_EXITS = 20004
ERROR_DHCP_SUBNET_NOT_PRESENT = 20005
ERROR_DHCP_PRIMARY_NOT_FOUND = 20006
ERROR_DHCP_ELEMENT_CANT_REMOVE = 20007
ERROR_DHCP_OPTION_EXITS = 20009
ERROR_DHCP_OPTION_NOT_PRESENT = 20010
ERROR_DHCP_ADDRESS_NOT_AVAILABLE = 20011
ERROR_DHCP_RANGE_FULL = 20012
ERROR_DHCP_JET_ERROR = 20013
ERROR_DHCP_CLIENT_EXISTS = 20014
ERROR_DHCP_INVALID_DHCP_MESSAGE = 20015
ERROR_DHCP_INVALID_DHCP_CLIENT = 20016
ERROR_DHCP_SERVICE_PAUSED = 20017
ERROR_DHCP_NOT_RESERVED_CLIENT = 20018
ERROR_DHCP_RESERVED_CLIENT = 20019
ERROR_DHCP_RANGE_TOO_SMALL = 20020
ERROR_DHCP_IPRANGE_EXITS = 20021
ERROR_DHCP_RESERVEDIP_EXITS = 20022
ERROR_DHCP_INVALID_RANGE = 20023
ERROR_DHCP_RANGE_EXTENDED = 20024
ERROR_EXTEND_TOO_SMALL = 20025
WARNING_EXTENDED_LESS = 20026
ERROR_DHCP_JET_CONV_REQUIRED = 20027
ERROR_SERVER_INVALID_BOOT_FILE_TABLE = 20028
ERROR_SERVER_UNKNOWN_BOOT_FILE_NAME = 20029
ERROR_DHCP_SUPER_SCOPE_NAME_TOO_LONG = 20030
ERROR_DHCP_IP_ADDRESS_IN_USE = 20032
ERROR_DHCP_LOG_FILE_PATH_TOO_LONG = 20033
ERROR_DHCP_UNSUPPORTED_CLIENT = 20034
ERROR_DHCP_JET97_CONV_REQUIRED = 20036
ERROR_DHCP_ROGUE_INIT_FAILED = 20037
ERROR_DHCP_ROGUE_SAMSHUTDOWN = 20038
ERROR_DHCP_ROGUE_NOT_AUTHORIZED = 20039
ERROR_DHCP_ROGUE_DS_UNREACHABLE = 20040
ERROR_DHCP_ROGUE_DS_CONFLICT = 20041
ERROR_DHCP_ROGUE_NOT_OUR_ENTERPRISE = 20042
ERROR_DHCP_ROGUE_STANDALONE_IN_DS = 20043
ERROR_DHCP_CLASS_NOT_FOUND = 20044
ERROR_DHCP_CLASS_ALREADY_EXISTS = 20045
ERROR_DHCP_SCOPE_NAME_TOO_LONG = 20046
ERROR_DHCP_DEFAULT_SCOPE_EXITS = 20047
ERROR_DHCP_CANT_CHANGE_ATTRIBUTE = 20048
ERROR_DHCP_IPRANGE_CONV_ILLEGAL = 20049
ERROR_DHCP_NETWORK_CHANGED = 20050
ERROR_DHCP_CANNOT_MODIFY_BINDINGS = 20051
ERROR_DHCP_SUBNET_EXISTS = 20052
ERROR_DHCP_MSCOPE_EXISTS = 20053
ERROR_MSCOPE_RANGE_TOO_SMALL = 20054
ERROR_DHCP_EXEMPTION_EXISTS = 20055
ERROR_DHCP_EXEMPTION_NOT_PRESENT = 20056
ERROR_DHCP_INVALID_PARAMETER_OPTION32 = 20057
ERROR_DDS_NO_DS_AVAILABLE = 20070
ERROR_DDS_NO_DHCP_ROOT = 20071
ERROR_DDS_UNEXPECTED_ERROR = 20072
ERROR_DDS_TOO_MANY_ERRORS = 20073
ERROR_DDS_DHCP_SERVER_NOT_FOUND = 20074
ERROR_DDS_OPTION_ALREADY_EXISTS = 20075
ERROR_DDS_OPTION_DOES_NOT_EXIST = 20076
ERROR_DDS_CLASS_EXISTS = 20077
ERROR_DDS_CLASS_DOES_NOT_EXIST = 20078
ERROR_DDS_SERVER_ALREADY_EXISTS = 20079
ERROR_DDS_SERVER_DOES_NOT_EXIST = 20080
ERROR_DDS_SERVER_ADDRESS_MISMATCH = 20081
ERROR_DDS_SUBNET_EXISTS = 20082
ERROR_DDS_SUBNET_HAS_DIFF_SSCOPE = 20083
ERROR_DDS_SUBNET_NOT_PRESENT = 20084
ERROR_DDS_RESERVATION_NOT_PRESENT = 20085
ERROR_DDS_RESERVATION_CONFLICT = 20086
ERROR_DDS_POSSIBLE_RANGE_CONFLICT = 20087
ERROR_DDS_RANGE_DOES_NOT_EXIST = 20088
ERROR_DHCP_DELETE_BUILTIN_CLASS = 20089
ERROR_DHCP_INVALID_SUBNET_PREFIX = 20091
ERROR_DHCP_INVALID_DELAY = 20092
ERROR_DHCP_LINKLAYER_ADDRESS_EXISTS = 20093
ERROR_DHCP_LINKLAYER_ADDRESS_RESERVATION_EXISTS = 20094
ERROR_DHCP_LINKLAYER_ADDRESS_DOES_NOT_EXIST = 20095
ERROR_DHCP_HARDWARE_ADDRESS_TYPE_ALREADY_EXEMPT = 20101
ERROR_DHCP_UNDEFINED_HARDWARE_ADDRESS_TYPE = 20102
ERROR_DHCP_OPTION_TYPE_MISMATCH = 20103
ERROR_DHCP_POLICY_BAD_PARENT_EXPR = 20104
ERROR_DHCP_POLICY_EXISTS = 20105
ERROR_DHCP_POLICY_RANGE_EXISTS = 20106
ERROR_DHCP_POLICY_RANGE_BAD = 20107
ERROR_DHCP_RANGE_INVALID_IN_SERVER_POLICY = 20108
ERROR_DHCP_INVALID_POLICY_EXPRESSION = 20109
ERROR_DHCP_INVALID_PROCESSING_ORDER = 20110
ERROR_DHCP_POLICY_NOT_FOUND = 20111
ERROR_SCOPE_RANGE_POLICY_RANGE_CONFLICT = 20112
ERROR_DHCP_FO_SCOPE_ALREADY_IN_RELATIONSHIP = 20113
ERROR_DHCP_FO_RELATIONSHIP_EXISTS = 20114
ERROR_DHCP_FO_RELATIONSHIP_DOES_NOT_EXIST = 20115
ERROR_DHCP_FO_SCOPE_NOT_IN_RELATIONSHIP = 20116
ERROR_DHCP_FO_RELATION_IS_SECONDARY = 20117
ERROR_DHCP_FO_NOT_SUPPORTED = 20118
ERROR_DHCP_FO_TIME_OUT_OF_SYNC = 20119
ERROR_DHCP_FO_STATE_NOT_NORMAL = 20120
ERROR_DHCP_NO_ADMIN_PERMISSION = 20121
ERROR_DHCP_SERVER_NOT_REACHABLE = 20122
ERROR_DHCP_SERVER_NOT_RUNNING = 20123
ERROR_DHCP_SERVER_NAME_NOT_RESOLVED = 20124
ERROR_DHCP_FO_RELATIONSHIP_NAME_TOO_LONG = 20125
ERROR_DHCP_REACHED_END_OF_SELECTION = 20126
ERROR_DHCP_FO_ADDSCOPE_LEASES_NOT_SYNCED = 20127
ERROR_DHCP_FO_MAX_RELATIONSHIPS = 20128
ERROR_DHCP_FO_IPRANGE_TYPE_CONV_ILLEGAL = 20129
ERROR_DHCP_FO_MAX_ADD_SCOPES = 20130
ERROR_DHCP_FO_BOOT_NOT_SUPPORTED = 20131
ERROR_DHCP_FO_RANGE_PART_OF_REL = 20132
ERROR_DHCP_FO_SCOPE_SYNC_IN_PROGRESS = 20133
ERROR_DHCP_FO_FEATURE_NOT_SUPPORTED = 20134
ERROR_DHCP_POLICY_FQDN_RANGE_UNSUPPORTED = 20135
ERROR_DHCP_POLICY_FQDN_OPTION_UNSUPPORTED = 20136
ERROR_DHCP_POLICY_EDIT_FQDN_UNSUPPORTED = 20137
ERROR_DHCP_NAP_NOT_SUPPORTED = 20138
ERROR_LAST_DHCP_SERVER_ERROR = 20139

class DCERPCSessionError(DCERPCException):
    ERROR_MESSAGES = {ERROR_DHCP_JET_ERROR: ('ERROR_DHCP_JET_ERROR', 'An error occurred while accessing the DHCP server database.'), ERROR_DHCP_SUBNET_NOT_PRESENT: ('ERROR_DHCP_SUBNET_NOT_PRESENT', 'The specified IPv4 subnet does not exist.'), ERROR_DHCP_SUBNET_EXISTS: ('ERROR_DHCP_SUBNET_EXISTS', 'The IPv4 scope parameters are incorrect. Either the IPv4 scope already exists, corresponding to the SubnetAddress and SubnetMask members of the structure DHCP_SUBNET_INFO (section 2.2.1.2.8), or there is a range overlap of IPv4 addresses between those associated with the SubnetAddress and SubnetMask fields of the new IPv4 scope and the subnet address and mask of an already existing IPv4 scope'), ERROR_DHCP_INVALID_DHCP_CLIENT: ('ERROR_DHCP_INVALID_DHCP_CLIENT', 'The DHCP server received an invalid message from the client.')}

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            print('Hello World!')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            while True:
                i = 10
        key = self.error_code
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'DHCPM SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        elif key in self.ERROR_MESSAGES:
            error_msg_short = self.ERROR_MESSAGES[key][0]
            error_msg_verbose = self.ERROR_MESSAGES[key][1]
            return 'DHCPM SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'DHCPM SessionError: unknown error code: 0x%x' % self.error_code

class DHCP_SEARCH_INFO_TYPE(NDRENUM):

    class enumItems(Enum):
        DhcpClientIpAddress = 0
        DhcpClientHardwareAddress = 1
        DhcpClientName = 2

class QuarantineStatus(NDRENUM):

    class enumItems(Enum):
        NOQUARANTINE = 0
        RESTRICTEDACCESS = 1
        DROPPACKET = 2
        PROBATION = 3
        EXEMPT = 4
        DEFAULTQUARSETTING = 5
        NOQUARINFO = 6

class DHCP_HOST_INFO(NDRSTRUCT):
    structure = (('IpAddress', DHCP_IP_ADDRESS), ('NetBiosName', LPWSTR), ('HostName', LPWSTR))

class BYTE_ARRAY(NDRUniConformantArray):
    item = 'c'

class PBYTE_ARRAY(NDRPOINTER):
    referent = (('Data', BYTE_ARRAY),)

class DHCP_BINARY_DATA(NDRSTRUCT):
    structure = (('DataLength', DWORD), ('Data_', PBYTE_ARRAY))
DHCP_CLIENT_UID = DHCP_BINARY_DATA

class DATE_TIME(NDRSTRUCT):
    structure = (('dwLowDateTime', DWORD), ('dwHighDateTime', DWORD))

class DHCP_CLIENT_INFO_VQ(NDRSTRUCT):
    structure = (('ClientIpAddress', DHCP_IP_ADDRESS), ('SubnetMask', DHCP_IP_MASK), ('ClientHardwareAddress', DHCP_CLIENT_UID), ('ClientName', LPWSTR), ('ClientComment', LPWSTR), ('ClientLeaseExpires', DATE_TIME), ('OwnerHost', DHCP_HOST_INFO), ('bClientType', BYTE), ('AddressState', BYTE), ('Status', QuarantineStatus), ('ProbationEnds', DATE_TIME), ('QuarantineCapable', BOOL))

class DHCP_CLIENT_SEARCH_UNION(NDRUNION):
    union = {DHCP_SEARCH_INFO_TYPE.DhcpClientIpAddress: ('ClientIpAddress', DHCP_IP_ADDRESS), DHCP_SEARCH_INFO_TYPE.DhcpClientHardwareAddress: ('ClientHardwareAddress', DHCP_CLIENT_UID), DHCP_SEARCH_INFO_TYPE.DhcpClientName: ('ClientName', LPWSTR)}

class DHCP_SEARCH_INFO(NDRSTRUCT):
    structure = (('SearchType', DHCP_SEARCH_INFO_TYPE), ('SearchInfo', DHCP_CLIENT_SEARCH_UNION))

class DHCP_CLIENT_INFO_V4(NDRSTRUCT):
    structure = (('ClientIpAddress', DHCP_IP_ADDRESS), ('SubnetMask', DHCP_IP_MASK), ('ClientHardwareAddress', DHCP_CLIENT_UID), ('ClientName', LPWSTR), ('ClientComment', LPWSTR), ('ClientLeaseExpires', DATE_TIME), ('OwnerHost', DHCP_HOST_INFO), ('bClientType', BYTE))

class DHCP_CLIENT_INFO_V5(NDRSTRUCT):
    structure = (('ClientIpAddress', DHCP_IP_ADDRESS), ('SubnetMask', DHCP_IP_MASK), ('ClientHardwareAddress', DHCP_CLIENT_UID), ('ClientName', LPWSTR), ('ClientComment', LPWSTR), ('ClientLeaseExpires', DATE_TIME), ('OwnerHost', DHCP_HOST_INFO), ('bClientType', BYTE), ('AddressState', BYTE))

class LPDHCP_CLIENT_INFO_V4(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_V4),)

class LPDHCP_CLIENT_INFO_V5(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_V5),)

class DHCP_CLIENT_INFO_PB(NDRSTRUCT):
    structure = (('ClientIpAddress', DHCP_IP_ADDRESS), ('SubnetMask', DHCP_IP_MASK), ('ClientHardwareAddress', DHCP_CLIENT_UID), ('ClientName', LPWSTR), ('ClientComment', LPWSTR), ('ClientLeaseExpires', DATE_TIME), ('OwnerHost', DHCP_HOST_INFO), ('bClientType', BYTE), ('AddressState', BYTE), ('Status', QuarantineStatus), ('ProbationEnds', DATE_TIME), ('QuarantineCapable', BOOL), ('FilterStatus', DWORD), ('PolicyName', LPWSTR))

class LPDHCP_CLIENT_INFO_PB(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_PB),)

class LPDHCP_CLIENT_INFO_VQ(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_VQ),)

class DHCP_CLIENT_INFO_VQ_ARRAY(NDRUniConformantArray):
    item = LPDHCP_CLIENT_INFO_VQ

class LPDHCP_CLIENT_INFO_VQ_ARRAY(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_VQ_ARRAY),)

class DHCP_CLIENT_INFO_ARRAY_VQ(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Clients', LPDHCP_CLIENT_INFO_VQ_ARRAY))

class LPDHCP_CLIENT_INFO_ARRAY_VQ(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_ARRAY_VQ),)

class DHCP_CLIENT_INFO_V4_ARRAY(NDRUniConformantArray):
    item = LPDHCP_CLIENT_INFO_V4

class DHCP_CLIENT_INFO_V5_ARRAY(NDRUniConformantArray):
    item = LPDHCP_CLIENT_INFO_V5

class LPDHCP_CLIENT_INFO_V4_ARRAY(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_V4_ARRAY),)

class LPDHCP_CLIENT_INFO_V5_ARRAY(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_V5_ARRAY),)

class DHCP_CLIENT_INFO_ARRAY_V4(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Clients', LPDHCP_CLIENT_INFO_V4_ARRAY))

class DHCP_CLIENT_INFO_ARRAY_V5(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Clients', LPDHCP_CLIENT_INFO_V5_ARRAY))

class LPDHCP_CLIENT_INFO_ARRAY_V5(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_ARRAY_V5),)

class LPDHCP_CLIENT_INFO_ARRAY_V4(NDRPOINTER):
    referent = (('Data', DHCP_CLIENT_INFO_ARRAY_V4),)

class DHCP_IP_ADDRESS_ARRAY(NDRUniConformantArray):
    item = DHCP_IP_ADDRESS

class LPDHCP_IP_ADDRESS_ARRAY(NDRPOINTER):
    referent = (('Data', DHCP_IP_ADDRESS_ARRAY),)

class DHCP_IP_ARRAY(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Elements', LPDHCP_IP_ADDRESS_ARRAY))

class DHCP_SUBNET_STATE(NDRENUM):

    class enumItems(Enum):
        DhcpSubnetEnabled = 0
        DhcpSubnetDisabled = 1
        DhcpSubnetEnabledSwitched = 2
        DhcpSubnetDisabledSwitched = 3
        DhcpSubnetInvalidState = 4

class DHCP_SUBNET_INFO(NDRSTRUCT):
    structure = (('SubnetAddress', DHCP_IP_ADDRESS), ('SubnetMask', DHCP_IP_MASK), ('SubnetName', LPWSTR), ('SubnetComment', LPWSTR), ('PrimaryHost', DHCP_HOST_INFO), ('SubnetState', DHCP_SUBNET_STATE))

class LPDHCP_SUBNET_INFO(NDRPOINTER):
    referent = (('Data', DHCP_SUBNET_INFO),)

class DHCP_OPTION_SCOPE_TYPE(NDRENUM):

    class enumItems(Enum):
        DhcpDefaultOptions = 0
        DhcpGlobalOptions = 1
        DhcpSubnetOptions = 2
        DhcpReservedOptions = 3
        DhcpMScopeOptions = 4

class DHCP_RESERVED_SCOPE(NDRSTRUCT):
    structure = (('ReservedIpAddress', DHCP_IP_ADDRESS), ('ReservedIpSubnetAddress', DHCP_IP_ADDRESS))

class DHCP_OPTION_SCOPE_UNION(NDRUNION):
    union = {DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions: (), DHCP_OPTION_SCOPE_TYPE.DhcpGlobalOptions: (), DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions: ('SubnetScopeInfo', DHCP_IP_ADDRESS), DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions: ('ReservedScopeInfo', DHCP_RESERVED_SCOPE), DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions: ('MScopeInfo', LPWSTR)}

class DHCP_OPTION_SCOPE_INFO(NDRSTRUCT):
    structure = (('ScopeType', DHCP_OPTION_SCOPE_TYPE), ('ScopeInfo', DHCP_OPTION_SCOPE_UNION))

class LPDHCP_OPTION_SCOPE_INFO(NDRPOINTER):
    referent = ('Data', DHCP_OPTION_SCOPE_INFO)

class DWORD_DWORD(NDRSTRUCT):
    structure = (('DWord1', DWORD), ('DWord2', DWORD))

class DHCP_BOOTP_IP_RANGE(NDRSTRUCT):
    structure = (('StartAddress', DHCP_IP_ADDRESS), ('EndAddress', DHCP_IP_ADDRESS), ('BootpAllocated', ULONG), ('MaxBootpAllowed', DHCP_IP_ADDRESS), ('MaxBootpAllowed', ULONG))

class DHCP_IP_RESERVATION_V4(NDRSTRUCT):
    structure = (('ReservedIpAddress', DHCP_IP_ADDRESS), ('ReservedForClient', DHCP_CLIENT_UID), ('bAllowedClientTypes', BYTE))

class DHCP_IP_RANGE(NDRSTRUCT):
    structure = (('StartAddress', DHCP_IP_ADDRESS), ('EndAddress', DHCP_IP_ADDRESS))

class DHCP_IP_CLUSTER(NDRSTRUCT):
    structure = (('ClusterAddress', DHCP_IP_ADDRESS), ('ClusterMask', DWORD))

class DHCP_SUBNET_ELEMENT_TYPE(NDRENUM):

    class enumItems(Enum):
        DhcpIpRanges = 0
        DhcpSecondaryHosts = 1
        DhcpReservedIps = 2
        DhcpExcludedIpRanges = 3
        DhcpIpUsedClusters = 4
        DhcpIpRangesDhcpOnly = 5
        DhcpIpRangesDhcpBootp = 6
        DhcpIpRangesBootpOnly = 7

class DHCP_SUBNET_ELEMENT_UNION_V5(NDRUNION):
    union = {DHCP_SUBNET_ELEMENT_TYPE.DhcpIpRanges: ('IpRange', DHCP_BOOTP_IP_RANGE), DHCP_SUBNET_ELEMENT_TYPE.DhcpSecondaryHosts: ('SecondaryHost', DHCP_HOST_INFO), DHCP_SUBNET_ELEMENT_TYPE.DhcpReservedIps: ('ReservedIp', DHCP_IP_RESERVATION_V4), DHCP_SUBNET_ELEMENT_TYPE.DhcpExcludedIpRanges: ('ExcludeIpRange', DHCP_IP_RANGE), DHCP_SUBNET_ELEMENT_TYPE.DhcpIpUsedClusters: ('IpUsedCluster', DHCP_IP_CLUSTER)}

class DHCP_SUBNET_ELEMENT_DATA_V5(NDRSTRUCT):
    structure = (('ElementType', DHCP_SUBNET_ELEMENT_TYPE), ('Element', DHCP_SUBNET_ELEMENT_UNION_V5))

class LPDHCP_SUBNET_ELEMENT_DATA_V5(NDRUniConformantArray):
    item = DHCP_SUBNET_ELEMENT_DATA_V5

class DHCP_SUBNET_ELEMENT_INFO_ARRAY_V5(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Elements', LPDHCP_SUBNET_ELEMENT_DATA_V5))

class LPDHCP_SUBNET_ELEMENT_INFO_ARRAY_V5(NDRPOINTER):
    referent = ('Data', DHCP_SUBNET_ELEMENT_INFO_ARRAY_V5)

class DHCP_OPTION_DATA_TYPE(NDRENUM):

    class enumItems(Enum):
        DhcpByteOption = 0
        DhcpWordOption = 1
        DhcpDWordOption = 2
        DhcpDWordDWordOption = 3
        DhcpIpAddressOption = 4
        DhcpStringDataOption = 5
        DhcpBinaryDataOption = 6
        DhcpEncapsulatedDataOption = 7
        DhcpIpv6AddressOption = 8

class DHCP_OPTION_ELEMENT_UNION(NDRUNION):
    commonHdr = (('tag', DHCP_OPTION_DATA_TYPE),)
    union = {DHCP_OPTION_DATA_TYPE.DhcpByteOption: ('ByteOption', BYTE), DHCP_OPTION_DATA_TYPE.DhcpWordOption: ('WordOption', WORD), DHCP_OPTION_DATA_TYPE.DhcpDWordOption: ('DWordOption', DWORD), DHCP_OPTION_DATA_TYPE.DhcpDWordDWordOption: ('DWordDWordOption', DWORD_DWORD), DHCP_OPTION_DATA_TYPE.DhcpIpAddressOption: ('IpAddressOption', DHCP_IP_ADDRESS), DHCP_OPTION_DATA_TYPE.DhcpStringDataOption: ('StringDataOption', LPWSTR), DHCP_OPTION_DATA_TYPE.DhcpBinaryDataOption: ('BinaryDataOption', DHCP_BINARY_DATA), DHCP_OPTION_DATA_TYPE.DhcpEncapsulatedDataOption: ('EncapsulatedDataOption', DHCP_BINARY_DATA), DHCP_OPTION_DATA_TYPE.DhcpIpv6AddressOption: ('Ipv6AddressDataOption', LPWSTR)}

class DHCP_OPTION_DATA_ELEMENT(NDRSTRUCT):
    structure = (('OptionType', DHCP_OPTION_DATA_TYPE), ('Element', DHCP_OPTION_ELEMENT_UNION))

class DHCP_OPTION_DATA_ELEMENT_ARRAY2(NDRUniConformantArray):
    item = DHCP_OPTION_DATA_ELEMENT

class LPDHCP_OPTION_DATA_ELEMENT(NDRPOINTER):
    referent = (('Data', DHCP_OPTION_DATA_ELEMENT_ARRAY2),)

class DHCP_OPTION_DATA(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Elements', LPDHCP_OPTION_DATA_ELEMENT))

class DHCP_OPTION_VALUE(NDRSTRUCT):
    structure = (('OptionID', DHCP_OPTION_ID), ('Value', DHCP_OPTION_DATA))

class PDHCP_OPTION_VALUE(NDRPOINTER):
    referent = (('Data', DHCP_OPTION_VALUE),)

class DHCP_OPTION_VALUE_ARRAY2(NDRUniConformantArray):
    item = DHCP_OPTION_VALUE

class LPDHCP_OPTION_VALUE(NDRPOINTER):
    referent = (('Data', DHCP_OPTION_VALUE_ARRAY2),)

class DHCP_OPTION_VALUE_ARRAY(NDRSTRUCT):
    structure = (('NumElements', DWORD), ('Values', LPDHCP_OPTION_VALUE))

class LPDHCP_OPTION_VALUE_ARRAY(NDRPOINTER):
    referent = (('Data', DHCP_OPTION_VALUE_ARRAY),)

class DHCP_ALL_OPTION_VALUES(NDRSTRUCT):
    structure = (('ClassName', LPWSTR), ('VendorName', LPWSTR), ('IsVendor', BOOL), ('OptionsArray', LPDHCP_OPTION_VALUE_ARRAY))

class OPTION_VALUES_ARRAY(NDRUniConformantArray):
    item = DHCP_ALL_OPTION_VALUES

class LPOPTION_VALUES_ARRAY(NDRPOINTER):
    referent = (('Data', OPTION_VALUES_ARRAY),)

class DHCP_ALL_OPTIONS_VALUES(NDRSTRUCT):
    structure = (('Flags', DWORD), ('NumElements', DWORD), ('Options', LPOPTION_VALUES_ARRAY))

class LPDHCP_ALL_OPTION_VALUES(NDRPOINTER):
    referent = (('Data', DHCP_ALL_OPTIONS_VALUES),)

class DhcpGetSubnetInfo(NDRCALL):
    opnum = 2
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SubnetAddress', DHCP_IP_ADDRESS))

class DhcpGetSubnetInfoResponse(NDRCALL):
    structure = (('SubnetInfo', LPDHCP_SUBNET_INFO), ('ErrorCode', ULONG))

class DhcpEnumSubnets(NDRCALL):
    opnum = 3
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumSubnetsResponse(NDRCALL):
    structure = (('ResumeHandle', LPDWORD), ('EnumInfo', DHCP_IP_ARRAY), ('EnumRead', DWORD), ('EnumTotal', DWORD), ('ErrorCode', ULONG))

class DhcpGetOptionValue(NDRCALL):
    opnum = 13
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('OptionID', DHCP_OPTION_ID), ('ScopeInfo', DHCP_OPTION_SCOPE_INFO))

class DhcpGetOptionValueResponse(NDRCALL):
    structure = (('OptionValue', PDHCP_OPTION_VALUE), ('ErrorCode', ULONG))

class DhcpEnumOptionValues(NDRCALL):
    opnum = 14
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('ScopeInfo', DHCP_OPTION_SCOPE_INFO), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumOptionValuesResponse(NDRCALL):
    structure = (('ResumeHandle', DWORD), ('OptionValues', LPDHCP_OPTION_VALUE_ARRAY), ('OptionsRead', DWORD), ('OptionsTotal', DWORD), ('ErrorCode', ULONG))

class DhcpGetClientInfoV4(NDRCALL):
    opnum = 34
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SearchInfo', DHCP_SEARCH_INFO))

class DhcpGetClientInfoV4Response(NDRCALL):
    structure = (('ClientInfo', LPDHCP_CLIENT_INFO_V4), ('ErrorCode', ULONG))

class DhcpEnumSubnetClientsV4(NDRCALL):
    opnum = 35
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SubnetAddress', DHCP_IP_ADDRESS), ('ResumeHandle', DWORD), ('PreferredMaximum', DWORD))

class DhcpEnumSubnetClientsV4Response(NDRCALL):
    structure = (('ResumeHandle', LPDWORD), ('ClientInfo', LPDHCP_CLIENT_INFO_ARRAY_V4), ('ClientsRead', DWORD), ('ClientsTotal', DWORD), ('ErrorCode', ULONG))

class DhcpEnumSubnetClientsV5(NDRCALL):
    opnum = 0
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SubnetAddress', DHCP_IP_ADDRESS), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumSubnetClientsV5Response(NDRCALL):
    structure = (('ResumeHandle', DWORD), ('ClientsInfo', LPDHCP_CLIENT_INFO_ARRAY_V5), ('ClientsRead', DWORD), ('ClientsTotal', DWORD))

class DhcpGetOptionValueV5(NDRCALL):
    opnum = 21
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('Flags', DWORD), ('OptionID', DHCP_OPTION_ID), ('ClassName', LPWSTR), ('VendorName', LPWSTR), ('ScopeInfo', DHCP_OPTION_SCOPE_INFO))

class DhcpGetOptionValueV5Response(NDRCALL):
    structure = (('OptionValue', PDHCP_OPTION_VALUE), ('ErrorCode', ULONG))

class DhcpEnumOptionValuesV5(NDRCALL):
    opnum = 22
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('Flags', DWORD), ('ClassName', LPWSTR), ('VendorName', LPWSTR), ('ScopeInfo', DHCP_OPTION_SCOPE_INFO), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumOptionValuesV5Response(NDRCALL):
    structure = (('ResumeHandle', DWORD), ('OptionValues', LPDHCP_OPTION_VALUE_ARRAY), ('OptionsRead', DWORD), ('OptionsTotal', DWORD), ('ErrorCode', ULONG))

class DhcpGetAllOptionValues(NDRCALL):
    opnum = 30
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('Flags', DWORD), ('ScopeInfo', DHCP_OPTION_SCOPE_INFO))

class DhcpGetAllOptionValuesResponse(NDRCALL):
    structure = (('Values', LPDHCP_ALL_OPTION_VALUES), ('ErrorCode', ULONG))

class DhcpEnumSubnetElementsV5(NDRCALL):
    opnum = 38
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SubnetAddress', DHCP_IP_ADDRESS), ('EnumElementType', DHCP_SUBNET_ELEMENT_TYPE), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumSubnetElementsV5Response(NDRCALL):
    structure = (('ResumeHandle', DWORD), ('EnumElementInfo', LPDHCP_SUBNET_ELEMENT_INFO_ARRAY_V5), ('ElementsRead', DWORD), ('ElementsTotal', DWORD), ('ErrorCode', ULONG))

class DhcpEnumSubnetClientsVQ(NDRCALL):
    opnum = 47
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SubnetAddress', DHCP_IP_ADDRESS), ('ResumeHandle', LPDWORD), ('PreferredMaximum', DWORD))

class DhcpEnumSubnetClientsVQResponse(NDRCALL):
    structure = (('ResumeHandle', LPDWORD), ('ClientInfo', LPDHCP_CLIENT_INFO_ARRAY_VQ), ('ClientsRead', DWORD), ('ClientsTotal', DWORD), ('ErrorCode', ULONG))

class DhcpV4GetClientInfo(NDRCALL):
    opnum = 123
    structure = (('ServerIpAddress', DHCP_SRV_HANDLE), ('SearchInfo', DHCP_SEARCH_INFO))

class DhcpV4GetClientInfoResponse(NDRCALL):
    structure = (('ClientInfo', LPDHCP_CLIENT_INFO_PB), ('ErrorCode', ULONG))
OPNUMS = {0: (DhcpEnumSubnetClientsV5, DhcpEnumSubnetClientsV5Response), 2: (DhcpGetSubnetInfo, DhcpGetSubnetInfoResponse), 3: (DhcpEnumSubnets, DhcpEnumSubnetsResponse), 13: (DhcpGetOptionValue, DhcpGetOptionValueResponse), 14: (DhcpEnumOptionValues, DhcpEnumOptionValuesResponse), 21: (DhcpGetOptionValueV5, DhcpGetOptionValueV5Response), 22: (DhcpEnumOptionValuesV5, DhcpEnumOptionValuesV5Response), 30: (DhcpGetAllOptionValues, DhcpGetAllOptionValuesResponse), 34: (DhcpGetClientInfoV4, DhcpGetClientInfoV4Response), 35: (DhcpEnumSubnetClientsV4, DhcpEnumSubnetClientsV4Response), 38: (DhcpEnumSubnetElementsV5, DhcpEnumSubnetElementsV5Response), 47: (DhcpEnumSubnetClientsVQ, DhcpEnumSubnetClientsVQResponse), 123: (DhcpV4GetClientInfo, DhcpV4GetClientInfoResponse)}

def hDhcpGetClientInfoV4(dce, searchType, searchValue):
    if False:
        while True:
            i = 10
    request = DhcpGetClientInfoV4()
    request['ServerIpAddress'] = NULL
    request['SearchInfo']['SearchType'] = searchType
    request['SearchInfo']['SearchInfo']['tag'] = searchType
    if searchType == DHCP_SEARCH_INFO_TYPE.DhcpClientIpAddress:
        request['SearchInfo']['SearchInfo']['ClientIpAddress'] = searchValue
    elif searchType == DHCP_SEARCH_INFO_TYPE.DhcpClientHardwareAddress:
        request['SearchInfo']['SearchInfo']['ClientHardwareAddress'] = searchValue
    else:
        request['SearchInfo']['SearchInfo']['ClientName'] = searchValue
    return dce.request(request)

def hDhcpGetSubnetInfo(dce, subnetaddress):
    if False:
        i = 10
        return i + 15
    request = DhcpGetSubnetInfo()
    request['ServerIpAddress'] = NULL
    request['SubnetAddress'] = subnetaddress
    resp = dce.request(request)
    return resp

def hDhcpGetOptionValue(dce, optionID, scopetype=DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions, options=NULL):
    if False:
        while True:
            i = 10
    request = DhcpGetOptionValue()
    request['ServerIpAddress'] = NULL
    request['OptionID'] = optionID
    request['ScopeInfo']['ScopeType'] = scopetype
    if scopetype != DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions and scopetype != DHCP_OPTION_SCOPE_TYPE.DhcpGlobalOptions:
        request['ScopeInfo']['ScopeInfo']['tag'] = scopetype
    if scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions:
        request['ScopeInfo']['ScopeInfo']['SubnetScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions:
        request['ScopeInfo']['ScopeInfo']['ReservedScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions:
        request['ScopeInfo']['ScopeInfo']['MScopeInfo'] = options
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumOptionValues(dce, scopetype=DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions, options=NULL, preferredMaximum=4294967295):
    if False:
        return 10
    request = DhcpEnumOptionValues()
    request['ServerIpAddress'] = NULL
    request['ScopeInfo']['ScopeType'] = scopetype
    if scopetype != DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions and scopetype != DHCP_OPTION_SCOPE_TYPE.DhcpGlobalOptions:
        request['ScopeInfo']['ScopeInfo']['tag'] = scopetype
    if scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions:
        request['ScopeInfo']['ScopeInfo']['SubnetScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions:
        request['ScopeInfo']['ScopeInfo']['ReservedScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions:
        request['ScopeInfo']['ScopeInfo']['MScopeInfo'] = options
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumOptionValuesV5(dce, flags=DHCP_FLAGS_OPTION_DEFAULT, classname=NULL, vendorname=NULL, scopetype=DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions, options=NULL, preferredMaximum=4294967295):
    if False:
        i = 10
        return i + 15
    request = DhcpEnumOptionValuesV5()
    request['ServerIpAddress'] = NULL
    request['Flags'] = flags
    request['ClassName'] = classname
    request['VendorName'] = vendorname
    request['ScopeInfo']['ScopeType'] = scopetype
    request['ScopeInfo']['ScopeInfo']['tag'] = scopetype
    if scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions:
        request['ScopeInfo']['ScopeInfo']['SubnetScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions:
        request['ScopeInfo']['ScopeInfo']['ReservedScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions:
        request['ScopeInfo']['ScopeInfo']['MScopeInfo'] = options
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpGetOptionValueV5(dce, option_id, flags=DHCP_FLAGS_OPTION_DEFAULT, classname=NULL, vendorname=NULL, scopetype=DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions, options=NULL):
    if False:
        i = 10
        return i + 15
    request = DhcpGetOptionValueV5()
    request['ServerIpAddress'] = NULL
    request['Flags'] = flags
    request['OptionID'] = option_id
    request['ClassName'] = classname
    request['VendorName'] = vendorname
    request['ScopeInfo']['ScopeType'] = scopetype
    request['ScopeInfo']['ScopeInfo']['tag'] = scopetype
    if scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions:
        request['ScopeInfo']['ScopeInfo']['SubnetScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions:
        request['ScopeInfo']['ScopeInfo']['ReservedScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions:
        request['ScopeInfo']['ScopeInfo']['MScopeInfo'] = options
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpGetAllOptionValues(dce, scopetype=DHCP_OPTION_SCOPE_TYPE.DhcpDefaultOptions, options=NULL):
    if False:
        print('Hello World!')
    request = DhcpGetAllOptionValues()
    request['ServerIpAddress'] = NULL
    request['Flags'] = NULL
    request['ScopeInfo']['ScopeType'] = scopetype
    request['ScopeInfo']['ScopeInfo']['tag'] = scopetype
    if scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions:
        request['ScopeInfo']['ScopeInfo']['SubnetScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpReservedOptions:
        request['ScopeInfo']['ScopeInfo']['ReservedScopeInfo'] = options
    elif scopetype == DHCP_OPTION_SCOPE_TYPE.DhcpMScopeOptions:
        request['ScopeInfo']['ScopeInfo']['MScopeInfo'] = options
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumSubnets(dce, preferredMaximum=4294967295):
    if False:
        i = 10
        return i + 15
    request = DhcpEnumSubnets()
    request['ServerIpAddress'] = NULL
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_MORE_ENTRIES') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumSubnetClientsVQ(dce, preferredMaximum=4294967295):
    if False:
        return 10
    request = DhcpEnumSubnetClientsVQ()
    request['ServerIpAddress'] = NULL
    request['SubnetAddress'] = NULL
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_MORE_ENTRIES') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumSubnetClientsV4(dce, preferredMaximum=4294967295):
    if False:
        i = 10
        return i + 15
    request = DhcpEnumSubnetClientsV4()
    request['ServerIpAddress'] = NULL
    request['SubnetAddress'] = NULL
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_MORE_ENTRIES') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumSubnetClientsV5(dce, subnetAddress=0, preferredMaximum=4294967295):
    if False:
        for i in range(10):
            print('nop')
    request = DhcpEnumSubnetClientsV5()
    request['ServerIpAddress'] = NULL
    request['SubnetAddress'] = subnetAddress
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCSessionError as e:
            if str(e).find('STATUS_MORE_ENTRIES') < 0:
                raise
            resp = e.get_packet()
        return resp

def hDhcpEnumSubnetElementsV5(dce, subnet_address, element_type=DHCP_SUBNET_ELEMENT_TYPE.DhcpIpRanges, preferredMaximum=4294967295):
    if False:
        i = 10
        return i + 15
    request = DhcpEnumSubnetElementsV5()
    request['ServerIpAddress'] = NULL
    request['SubnetAddress'] = subnet_address
    request['EnumElementType'] = element_type
    request['ResumeHandle'] = NULL
    request['PreferredMaximum'] = preferredMaximum
    status = system_errors.ERROR_MORE_DATA
    while status == system_errors.ERROR_MORE_DATA:
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_MORE_ITEMS') < 0:
                raise
            resp = e.get_packet()
        return resp