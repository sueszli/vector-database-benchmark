from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRUNION, NDRPOINTER, NDRENUM
from impacket.dcerpc.v5.dtypes import UINT, LPWSTR, GUID
from impacket import system_errors
from impacket.dcerpc.v5.enum import Enum
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_DSSP = uuidtup_to_bin(('3919286A-B10C-11D0-9BA8-00C04FD92EF5', '0.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            for i in range(10):
                print('nop')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        key = self.error_code
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'DSSP SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'DSSP SessionError: unknown error code: 0x%x' % self.error_code
DSROLE_PRIMARY_DS_RUNNING = 1
DSROLE_PRIMARY_DS_MIXED_MODE = 2
DSROLE_PRIMARY_DS_READONLY = 8
DSROLE_PRIMARY_DOMAIN_GUID_PRESENT = 16777216
DSROLE_UPGRADE_IN_PROGRESS = 4

class DSROLE_MACHINE_ROLE(NDRENUM):

    class enumItems(Enum):
        DsRole_RoleStandaloneWorkstation = 0
        DsRole_RoleMemberWorkstation = 1
        DsRole_RoleStandaloneServer = 2
        DsRole_RoleMemberServer = 3
        DsRole_RoleBackupDomainController = 4
        DsRole_RolePrimaryDomainController = 5

class DSROLER_PRIMARY_DOMAIN_INFO_BASIC(NDRSTRUCT):
    structure = (('MachineRole', DSROLE_MACHINE_ROLE), ('Flags', UINT), ('DomainNameFlat', LPWSTR), ('DomainNameDns', LPWSTR), ('DomainForestName', LPWSTR), ('DomainGuid', GUID))

class PDSROLER_PRIMARY_DOMAIN_INFO_BASIC(NDRPOINTER):
    referent = (('Data', DSROLER_PRIMARY_DOMAIN_INFO_BASIC),)

class DSROLE_OPERATION_STATE(NDRENUM):

    class enumItems(Enum):
        DsRoleOperationIdle = 0
        DsRoleOperationActive = 1
        DsRoleOperationNeedReboot = 2

class DSROLE_OPERATION_STATE_INFO(NDRSTRUCT):
    structure = (('OperationState', DSROLE_OPERATION_STATE),)

class PDSROLE_OPERATION_STATE_INFO(NDRPOINTER):
    referent = (('Data', DSROLE_OPERATION_STATE_INFO),)

class DSROLE_SERVER_STATE(NDRENUM):

    class enumItems(Enum):
        DsRoleServerUnknown = 0
        DsRoleServerPrimary = 1
        DsRoleServerBackup = 2

class PDSROLE_SERVER_STATE(NDRPOINTER):
    referent = (('Data', DSROLE_SERVER_STATE),)

class DSROLE_UPGRADE_STATUS_INFO(NDRSTRUCT):
    structure = (('OperationState', UINT), ('PreviousServerState', DSROLE_SERVER_STATE))

class PDSROLE_UPGRADE_STATUS_INFO(NDRPOINTER):
    referent = (('Data', DSROLE_UPGRADE_STATUS_INFO),)

class DSROLE_PRIMARY_DOMAIN_INFO_LEVEL(NDRENUM):

    class enumItems(Enum):
        DsRolePrimaryDomainInfoBasic = 1
        DsRoleUpgradeStatus = 2
        DsRoleOperationState = 3

class DSROLER_PRIMARY_DOMAIN_INFORMATION(NDRUNION):
    commonHdr = (('tag', DSROLE_PRIMARY_DOMAIN_INFO_LEVEL),)
    union = {DSROLE_PRIMARY_DOMAIN_INFO_LEVEL.DsRolePrimaryDomainInfoBasic: ('DomainInfoBasic', DSROLER_PRIMARY_DOMAIN_INFO_BASIC), DSROLE_PRIMARY_DOMAIN_INFO_LEVEL.DsRoleUpgradeStatus: ('UpgradStatusInfo', DSROLE_UPGRADE_STATUS_INFO), DSROLE_PRIMARY_DOMAIN_INFO_LEVEL.DsRoleOperationState: ('OperationStateInfo', DSROLE_OPERATION_STATE_INFO)}

class PDSROLER_PRIMARY_DOMAIN_INFORMATION(NDRPOINTER):
    referent = (('Data', DSROLER_PRIMARY_DOMAIN_INFORMATION),)

class DsRolerGetPrimaryDomainInformation(NDRCALL):
    opnum = 0
    structure = (('InfoLevel', DSROLE_PRIMARY_DOMAIN_INFO_LEVEL),)

class DsRolerGetPrimaryDomainInformationResponse(NDRCALL):
    structure = (('DomainInfo', PDSROLER_PRIMARY_DOMAIN_INFORMATION),)
OPNUMS = {0: (DsRolerGetPrimaryDomainInformation, DsRolerGetPrimaryDomainInformationResponse)}

def hDsRolerGetPrimaryDomainInformation(dce, infoLevel):
    if False:
        print('Hello World!')
    request = DsRolerGetPrimaryDomainInformation()
    request['InfoLevel'] = infoLevel
    return dce.request(request)