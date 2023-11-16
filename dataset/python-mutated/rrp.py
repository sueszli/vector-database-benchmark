from struct import unpack, pack
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRPOINTER, NDRUniConformantVaryingArray, NDRUniConformantArray
from impacket.dcerpc.v5.dtypes import DWORD, UUID, ULONG, LPULONG, BOOLEAN, SECURITY_INFORMATION, PFILETIME, RPC_UNICODE_STRING, FILETIME, NULL, MAXIMUM_ALLOWED, OWNER_SECURITY_INFORMATION, PWCHAR, PRPC_UNICODE_STRING
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket import system_errors, LOG
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_RRP = uuidtup_to_bin(('338CD001-2244-31F1-AAAA-900038001003', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            for i in range(10):
                print('nop')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            while True:
                i = 10
        key = self.error_code
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'RRP SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'RRP SessionError: unknown error code: 0x%x' % self.error_code
PREGISTRY_SERVER_NAME = PWCHAR
error_status_t = ULONG
RRP_UNICODE_STRING = RPC_UNICODE_STRING
PRRP_UNICODE_STRING = PRPC_UNICODE_STRING
REGSAM = ULONG
KEY_QUERY_VALUE = 1
KEY_SET_VALUE = 2
KEY_CREATE_SUB_KEY = 4
KEY_ENUMERATE_SUB_KEYS = 8
KEY_CREATE_LINK = 32
KEY_WOW64_64KEY = 256
KEY_WOW64_32KEY = 512
KEY_READ = 131097
REG_BINARY = 3
REG_DWORD = 4
REG_DWORD_LITTLE_ENDIAN = 4
REG_DWORD_BIG_ENDIAN = 5
REG_EXPAND_SZ = 2
REG_LINK = 6
REG_MULTI_SZ = 7
REG_NONE = 0
REG_QWORD = 11
REG_QWORD_LITTLE_ENDIAN = 11
REG_SZ = 1
REG_OPTION_BACKUP_RESTORE = 4
REG_OPTION_OPEN_LINK = 8
REG_CREATED_NEW_KEY = 1
REG_OPENED_EXISTING_KEY = 2
REG_WHOLE_HIVE_VOLATILE = 1
REG_REFRESH_HIVE = 2
REG_NO_LAZY_FLUSH = 4
REG_FORCE_RESTORE = 8

class RPC_HKEY(NDRSTRUCT):
    structure = (('context_handle_attributes', ULONG), ('context_handle_uuid', UUID))

    def __init__(self, data=None, isNDR64=False):
        if False:
            i = 10
            return i + 15
        NDRSTRUCT.__init__(self, data, isNDR64)
        self['context_handle_uuid'] = b'\x00' * 16

    def isNull(self):
        if False:
            return 10
        return self['context_handle_uuid'] == b'\x00' * 16

class RVALENT(NDRSTRUCT):
    structure = (('ve_valuename', PRRP_UNICODE_STRING), ('ve_valuelen', DWORD), ('ve_valueptr', DWORD), ('ve_type', DWORD))

class RVALENT_ARRAY(NDRUniConformantVaryingArray):
    item = RVALENT

class BYTE_ARRAY(NDRUniConformantVaryingArray):
    pass

class PBYTE_ARRAY(NDRPOINTER):
    referent = (('Data', BYTE_ARRAY),)

class RPC_SECURITY_DESCRIPTOR(NDRSTRUCT):
    structure = (('lpSecurityDescriptor', PBYTE_ARRAY), ('cbInSecurityDescriptor', DWORD), ('cbOutSecurityDescriptor', DWORD))

class RPC_SECURITY_ATTRIBUTES(NDRSTRUCT):
    structure = (('nLength', DWORD), ('RpcSecurityDescriptor', RPC_SECURITY_DESCRIPTOR), ('bInheritHandle', BOOLEAN))

class PRPC_SECURITY_ATTRIBUTES(NDRPOINTER):
    referent = (('Data', RPC_SECURITY_ATTRIBUTES),)

class OpenClassesRoot(NDRCALL):
    opnum = 0
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenClassesRootResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class OpenCurrentUser(NDRCALL):
    opnum = 1
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenCurrentUserResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class OpenLocalMachine(NDRCALL):
    opnum = 2
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenLocalMachineResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class OpenPerformanceData(NDRCALL):
    opnum = 3
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenPerformanceDataResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class OpenUsers(NDRCALL):
    opnum = 4
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenUsersResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class BaseRegCloseKey(NDRCALL):
    opnum = 5
    structure = (('hKey', RPC_HKEY),)

class BaseRegCloseKeyResponse(NDRCALL):
    structure = (('hKey', RPC_HKEY), ('ErrorCode', error_status_t))

class BaseRegCreateKey(NDRCALL):
    opnum = 6
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING), ('lpClass', RRP_UNICODE_STRING), ('dwOptions', DWORD), ('samDesired', REGSAM), ('lpSecurityAttributes', PRPC_SECURITY_ATTRIBUTES), ('lpdwDisposition', LPULONG))

class BaseRegCreateKeyResponse(NDRCALL):
    structure = (('phkResult', RPC_HKEY), ('lpdwDisposition', LPULONG), ('ErrorCode', error_status_t))

class BaseRegDeleteKey(NDRCALL):
    opnum = 7
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING))

class BaseRegDeleteKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegDeleteValue(NDRCALL):
    opnum = 8
    structure = (('hKey', RPC_HKEY), ('lpValueName', RRP_UNICODE_STRING))

class BaseRegDeleteValueResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegEnumKey(NDRCALL):
    opnum = 9
    structure = (('hKey', RPC_HKEY), ('dwIndex', DWORD), ('lpNameIn', RRP_UNICODE_STRING), ('lpClassIn', PRRP_UNICODE_STRING), ('lpftLastWriteTime', PFILETIME))

class BaseRegEnumKeyResponse(NDRCALL):
    structure = (('lpNameOut', RRP_UNICODE_STRING), ('lplpClassOut', PRRP_UNICODE_STRING), ('lpftLastWriteTime', PFILETIME), ('ErrorCode', error_status_t))

class BaseRegEnumValue(NDRCALL):
    opnum = 10
    structure = (('hKey', RPC_HKEY), ('dwIndex', DWORD), ('lpValueNameIn', RRP_UNICODE_STRING), ('lpType', LPULONG), ('lpData', PBYTE_ARRAY), ('lpcbData', LPULONG), ('lpcbLen', LPULONG))

class BaseRegEnumValueResponse(NDRCALL):
    structure = (('lpValueNameOut', RRP_UNICODE_STRING), ('lpType', LPULONG), ('lpData', PBYTE_ARRAY), ('lpcbData', LPULONG), ('lpcbLen', LPULONG), ('ErrorCode', error_status_t))

class BaseRegFlushKey(NDRCALL):
    opnum = 11
    structure = (('hKey', RPC_HKEY),)

class BaseRegFlushKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegGetKeySecurity(NDRCALL):
    opnum = 12
    structure = (('hKey', RPC_HKEY), ('SecurityInformation', SECURITY_INFORMATION), ('pRpcSecurityDescriptorIn', RPC_SECURITY_DESCRIPTOR))

class BaseRegGetKeySecurityResponse(NDRCALL):
    structure = (('pRpcSecurityDescriptorOut', RPC_SECURITY_DESCRIPTOR), ('ErrorCode', error_status_t))

class BaseRegLoadKey(NDRCALL):
    opnum = 13
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING), ('lpFile', RRP_UNICODE_STRING))

class BaseRegLoadKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegOpenKey(NDRCALL):
    opnum = 15
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING), ('dwOptions', DWORD), ('samDesired', REGSAM))

class BaseRegOpenKeyResponse(NDRCALL):
    structure = (('phkResult', RPC_HKEY), ('ErrorCode', error_status_t))

class BaseRegQueryInfoKey(NDRCALL):
    opnum = 16
    structure = (('hKey', RPC_HKEY), ('lpClassIn', RRP_UNICODE_STRING))

class BaseRegQueryInfoKeyResponse(NDRCALL):
    structure = (('lpClassOut', RPC_UNICODE_STRING), ('lpcSubKeys', DWORD), ('lpcbMaxSubKeyLen', DWORD), ('lpcbMaxClassLen', DWORD), ('lpcValues', DWORD), ('lpcbMaxValueNameLen', DWORD), ('lpcbMaxValueLen', DWORD), ('lpcbSecurityDescriptor', DWORD), ('lpftLastWriteTime', FILETIME), ('ErrorCode', error_status_t))

class BaseRegQueryValue(NDRCALL):
    opnum = 17
    structure = (('hKey', RPC_HKEY), ('lpValueName', RRP_UNICODE_STRING), ('lpType', LPULONG), ('lpData', PBYTE_ARRAY), ('lpcbData', LPULONG), ('lpcbLen', LPULONG))

class BaseRegQueryValueResponse(NDRCALL):
    structure = (('lpType', LPULONG), ('lpData', PBYTE_ARRAY), ('lpcbData', LPULONG), ('lpcbLen', LPULONG), ('ErrorCode', error_status_t))

class BaseRegReplaceKey(NDRCALL):
    opnum = 18
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING), ('lpNewFile', RRP_UNICODE_STRING), ('lpOldFile', RRP_UNICODE_STRING))

class BaseRegReplaceKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegRestoreKey(NDRCALL):
    opnum = 19
    structure = (('hKey', RPC_HKEY), ('lpFile', RRP_UNICODE_STRING), ('Flags', DWORD))

class BaseRegRestoreKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegSaveKey(NDRCALL):
    opnum = 20
    structure = (('hKey', RPC_HKEY), ('lpFile', RRP_UNICODE_STRING), ('pSecurityAttributes', PRPC_SECURITY_ATTRIBUTES))

class BaseRegSaveKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegSetKeySecurity(NDRCALL):
    opnum = 21
    structure = (('hKey', RPC_HKEY), ('SecurityInformation', SECURITY_INFORMATION), ('pRpcSecurityDescriptor', RPC_SECURITY_DESCRIPTOR))

class BaseRegSetKeySecurityResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegSetValue(NDRCALL):
    opnum = 22
    structure = (('hKey', RPC_HKEY), ('lpValueName', RRP_UNICODE_STRING), ('dwType', DWORD), ('lpData', NDRUniConformantArray), ('cbData', DWORD))

class BaseRegSetValueResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegUnLoadKey(NDRCALL):
    opnum = 23
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING))

class BaseRegUnLoadKeyResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class BaseRegGetVersion(NDRCALL):
    opnum = 26
    structure = (('hKey', RPC_HKEY),)

class BaseRegGetVersionResponse(NDRCALL):
    structure = (('lpdwVersion', DWORD), ('ErrorCode', error_status_t))

class OpenCurrentConfig(NDRCALL):
    opnum = 27
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenCurrentConfigResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class BaseRegQueryMultipleValues(NDRCALL):
    opnum = 29
    structure = (('hKey', RPC_HKEY), ('val_listIn', RVALENT_ARRAY), ('num_vals', DWORD), ('lpvalueBuf', PBYTE_ARRAY), ('ldwTotsize', DWORD))

class BaseRegQueryMultipleValuesResponse(NDRCALL):
    structure = (('val_listOut', RVALENT_ARRAY), ('lpvalueBuf', PBYTE_ARRAY), ('ldwTotsize', DWORD), ('ErrorCode', error_status_t))

class BaseRegSaveKeyEx(NDRCALL):
    opnum = 31
    structure = (('hKey', RPC_HKEY), ('lpFile', RRP_UNICODE_STRING), ('pSecurityAttributes', PRPC_SECURITY_ATTRIBUTES), ('Flags', DWORD))

class BaseRegSaveKeyExResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)

class OpenPerformanceText(NDRCALL):
    opnum = 32
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenPerformanceTextResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class OpenPerformanceNlsText(NDRCALL):
    opnum = 33
    structure = (('ServerName', PREGISTRY_SERVER_NAME), ('samDesired', REGSAM))

class OpenPerformanceNlsTextResponse(NDRCALL):
    structure = (('phKey', RPC_HKEY), ('ErrorCode', error_status_t))

class BaseRegQueryMultipleValues2(NDRCALL):
    opnum = 34
    structure = (('hKey', RPC_HKEY), ('val_listIn', RVALENT_ARRAY), ('num_vals', DWORD), ('lpvalueBuf', PBYTE_ARRAY), ('ldwTotsize', DWORD))

class BaseRegQueryMultipleValues2Response(NDRCALL):
    structure = (('val_listOut', RVALENT_ARRAY), ('lpvalueBuf', PBYTE_ARRAY), ('ldwRequiredSize', DWORD), ('ErrorCode', error_status_t))

class BaseRegDeleteKeyEx(NDRCALL):
    opnum = 35
    structure = (('hKey', RPC_HKEY), ('lpSubKey', RRP_UNICODE_STRING), ('AccessMask', REGSAM), ('Reserved', DWORD))

class BaseRegDeleteKeyExResponse(NDRCALL):
    structure = (('ErrorCode', error_status_t),)
OPNUMS = {0: (OpenClassesRoot, OpenClassesRootResponse), 1: (OpenCurrentUser, OpenCurrentUserResponse), 2: (OpenLocalMachine, OpenLocalMachineResponse), 3: (OpenPerformanceData, OpenPerformanceDataResponse), 4: (OpenUsers, OpenUsersResponse), 5: (BaseRegCloseKey, BaseRegCloseKeyResponse), 6: (BaseRegCreateKey, BaseRegCreateKeyResponse), 7: (BaseRegDeleteKey, BaseRegDeleteKeyResponse), 8: (BaseRegDeleteValue, BaseRegDeleteValueResponse), 9: (BaseRegEnumKey, BaseRegEnumKeyResponse), 10: (BaseRegEnumValue, BaseRegEnumValueResponse), 11: (BaseRegFlushKey, BaseRegFlushKeyResponse), 12: (BaseRegGetKeySecurity, BaseRegGetKeySecurityResponse), 13: (BaseRegLoadKey, BaseRegLoadKeyResponse), 15: (BaseRegOpenKey, BaseRegOpenKeyResponse), 16: (BaseRegQueryInfoKey, BaseRegQueryInfoKeyResponse), 17: (BaseRegQueryValue, BaseRegQueryValueResponse), 18: (BaseRegReplaceKey, BaseRegReplaceKeyResponse), 19: (BaseRegRestoreKey, BaseRegRestoreKeyResponse), 20: (BaseRegSaveKey, BaseRegSaveKeyResponse), 21: (BaseRegSetKeySecurity, BaseRegSetKeySecurityResponse), 22: (BaseRegSetValue, BaseRegSetValueResponse), 23: (BaseRegUnLoadKey, BaseRegUnLoadKeyResponse), 26: (BaseRegGetVersion, BaseRegGetVersionResponse), 27: (OpenCurrentConfig, OpenCurrentConfigResponse), 29: (BaseRegQueryMultipleValues, BaseRegQueryMultipleValuesResponse), 31: (BaseRegSaveKeyEx, BaseRegSaveKeyExResponse), 32: (OpenPerformanceText, OpenPerformanceTextResponse), 33: (OpenPerformanceNlsText, OpenPerformanceNlsTextResponse), 34: (BaseRegQueryMultipleValues2, BaseRegQueryMultipleValues2Response), 35: (BaseRegDeleteKeyEx, BaseRegDeleteKeyExResponse)}

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

def packValue(valueType, value):
    if False:
        while True:
            i = 10
    if valueType == REG_DWORD:
        retData = pack('<L', value)
    elif valueType == REG_DWORD_BIG_ENDIAN:
        retData = pack('>L', value)
    elif valueType == REG_EXPAND_SZ:
        try:
            retData = value.encode('utf-16le')
        except UnicodeDecodeError:
            import sys
            retData = value.decode(sys.getfilesystemencoding()).encode('utf-16le')
    elif valueType == REG_MULTI_SZ:
        try:
            retData = value.encode('utf-16le')
        except UnicodeDecodeError:
            import sys
            retData = value.decode(sys.getfilesystemencoding()).encode('utf-16le')
    elif valueType == REG_QWORD:
        retData = pack('<Q', value)
    elif valueType == REG_QWORD_LITTLE_ENDIAN:
        retData = pack('>Q', value)
    elif valueType == REG_SZ:
        try:
            retData = value.encode('utf-16le')
        except UnicodeDecodeError:
            import sys
            retData = value.decode(sys.getfilesystemencoding()).encode('utf-16le')
    else:
        retData = value
    return retData

def unpackValue(valueType, value):
    if False:
        while True:
            i = 10
    if valueType == REG_DWORD:
        retData = unpack('<L', b''.join(value))[0]
    elif valueType == REG_DWORD_BIG_ENDIAN:
        retData = unpack('>L', b''.join(value))[0]
    elif valueType == REG_EXPAND_SZ:
        retData = b''.join(value).decode('utf-16le')
    elif valueType == REG_MULTI_SZ:
        retData = b''.join(value).decode('utf-16le')
    elif valueType == REG_QWORD:
        retData = unpack('<Q', b''.join(value))[0]
    elif valueType == REG_QWORD_LITTLE_ENDIAN:
        retData = unpack('>Q', b''.join(value))[0]
    elif valueType == REG_SZ:
        retData = b''.join(value).decode('utf-16le')
    else:
        retData = b''.join(value)
    return retData

def hOpenClassesRoot(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        return 10
    request = OpenClassesRoot()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hOpenCurrentUser(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        return 10
    request = OpenCurrentUser()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hOpenLocalMachine(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        return 10
    request = OpenLocalMachine()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hOpenPerformanceData(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        i = 10
        return i + 15
    request = OpenPerformanceData()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hOpenUsers(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        i = 10
        return i + 15
    request = OpenUsers()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hBaseRegCloseKey(dce, hKey):
    if False:
        i = 10
        return i + 15
    request = BaseRegCloseKey()
    request['hKey'] = hKey
    return dce.request(request)

def hBaseRegCreateKey(dce, hKey, lpSubKey, lpClass=NULL, dwOptions=1, samDesired=MAXIMUM_ALLOWED, lpSecurityAttributes=NULL, lpdwDisposition=REG_CREATED_NEW_KEY):
    if False:
        return 10
    request = BaseRegCreateKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    request['lpClass'] = checkNullString(lpClass)
    request['dwOptions'] = dwOptions
    request['samDesired'] = samDesired
    if lpSecurityAttributes == NULL:
        request['lpSecurityAttributes']['RpcSecurityDescriptor']['lpSecurityDescriptor'] = NULL
    else:
        request['lpSecurityAttributes'] = lpSecurityAttributes
    request['lpdwDisposition'] = lpdwDisposition
    return dce.request(request)

def hBaseRegDeleteKey(dce, hKey, lpSubKey):
    if False:
        i = 10
        return i + 15
    request = BaseRegDeleteKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    return dce.request(request)

def hBaseRegEnumKey(dce, hKey, dwIndex, lpftLastWriteTime=NULL):
    if False:
        i = 10
        return i + 15
    request = BaseRegEnumKey()
    request['hKey'] = hKey
    request['dwIndex'] = dwIndex
    request.fields['lpNameIn'].fields['MaximumLength'] = 1024
    request.fields['lpNameIn'].fields['Data'].fields['Data'].fields['MaximumCount'] = 1024 // 2
    request['lpClassIn'] = ' ' * 64
    request['lpftLastWriteTime'] = lpftLastWriteTime
    return dce.request(request)

def hBaseRegEnumValue(dce, hKey, dwIndex, dataLen=256):
    if False:
        return 10
    request = BaseRegEnumValue()
    request['hKey'] = hKey
    request['dwIndex'] = dwIndex
    retries = 1
    while True:
        try:
            request.fields['lpValueNameIn'].fields['MaximumLength'] = dataLen * 2
            request.fields['lpValueNameIn'].fields['Data'].fields['Data'].fields['MaximumCount'] = dataLen
            request['lpData'] = b' ' * dataLen
            request['lpcbData'] = dataLen
            request['lpcbLen'] = dataLen
            resp = dce.request(request)
        except DCERPCSessionError as e:
            if retries > 1:
                LOG.debug('Too many retries when calling hBaseRegEnumValue, aborting')
                raise
            if e.get_error_code() == system_errors.ERROR_MORE_DATA:
                retries += 1
                dataLen = e.get_packet()['lpcbData']
                continue
            else:
                raise
        else:
            break
    return resp

def hBaseRegFlushKey(dce, hKey):
    if False:
        print('Hello World!')
    request = BaseRegFlushKey()
    request['hKey'] = hKey
    return dce.request(request)

def hBaseRegGetKeySecurity(dce, hKey, securityInformation=OWNER_SECURITY_INFORMATION):
    if False:
        while True:
            i = 10
    request = BaseRegGetKeySecurity()
    request['hKey'] = hKey
    request['SecurityInformation'] = securityInformation
    request['pRpcSecurityDescriptorIn']['lpSecurityDescriptor'] = NULL
    request['pRpcSecurityDescriptorIn']['cbInSecurityDescriptor'] = 1024
    return dce.request(request)

def hBaseRegLoadKey(dce, hKey, lpSubKey, lpFile):
    if False:
        i = 10
        return i + 15
    request = BaseRegLoadKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    request['lpFile'] = checkNullString(lpFile)
    return dce.request(request)

def hBaseRegUnLoadKey(dce, hKey, lpSubKey):
    if False:
        while True:
            i = 10
    request = BaseRegUnLoadKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    return dce.request(request)

def hBaseRegOpenKey(dce, hKey, lpSubKey, dwOptions=1, samDesired=MAXIMUM_ALLOWED):
    if False:
        print('Hello World!')
    request = BaseRegOpenKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    request['dwOptions'] = dwOptions
    request['samDesired'] = samDesired
    return dce.request(request)

def hBaseRegQueryInfoKey(dce, hKey):
    if False:
        return 10
    request = BaseRegQueryInfoKey()
    request['hKey'] = hKey
    request.fields['lpClassIn'].fields['MaximumLength'] = 1024
    request.fields['lpClassIn'].fields['Data'].fields['Data'].fields['MaximumCount'] = 1024 // 2
    return dce.request(request)

def hBaseRegQueryValue(dce, hKey, lpValueName, dataLen=512):
    if False:
        return 10
    request = BaseRegQueryValue()
    request['hKey'] = hKey
    request['lpValueName'] = checkNullString(lpValueName)
    retries = 1
    while True:
        try:
            request['lpData'] = b' ' * dataLen
            request['lpcbData'] = dataLen
            request['lpcbLen'] = dataLen
            resp = dce.request(request)
        except DCERPCSessionError as e:
            if retries > 1:
                LOG.debug('Too many retries when calling hBaseRegQueryValue, aborting')
                raise
            if e.get_error_code() == system_errors.ERROR_MORE_DATA:
                dataLen = e.get_packet()['lpcbData']
                continue
            else:
                raise
        else:
            break
    return (resp['lpType'], unpackValue(resp['lpType'], resp['lpData']))

def hBaseRegReplaceKey(dce, hKey, lpSubKey, lpNewFile, lpOldFile):
    if False:
        return 10
    request = BaseRegReplaceKey()
    request['hKey'] = hKey
    request['lpSubKey'] = checkNullString(lpSubKey)
    request['lpNewFile'] = checkNullString(lpNewFile)
    request['lpOldFile'] = checkNullString(lpOldFile)
    return dce.request(request)

def hBaseRegRestoreKey(dce, hKey, lpFile, flags=REG_REFRESH_HIVE):
    if False:
        while True:
            i = 10
    request = BaseRegRestoreKey()
    request['hKey'] = hKey
    request['lpFile'] = checkNullString(lpFile)
    request['Flags'] = flags
    return dce.request(request)

def hBaseRegSaveKey(dce, hKey, lpFile, pSecurityAttributes=NULL):
    if False:
        for i in range(10):
            print('nop')
    request = BaseRegSaveKey()
    request['hKey'] = hKey
    request['lpFile'] = checkNullString(lpFile)
    request['pSecurityAttributes'] = pSecurityAttributes
    return dce.request(request)

def hBaseRegSetValue(dce, hKey, lpValueName, dwType, lpData):
    if False:
        return 10
    request = BaseRegSetValue()
    request['hKey'] = hKey
    request['lpValueName'] = checkNullString(lpValueName)
    request['dwType'] = dwType
    request['lpData'] = packValue(dwType, lpData)
    request['cbData'] = len(request['lpData'])
    return dce.request(request)

def hBaseRegGetVersion(dce, hKey):
    if False:
        while True:
            i = 10
    request = BaseRegGetVersion()
    request['hKey'] = hKey
    return dce.request(request)

def hOpenCurrentConfig(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        print('Hello World!')
    request = OpenCurrentConfig()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hBaseRegQueryMultipleValues(dce, hKey, val_listIn):
    if False:
        while True:
            i = 10
    request = BaseRegQueryMultipleValues()
    request['hKey'] = hKey
    for item in val_listIn:
        itemn = RVALENT()
        itemn['ve_valuename'] = checkNullString(item['ValueName'])
        itemn['ve_valuelen'] = len(itemn['ve_valuename'])
        itemn['ve_valueptr'] = NULL
        itemn['ve_type'] = item['ValueType']
        request['val_listIn'].append(itemn)
    request['num_vals'] = len(request['val_listIn'])
    request['lpvalueBuf'] = list(b' ' * 128)
    request['ldwTotsize'] = 128
    resp = dce.request(request)
    retVal = list()
    for item in resp['val_listOut']:
        itemn = dict()
        itemn['ValueName'] = item['ve_valuename']
        itemn['ValueData'] = unpackValue(item['ve_type'], resp['lpvalueBuf'][item['ve_valueptr']:item['ve_valueptr'] + item['ve_valuelen']])
        retVal.append(itemn)
    return retVal

def hBaseRegSaveKeyEx(dce, hKey, lpFile, pSecurityAttributes=NULL, flags=1):
    if False:
        i = 10
        return i + 15
    request = BaseRegSaveKeyEx()
    request['hKey'] = hKey
    request['lpFile'] = checkNullString(lpFile)
    request['pSecurityAttributes'] = pSecurityAttributes
    request['Flags'] = flags
    return dce.request(request)

def hOpenPerformanceText(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        for i in range(10):
            print('nop')
    request = OpenPerformanceText()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hOpenPerformanceNlsText(dce, samDesired=MAXIMUM_ALLOWED):
    if False:
        for i in range(10):
            print('nop')
    request = OpenPerformanceNlsText()
    request['ServerName'] = NULL
    request['samDesired'] = samDesired
    return dce.request(request)

def hBaseRegDeleteValue(dce, hKey, lpValueName):
    if False:
        while True:
            i = 10
    request = BaseRegDeleteValue()
    request['hKey'] = hKey
    request['lpValueName'] = checkNullString(lpValueName)
    return dce.request(request)