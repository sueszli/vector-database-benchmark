from impacket import system_errors
from impacket.dcerpc.v5.dtypes import ULONGLONG, UINT, USHORT, LPWSTR, DWORD, ULONG, NULL
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRUNION, NDRPOINTER, NDRUniConformantArray
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.uuid import uuidtup_to_bin, string_to_bin
MSRPC_UUID_PAR = uuidtup_to_bin(('76F03F96-CDFD-44FC-A22C-64950A001209', '1.0'))
MSRPC_UUID_WINSPOOL = string_to_bin('9940CA8E-512F-4C58-88A9-61098D6896BD')

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
            return 'RPRN SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'RPRN SessionError: unknown error code: 0x%x' % self.error_code
STRING_HANDLE = LPWSTR

class PSTRING_HANDLE(NDRPOINTER):
    referent = (('Data', STRING_HANDLE),)
JOB_ACCESS_ADMINISTER = 16
JOB_ACCESS_READ = 32
JOB_EXECUTE = 131088
JOB_READ = 131104
JOB_WRITE = 131088
JOB_ALL_ACCESS = 983088
PRINTER_ACCESS_ADMINISTER = 4
PRINTER_ACCESS_USE = 8
PRINTER_ACCESS_MANAGE_LIMITED = 64
PRINTER_ALL_ACCESS = 983052
PRINTER_EXECUTE = 131080
PRINTER_READ = 131080
PRINTER_WRITE = 131080
SERVER_ACCESS_ADMINISTER = 1
SERVER_ACCESS_ENUMERATE = 2
SERVER_ALL_ACCESS = 983043
SERVER_EXECUTE = 131074
SERVER_READ = 131074
SERVER_WRITE = 131075
SPECIFIC_RIGHTS_ALL = 65535
STANDARD_RIGHTS_ALL = 2031616
STANDARD_RIGHTS_EXECUTE = 131072
STANDARD_RIGHTS_READ = 131072
STANDARD_RIGHTS_REQUIRED = 983040
STANDARD_RIGHTS_WRITE = 131072
SYNCHRONIZE = 1048576
DELETE = 65536
READ_CONTROL = 131072
WRITE_DAC = 262144
WRITE_OWNER = 524288
GENERIC_READ = 2147483648
GENERIC_WRITE = 1073741824
GENERIC_EXECUTE = 536870912
GENERIC_ALL = 268435456
PRINTER_CHANGE_SET_PRINTER = 2
PRINTER_CHANGE_DELETE_PRINTER = 4
PRINTER_CHANGE_PRINTER = 255
PRINTER_CHANGE_ADD_JOB = 256
PRINTER_CHANGE_SET_JOB = 512
PRINTER_CHANGE_DELETE_JOB = 1024
PRINTER_CHANGE_WRITE_JOB = 2048
PRINTER_CHANGE_JOB = 65280
PRINTER_CHANGE_SET_PRINTER_DRIVER = 536870912
PRINTER_CHANGE_TIMEOUT = 2147483648
PRINTER_CHANGE_ALL = 2004353023
PRINTER_CHANGE_ALL_2 = 2138570751
PRINTER_CHANGE_ADD_PRINTER_DRIVER = 268435456
PRINTER_CHANGE_DELETE_PRINTER_DRIVER = 1073741824
PRINTER_CHANGE_PRINTER_DRIVER = 1879048192
PRINTER_CHANGE_ADD_FORM = 65536
PRINTER_CHANGE_DELETE_FORM = 262144
PRINTER_CHANGE_SET_FORM = 131072
PRINTER_CHANGE_FORM = 458752
PRINTER_CHANGE_ADD_PORT = 1048576
PRINTER_CHANGE_CONFIGURE_PORT = 2097152
PRINTER_CHANGE_DELETE_PORT = 4194304
PRINTER_CHANGE_PORT = 7340032
PRINTER_CHANGE_ADD_PRINT_PROCESSOR = 16777216
PRINTER_CHANGE_DELETE_PRINT_PROCESSOR = 67108864
PRINTER_CHANGE_PRINT_PROCESSOR = 117440512
PRINTER_CHANGE_ADD_PRINTER = 1
PRINTER_CHANGE_FAILED_CONNECTION_PRINTER = 8
PRINTER_CHANGE_SERVER = 134217728
PRINTER_ENUM_LOCAL = 2
PRINTER_ENUM_CONNECTIONS = 4
PRINTER_ENUM_NAME = 8
PRINTER_ENUM_REMOTE = 16
PRINTER_ENUM_SHARED = 32
PRINTER_ENUM_NETWORK = 64
PRINTER_ENUM_EXPAND = 16384
PRINTER_ENUM_CONTAINER = 32768
PRINTER_ENUM_ICON1 = 65536
PRINTER_ENUM_ICON2 = 131072
PRINTER_ENUM_ICON3 = 262144
PRINTER_ENUM_ICON8 = 8388608
PRINTER_ENUM_HIDE = 16777216
PRINTER_NOTIFY_CATEGORY_2D = 0
PRINTER_NOTIFY_CATEGORY_ALL = 65536
PRINTER_NOTIFY_CATEGORY_3D = 131072
APD_STRICT_UPGRADE = 1
APD_STRICT_DOWNGRADE = 2
APD_COPY_ALL_FILES = 4
APD_COPY_NEW_FILES = 8
APD_COPY_FROM_DIRECTORY = 16
APD_DONT_COPY_FILES_TO_CLUSTER = 4096
APD_COPY_TO_ALL_SPOOLERS = 8192
APD_INSTALL_WARNED_DRIVER = 32768
APD_RETURN_BLOCKING_STATUS_CODE = 65536

class PRINTER_HANDLE(NDRSTRUCT):
    structure = (('Data', '20s=b""'),)

    def getAlignment(self):
        if False:
            print('Hello World!')
        if self._isNDR64 is True:
            return 8
        else:
            return 4

class BYTE_ARRAY(NDRUniConformantArray):
    item = 'c'

class PBYTE_ARRAY(NDRPOINTER):
    referent = (('Data', BYTE_ARRAY),)

class DEVMODE_CONTAINER(NDRSTRUCT):
    structure = (('cbBuf', DWORD), ('pDevMode', PBYTE_ARRAY))

class SPLCLIENT_INFO_1(NDRSTRUCT):
    structure = (('dwSize', DWORD), ('pMachineName', LPWSTR), ('pUserName', LPWSTR), ('dwBuildNum', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('wProcessorArchitecture', USHORT))

class PSPLCLIENT_INFO_1(NDRPOINTER):
    referent = (('Data', SPLCLIENT_INFO_1),)

class SPLCLIENT_INFO_2(NDRSTRUCT):
    structure = (('notUsed', ULONGLONG),)

class PSPLCLIENT_INFO_2(NDRPOINTER):
    referent = (('Data', SPLCLIENT_INFO_2),)

class SPLCLIENT_INFO_3(NDRSTRUCT):
    structure = (('cbSize', UINT), ('dwFlags', DWORD), ('dwFlags', DWORD), ('pMachineName', LPWSTR), ('pUserName', LPWSTR), ('dwBuildNum', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('wProcessorArchitecture', USHORT), ('hSplPrinter', ULONGLONG))

class PSPLCLIENT_INFO_3(NDRPOINTER):
    referent = (('Data', SPLCLIENT_INFO_3),)

class DRIVER_INFO_1(NDRSTRUCT):
    structure = (('pName', STRING_HANDLE),)

class PDRIVER_INFO_1(NDRPOINTER):
    referent = (('Data', DRIVER_INFO_1),)

class DRIVER_INFO_2(NDRSTRUCT):
    structure = (('cVersion', DWORD), ('pName', LPWSTR), ('pEnvironment', LPWSTR), ('pDriverPath', LPWSTR), ('pDataFile', LPWSTR), ('pConfigFile', LPWSTR))

class PDRIVER_INFO_2(NDRPOINTER):
    referent = (('Data', DRIVER_INFO_2),)

class DRIVER_INFO_UNION(NDRUNION):
    commonHdr = (('tag', ULONG),)
    union = {1: ('pNotUsed', PDRIVER_INFO_1), 2: ('Level2', PDRIVER_INFO_2)}

class DRIVER_CONTAINER(NDRSTRUCT):
    structure = (('Level', DWORD), ('DriverInfo', DRIVER_INFO_UNION))

class CLIENT_INFO_UNION(NDRUNION):
    commonHdr = (('tag', ULONG),)
    union = {1: ('pClientInfo1', PSPLCLIENT_INFO_1), 2: ('pNotUsed1', PSPLCLIENT_INFO_2), 3: ('pNotUsed2', PSPLCLIENT_INFO_3)}

class SPLCLIENT_CONTAINER(NDRSTRUCT):
    structure = (('Level', DWORD), ('ClientInfo', CLIENT_INFO_UNION))

class USHORT_ARRAY(NDRUniConformantArray):
    item = '<H'

class PUSHORT_ARRAY(NDRPOINTER):
    referent = (('Data', USHORT_ARRAY),)

class RpcAsync_V2_NOTIFY_OPTIONS_TYPE(NDRSTRUCT):
    structure = (('Type', USHORT), ('Reserved0', USHORT), ('Reserved1', DWORD), ('Reserved2', DWORD), ('Count', DWORD), ('pFields', PUSHORT_ARRAY))

class PRPC_V2_NOTIFY_OPTIONS_TYPE_ARRAY(NDRPOINTER):
    referent = (('Data', RpcAsync_V2_NOTIFY_OPTIONS_TYPE),)

class RpcAsync_V2_NOTIFY_OPTIONS(NDRSTRUCT):
    structure = (('Version', DWORD), ('Reserved', DWORD), ('Count', DWORD), ('pTypes', PRPC_V2_NOTIFY_OPTIONS_TYPE_ARRAY))

class PRPC_V2_NOTIFY_OPTIONS(NDRPOINTER):
    referent = (('Data', RpcAsync_V2_NOTIFY_OPTIONS),)

class RpcAsyncEnumPrinters(NDRCALL):
    opnum = 38
    structure = (('Flags', DWORD), ('Name', STRING_HANDLE), ('Level', DWORD), ('pPrinterEnum', PBYTE_ARRAY), ('cbBuf', DWORD))

class RpcAsyncEnumPrintersResponse(NDRCALL):
    structure = (('pPrinterEnum', PBYTE_ARRAY), ('pcbNeeded', DWORD), ('pcReturned', DWORD), ('ErrorCode', ULONG))

class RpcAsyncOpenPrinter(NDRCALL):
    opnum = 0
    structure = (('pPrinterName', STRING_HANDLE), ('pDatatype', LPWSTR), ('pDevModeContainer', DEVMODE_CONTAINER), ('AccessRequired', DWORD), ('pClientInfo', SPLCLIENT_CONTAINER))

class RpcAsyncOpenPrinterResponse(NDRCALL):
    structure = (('pHandle', PRINTER_HANDLE), ('ErrorCode', ULONG))

class RpcAsyncClosePrinter(NDRCALL):
    opnum = 20
    structure = (('phPrinter', PRINTER_HANDLE),)

class RpcAsyncClosePrinterResponse(NDRCALL):
    structure = (('phPrinter', PRINTER_HANDLE), ('ErrorCode', ULONG))

class RpcAsyncEnumPrinterDrivers(NDRCALL):
    opnum = 40
    structure = (('pName', STRING_HANDLE), ('pEnvironment', LPWSTR), ('Level', DWORD), ('pDrivers', PBYTE_ARRAY), ('cbBuf', DWORD))

class RpcAsyncEnumPrinterDriversResponse(NDRCALL):
    structure = (('pDrivers', PBYTE_ARRAY), ('pcbNeeded', DWORD), ('pcReturned', DWORD), ('ErrorCode', ULONG))

class RpcAsyncGetPrinterDriverDirectory(NDRCALL):
    opnum = 41
    structure = (('pName', STRING_HANDLE), ('pEnvironment', LPWSTR), ('Level', DWORD), ('pDriverDirectory', PBYTE_ARRAY), ('cbBuf', DWORD))

class RpcAsyncGetPrinterDriverDirectoryResponse(NDRCALL):
    structure = (('pDriverDirectory', PBYTE_ARRAY), ('pcbNeeded', DWORD), ('ErrorCode', ULONG))

class RpcAsyncAddPrinterDriver(NDRCALL):
    opnum = 39
    structure = (('pName', STRING_HANDLE), ('pDriverContainer', DRIVER_CONTAINER), ('dwFileCopyFlags', DWORD))

class RpcAsyncAddPrinterDriverResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)
OPNUMS = {0: (RpcAsyncOpenPrinter, RpcAsyncOpenPrinterResponse), 20: (RpcAsyncClosePrinter, RpcAsyncClosePrinterResponse), 38: (RpcAsyncEnumPrinters, RpcAsyncEnumPrintersResponse), 39: (RpcAsyncAddPrinterDriver, RpcAsyncAddPrinterDriver), 40: (RpcAsyncEnumPrinterDrivers, RpcAsyncEnumPrinterDriversResponse), 41: (RpcAsyncGetPrinterDriverDirectory, RpcAsyncGetPrinterDriverDirectoryResponse)}

def checkNullString(string):
    if False:
        while True:
            i = 10
    if string == NULL:
        return string
    if string[-1:] != '\x00':
        return string + '\x00'
    else:
        return string

def hRpcAsyncClosePrinter(dce, phPrinter):
    if False:
        i = 10
        return i + 15
    '\n    RpcClosePrinter closes a handle to a printer object, server object, job object, or port object.\n    Full Documentation: https://msdn.microsoft.com/en-us/library/cc244768.aspx\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param PRINTER_HANDLE phPrinter: A handle to a printer object, server object, job object, or port object.\n\n    :return: a RpcClosePrinterResponse instance, raises DCERPCSessionError on error.\n    '
    request = RpcAsyncClosePrinter()
    request['phPrinter'] = phPrinter
    return dce.request(request, MSRPC_UUID_WINSPOOL)

def hRpcAsyncOpenPrinter(dce, printerName, pDatatype=NULL, pDevModeContainer=NULL, accessRequired=SERVER_READ, pClientInfo=NULL):
    if False:
        print('Hello World!')
    '\n    RpcOpenPrinterEx retrieves a handle for a printer, port, port monitor, print job, or print server\n    Full Documentation: https://msdn.microsoft.com/en-us/library/cc244809.aspx\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param string printerName: A string for a printer connection, printer object, server object, job object, port\n    object, or port monitor object. This MUST be a Domain Name System (DNS), NetBIOS, Internet Protocol version 4\n    (IPv4), Internet Protocol version 6 (IPv6), or Universal Naming Convention (UNC) name that remote procedure\n    call (RpcAsync) binds to, and it MUST uniquely identify a print server on the network.\n    :param string pDatatype: A string that specifies the data type to be associated with the printer handle.\n    :param DEVMODE_CONTAINER pDevModeContainer: A DEVMODE_CONTAINER structure. This parameter MUST adhere to the specification in\n    DEVMODE_CONTAINER Parameters (section 3.1.4.1.8.1).\n    :param int accessRequired: The access level that the client requires for interacting with the object to which a\n    handle is being opened.\n    :param SPLCLIENT_CONTAINER pClientInfo: This parameter MUST adhere to the specification in SPLCLIENT_CONTAINER Parameters.\n\n    :return: a RpcOpenPrinterExResponse instance, raises DCERPCSessionError on error.\n    '
    request = RpcAsyncOpenPrinter()
    request['pPrinterName'] = checkNullString(printerName)
    request['pDatatype'] = pDatatype
    if pDevModeContainer is NULL:
        request['pDevModeContainer']['pDevMode'] = NULL
    else:
        request['pDevModeContainer'] = pDevModeContainer
    request['AccessRequired'] = accessRequired
    if pClientInfo is NULL:
        raise Exception('pClientInfo cannot be NULL')
    request['pClientInfo'] = pClientInfo
    return dce.request(request, MSRPC_UUID_WINSPOOL)

def hRpcAsyncEnumPrinters(dce, flags, name=NULL, level=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    RpcEnumPrinters enumerates available printers, print servers, domains, or print providers.\n    Full Documentation: https://msdn.microsoft.com/en-us/library/cc244794.aspx\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param int flags: The types of print objects that this method enumerates. The value of this parameter is the\n    result of a bitwise OR of one or more of the Printer Enumeration Flags (section 2.2.3.7).\n    :param string name: NULL or a server name parameter as specified in Printer Server Name Parameters (section 3.1.4.1.4).\n    :param level: The level of printer information structure.\n\n    :return: a RpcEnumPrintersResponse instance, raises DCERPCSessionError on error.\n    '
    request = RpcAsyncEnumPrinters()
    request['Flags'] = flags
    request['Name'] = name
    request['pPrinterEnum'] = NULL
    request['Level'] = level
    bytesNeeded = 0
    try:
        dce.request(request, MSRPC_UUID_WINSPOOL)
    except DCERPCSessionError as e:
        if str(e).find('ERROR_INSUFFICIENT_BUFFER') < 0:
            raise
        bytesNeeded = e.get_packet()['pcbNeeded']
    request = RpcAsyncEnumPrinters()
    request['Flags'] = flags
    request['Name'] = name
    request['Level'] = level
    request['cbBuf'] = bytesNeeded
    request['pPrinterEnum'] = b'a' * bytesNeeded
    return dce.request(request, MSRPC_UUID_WINSPOOL)

def hRpcAsyncAddPrinterDriver(dce, pName, pDriverContainer, dwFileCopyFlags):
    if False:
        return 10
    '\n    RpcAddPrinterDriverEx installs a printer driver on the print server\n    Full Documentation: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-rprn/b96cc497-59e5-4510-ab04-5484993b259b\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param pName\n    :param pDriverContainer\n    :param dwFileCopyFlags\n\n    :return: raises DCERPCSessionError on error.\n    '
    request = RpcAsyncAddPrinterDriver()
    request['pName'] = checkNullString(pName)
    request['pDriverContainer'] = pDriverContainer
    request['dwFileCopyFlags'] = dwFileCopyFlags
    return dce.request(request, MSRPC_UUID_WINSPOOL)

def hRpcAsyncEnumPrinterDrivers(dce, pName, pEnvironment, Level):
    if False:
        while True:
            i = 10
    '\n    RpcEnumPrinterDrivers enumerates the printer drivers installed on a specified print server.\n    Full Documentation: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-rprn/857d00ac-3682-4a0d-86ca-3d3c372e5e4a\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param pName\n    :param pEnvironment\n    :param Level\n    :param pDrivers\n    :param cbBuf\n    :param pcbNeeded\n    :param pcReturned\n\n    :return: raises DCERPCSessionError on error.\n    '
    request = RpcAsyncEnumPrinterDrivers()
    request['pName'] = checkNullString(pName)
    request['pEnvironment'] = pEnvironment
    request['Level'] = Level
    request['pDrivers'] = NULL
    request['cbBuf'] = 0
    try:
        dce.request(request, MSRPC_UUID_WINSPOOL)
    except DCERPCSessionError as e:
        if str(e).find('ERROR_INSUFFICIENT_BUFFER') < 0:
            raise
        bytesNeeded = e.get_packet()['pcbNeeded']
    request = RpcAsyncEnumPrinterDrivers()
    request['pName'] = checkNullString(pName)
    request['pEnvironment'] = pEnvironment
    request['Level'] = Level
    request['pDrivers'] = b'a' * bytesNeeded
    request['cbBuf'] = bytesNeeded
    return dce.request(request, MSRPC_UUID_WINSPOOL)

def hRpcAsyncGetPrinterDriverDirectory(dce, pName, pEnvironment, Level):
    if False:
        i = 10
        return i + 15
    '\n    RpcAsyncGetPrinterDriverDirectory retrieves the path of the printer driver directory.\n    Full Documentation: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-par/92206fb2-dd31-47f4-8d12-4cd239b71d78\n\n    :param DCERPC_v5 dce: a connected DCE instance.\n    :param pName\n    :param pEnvironment\n    :param Level\n    :param pDriverDirectory\n    :param cbBuf\n    :param pcbNeeded\n\n    :return: raises DCERPCSessionError on error.\n    '
    request = RpcAsyncGetPrinterDriverDirectory()
    request['pName'] = checkNullString(pName)
    request['pEnvironment'] = pEnvironment
    request['Level'] = Level
    request['pDriverDirectory'] = NULL
    request['cbBuf'] = 0
    try:
        dce.request(request, MSRPC_UUID_WINSPOOL)
    except DCERPCSessionError as e:
        if str(e).find('ERROR_INSUFFICIENT_BUFFER') < 0:
            raise
        bytesNeeded = e.get_packet()['pcbNeeded']
    request = RpcAsyncGetPrinterDriverDirectory()
    request['pName'] = checkNullString(pName)
    request['pEnvironment'] = pEnvironment
    request['Level'] = Level
    request['pDriverDirectory'] = b'a' * bytesNeeded
    request['cbBuf'] = bytesNeeded
    return dce.request(request, MSRPC_UUID_WINSPOOL)