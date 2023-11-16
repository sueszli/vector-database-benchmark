from impacket.dcerpc.v5.ndr import NDRCALL, NDRUniConformantArray
from impacket.dcerpc.v5.dtypes import DWORD, LPWSTR, ULONG, WSTR, NULL
from impacket import hresult_errors
from impacket.uuid import uuidtup_to_bin
from impacket.dcerpc.v5.rpcrt import DCERPCException
MSRPC_UUID_SASEC = uuidtup_to_bin(('378E52B0-C0A9-11CF-822D-00AA0051E40F', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            while True:
                i = 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        key = self.error_code
        if key in hresult_errors.ERROR_MESSAGES:
            error_msg_short = hresult_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = hresult_errors.ERROR_MESSAGES[key][1]
            return 'TSCH SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'TSCH SessionError: unknown error code: 0x%x' % self.error_code
SASEC_HANDLE = WSTR
PSASEC_HANDLE = LPWSTR
MAX_BUFFER_SIZE = 273
TASK_FLAG_RUN_ONLY_IF_LOGGED_ON = 262144

class WORD_ARRAY(NDRUniConformantArray):
    item = '<H'

class SASetAccountInformation(NDRCALL):
    opnum = 0
    structure = (('Handle', PSASEC_HANDLE), ('pwszJobName', WSTR), ('pwszAccount', WSTR), ('pwszPassword', LPWSTR), ('dwJobFlags', DWORD))

class SASetAccountInformationResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SASetNSAccountInformation(NDRCALL):
    opnum = 1
    structure = (('Handle', PSASEC_HANDLE), ('pwszAccount', LPWSTR), ('pwszPassword', LPWSTR))

class SASetNSAccountInformationResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class SAGetNSAccountInformation(NDRCALL):
    opnum = 2
    structure = (('Handle', PSASEC_HANDLE), ('ccBufferSize', DWORD), ('wszBuffer', WORD_ARRAY))

class SAGetNSAccountInformationResponse(NDRCALL):
    structure = (('wszBuffer', WORD_ARRAY), ('ErrorCode', ULONG))

class SAGetAccountInformation(NDRCALL):
    opnum = 3
    structure = (('Handle', PSASEC_HANDLE), ('pwszJobName', WSTR), ('ccBufferSize', DWORD), ('wszBuffer', WORD_ARRAY))

class SAGetAccountInformationResponse(NDRCALL):
    structure = (('wszBuffer', WORD_ARRAY), ('ErrorCode', ULONG))
OPNUMS = {0: (SASetAccountInformation, SASetAccountInformationResponse), 1: (SASetNSAccountInformation, SASetNSAccountInformationResponse), 2: (SAGetNSAccountInformation, SAGetNSAccountInformationResponse), 3: (SAGetAccountInformation, SAGetAccountInformationResponse)}

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

def hSASetAccountInformation(dce, handle, pwszJobName, pwszAccount, pwszPassword, dwJobFlags=0):
    if False:
        while True:
            i = 10
    request = SASetAccountInformation()
    request['Handle'] = handle
    request['pwszJobName'] = checkNullString(pwszJobName)
    request['pwszAccount'] = checkNullString(pwszAccount)
    request['pwszPassword'] = checkNullString(pwszPassword)
    request['dwJobFlags'] = dwJobFlags
    return dce.request(request)

def hSASetNSAccountInformation(dce, handle, pwszAccount, pwszPassword):
    if False:
        for i in range(10):
            print('nop')
    request = SASetNSAccountInformation()
    request['Handle'] = handle
    request['pwszAccount'] = checkNullString(pwszAccount)
    request['pwszPassword'] = checkNullString(pwszPassword)
    return dce.request(request)

def hSAGetNSAccountInformation(dce, handle, ccBufferSize=MAX_BUFFER_SIZE):
    if False:
        for i in range(10):
            print('nop')
    request = SAGetNSAccountInformation()
    request['Handle'] = handle
    request['ccBufferSize'] = ccBufferSize
    for _ in range(ccBufferSize):
        request['wszBuffer'].append(0)
    return dce.request(request)

def hSAGetAccountInformation(dce, handle, pwszJobName, ccBufferSize=MAX_BUFFER_SIZE):
    if False:
        for i in range(10):
            print('nop')
    request = SAGetAccountInformation()
    request['Handle'] = handle
    request['pwszJobName'] = checkNullString(pwszJobName)
    request['ccBufferSize'] = ccBufferSize
    for _ in range(ccBufferSize):
        request['wszBuffer'].append(0)
    return dce.request(request)