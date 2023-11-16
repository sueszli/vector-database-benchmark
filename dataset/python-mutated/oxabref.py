from impacket import hresult_errors, mapi_constants
from impacket.dcerpc.v5.dtypes import NULL, STR, ULONG
from impacket.dcerpc.v5.ndr import NDRCALL, NDRPOINTER
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_OXABREF = uuidtup_to_bin(('1544F5E0-613C-11D1-93DF-00C04FD7BD09', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            return 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            return 10
        key = self.error_code
        if key in mapi_constants.ERROR_MESSAGES:
            error_msg_short = mapi_constants.ERROR_MESSAGES[key]
            return 'OXABREF SessionError: code: 0x%x - %s' % (self.error_code, error_msg_short)
        elif key in hresult_errors.ERROR_MESSAGES:
            error_msg_short = hresult_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = hresult_errors.ERROR_MESSAGES[key][1]
            return 'OXABREF SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'OXABREF SessionError: unknown error code: 0x%x' % self.error_code

class PUCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', STR),)

class PPUCHAR_ARRAY(NDRPOINTER):
    referent = (('Data', PUCHAR_ARRAY),)

class RfrGetNewDSA(NDRCALL):
    opnum = 0
    structure = (('ulFlags', ULONG), ('pUserDN', STR), ('ppszUnused', PPUCHAR_ARRAY), ('ppszServer', PPUCHAR_ARRAY))

class RfrGetNewDSAResponse(NDRCALL):
    structure = (('ppszUnused', PPUCHAR_ARRAY), ('ppszServer', PPUCHAR_ARRAY))

class RfrGetFQDNFromServerDN(NDRCALL):
    opnum = 1
    structure = (('ulFlags', ULONG), ('cbMailboxServerDN', ULONG), ('szMailboxServerDN', STR))

class RfrGetFQDNFromServerDNResponse(NDRCALL):
    structure = (('ppszServerFQDN', PUCHAR_ARRAY), ('ErrorCode', ULONG))
OPNUMS = {0: (RfrGetNewDSA, RfrGetNewDSAResponse), 1: (RfrGetFQDNFromServerDN, RfrGetFQDNFromServerDNResponse)}

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

def hRfrGetNewDSA(dce, pUserDN=''):
    if False:
        print('Hello World!')
    request = RfrGetNewDSA()
    request['ulFlags'] = 0
    request['pUserDN'] = checkNullString(pUserDN)
    request['ppszUnused'] = NULL
    request['ppszServer'] = '\x00'
    resp = dce.request(request)
    resp['ppszServer'] = resp['ppszServer'][:-1]
    if request['ppszUnused'] != NULL:
        resp['ppszUnused'] = resp['ppszUnused'][:-1]
    return resp

def hRfrGetFQDNFromServerDN(dce, szMailboxServerDN):
    if False:
        while True:
            i = 10
    szMailboxServerDN = checkNullString(szMailboxServerDN)
    request = RfrGetFQDNFromServerDN()
    request['ulFlags'] = 0
    request['szMailboxServerDN'] = szMailboxServerDN
    request['cbMailboxServerDN'] = len(szMailboxServerDN)
    resp = dce.request(request)
    resp['ppszServerFQDN'] = resp['ppszServerFQDN'][:-1]
    return resp