from impacket import nt_errors
from impacket.dcerpc.v5.dtypes import ULONG, LONG, PRPC_SID, RPC_UNICODE_STRING, LPWSTR, PRPC_UNICODE_STRING, NTSTATUS, NULL
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.lsad import LSAPR_HANDLE, PLSAPR_TRUST_INFORMATION_ARRAY
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRENUM, NDRPOINTER, NDRUniConformantArray
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.samr import SID_NAME_USE
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_LSAT = uuidtup_to_bin(('12345778-1234-ABCD-EF00-0123456789AB', '0.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            return 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        key = self.error_code
        if key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'LSAT SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'LSAT SessionError: unknown error code: 0x%x' % self.error_code
POLICY_LOOKUP_NAMES = 2048

class LSAPR_REFERENCED_DOMAIN_LIST(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Domains', PLSAPR_TRUST_INFORMATION_ARRAY), ('MaxEntries', ULONG))

class PLSAPR_REFERENCED_DOMAIN_LIST(NDRPOINTER):
    referent = (('Data', LSAPR_REFERENCED_DOMAIN_LIST),)

class LSA_TRANSLATED_SID(NDRSTRUCT):
    structure = (('Use', SID_NAME_USE), ('RelativeId', ULONG), ('DomainIndex', LONG))

class LSA_TRANSLATED_SID_ARRAY(NDRUniConformantArray):
    item = LSA_TRANSLATED_SID

class PLSA_TRANSLATED_SID_ARRAY(NDRPOINTER):
    referent = (('Data', LSA_TRANSLATED_SID_ARRAY),)

class LSAPR_TRANSLATED_SIDS(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Sids', PLSA_TRANSLATED_SID_ARRAY))

class LSAP_LOOKUP_LEVEL(NDRENUM):

    class enumItems(Enum):
        LsapLookupWksta = 1
        LsapLookupPDC = 2
        LsapLookupTDL = 3
        LsapLookupGC = 4
        LsapLookupXForestReferral = 5
        LsapLookupXForestResolve = 6
        LsapLookupRODCReferralToFullDC = 7

class LSAPR_SID_INFORMATION(NDRSTRUCT):
    structure = (('Sid', PRPC_SID),)

class LSAPR_SID_INFORMATION_ARRAY(NDRUniConformantArray):
    item = LSAPR_SID_INFORMATION

class PLSAPR_SID_INFORMATION_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_SID_INFORMATION_ARRAY),)

class LSAPR_SID_ENUM_BUFFER(NDRSTRUCT):
    structure = (('Entries', ULONG), ('SidInfo', PLSAPR_SID_INFORMATION_ARRAY))

class LSAPR_TRANSLATED_NAME(NDRSTRUCT):
    structure = (('Use', SID_NAME_USE), ('Name', RPC_UNICODE_STRING), ('DomainIndex', LONG))

class LSAPR_TRANSLATED_NAME_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRANSLATED_NAME

class PLSAPR_TRANSLATED_NAME_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRANSLATED_NAME_ARRAY),)

class LSAPR_TRANSLATED_NAMES(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Names', PLSAPR_TRANSLATED_NAME_ARRAY))

class LSAPR_TRANSLATED_NAME_EX(NDRSTRUCT):
    structure = (('Use', SID_NAME_USE), ('Name', RPC_UNICODE_STRING), ('DomainIndex', LONG), ('Flags', ULONG))

class LSAPR_TRANSLATED_NAME_EX_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRANSLATED_NAME_EX

class PLSAPR_TRANSLATED_NAME_EX_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRANSLATED_NAME_EX_ARRAY),)

class LSAPR_TRANSLATED_NAMES_EX(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Names', PLSAPR_TRANSLATED_NAME_EX_ARRAY))

class LSAPR_TRANSLATED_SID_EX(NDRSTRUCT):
    structure = (('Use', SID_NAME_USE), ('RelativeId', ULONG), ('DomainIndex', LONG), ('Flags', ULONG))

class LSAPR_TRANSLATED_SID_EX_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRANSLATED_SID_EX

class PLSAPR_TRANSLATED_SID_EX_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRANSLATED_SID_EX_ARRAY),)

class LSAPR_TRANSLATED_SIDS_EX(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Sids', PLSAPR_TRANSLATED_SID_EX_ARRAY))

class LSAPR_TRANSLATED_SID_EX2(NDRSTRUCT):
    structure = (('Use', SID_NAME_USE), ('Sid', PRPC_SID), ('DomainIndex', LONG), ('Flags', ULONG))

class LSAPR_TRANSLATED_SID_EX2_ARRAY(NDRUniConformantArray):
    item = LSAPR_TRANSLATED_SID_EX2

class PLSAPR_TRANSLATED_SID_EX2_ARRAY(NDRPOINTER):
    referent = (('Data', LSAPR_TRANSLATED_SID_EX2_ARRAY),)

class LSAPR_TRANSLATED_SIDS_EX2(NDRSTRUCT):
    structure = (('Entries', ULONG), ('Sids', PLSAPR_TRANSLATED_SID_EX2_ARRAY))

class RPC_UNICODE_STRING_ARRAY(NDRUniConformantArray):
    item = RPC_UNICODE_STRING

class LsarGetUserName(NDRCALL):
    opnum = 45
    structure = (('SystemName', LPWSTR), ('UserName', PRPC_UNICODE_STRING), ('DomainName', PRPC_UNICODE_STRING))

class LsarGetUserNameResponse(NDRCALL):
    structure = (('UserName', PRPC_UNICODE_STRING), ('DomainName', PRPC_UNICODE_STRING), ('ErrorCode', NTSTATUS))

class LsarLookupNames4(NDRCALL):
    opnum = 77
    structure = (('Count', ULONG), ('Names', RPC_UNICODE_STRING_ARRAY), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX2), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG), ('LookupOptions', ULONG), ('ClientRevision', ULONG))

class LsarLookupNames4Response(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX2), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupNames3(NDRCALL):
    opnum = 68
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Count', ULONG), ('Names', RPC_UNICODE_STRING_ARRAY), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX2), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG), ('LookupOptions', ULONG), ('ClientRevision', ULONG))

class LsarLookupNames3Response(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX2), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupNames2(NDRCALL):
    opnum = 58
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Count', ULONG), ('Names', RPC_UNICODE_STRING_ARRAY), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG), ('LookupOptions', ULONG), ('ClientRevision', ULONG))

class LsarLookupNames2Response(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedSids', LSAPR_TRANSLATED_SIDS_EX), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupNames(NDRCALL):
    opnum = 14
    structure = (('PolicyHandle', LSAPR_HANDLE), ('Count', ULONG), ('Names', RPC_UNICODE_STRING_ARRAY), ('TranslatedSids', LSAPR_TRANSLATED_SIDS), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG))

class LsarLookupNamesResponse(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedSids', LSAPR_TRANSLATED_SIDS), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupSids3(NDRCALL):
    opnum = 76
    structure = (('SidEnumBuffer', LSAPR_SID_ENUM_BUFFER), ('TranslatedNames', LSAPR_TRANSLATED_NAMES_EX), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG), ('LookupOptions', ULONG), ('ClientRevision', ULONG))

class LsarLookupSids3Response(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedNames', LSAPR_TRANSLATED_NAMES_EX), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupSids2(NDRCALL):
    opnum = 57
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SidEnumBuffer', LSAPR_SID_ENUM_BUFFER), ('TranslatedNames', LSAPR_TRANSLATED_NAMES_EX), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG), ('LookupOptions', ULONG), ('ClientRevision', ULONG))

class LsarLookupSids2Response(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedNames', LSAPR_TRANSLATED_NAMES_EX), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))

class LsarLookupSids(NDRCALL):
    opnum = 15
    structure = (('PolicyHandle', LSAPR_HANDLE), ('SidEnumBuffer', LSAPR_SID_ENUM_BUFFER), ('TranslatedNames', LSAPR_TRANSLATED_NAMES), ('LookupLevel', LSAP_LOOKUP_LEVEL), ('MappedCount', ULONG))

class LsarLookupSidsResponse(NDRCALL):
    structure = (('ReferencedDomains', PLSAPR_REFERENCED_DOMAIN_LIST), ('TranslatedNames', LSAPR_TRANSLATED_NAMES), ('MappedCount', ULONG), ('ErrorCode', NTSTATUS))
OPNUMS = {14: (LsarLookupNames, LsarLookupNamesResponse), 15: (LsarLookupSids, LsarLookupSidsResponse), 45: (LsarGetUserName, LsarGetUserNameResponse), 57: (LsarLookupSids2, LsarLookupSids2Response), 58: (LsarLookupNames2, LsarLookupNames2Response), 68: (LsarLookupNames3, LsarLookupNames3Response), 76: (LsarLookupSids3, LsarLookupSids3Response), 77: (LsarLookupNames4, LsarLookupNames4Response)}

def hLsarGetUserName(dce, userName=NULL, domainName=NULL):
    if False:
        for i in range(10):
            print('nop')
    request = LsarGetUserName()
    request['SystemName'] = NULL
    request['UserName'] = userName
    request['DomainName'] = domainName
    return dce.request(request)

def hLsarLookupNames4(dce, names, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta, lookupOptions=0, clientRevision=1):
    if False:
        return 10
    request = LsarLookupNames4()
    request['Count'] = len(names)
    for name in names:
        itemn = RPC_UNICODE_STRING()
        itemn['Data'] = name
        request['Names'].append(itemn)
    request['TranslatedSids']['Sids'] = NULL
    request['LookupLevel'] = lookupLevel
    request['LookupOptions'] = lookupOptions
    request['ClientRevision'] = clientRevision
    return dce.request(request)

def hLsarLookupNames3(dce, policyHandle, names, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta, lookupOptions=0, clientRevision=1):
    if False:
        print('Hello World!')
    request = LsarLookupNames3()
    request['PolicyHandle'] = policyHandle
    request['Count'] = len(names)
    for name in names:
        itemn = RPC_UNICODE_STRING()
        itemn['Data'] = name
        request['Names'].append(itemn)
    request['TranslatedSids']['Sids'] = NULL
    request['LookupLevel'] = lookupLevel
    request['LookupOptions'] = lookupOptions
    request['ClientRevision'] = clientRevision
    return dce.request(request)

def hLsarLookupNames2(dce, policyHandle, names, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta, lookupOptions=0, clientRevision=1):
    if False:
        print('Hello World!')
    request = LsarLookupNames2()
    request['PolicyHandle'] = policyHandle
    request['Count'] = len(names)
    for name in names:
        itemn = RPC_UNICODE_STRING()
        itemn['Data'] = name
        request['Names'].append(itemn)
    request['TranslatedSids']['Sids'] = NULL
    request['LookupLevel'] = lookupLevel
    request['LookupOptions'] = lookupOptions
    request['ClientRevision'] = clientRevision
    return dce.request(request)

def hLsarLookupNames(dce, policyHandle, names, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta):
    if False:
        for i in range(10):
            print('nop')
    request = LsarLookupNames()
    request['PolicyHandle'] = policyHandle
    request['Count'] = len(names)
    for name in names:
        itemn = RPC_UNICODE_STRING()
        itemn['Data'] = name
        request['Names'].append(itemn)
    request['TranslatedSids']['Sids'] = NULL
    request['LookupLevel'] = lookupLevel
    return dce.request(request)

def hLsarLookupSids2(dce, policyHandle, sids, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta, lookupOptions=0, clientRevision=1):
    if False:
        for i in range(10):
            print('nop')
    request = LsarLookupSids2()
    request['PolicyHandle'] = policyHandle
    request['SidEnumBuffer']['Entries'] = len(sids)
    for sid in sids:
        itemn = LSAPR_SID_INFORMATION()
        itemn['Sid'].fromCanonical(sid)
        request['SidEnumBuffer']['SidInfo'].append(itemn)
    request['TranslatedNames']['Names'] = NULL
    request['LookupLevel'] = lookupLevel
    request['LookupOptions'] = lookupOptions
    request['ClientRevision'] = clientRevision
    return dce.request(request)

def hLsarLookupSids(dce, policyHandle, sids, lookupLevel=LSAP_LOOKUP_LEVEL.LsapLookupWksta):
    if False:
        for i in range(10):
            print('nop')
    request = LsarLookupSids()
    request['PolicyHandle'] = policyHandle
    request['SidEnumBuffer']['Entries'] = len(sids)
    for sid in sids:
        itemn = LSAPR_SID_INFORMATION()
        itemn['Sid'].fromCanonical(sid)
        request['SidEnumBuffer']['SidInfo'].append(itemn)
    request['TranslatedNames']['Names'] = NULL
    request['LookupLevel'] = lookupLevel
    return dce.request(request)