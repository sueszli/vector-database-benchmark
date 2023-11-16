from __future__ import division
from __future__ import print_function
import random
from struct import pack, unpack
from impacket import LOG
from impacket import hresult_errors
from impacket.dcerpc.v5.dcomrt import DCOMCALL, DCOMANSWER, IRemUnknown2, PMInterfacePointer, INTERFACE, MInterfacePointer, MInterfacePointer_ARRAY, BYTE_ARRAY, PPMInterfacePointer
from impacket.dcerpc.v5.dtypes import LPWSTR, ULONG, DWORD, SHORT, GUID, USHORT, LONG, WSTR, BYTE, LONGLONG, FLOAT, DOUBLE, HRESULT, PSHORT, PLONG, PLONGLONG, PFLOAT, PDOUBLE, PHRESULT, CHAR, ULONGLONG, INT, UINT, PCHAR, PUSHORT, PULONG, PULONGLONG, PINT, PUINT, NULL
from impacket.dcerpc.v5.enum import Enum
from impacket.dcerpc.v5.ndr import NDRSTRUCT, NDRUniConformantArray, NDRPOINTER, NDRENUM, NDRUSHORT, NDRUNION, NDRUniConformantVaryingArray, NDR
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.uuid import string_to_bin

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            print('Hello World!')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.error_code in hresult_errors.ERROR_MESSAGES:
            error_msg_short = hresult_errors.ERROR_MESSAGES[self.error_code][0]
            error_msg_verbose = hresult_errors.ERROR_MESSAGES[self.error_code][1]
            return 'OAUT SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'OAUT SessionError: unknown error code: 0x%x' % self.error_code
IID_IDispatch = string_to_bin('00020400-0000-0000-C000-000000000046')
IID_ITypeInfo = string_to_bin('00020401-0000-0000-C000-000000000046')
IID_ITypeComp = string_to_bin('00020403-0000-0000-C000-000000000046')
IID_NULL = string_to_bin('00000000-0000-0000-0000-000000000000')
error_status_t = ULONG
LCID = DWORD
WORD = NDRUSHORT
IID = GUID
LPOLESTR = LPWSTR
OLESTR = WSTR
REFIID = IID
DATE = DOUBLE

class PDATE(NDRPOINTER):
    referent = (('Data', DATE),)
VARIANT_BOOL = USHORT

class PVARIANT_BOOL(NDRPOINTER):
    referent = (('Data', VARIANT_BOOL),)
DISPATCH_METHOD = 1
DISPATCH_PROPERTYGET = 2
DISPATCH_PROPERTYPUT = 4
DISPATCH_PROPERTYPUTREF = 8
DISPATCH_zeroVarResult = 131072
DISPATCH_zeroExcepInfo = 262144
DISPATCH_zeroArgErr = 524288

class DECIMAL(NDRSTRUCT):
    structure = (('wReserved', WORD), ('scale', BYTE), ('sign', BYTE), ('Hi32', ULONG), ('Lo64', ULONGLONG))

class PDECIMAL(NDRPOINTER):
    referent = (('Data', DECIMAL),)

class VARENUM(NDRENUM):

    class enumItems(Enum):
        VT_EMPTY = 0
        VT_NULL = 1
        VT_I2 = 2
        VT_I4 = 3
        VT_R4 = 4
        VT_R8 = 5
        VT_CY = 6
        VT_DATE = 7
        VT_BSTR = 8
        VT_DISPATCH = 9
        VT_ERROR = 10
        VT_BOOL = 11
        VT_VARIANT = 12
        VT_UNKNOWN = 13
        VT_DECIMAL = 14
        VT_I1 = 16
        VT_UI1 = 17
        VT_UI2 = 18
        VT_UI4 = 19
        VT_I8 = 20
        VT_UI8 = 21
        VT_INT = 22
        VT_UINT = 23
        VT_VOID = 24
        VT_HRESULT = 25
        VT_PTR = 26
        VT_SAFEARRAY = 27
        VT_CARRAY = 28
        VT_USERDEFINED = 29
        VT_LPSTR = 30
        VT_LPWSTR = 31
        VT_RECORD = 36
        VT_INT_PTR = 37
        VT_UINT_PTR = 38
        VT_ARRAY = 8192
        VT_BYREF = 16384
        VT_RECORD_OR_VT_BYREF = VT_RECORD | VT_BYREF
        VT_UI1_OR_VT_BYREF = VT_UI1 | VT_BYREF
        VT_I2_OR_VT_BYREF = VT_I2 | VT_BYREF
        VT_I4_OR_VT_BYREF = VT_I4 | VT_BYREF
        VT_I8_OR_VT_BYREF = VT_I8 | VT_BYREF
        VT_R4_OR_VT_BYREF = VT_R4 | VT_BYREF
        VT_R8_OR_VT_BYREF = VT_R8 | VT_BYREF
        VT_BOOL_OR_VT_BYREF = VT_BOOL | VT_BYREF
        VT_ERROR_OR_VT_BYREF = VT_ERROR | VT_BYREF
        VT_CY_OR_VT_BYREF = VT_CY | VT_BYREF
        VT_DATE_OR_VT_BYREF = VT_DATE | VT_BYREF
        VT_BSTR_OR_VT_BYREF = VT_BSTR | VT_BYREF
        VT_UNKNOWN_OR_VT_BYREF = VT_UNKNOWN | VT_BYREF
        VT_DISPATCH_OR_VT_BYREF = VT_DISPATCH | VT_BYREF
        VT_ARRAY_OR_VT_BYREF = VT_ARRAY | VT_BYREF
        VT_VARIANT_OR_VT_BYREF = VT_VARIANT | VT_BYREF
        VT_I1_OR_VT_BYREF = VT_I1 | VT_BYREF
        VT_UI2_OR_VT_BYREF = VT_UI2 | VT_BYREF
        VT_UI4_OR_VT_BYREF = VT_UI4 | VT_BYREF
        VT_UI8_OR_VT_BYREF = VT_UI8 | VT_BYREF
        VT_INT_OR_VT_BYREF = VT_INT | VT_BYREF
        VT_UINT_OR_VT_BYREF = VT_UINT | VT_BYREF
        VT_DECIMAL_OR_VT_BYREF = VT_DECIMAL | VT_BYREF

class SF_TYPE(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        SF_ERROR = VARENUM.VT_ERROR
        SF_I1 = VARENUM.VT_I1
        SF_I2 = VARENUM.VT_I2
        SF_I4 = VARENUM.VT_I4
        SF_I8 = VARENUM.VT_I8
        SF_BSTR = VARENUM.VT_BSTR
        SF_UNKNOWN = VARENUM.VT_UNKNOWN
        SF_DISPATCH = VARENUM.VT_DISPATCH
        SF_VARIANT = VARENUM.VT_VARIANT
        SF_RECORD = VARENUM.VT_RECORD
        SF_HAVEIID = VARENUM.VT_UNKNOWN | 32768

class CALLCONV(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        CC_CDECL = 1
        CC_PASCAL = 2
        CC_STDCALL = 4

class FUNCKIND(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        FUNC_PUREVIRTUAL = 1
        FUNC_STATIC = 3
        FUNC_DISPATCH = 4

class INVOKEKIND(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        INVOKE_FUNC = 1
        INVOKE_PROPERTYGET = 2
        INVOKE_PROPERTYPUT = 4
        INVOKE_PROPERTYPUTREF = 8

class TYPEKIND(NDRENUM):
    structure = (('Data', '<L'),)

    class enumItems(Enum):
        TKIND_ENUM = 0
        TKIND_RECORD = 1
        TKIND_MODULE = 2
        TKIND_INTERFACE = 3
        TKIND_DISPATCH = 4
        TKIND_COCLASS = 5
        TKIND_ALIAS = 6
        TKIND_UNION = 7

class USHORT_ARRAY(NDRUniConformantArray):
    item = '<H'

class FLAGGED_WORD_BLOB(NDRSTRUCT):
    structure = (('cBytes', ULONG), ('clSize', ULONG), ('asData', USHORT_ARRAY))

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if key == 'asData':
            value = value
            array = list()
            for letter in value:
                encoded = letter.encode('utf-16le')
                array.append(unpack('<H', encoded)[0])
            self.fields[key]['Data'] = array
            self['cBytes'] = len(value) * 2
            self['clSize'] = len(value)
            self.data = None
        else:
            return NDRSTRUCT.__setitem__(self, key, value)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if key == 'asData':
            value = ''
            for letter in self.fields['asData']['Data']:
                value += pack('<H', letter).decode('utf-16le')
            return value
        else:
            return NDRSTRUCT.__getitem__(self, key)

    def dump(self, msg=None, indent=0):
        if False:
            i = 10
            return i + 15
        if msg is None:
            msg = self.__class__.__name__
        ind = ' ' * indent
        if msg != '':
            print('%s' % msg)
        value = ''
        print('%sasData: %s' % (ind, self['asData']), end=' ')

class BSTR(NDRPOINTER):
    referent = (('Data', FLAGGED_WORD_BLOB),)

class PBSTR(NDRPOINTER):
    referent = (('Data', BSTR),)

class CURRENCY(NDRSTRUCT):
    structure = (('int64', LONGLONG),)

class PCURRENCY(NDRPOINTER):
    referent = (('Data', CURRENCY),)

class _wireBRECORD(NDRSTRUCT):
    structure = (('fFlags', LONGLONG), ('clSize', LONGLONG), ('pRecInfo', MInterfacePointer), ('pRecord', BYTE_ARRAY))

class BRECORD(NDRPOINTER):
    referent = (('Data', _wireBRECORD),)

class SAFEARRAYBOUND(NDRSTRUCT):
    structure = (('cElements', ULONG), ('lLbound', LONG))

class PSAFEARRAYBOUND(NDRPOINTER):
    referent = (('Data', SAFEARRAYBOUND),)

class BSTR_ARRAY(NDRUniConformantArray):
    item = BSTR

class PBSTR_ARRAY(NDRPOINTER):
    referent = (('Data', BSTR_ARRAY),)

class SAFEARR_BSTR(NDRSTRUCT):
    structure = (('Size', ULONG), ('aBstr', PBSTR_ARRAY))

class SAFEARR_UNKNOWN(NDRSTRUCT):
    structure = (('Size', ULONG), ('apUnknown', MInterfacePointer_ARRAY))

class SAFEARR_DISPATCH(NDRSTRUCT):
    structure = (('Size', ULONG), ('apDispatch', MInterfacePointer_ARRAY))

class BRECORD_ARRAY(NDRUniConformantArray):
    item = BRECORD

class SAFEARR_BRECORD(NDRSTRUCT):
    structure = (('Size', ULONG), ('aRecord', BRECORD_ARRAY))

class SAFEARR_HAVEIID(NDRSTRUCT):
    structure = (('Size', ULONG), ('apUnknown', MInterfacePointer_ARRAY), ('iid', IID))

class BYTE_SIZEDARR(NDRSTRUCT):
    structure = (('clSize', ULONG), ('pData', BYTE_ARRAY))

class WORD_ARRAY(NDRUniConformantArray):
    item = '<H'

class WORD_SIZEDARR(NDRSTRUCT):
    structure = (('clSize', ULONG), ('pData', WORD_ARRAY))

class DWORD_ARRAY(NDRUniConformantArray):
    item = '<L'

class DWORD_SIZEDARR(NDRSTRUCT):
    structure = (('clSize', ULONG), ('pData', DWORD_ARRAY))

class HYPER_ARRAY(NDRUniConformantArray):
    item = '<Q'

class HYPER_SIZEDARR(NDRSTRUCT):
    structure = (('clSize', ULONG), ('pData', HYPER_ARRAY))
HREFTYPE = DWORD

class VARIANT_ARRAY(NDRUniConformantArray):

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRUniConformantArray.__init__(self, data, isNDR64)
        self.item = VARIANT

class PVARIANT_ARRAY(NDRPOINTER):
    referent = (('Data', VARIANT_ARRAY),)

class PVARIANT(NDRPOINTER):

    def __init__(self, data=None, isNDR64=False):
        if False:
            for i in range(10):
                print('nop')
        NDRPOINTER.__init__(self, data, isNDR64)
        self.referent = (('Data', VARIANT),)

class SAFEARR_VARIANT(NDRSTRUCT):
    structure = (('Size', ULONG), ('aVariant', VARIANT_ARRAY))

class SAFEARRAYUNION(NDRUNION):
    commonHdr = (('tag', ULONG),)
    union = {SF_TYPE.SF_BSTR: ('BstrStr', SAFEARR_BSTR), SF_TYPE.SF_UNKNOWN: ('UnknownStr', SAFEARR_UNKNOWN), SF_TYPE.SF_DISPATCH: ('DispatchStr', SAFEARR_DISPATCH), SF_TYPE.SF_VARIANT: ('VariantStr', SAFEARR_VARIANT), SF_TYPE.SF_RECORD: ('RecordStr', SAFEARR_BRECORD), SF_TYPE.SF_HAVEIID: ('HaveIidStr', SAFEARR_HAVEIID), SF_TYPE.SF_I1: ('ByteStr', BYTE_SIZEDARR), SF_TYPE.SF_I2: ('WordStr', WORD_SIZEDARR), SF_TYPE.SF_I4: ('LongStr', DWORD_SIZEDARR), SF_TYPE.SF_I8: ('HyperStr', HYPER_SIZEDARR)}

class SAFEARRAYBOUND_ARRAY(NDRUniConformantArray):
    item = SAFEARRAYBOUND

class PSAFEARRAYBOUND_ARRAY(NDRPOINTER):
    referent = (('Data', SAFEARRAYBOUND_ARRAY),)

class SAFEARRAY(NDRSTRUCT):
    structure = (('cDims', USHORT), ('fFeatures', USHORT), ('cbElements', ULONG), ('cLocks', ULONG), ('uArrayStructs', SAFEARRAYUNION), ('rgsabound', SAFEARRAYBOUND_ARRAY))

class PSAFEARRAY(NDRPOINTER):
    referent = (('Data', SAFEARRAY),)

class EMPTY(NDR):
    align = 0
    structure = ()

class varUnion(NDRUNION):
    commonHdr = (('tag', ULONG),)
    union = {VARENUM.VT_I8: ('llVal', LONGLONG), VARENUM.VT_I4: ('lVal', LONG), VARENUM.VT_UI1: ('bVal', BYTE), VARENUM.VT_I2: ('iVal', SHORT), VARENUM.VT_R4: ('fltVal', FLOAT), VARENUM.VT_R8: ('dblVal', DOUBLE), VARENUM.VT_BOOL: ('boolVal', VARIANT_BOOL), VARENUM.VT_ERROR: ('scode', HRESULT), VARENUM.VT_CY: ('cyVal', CURRENCY), VARENUM.VT_DATE: ('date', DATE), VARENUM.VT_BSTR: ('bstrVal', BSTR), VARENUM.VT_UNKNOWN: ('punkVal', PMInterfacePointer), VARENUM.VT_DISPATCH: ('pdispVal', PMInterfacePointer), VARENUM.VT_ARRAY: ('parray', SAFEARRAY), VARENUM.VT_RECORD: ('brecVal', BRECORD), VARENUM.VT_RECORD_OR_VT_BYREF: ('brecVal', BRECORD), VARENUM.VT_UI1_OR_VT_BYREF: ('pbVal', BYTE), VARENUM.VT_I2_OR_VT_BYREF: ('piVal', PSHORT), VARENUM.VT_I4_OR_VT_BYREF: ('plVal', PLONG), VARENUM.VT_I8_OR_VT_BYREF: ('pllVal', PLONGLONG), VARENUM.VT_R4_OR_VT_BYREF: ('pfltVal', PFLOAT), VARENUM.VT_R8_OR_VT_BYREF: ('pdblVal', PDOUBLE), VARENUM.VT_BOOL_OR_VT_BYREF: ('pboolVal', PVARIANT_BOOL), VARENUM.VT_ERROR_OR_VT_BYREF: ('pscode', PHRESULT), VARENUM.VT_CY_OR_VT_BYREF: ('pcyVal', PCURRENCY), VARENUM.VT_DATE_OR_VT_BYREF: ('pdate', PDATE), VARENUM.VT_BSTR_OR_VT_BYREF: ('pbstrVal', PBSTR), VARENUM.VT_UNKNOWN_OR_VT_BYREF: ('ppunkVal', PPMInterfacePointer), VARENUM.VT_DISPATCH_OR_VT_BYREF: ('ppdispVal', PPMInterfacePointer), VARENUM.VT_ARRAY_OR_VT_BYREF: ('pparray', PSAFEARRAY), VARENUM.VT_VARIANT_OR_VT_BYREF: ('pvarVal', PVARIANT), VARENUM.VT_I1: ('cVal', CHAR), VARENUM.VT_UI2: ('uiVal', USHORT), VARENUM.VT_UI4: ('ulVal', ULONG), VARENUM.VT_UI8: ('ullVal', ULONGLONG), VARENUM.VT_INT: ('intVal', INT), VARENUM.VT_UINT: ('uintVal', UINT), VARENUM.VT_DECIMAL: ('decVal', DECIMAL), VARENUM.VT_I1_OR_VT_BYREF: ('pcVal', PCHAR), VARENUM.VT_UI2_OR_VT_BYREF: ('puiVal', PUSHORT), VARENUM.VT_UI4_OR_VT_BYREF: ('pulVal', PULONG), VARENUM.VT_UI8_OR_VT_BYREF: ('pullVal', PULONGLONG), VARENUM.VT_INT_OR_VT_BYREF: ('pintVal', PINT), VARENUM.VT_UINT_OR_VT_BYREF: ('puintVal', PUINT), VARENUM.VT_DECIMAL_OR_VT_BYREF: ('pdecVal', PDECIMAL), VARENUM.VT_EMPTY: ('empty', EMPTY), VARENUM.VT_NULL: ('null', EMPTY)}

class wireVARIANTStr(NDRSTRUCT):
    structure = (('clSize', DWORD), ('rpcReserved', DWORD), ('vt', USHORT), ('wReserved1', USHORT), ('wReserved2', USHORT), ('wReserved3', USHORT), ('_varUnion', varUnion))

    def getAlignment(self):
        if False:
            return 10
        return 8

class VARIANT(NDRPOINTER):
    referent = (('Data', wireVARIANTStr),)

class PVARIANT(NDRPOINTER):
    referent = (('Data', VARIANT),)
DISPID = LONG

class DISPID_ARRAY(NDRUniConformantArray):
    item = '<L'

class PDISPID_ARRAY(NDRPOINTER):
    referent = (('Data', DISPID_ARRAY),)

class DISPPARAMS(NDRSTRUCT):
    structure = (('rgvarg', PVARIANT_ARRAY), ('rgdispidNamedArgs', PDISPID_ARRAY), ('cArgs', UINT), ('cNamedArgs', UINT))

class EXCEPINFO(NDRSTRUCT):
    structure = (('wCode', WORD), ('wReserved', WORD), ('bstrSource', BSTR), ('bstrDescription', BSTR), ('bstrHelpFile', BSTR), ('dwHelpContext', DWORD), ('pvReserved', ULONG), ('pfnDeferredFillIn', ULONG), ('scode', HRESULT))
MEMBERID = DISPID

class ARRAYDESC(NDRSTRUCT):

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        self.structure = (('tdescElem', TYPEDESC), ('cDims', USHORT), ('rgbounds', SAFEARRAYBOUND_ARRAY))

class tdUnion(NDRUNION):
    notAlign = True
    commonHdr = (('tag', USHORT),)

    def __init__(self, data=None, isNDR64=False, topLevel=False):
        if False:
            i = 10
            return i + 15
        NDRUNION.__init__(self, None, isNDR64=isNDR64, topLevel=topLevel)
        self.union = {VARENUM.VT_PTR: ('lptdesc', PTYPEDESC), VARENUM.VT_SAFEARRAY: ('lptdesc', PTYPEDESC), VARENUM.VT_CARRAY: ('lpadesc', ARRAYDESC), VARENUM.VT_USERDEFINED: ('hreftype', HREFTYPE), 'default': None}

class TYPEDESC(NDRSTRUCT):
    structure = (('vtType', tdUnion), ('vt', VARENUM))

    def getAlignment(self):
        if False:
            print('Hello World!')
        return 4

class PTYPEDESC(NDRPOINTER):
    referent = (('Data', TYPEDESC),)

    def __init__(self, data=None, isNDR64=False, topLevel=False):
        if False:
            for i in range(10):
                print('nop')
        ret = NDRPOINTER.__init__(self, None, isNDR64=isNDR64, topLevel=False)
        if data is None:
            self.fields['ReferentID'] = random.randint(1, 65535)
        else:
            self.fromString(data)
SCODE = LONG

class SCODE_ARRAY(NDRUniConformantArray):
    item = SCODE

class PSCODE_ARRAY(NDRPOINTER):
    referent = (('Data', SCODE_ARRAY),)

class PARAMDESCEX(NDRSTRUCT):
    structure = (('cBytes', ULONG), ('varDefaultValue', VARIANT))

class PPARAMDESCEX(NDRPOINTER):
    referent = (('Data', PARAMDESCEX),)

class PARAMDESC(NDRSTRUCT):
    structure = (('pparamdescex', PPARAMDESCEX), ('wParamFlags', USHORT))

class ELEMDESC(NDRSTRUCT):
    structure = (('tdesc', TYPEDESC), ('paramdesc', PARAMDESC))

class ELEMDESC_ARRAY(NDRUniConformantArray):
    item = ELEMDESC

class PELEMDESC_ARRAY(NDRPOINTER):
    referent = (('Data', ELEMDESC_ARRAY),)

class FUNCDESC(NDRSTRUCT):
    structure = (('memid', MEMBERID), ('lReserved1', PSCODE_ARRAY), ('lprgelemdescParam', PELEMDESC_ARRAY), ('funckind', FUNCKIND), ('invkind', INVOKEKIND), ('callconv', CALLCONV), ('cParams', SHORT), ('cParamsOpt', SHORT), ('oVft', SHORT), ('cReserved2', SHORT), ('elemdescFunc', ELEMDESC), ('wFuncFlags', WORD))

class LPFUNCDESC(NDRPOINTER):
    referent = (('Data', FUNCDESC),)

class TYPEATTR(NDRSTRUCT):
    structure = (('guid', GUID), ('lcid', LCID), ('dwReserved1', DWORD), ('dwReserved2', DWORD), ('dwReserved3', DWORD), ('lpstrReserved4', LPOLESTR), ('cbSizeInstance', ULONG), ('typeKind', TYPEKIND), ('cFuncs', WORD), ('cVars', WORD), ('cImplTypes', WORD), ('cbSizeVft', WORD), ('cbAlignment', WORD), ('wTypeFlags', WORD), ('wMajorVerNum', WORD), ('wMinorVerNum', WORD), ('tdescAlias', TYPEDESC), ('dwReserved5', DWORD), ('dwReserved6', WORD))

class PTYPEATTR(NDRPOINTER):
    referent = (('Data', TYPEATTR),)

class BSTR_ARRAY_CV(NDRUniConformantVaryingArray):
    item = BSTR

class UINT_ARRAY(NDRUniConformantArray):
    item = '<L'

class OLESTR_ARRAY(NDRUniConformantArray):
    item = LPOLESTR

class IDispatch_GetTypeInfoCount(DCOMCALL):
    opnum = 3
    structure = (('pwszMachineName', LPWSTR),)

class IDispatch_GetTypeInfoCountResponse(DCOMANSWER):
    structure = (('pctinfo', ULONG), ('ErrorCode', error_status_t))

class IDispatch_GetTypeInfo(DCOMCALL):
    opnum = 4
    structure = (('iTInfo', ULONG), ('lcid', DWORD))

class IDispatch_GetTypeInfoResponse(DCOMANSWER):
    structure = (('ppTInfo', PMInterfacePointer), ('ErrorCode', error_status_t))

class IDispatch_GetIDsOfNames(DCOMCALL):
    opnum = 5
    structure = (('riid', REFIID), ('rgszNames', OLESTR_ARRAY), ('cNames', UINT), ('lcid', LCID))

class IDispatch_GetIDsOfNamesResponse(DCOMANSWER):
    structure = (('rgDispId', DISPID_ARRAY), ('ErrorCode', error_status_t))

class IDispatch_Invoke(DCOMCALL):
    opnum = 6
    structure = (('dispIdMember', DISPID), ('riid', REFIID), ('lcid', LCID), ('dwFlags', DWORD), ('pDispParams', DISPPARAMS), ('cVarRef', UINT), ('rgVarRefIdx', UINT_ARRAY), ('rgVarRef', VARIANT_ARRAY))

class IDispatch_InvokeResponse(DCOMANSWER):
    structure = (('pVarResult', VARIANT), ('pExcepInfo', EXCEPINFO), ('pArgErr', UINT), ('ErrorCode', error_status_t))

class ITypeInfo_GetTypeAttr(DCOMCALL):
    opnum = 3
    structure = ()

class ITypeInfo_GetTypeAttrResponse(DCOMANSWER):
    structure = (('ppTypeAttr', PTYPEATTR), ('pReserved', DWORD), ('ErrorCode', error_status_t))

class ITypeInfo_GetTypeComp(DCOMCALL):
    opnum = 4
    structure = ()

class ITypeInfo_GetTypeCompResponse(DCOMANSWER):
    structure = (('ppTComp', PMInterfacePointer), ('ErrorCode', error_status_t))

class ITypeInfo_GetFuncDesc(DCOMCALL):
    opnum = 5
    structure = (('index', UINT),)

class ITypeInfo_GetFuncDescResponse(DCOMANSWER):
    structure = (('ppFuncDesc', LPFUNCDESC), ('pReserved', DWORD), ('ErrorCode', error_status_t))

class ITypeInfo_GetNames(DCOMCALL):
    opnum = 7
    structure = (('memid', MEMBERID), ('cMaxNames', UINT))

class ITypeInfo_GetNamesResponse(DCOMANSWER):
    structure = (('rgBstrNames', BSTR_ARRAY_CV), ('pcNames', UINT), ('ErrorCode', error_status_t))

class ITypeInfo_GetDocumentation(DCOMCALL):
    opnum = 12
    structure = (('memid', MEMBERID), ('refPtrFlags', DWORD))

class ITypeInfo_GetDocumentationResponse(DCOMANSWER):
    structure = (('pBstrName', BSTR), ('pBstrDocString', BSTR), ('pdwHelpContext', DWORD), ('ErrorCode', error_status_t))
OPNUMS = {}

def enumerateMethods(iInterface):
    if False:
        while True:
            i = 10
    methods = dict()
    typeInfoCount = iInterface.GetTypeInfoCount()
    if typeInfoCount['pctinfo'] == 0:
        LOG.error('Automation Server does not support type information for this object')
        return {}
    iTypeInfo = iInterface.GetTypeInfo()
    iTypeAttr = iTypeInfo.GetTypeAttr()
    for x in range(iTypeAttr['ppTypeAttr']['cFuncs']):
        funcDesc = iTypeInfo.GetFuncDesc(x)
        names = iTypeInfo.GetNames(funcDesc['ppFuncDesc']['memid'], 255)
        print(names['rgBstrNames'][0]['asData'])
        funcDesc.dump()
        print('=' * 80)
        if names['pcNames'] > 0:
            name = names['rgBstrNames'][0]['asData']
            methods[name] = {}
            for param in range(1, names['pcNames']):
                methods[name][names['rgBstrNames'][param]['asData']] = ''
        if funcDesc['ppFuncDesc']['elemdescFunc'] != NULL:
            methods[name]['ret'] = funcDesc['ppFuncDesc']['elemdescFunc']['tdesc']['vt']
    return methods

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

class ITypeComp(IRemUnknown2):

    def __init__(self, interface):
        if False:
            while True:
                i = 10
        IRemUnknown2.__init__(self, interface)
        self._iid = IID_ITypeComp

class ITypeInfo(IRemUnknown2):

    def __init__(self, interface):
        if False:
            return 10
        IRemUnknown2.__init__(self, interface)
        self._iid = IID_ITypeInfo

    def GetTypeAttr(self):
        if False:
            i = 10
            return i + 15
        request = ITypeInfo_GetTypeAttr()
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp

    def GetTypeComp(self):
        if False:
            print('Hello World!')
        request = ITypeInfo_GetTypeComp()
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return ITypeComp(INTERFACE(self.get_cinstance(), b''.join(resp['ppTComp']['abData']), self.get_ipidRemUnknown(), target=self.get_target()))

    def GetFuncDesc(self, index):
        if False:
            i = 10
            return i + 15
        request = ITypeInfo_GetFuncDesc()
        request['index'] = index
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp

    def GetNames(self, memid, cMaxNames=10):
        if False:
            while True:
                i = 10
        request = ITypeInfo_GetNames()
        request['memid'] = memid
        request['cMaxNames'] = cMaxNames
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp

    def GetDocumentation(self, memid, refPtrFlags=15):
        if False:
            for i in range(10):
                print('nop')
        request = ITypeInfo_GetDocumentation()
        request['memid'] = memid
        request['refPtrFlags'] = refPtrFlags
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp

class IDispatch(IRemUnknown2):

    def __init__(self, interface):
        if False:
            return 10
        IRemUnknown2.__init__(self, interface)
        self._iid = IID_IDispatch

    def GetTypeInfoCount(self):
        if False:
            return 10
        request = IDispatch_GetTypeInfoCount()
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp

    def GetTypeInfo(self):
        if False:
            while True:
                i = 10
        request = IDispatch_GetTypeInfo()
        request['iTInfo'] = 0
        request['lcid'] = 0
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return ITypeInfo(INTERFACE(self.get_cinstance(), b''.join(resp['ppTInfo']['abData']), self.get_ipidRemUnknown(), target=self.get_target()))

    def GetIDsOfNames(self, rgszNames, lcid=0):
        if False:
            for i in range(10):
                print('nop')
        request = IDispatch_GetIDsOfNames()
        request['riid'] = IID_NULL
        for name in rgszNames:
            tmpName = LPOLESTR()
            tmpName['Data'] = checkNullString(name)
            request['rgszNames'].append(tmpName)
        request['cNames'] = len(rgszNames)
        request['lcid'] = lcid
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        IDs = list()
        for id in resp['rgDispId']:
            IDs.append(id)
        return IDs

    def Invoke(self, dispIdMember, lcid, dwFlags, pDispParams, cVarRef, rgVarRefIdx, rgVarRef):
        if False:
            return 10
        request = IDispatch_Invoke()
        request['dispIdMember'] = dispIdMember
        request['riid'] = IID_NULL
        request['lcid'] = lcid
        request['dwFlags'] = dwFlags
        request['pDispParams'] = pDispParams
        request['cVarRef'] = cVarRef
        request['rgVarRefIdx'] = rgVarRefIdx
        request['rgVarRef'] = rgVarRefIdx
        resp = self.request(request, iid=self._iid, uuid=self.get_iPid())
        return resp