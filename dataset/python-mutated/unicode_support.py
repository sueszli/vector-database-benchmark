"""
This module contains support functions for more advanced unicode operations.
This is not a public API and is for Numba internal use only. Most of the
functions are relatively straightforward translations of the functions with the
same name in CPython.
"""
from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import impl_ret_untracked
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
typerecord = namedtuple('typerecord', 'upper lower title decimal digit flags')
_Py_UCS4 = types.uint32
_Py_TAB = 9
_Py_LINEFEED = 10
_Py_CARRIAGE_RETURN = 13
_Py_SPACE = 32

class _PyUnicode_TyperecordMasks(IntEnum):
    ALPHA_MASK = 1
    DECIMAL_MASK = 2
    DIGIT_MASK = 4
    LOWER_MASK = 8
    LINEBREAK_MASK = 16
    SPACE_MASK = 32
    TITLE_MASK = 64
    UPPER_MASK = 128
    XID_START_MASK = 256
    XID_CONTINUE_MASK = 512
    PRINTABLE_MASK = 1024
    NUMERIC_MASK = 2048
    CASE_IGNORABLE_MASK = 4096
    CASED_MASK = 8192
    EXTENDED_CASE_MASK = 16384

def _PyUnicode_gettyperecord(a):
    if False:
        i = 10
        return i + 15
    raise RuntimeError('Calling the Python definition is invalid')

@intrinsic
def _gettyperecord_impl(typingctx, codepoint):
    if False:
        i = 10
        return i + 15
    '\n    Provides the binding to numba_gettyperecord, returns a `typerecord`\n    namedtuple of properties from the codepoint.\n    '
    if not isinstance(codepoint, types.Integer):
        raise TypingError('codepoint must be an integer')

    def details(context, builder, signature, args):
        if False:
            i = 10
            return i + 15
        ll_void = context.get_value_type(types.void)
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        ll_intc_ptr = ll_intc.as_pointer()
        ll_uchar = context.get_value_type(types.uchar)
        ll_uchar_ptr = ll_uchar.as_pointer()
        ll_ushort = context.get_value_type(types.ushort)
        ll_ushort_ptr = ll_ushort.as_pointer()
        fnty = llvmlite.ir.FunctionType(ll_void, [ll_Py_UCS4, ll_intc_ptr, ll_intc_ptr, ll_intc_ptr, ll_uchar_ptr, ll_uchar_ptr, ll_ushort_ptr])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name='numba_gettyperecord')
        upper = cgutils.alloca_once(builder, ll_intc, name='upper')
        lower = cgutils.alloca_once(builder, ll_intc, name='lower')
        title = cgutils.alloca_once(builder, ll_intc, name='title')
        decimal = cgutils.alloca_once(builder, ll_uchar, name='decimal')
        digit = cgutils.alloca_once(builder, ll_uchar, name='digit')
        flags = cgutils.alloca_once(builder, ll_ushort, name='flags')
        byref = [upper, lower, title, decimal, digit, flags]
        builder.call(fn, [args[0]] + byref)
        buf = []
        for x in byref:
            buf.append(builder.load(x))
        res = context.make_tuple(builder, signature.return_type, tuple(buf))
        return impl_ret_untracked(context, builder, signature.return_type, res)
    tupty = types.NamedTuple([types.intc, types.intc, types.intc, types.uchar, types.uchar, types.ushort], typerecord)
    sig = tupty(_Py_UCS4)
    return (sig, details)

@overload(_PyUnicode_gettyperecord)
def gettyperecord_impl(a):
    if False:
        return 10
    '\n    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept\n    single character strings and code points.\n    '
    if isinstance(a, types.UnicodeType):
        from numba.cpython.unicode import _get_code_point

        def impl(a):
            if False:
                for i in range(10):
                    print('nop')
            if len(a) > 1:
                msg = 'gettyperecord takes a single unicode character'
                raise ValueError(msg)
            code_point = _get_code_point(a, 0)
            data = _gettyperecord_impl(_Py_UCS4(code_point))
            return data
        return impl
    if isinstance(a, types.Integer):
        return lambda a: _gettyperecord_impl(_Py_UCS4(a))

@intrinsic
def _PyUnicode_ExtendedCase(typingctx, index):
    if False:
        print('Hello World!')
    '\n    Accessor function for the _PyUnicode_ExtendedCase array, binds to\n    numba_get_PyUnicode_ExtendedCase which wraps the array and does the lookup\n    '
    if not isinstance(index, types.Integer):
        raise TypingError('Expected an index')

    def details(context, builder, signature, args):
        if False:
            for i in range(10):
                print('nop')
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        fnty = llvmlite.ir.FunctionType(ll_Py_UCS4, [ll_intc])
        fn = cgutils.get_or_insert_function(builder.module, fnty, name='numba_get_PyUnicode_ExtendedCase')
        return builder.call(fn, [args[0]])
    sig = _Py_UCS4(types.intc)
    return (sig, details)

@register_jitable
def _PyUnicode_ToTitlecase(ch):
    if False:
        print('Hello World!')
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        return _PyUnicode_ExtendedCase(ctype.title & 65535)
    return ch + ctype.title

@register_jitable
def _PyUnicode_IsTitlecase(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.TITLE_MASK != 0

@register_jitable
def _PyUnicode_IsXidStart(ch):
    if False:
        return 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.XID_START_MASK != 0

@register_jitable
def _PyUnicode_IsXidContinue(ch):
    if False:
        for i in range(10):
            print('nop')
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.XID_CONTINUE_MASK != 0

@register_jitable
def _PyUnicode_ToDecimalDigit(ch):
    if False:
        for i in range(10):
            print('nop')
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.DECIMAL_MASK:
        return ctype.decimal
    return -1

@register_jitable
def _PyUnicode_ToDigit(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.DIGIT_MASK:
        return ctype.digit
    return -1

@register_jitable
def _PyUnicode_IsNumeric(ch):
    if False:
        i = 10
        return i + 15
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.NUMERIC_MASK != 0

@register_jitable
def _PyUnicode_IsPrintable(ch):
    if False:
        print('Hello World!')
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.PRINTABLE_MASK != 0

@register_jitable
def _PyUnicode_IsLowercase(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.LOWER_MASK != 0

@register_jitable
def _PyUnicode_IsUppercase(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.UPPER_MASK != 0

@register_jitable
def _PyUnicode_IsLineBreak(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.LINEBREAK_MASK != 0

@register_jitable
def _PyUnicode_ToUppercase(ch):
    if False:
        return 10
    raise NotImplementedError

@register_jitable
def _PyUnicode_ToLowercase(ch):
    if False:
        return 10
    raise NotImplementedError

@register_jitable
def _PyUnicode_ToLowerFull(ch, res):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.lower & 65535
        n = ctype.lower >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.lower
    return 1

@register_jitable
def _PyUnicode_ToTitleFull(ch, res):
    if False:
        print('Hello World!')
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.title & 65535
        n = ctype.title >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.title
    return 1

@register_jitable
def _PyUnicode_ToUpperFull(ch, res):
    if False:
        i = 10
        return i + 15
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.upper & 65535
        n = ctype.upper >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.upper
    return 1

@register_jitable
def _PyUnicode_ToFoldedFull(ch, res):
    if False:
        for i in range(10):
            print('nop')
    ctype = _PyUnicode_gettyperecord(ch)
    extended_case_mask = _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK
    if ctype.flags & extended_case_mask and ctype.lower >> 20 & 7:
        index = (ctype.lower & 65535) + (ctype.lower >> 24)
        n = ctype.lower >> 20 & 7
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    return _PyUnicode_ToLowerFull(ch, res)

@register_jitable
def _PyUnicode_IsCased(ch):
    if False:
        while True:
            i = 10
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.CASED_MASK != 0

@register_jitable
def _PyUnicode_IsCaseIgnorable(ch):
    if False:
        for i in range(10):
            print('nop')
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.CASE_IGNORABLE_MASK != 0

@register_jitable
def _PyUnicode_IsDigit(ch):
    if False:
        while True:
            i = 10
    if _PyUnicode_ToDigit(ch) < 0:
        return 0
    return 1

@register_jitable
def _PyUnicode_IsDecimalDigit(ch):
    if False:
        i = 10
        return i + 15
    if _PyUnicode_ToDecimalDigit(ch) < 0:
        return 0
    return 1

@register_jitable
def _PyUnicode_IsSpace(ch):
    if False:
        i = 10
        return i + 15
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.SPACE_MASK != 0

@register_jitable
def _PyUnicode_IsAlpha(ch):
    if False:
        for i in range(10):
            print('nop')
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.ALPHA_MASK != 0

class _PY_CTF(IntEnum):
    LOWER = 1
    UPPER = 2
    ALPHA = 1 | 2
    DIGIT = 4
    ALNUM = 1 | 2 | 4
    SPACE = 8
    XDIGIT = 16
_Py_ctype_table = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF.SPACE, _PY_CTF.SPACE, _PY_CTF.SPACE, _PY_CTF.SPACE, _PY_CTF.SPACE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF.SPACE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, _PY_CTF.DIGIT | _PY_CTF.XDIGIT, 0, 0, 0, 0, 0, 0, 0, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER | _PY_CTF.XDIGIT, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, _PY_CTF.UPPER, 0, 0, 0, 0, 0, 0, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER | _PY_CTF.XDIGIT, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, _PY_CTF.LOWER, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intc)
_Py_ctype_tolower = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255], dtype=np.uint8)
_Py_ctype_toupper = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255], dtype=np.uint8)

class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 1
    LINE_FEED = 2
    CARRIAGE_RETURN = 4
_Py_ctype_islinebreak = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF_LB.LINE_BREAK | _PY_CTF_LB.LINE_FEED, _PY_CTF_LB.LINE_BREAK, _PY_CTF_LB.LINE_BREAK, _PY_CTF_LB.LINE_BREAK | _PY_CTF_LB.CARRIAGE_RETURN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF_LB.LINE_BREAK, _PY_CTF_LB.LINE_BREAK, _PY_CTF_LB.LINE_BREAK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, _PY_CTF_LB.LINE_BREAK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intc)

@register_jitable
def _Py_CHARMASK(ch):
    if False:
        print('Hello World!')
    '\n    Equivalent to the CPython macro `Py_CHARMASK()`, masks off all but the\n    lowest 256 bits of ch.\n    '
    return types.uint8(ch) & types.uint8(255)

@register_jitable
def _Py_TOUPPER(ch):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent to the CPython macro `Py_TOUPPER()` converts an ASCII range\n    code point to the upper equivalent\n    '
    return _Py_ctype_toupper[_Py_CHARMASK(ch)]

@register_jitable
def _Py_TOLOWER(ch):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent to the CPython macro `Py_TOLOWER()` converts an ASCII range\n    code point to the lower equivalent\n    '
    return _Py_ctype_tolower[_Py_CHARMASK(ch)]

@register_jitable
def _Py_ISLOWER(ch):
    if False:
        for i in range(10):
            print('nop')
    '\n    Equivalent to the CPython macro `Py_ISLOWER()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.LOWER

@register_jitable
def _Py_ISUPPER(ch):
    if False:
        print('Hello World!')
    '\n    Equivalent to the CPython macro `Py_ISUPPER()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.UPPER

@register_jitable
def _Py_ISALPHA(ch):
    if False:
        return 10
    '\n    Equivalent to the CPython macro `Py_ISALPHA()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.ALPHA

@register_jitable
def _Py_ISDIGIT(ch):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent to the CPython macro `Py_ISDIGIT()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.DIGIT

@register_jitable
def _Py_ISXDIGIT(ch):
    if False:
        while True:
            i = 10
    '\n    Equivalent to the CPython macro `Py_ISXDIGIT()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.XDIGIT

@register_jitable
def _Py_ISALNUM(ch):
    if False:
        while True:
            i = 10
    '\n    Equivalent to the CPython macro `Py_ISALNUM()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.ALNUM

@register_jitable
def _Py_ISSPACE(ch):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent to the CPython macro `Py_ISSPACE()`\n    '
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.SPACE

@register_jitable
def _Py_ISLINEBREAK(ch):
    if False:
        i = 10
        return i + 15
    'Check if character is ASCII line break'
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.LINE_BREAK

@register_jitable
def _Py_ISLINEFEED(ch):
    if False:
        return 10
    'Check if character is line feed `\n`'
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.LINE_FEED

@register_jitable
def _Py_ISCARRIAGERETURN(ch):
    if False:
        while True:
            i = 10
    'Check if character is carriage return `\r`'
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.CARRIAGE_RETURN