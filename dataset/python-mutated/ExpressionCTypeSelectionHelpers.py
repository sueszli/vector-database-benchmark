""" Module for helpers to select types for operation arguments.

This is first used for comparisons and binary operations, but should see
general use too and expand beyond constant values, e.g. covering constant
values that are of behind conditions or variables.
"""
from nuitka.nodes.shapes.BuiltinTypeShapes import tshape_bytearray, tshape_bytes, tshape_float, tshape_int, tshape_long, tshape_str, tshape_unicode
from nuitka.PythonVersions import isPythonValidCLongValue, isPythonValidDigitValue, python_version
from .c_types.CTypeCFloats import CTypeCFloat
from .c_types.CTypeCLongs import CTypeCLong, CTypeCLongDigit
from .c_types.CTypePyObjectPointers import CTypePyObjectPtr

def _pickIntFamilyType(expression):
    if False:
        i = 10
        return i + 15
    if expression.isCompileTimeConstant():
        if python_version < 768:
            c_type = CTypeCLong
        elif isPythonValidDigitValue(expression.getCompileTimeConstant()):
            c_type = CTypeCLongDigit
        elif isPythonValidCLongValue(expression.getCompileTimeConstant()):
            c_type = CTypeCLong
        else:
            c_type = CTypePyObjectPtr
    else:
        c_type = CTypePyObjectPtr
    return c_type

def _pickFloatFamilyType(expression):
    if False:
        print('Hello World!')
    if expression.isCompileTimeConstant():
        c_type = CTypeCFloat
    else:
        c_type = CTypePyObjectPtr
    return c_type

def _pickStrFamilyType(expression):
    if False:
        return 10
    return CTypePyObjectPtr

def _pickBytesFamilyType(expression):
    if False:
        print('Hello World!')
    return CTypePyObjectPtr
_int_types_family = (tshape_int, tshape_long)
_float_types_family = (tshape_int, tshape_long, tshape_float)
_str_types_family = (tshape_str, tshape_unicode)
_bytes_types_family = (tshape_bytes,)
_float_argument_normalization = {(CTypePyObjectPtr, CTypeCFloat): False, (CTypeCFloat, CTypePyObjectPtr): True}
_long_argument_normalization = {(CTypePyObjectPtr, CTypeCLong): False, (CTypeCLong, CTypePyObjectPtr): True, (CTypePyObjectPtr, CTypeCLongDigit): False, (CTypeCLongDigit, CTypePyObjectPtr): True}
_str_argument_normalization = {}
_bytes_argument_normalization = {}

def decideExpressionCTypes(left, right, may_swap_arguments):
    if False:
        print('Hello World!')
    left_shape = left.getTypeShape()
    right_shape = right.getTypeShape()
    if left_shape in _int_types_family and right_shape in _int_types_family:
        may_swap_arguments = may_swap_arguments in ('number', 'always')
        left_c_type = _pickIntFamilyType(left)
        right_c_type = _pickIntFamilyType(right)
        needs_argument_swap = may_swap_arguments and left_c_type is not right_c_type and _long_argument_normalization[left_c_type, right_c_type]
        if may_swap_arguments and (not needs_argument_swap):
            if right_shape is tshape_long and left_shape is tshape_int:
                needs_argument_swap = True
        unknown_types = False
    elif left_shape in _float_types_family and right_shape in _float_types_family:
        may_swap_arguments = may_swap_arguments in ('number', 'always')
        left_c_type = _pickFloatFamilyType(left)
        right_c_type = _pickFloatFamilyType(right)
        needs_argument_swap = may_swap_arguments and left_c_type is not right_c_type and _float_argument_normalization[left_c_type, right_c_type]
        if may_swap_arguments and (not needs_argument_swap):
            if right_shape is tshape_float and left_shape in (tshape_int, tshape_long):
                needs_argument_swap = True
        unknown_types = False
    elif left_shape in _str_types_family and right_shape in _str_types_family:
        may_swap_arguments = may_swap_arguments == 'always'
        left_c_type = _pickStrFamilyType(left)
        right_c_type = _pickStrFamilyType(right)
        needs_argument_swap = may_swap_arguments and left_c_type is not right_c_type and _str_argument_normalization[left_c_type, right_c_type]
        if may_swap_arguments and (not needs_argument_swap) and (str is bytes):
            if right_shape is tshape_unicode and left_shape is tshape_str:
                needs_argument_swap = True
        unknown_types = False
    elif left_shape in _bytes_types_family and right_shape in _bytes_types_family:
        may_swap_arguments = may_swap_arguments == 'always'
        left_c_type = _pickBytesFamilyType(left)
        right_c_type = _pickBytesFamilyType(right)
        needs_argument_swap = may_swap_arguments and left_c_type is not right_c_type and _bytes_argument_normalization[left_c_type, right_c_type]
        if may_swap_arguments and (not needs_argument_swap):
            if right_shape is tshape_bytearray and left_shape is tshape_bytes:
                needs_argument_swap = True
        unknown_types = False
    else:
        left_c_type = right_c_type = CTypePyObjectPtr
        needs_argument_swap = False
        unknown_types = True
    return (unknown_types, needs_argument_swap, left_shape, right_shape, left_c_type, right_c_type)