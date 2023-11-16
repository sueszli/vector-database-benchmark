""" Shapes for Python built-in types.

"""
from nuitka.code_generation.c_types.CTypeNuitkaBooleans import CTypeNuitkaBoolEnum
from nuitka.code_generation.c_types.CTypeNuitkaInts import CTypeNuitkaIntOrLongStruct
from nuitka.code_generation.Reports import onMissingOperation
from nuitka.Constants import the_empty_unicode
from nuitka.Options import isExperimental
from nuitka.PythonVersions import python_version
from .ControlFlowDescriptions import ControlFlowDescriptionAddUnsupported, ControlFlowDescriptionBitandUnsupported, ControlFlowDescriptionBitorUnsupported, ControlFlowDescriptionBitxorUnsupported, ControlFlowDescriptionComparisonUnorderable, ControlFlowDescriptionDivmodUnsupported, ControlFlowDescriptionElementBasedEscape, ControlFlowDescriptionFloorDivUnsupported, ControlFlowDescriptionFormatError, ControlFlowDescriptionLshiftUnsupported, ControlFlowDescriptionMatmultUnsupported, ControlFlowDescriptionModUnsupported, ControlFlowDescriptionMulUnsupported, ControlFlowDescriptionNoEscape, ControlFlowDescriptionOldDivUnsupported, ControlFlowDescriptionPowUnsupported, ControlFlowDescriptionRshiftUnsupported, ControlFlowDescriptionSubUnsupported, ControlFlowDescriptionTrueDivUnsupported, ControlFlowDescriptionValueErrorNoEscape, ControlFlowDescriptionZeroDivisionNoEscape
from .ShapeMixins import ShapeContainerImmutableMixin, ShapeContainerMixin, ShapeContainerMutableMixin, ShapeIteratorMixin, ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeNumberMixin
from .StandardShapes import ShapeBase, ShapeLoopCompleteAlternative, ShapeLoopInitialAlternative, ShapeTypeUnknown, operation_result_unknown, tshape_unknown
add_shapes_none = {}
sub_shapes_none = {}
mult_shapes_none = {}
floordiv_shapes_none = {}
truediv_shapes_none = {}
olddiv_shapes_none = {}
mod_shapes_none = {}
divmod_shapes_none = {}
pow_shapes_none = {}
bitor_shapes_none = {}
bitand_shapes_none = {}
bitxor_shapes_none = {}
lshift_shapes_none = {}
rshift_shapes_none = {}
matmult_shapes_none = {}
add_shapes_bool = {}
sub_shapes_bool = {}
mult_shapes_bool = {}
floordiv_shapes_bool = {}
truediv_shapes_bool = {}
olddiv_shapes_bool = {}
mod_shapes_bool = {}
divmod_shapes_bool = {}
pow_shapes_bool = {}
bitor_shapes_bool = {}
bitand_shapes_bool = {}
bitxor_shapes_bool = {}
lshift_shapes_bool = {}
rshift_shapes_bool = {}
matmult_shapes_bool = matmult_shapes_none
add_shapes_int = {}
sub_shapes_int = {}
mult_shapes_int = {}
floordiv_shapes_int = {}
truediv_shapes_int = {}
olddiv_shapes_int = {}
mod_shapes_int = {}
divmod_shapes_int = divmod_shapes_bool
pow_shapes_int = {}
bitor_shapes_int = {}
bitand_shapes_int = {}
bitxor_shapes_int = {}
lshift_shapes_int = {}
rshift_shapes_int = {}
matmult_shapes_int = matmult_shapes_none
add_shapes_long = {}
sub_shapes_long = {}
mult_shapes_long = {}
floordiv_shapes_long = {}
truediv_shapes_long = truediv_shapes_int
olddiv_shapes_long = {}
mod_shapes_long = {}
divmod_shapes_long = divmod_shapes_bool
pow_shapes_long = {}
bitor_shapes_long = {}
bitand_shapes_long = {}
bitxor_shapes_long = {}
lshift_shapes_long = {}
rshift_shapes_long = {}
matmult_shapes_long = matmult_shapes_none
add_shapes_intorlong = {}
sub_shapes_intorlong = {}
mult_shapes_intorlong = {}
floordiv_shapes_intorlong = {}
truediv_shapes_intorlong = {}
olddiv_shapes_intorlong = {}
mod_shapes_intorlong = {}
divmod_shapes_intorlong = {}
pow_shapes_intorlong = {}
bitor_shapes_intorlong = {}
bitand_shapes_intorlong = {}
bitxor_shapes_intorlong = {}
lshift_shapes_intorlong = {}
rshift_shapes_intorlong = {}
matmult_shapes_intorlong = matmult_shapes_none
add_shapes_float = {}
sub_shapes_float = {}
mult_shapes_float = {}
floordiv_shapes_float = {}
truediv_shapes_float = {}
olddiv_shapes_float = {}
mod_shapes_float = {}
divmod_shapes_float = divmod_shapes_bool
pow_shapes_float = {}
bitor_shapes_float = bitor_shapes_none
bitand_shapes_float = bitand_shapes_none
bitxor_shapes_float = bitxor_shapes_none
lshift_shapes_float = lshift_shapes_none
rshift_shapes_float = rshift_shapes_none
matmult_shapes_float = matmult_shapes_none
add_shapes_complex = {}
sub_shapes_complex = {}
mult_shapes_complex = {}
if python_version < 768:
    floordiv_shapes_complex = {}
else:
    floordiv_shapes_complex = floordiv_shapes_none
truediv_shapes_complex = {}
olddiv_shapes_complex = {}
mod_shapes_complex = {}
divmod_shapes_complex = divmod_shapes_bool
pow_shapes_complex = {}
bitor_shapes_complex = bitor_shapes_none
bitand_shapes_complex = bitand_shapes_none
bitxor_shapes_complex = bitxor_shapes_none
lshift_shapes_complex = lshift_shapes_none
rshift_shapes_complex = rshift_shapes_none
matmult_shapes_complex = matmult_shapes_none
add_shapes_tuple = {}
sub_shapes_tuple = sub_shapes_none
mult_shapes_tuple = {}
floordiv_shapes_tuple = floordiv_shapes_none
truediv_shapes_tuple = truediv_shapes_none
olddiv_shapes_tuple = olddiv_shapes_none
mod_shapes_tuple = mod_shapes_none
divmod_shapes_tuple = divmod_shapes_none
pow_shapes_tuple = pow_shapes_none
bitor_shapes_tuple = bitor_shapes_none
bitand_shapes_tuple = bitand_shapes_none
bitxor_shapes_tuple = bitxor_shapes_none
lshift_shapes_tuple = lshift_shapes_none
rshift_shapes_tuple = rshift_shapes_none
matmult_shapes_tuple = matmult_shapes_none
add_shapes_list = {}
iadd_shapes_list = {}
sub_shapes_list = sub_shapes_none
mult_shapes_list = {}
floordiv_shapes_list = floordiv_shapes_none
truediv_shapes_list = truediv_shapes_none
olddiv_shapes_list = olddiv_shapes_none
mod_shapes_list = mod_shapes_none
divmod_shapes_list = divmod_shapes_none
pow_shapes_list = pow_shapes_none
bitor_shapes_list = bitor_shapes_none
bitand_shapes_list = bitand_shapes_none
bitxor_shapes_list = bitxor_shapes_none
lshift_shapes_list = lshift_shapes_none
rshift_shapes_list = rshift_shapes_none
matmult_shapes_list = matmult_shapes_none
add_shapes_set = {}
sub_shapes_set = {}
mult_shapes_set = mult_shapes_none
floordiv_shapes_set = floordiv_shapes_none
truediv_shapes_set = truediv_shapes_none
olddiv_shapes_set = olddiv_shapes_none
mod_shapes_set = mod_shapes_none
divmod_shapes_set = divmod_shapes_none
pow_shapes_set = pow_shapes_none
bitor_shapes_set = {}
bitand_shapes_set = {}
bitxor_shapes_set = {}
lshift_shapes_set = lshift_shapes_none
rshift_shapes_set = rshift_shapes_none
matmult_shapes_set = matmult_shapes_none
add_shapes_frozenset = {}
sub_shapes_frozenset = {}
mult_shapes_frozenset = mult_shapes_none
floordiv_shapes_frozenset = floordiv_shapes_none
truediv_shapes_frozenset = truediv_shapes_none
olddiv_shapes_frozenset = olddiv_shapes_none
mod_shapes_frozenset = mod_shapes_none
divmod_shapes_frozenset = divmod_shapes_none
pow_shapes_frozenset = pow_shapes_none
bitor_shapes_frozenset = {}
bitand_shapes_frozenset = {}
bitxor_shapes_frozenset = {}
lshift_shapes_frozenset = lshift_shapes_none
rshift_shapes_frozenset = rshift_shapes_none
matmult_shapes_frozenset = matmult_shapes_none
add_shapes_dict = {}
sub_shapes_dict = sub_shapes_none
mult_shapes_dict = mult_shapes_none
floordiv_shapes_dict = floordiv_shapes_none
truediv_shapes_dict = truediv_shapes_none
olddiv_shapes_dict = olddiv_shapes_none
mod_shapes_dict = mod_shapes_none
divmod_shapes_dict = divmod_shapes_none
pow_shapes_dict = pow_shapes_none
bitor_shapes_dict = dict(bitor_shapes_none)
ibitor_shapes_dict = dict(bitor_shapes_none)
bitand_shapes_dict = bitand_shapes_none
bitxor_shapes_dict = bitxor_shapes_none
lshift_shapes_dict = lshift_shapes_none
rshift_shapes_dict = rshift_shapes_none
matmult_shapes_dict = matmult_shapes_none
add_shapes_str = {}
sub_shapes_str = sub_shapes_none
mult_shapes_str = {}
floordiv_shapes_str = floordiv_shapes_none
truediv_shapes_str = truediv_shapes_none
olddiv_shapes_str = olddiv_shapes_none
mod_shapes_str = {}
divmod_shapes_str = divmod_shapes_none
pow_shapes_str = pow_shapes_none
bitor_shapes_str = bitor_shapes_none
bitand_shapes_str = bitand_shapes_none
bitxor_shapes_str = bitxor_shapes_none
lshift_shapes_str = lshift_shapes_none
rshift_shapes_str = rshift_shapes_none
matmult_shapes_str = matmult_shapes_none
add_shapes_bytes = {}
sub_shapes_bytes = sub_shapes_none
mult_shapes_bytes = {}
floordiv_shapes_bytes = floordiv_shapes_none
truediv_shapes_bytes = truediv_shapes_none
olddiv_shapes_bytes = olddiv_shapes_none
mod_shapes_bytes = {}
divmod_shapes_bytes = divmod_shapes_none
pow_shapes_bytes = pow_shapes_none
bitor_shapes_bytes = bitor_shapes_none
bitand_shapes_bytes = bitand_shapes_none
bitxor_shapes_bytes = bitxor_shapes_none
lshift_shapes_bytes = lshift_shapes_none
rshift_shapes_bytes = rshift_shapes_none
matmult_shapes_bytes = matmult_shapes_none
add_shapes_bytearray = {}
sub_shapes_bytearray = sub_shapes_none
mult_shapes_bytearray = {}
floordiv_shapes_bytearray = floordiv_shapes_none
truediv_shapes_bytearray = truediv_shapes_none
olddiv_shapes_bytearray = olddiv_shapes_none
mod_shapes_bytearray = {}
divmod_shapes_bytearray = divmod_shapes_none
pow_shapes_bytearray = pow_shapes_none
bitor_shapes_bytearray = bitor_shapes_none
bitand_shapes_bytearray = bitand_shapes_none
bitxor_shapes_bytearray = bitxor_shapes_none
lshift_shapes_bytearray = lshift_shapes_none
rshift_shapes_bytearray = rshift_shapes_none
matmult_shapes_bytearray = matmult_shapes_none
add_shapes_unicode = {}
sub_shapes_unicode = sub_shapes_none
mult_shapes_unicode = {}
floordiv_shapes_unicode = floordiv_shapes_none
truediv_shapes_unicode = truediv_shapes_none
olddiv_shapes_unicode = olddiv_shapes_none
mod_shapes_unicode = {}
divmod_shapes_unicode = divmod_shapes_none
pow_shapes_unicode = pow_shapes_none
bitor_shapes_unicode = bitor_shapes_none
bitand_shapes_unicode = bitand_shapes_none
bitxor_shapes_unicode = bitxor_shapes_none
lshift_shapes_unicode = lshift_shapes_none
rshift_shapes_unicode = rshift_shapes_none
matmult_shapes_unicode = matmult_shapes_none
add_shapes_strorunicode = {}
sub_shapes_strorunicode = {}
mult_shapes_strorunicode = {}
floordiv_shapes_strorunicode = {}
truediv_shapes_strorunicode = {}
olddiv_shapes_strorunicode = {}
mod_shapes_strorunicode = {}
divmod_shapes_strorunicode = {}
pow_shapes_strorunicode = {}
bitor_shapes_strorunicode = {}
bitand_shapes_strorunicode = {}
bitxor_shapes_strorunicode = {}
lshift_shapes_strorunicode = {}
rshift_shapes_strorunicode = {}
matmult_shapes_strorunicode = matmult_shapes_none

def _getComparisonLtShapeGeneric(self, right_shape):
    if False:
        while True:
            i = 10
    if type(right_shape) is ShapeLoopCompleteAlternative:
        return right_shape.getComparisonLtLShape(self)
    if type(right_shape) is ShapeLoopInitialAlternative:
        return operation_result_unknown
    onMissingOperation('Lt', self, right_shape)
    return operation_result_unknown

def _getComparisonEqShapeGeneric(self, right_shape):
    if False:
        while True:
            i = 10
    if type(right_shape) is ShapeLoopCompleteAlternative:
        return right_shape.getComparisonEqLShape(self)
    if type(right_shape) is ShapeLoopInitialAlternative:
        return operation_result_unknown
    onMissingOperation('Eq', self, right_shape)
    return operation_result_unknown

class ShapeTypeNoneType(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = None

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'NoneType'

    @staticmethod
    def hasShapeSlotHash():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            while True:
                i = 10
        return True
    add_shapes = add_shapes_none
    sub_shapes = sub_shapes_none
    mult_shapes = mult_shapes_none
    floordiv_shapes = floordiv_shapes_none
    truediv_shapes = truediv_shapes_none
    olddiv_shapes = olddiv_shapes_none
    mod_shapes = mod_shapes_none
    divmod_shapes = divmod_shapes_none
    pow_shapes = pow_shapes_none
    bitor_shapes = bitor_shapes_none
    bitand_shapes = bitand_shapes_none
    bitxor_shapes = bitxor_shapes_none
    lshift_shapes = lshift_shapes_none
    rshift_shapes = rshift_shapes_none
    matmult_shapes = matmult_shapes_none
    if python_version < 768:

        def getComparisonLtShape(self, right_shape):
            if False:
                i = 10
                return i + 15
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape.getTypeName() is not None:
                return operation_result_bool_noescape
            if right_shape in (tshape_int_or_long, tshape_str_or_unicode):
                return operation_result_bool_noescape
            return _getComparisonLtShapeGeneric(self, right_shape)
    else:

        def getComparisonLtShape(self, right_shape):
            if False:
                while True:
                    i = 10
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape.getTypeName() is not None:
                return operation_result_unorderable_comparison
            return _getComparisonLtShapeGeneric(self, right_shape)

        def getComparisonEqShape(self, right_shape):
            if False:
                return 10
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape.getTypeName() is not None:
                return operation_result_bool_noescape
            return _getComparisonEqShapeGeneric(self, right_shape)

        def getComparisonNeqShape(self, right_shape):
            if False:
                print('Hello World!')
            return self.getComparisonEqShape(right_shape)

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            return 10
        return ControlFlowDescriptionNoEscape

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        return hasattr(None, attribute_name)
tshape_none = ShapeTypeNoneType()

class ShapeTypeBool(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = True

    @staticmethod
    def getTypeName():
        if False:
            for i in range(10):
                print('nop')
        return 'bool'

    @staticmethod
    def getCType():
        if False:
            for i in range(10):
                print('nop')
        return CTypeNuitkaBoolEnum
    add_shapes = add_shapes_bool
    sub_shapes = sub_shapes_bool
    mult_shapes = mult_shapes_bool
    floordiv_shapes = floordiv_shapes_bool
    truediv_shapes = truediv_shapes_bool
    olddiv_shapes = olddiv_shapes_bool
    mod_shapes = mod_shapes_bool
    divmod_shapes = divmod_shapes_bool
    pow_shapes = pow_shapes_bool
    bitor_shapes = bitor_shapes_bool
    bitand_shapes = bitand_shapes_bool
    bitxor_shapes = bitxor_shapes_bool
    lshift_shapes = lshift_shapes_bool
    rshift_shapes = rshift_shapes_bool
    matmult_shapes = matmult_shapes_bool

    def getComparisonLtShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape in (tshape_int, tshape_long, tshape_int_or_long, tshape_bool, tshape_float):
            return operation_result_bool_noescape
        if right_shape is tshape_int_or_long_derived:
            return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            i = 10
            return i + 15
        return hasattr(True, attribute_name)
tshape_bool = ShapeTypeBool()

class ShapeTypeInt(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = 7

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'int'
    helper_code = 'INT' if python_version < 768 else 'LONG'
    add_shapes = add_shapes_int
    sub_shapes = sub_shapes_int
    mult_shapes = mult_shapes_int
    floordiv_shapes = floordiv_shapes_int
    truediv_shapes = truediv_shapes_int
    olddiv_shapes = olddiv_shapes_int
    mod_shapes = mod_shapes_int
    divmod_shapes = divmod_shapes_int
    pow_shapes = pow_shapes_int
    bitor_shapes = bitor_shapes_int
    bitand_shapes = bitand_shapes_int
    bitxor_shapes = bitxor_shapes_int
    lshift_shapes = lshift_shapes_int
    rshift_shapes = rshift_shapes_int
    matmult_shapes = matmult_shapes_int

    def getComparisonLtShape(self, right_shape):
        if False:
            print('Hello World!')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape in (tshape_int, tshape_long, tshape_int_or_long, tshape_bool, tshape_float):
            return operation_result_bool_noescape
        if right_shape in (tshape_long_derived, tshape_int_or_long_derived, tshape_float_derived):
            return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            i = 10
            return i + 15
        return hasattr(7, attribute_name)
tshape_int = ShapeTypeInt()
if python_version < 768:
    _the_typical_long_value = long(7)

    class ShapeTypeLong(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
        __slots__ = ()
        typical_value = _the_typical_long_value

        @staticmethod
        def getTypeName():
            if False:
                while True:
                    i = 10
            return 'long'
        helper_code = 'LONG' if python_version < 768 else 'INVALID'
        add_shapes = add_shapes_long
        sub_shapes = sub_shapes_long
        mult_shapes = mult_shapes_long
        floordiv_shapes = floordiv_shapes_long
        truediv_shapes = truediv_shapes_long
        olddiv_shapes = olddiv_shapes_long
        mod_shapes = mod_shapes_long
        divmod_shapes = divmod_shapes_long
        pow_shapes = pow_shapes_long
        bitor_shapes = bitor_shapes_long
        bitand_shapes = bitand_shapes_long
        bitxor_shapes = bitxor_shapes_long
        lshift_shapes = lshift_shapes_long
        rshift_shapes = rshift_shapes_long
        matmult_shapes = matmult_shapes_long

        def getComparisonLtShape(self, right_shape):
            if False:
                for i in range(10):
                    print('nop')
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape in (tshape_int, tshape_long, tshape_int_or_long, tshape_bool, tshape_float):
                return operation_result_bool_noescape
            if right_shape in (tshape_long_derived, tshape_int_or_long_derived):
                return operation_result_unknown
            return _getComparisonLtShapeGeneric(self, right_shape)

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                i = 10
                return i + 15
            return hasattr(_the_typical_long_value, attribute_name)
    tshape_long = ShapeTypeLong()

    class ShapeTypeLongDerived(ShapeTypeUnknown):
        __slots__ = ()

        @staticmethod
        def getTypeName():
            if False:
                while True:
                    i = 10
            return None
    tshape_long_derived = ShapeTypeLongDerived()

    class ShapeTypeIntOrLong(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
        __slots__ = ()
        if isExperimental('nuitka_ilong'):

            @staticmethod
            def getCType():
                if False:
                    i = 10
                    return i + 15
                return CTypeNuitkaIntOrLongStruct

        @staticmethod
        def emitAlternatives(emit):
            if False:
                for i in range(10):
                    print('nop')
            emit(tshape_int)
            emit(tshape_long)
        add_shapes = add_shapes_intorlong
        sub_shapes = sub_shapes_intorlong
        mult_shapes = mult_shapes_intorlong
        floordiv_shapes = floordiv_shapes_intorlong
        truediv_shapes = truediv_shapes_intorlong
        olddiv_shapes = olddiv_shapes_intorlong
        mod_shapes = mod_shapes_intorlong
        divmod_shapes = divmod_shapes_intorlong
        pow_shapes = pow_shapes_intorlong
        bitor_shapes = bitor_shapes_intorlong
        bitand_shapes = bitand_shapes_intorlong
        bitxor_shapes = bitxor_shapes_intorlong
        lshift_shapes = lshift_shapes_intorlong
        rshift_shapes = rshift_shapes_intorlong
        matmult_shapes = matmult_shapes_intorlong

        def getComparisonLtShape(self, right_shape):
            if False:
                return 10
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape in (tshape_int, tshape_long, tshape_int_or_long, tshape_bool, tshape_float):
                return operation_result_bool_noescape
            if right_shape is tshape_int_or_long_derived:
                return operation_result_unknown
            return _getComparisonLtShapeGeneric(self, right_shape)

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                print('Hello World!')
            return hasattr(7, attribute_name) and hasattr(_the_typical_long_value, attribute_name)
    tshape_int_or_long = ShapeTypeIntOrLong()
else:
    tshape_long = None
    tshape_long_derived = None
    tshape_int_or_long = tshape_int

class ShapeTypeIntOrLongDerived(ShapeTypeUnknown):
    __slots__ = ()
tshape_int_or_long_derived = ShapeTypeIntOrLongDerived()

class ShapeTypeFloat(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = 0.1

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'float'
    helper_code = 'FLOAT'
    add_shapes = add_shapes_float
    sub_shapes = sub_shapes_float
    mult_shapes = mult_shapes_float
    floordiv_shapes = floordiv_shapes_float
    truediv_shapes = truediv_shapes_float
    olddiv_shapes = olddiv_shapes_float
    mod_shapes = mod_shapes_float
    divmod_shapes = divmod_shapes_float
    pow_shapes = pow_shapes_float
    bitor_shapes = bitor_shapes_float
    bitand_shapes = bitand_shapes_float
    bitxor_shapes = bitxor_shapes_float
    lshift_shapes = lshift_shapes_float
    rshift_shapes = rshift_shapes_float
    matmult_shapes = matmult_shapes_float

    def getComparisonLtShape(self, right_shape):
        if False:
            print('Hello World!')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape in (tshape_float, tshape_long, tshape_int, tshape_int_or_long, tshape_bool):
            return operation_result_bool_noescape
        if right_shape is tshape_float_derived:
            return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)
tshape_float = ShapeTypeFloat()

class ShapeTypeFloatDerived(ShapeTypeUnknown):
    __slots__ = ()
tshape_float_derived = ShapeTypeFloatDerived()

class ShapeTypeComplex(ShapeNotContainerMixin, ShapeNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = 0j

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'complex'
    add_shapes = add_shapes_complex
    sub_shapes = sub_shapes_complex
    mult_shapes = mult_shapes_complex
    floordiv_shapes = floordiv_shapes_complex
    truediv_shapes = truediv_shapes_complex
    olddiv_shapes = olddiv_shapes_complex
    mod_shapes = mod_shapes_complex
    divmod_shapes = divmod_shapes_complex
    pow_shapes = pow_shapes_complex
    bitor_shapes = bitor_shapes_complex
    bitand_shapes = bitand_shapes_complex
    bitxor_shapes = bitxor_shapes_complex
    lshift_shapes = lshift_shapes_complex
    rshift_shapes = rshift_shapes_complex
    matmult_shapes = matmult_shapes_complex
tshape_complex = ShapeTypeComplex()

class ShapeTypeTuple(ShapeContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = ()

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'tuple'
    helper_code = 'TUPLE'

    @staticmethod
    def getShapeIter():
        if False:
            return 10
        return tshape_tuple_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            print('Hello World!')
        return True
    add_shapes = add_shapes_tuple
    sub_shapes = sub_shapes_tuple
    mult_shapes = mult_shapes_tuple
    floordiv_shapes = floordiv_shapes_tuple
    truediv_shapes = truediv_shapes_tuple
    olddiv_shapes = olddiv_shapes_tuple
    mod_shapes = mod_shapes_tuple
    divmod_shapes = divmod_shapes_tuple
    pow_shapes = pow_shapes_tuple
    bitor_shapes = bitor_shapes_tuple
    bitand_shapes = bitand_shapes_tuple
    bitxor_shapes = bitxor_shapes_tuple
    lshift_shapes = lshift_shapes_tuple
    rshift_shapes = rshift_shapes_tuple
    matmult_shapes = matmult_shapes_tuple

    def getComparisonLtShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return operation_result_unknown
tshape_tuple = ShapeTypeTuple()

class ShapeTypeNamedTuple(ShapeContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'namedtuple'
    helper_code = 'NAMEDTUPLE'

    @staticmethod
    def getShapeIter():
        if False:
            print('Hello World!')
        return tshape_tuple_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            i = 10
            return i + 15
        return True
    add_shapes = add_shapes_tuple
    sub_shapes = sub_shapes_tuple
    mult_shapes = mult_shapes_tuple
    floordiv_shapes = floordiv_shapes_tuple
    truediv_shapes = truediv_shapes_tuple
    olddiv_shapes = olddiv_shapes_tuple
    mod_shapes = mod_shapes_tuple
    divmod_shapes = divmod_shapes_tuple
    pow_shapes = pow_shapes_tuple
    bitor_shapes = bitor_shapes_tuple
    bitand_shapes = bitand_shapes_tuple
    bitxor_shapes = bitxor_shapes_tuple
    lshift_shapes = lshift_shapes_tuple
    rshift_shapes = rshift_shapes_tuple
    matmult_shapes = matmult_shapes_tuple

    def getComparisonLtShape(self, right_shape):
        if False:
            while True:
                i = 10
        return operation_result_unknown
tshape_namedtuple = ShapeTypeNamedTuple()

class ShapeTypeTupleIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_tuple.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'tupleiterator' if python_version < 768 else 'tuple_iterator'

    @staticmethod
    def getIteratedShape():
        if False:
            return 10
        return tshape_tuple
tshape_tuple_iterator = ShapeTypeTupleIterator()

class ShapeTypeList(ShapeContainerMutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = []

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'list'
    helper_code = 'LIST'

    @staticmethod
    def getShapeIter():
        if False:
            print('Hello World!')
        return tshape_list_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return True
    add_shapes = add_shapes_list
    sub_shapes = sub_shapes_list
    mult_shapes = mult_shapes_list
    floordiv_shapes = floordiv_shapes_list
    truediv_shapes = truediv_shapes_list
    olddiv_shapes = olddiv_shapes_list
    mod_shapes = mod_shapes_list
    divmod_shapes = divmod_shapes_list
    pow_shapes = pow_shapes_list
    bitor_shapes = bitor_shapes_list
    bitand_shapes = bitand_shapes_list
    bitxor_shapes = bitxor_shapes_list
    lshift_shapes = lshift_shapes_list
    rshift_shapes = rshift_shapes_list
    matmult_shapes = matmult_shapes_list
    iadd_shapes = iadd_shapes_list

    def getComparisonLtShape(self, right_shape):
        if False:
            print('Hello World!')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape in (tshape_list, tshape_tuple):
            return operation_result_bool_elementbased
        if right_shape is tshape_xrange:
            if python_version < 768:
                return operation_result_bool_elementbased
            else:
                return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)
tshape_list = ShapeTypeList()

class ShapeTypeListIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_list.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'listiterator' if python_version < 768 else 'list_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            print('Hello World!')
        return False
tshape_list_iterator = ShapeTypeListIterator()

class ShapeTypeSet(ShapeContainerMutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = set()

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'set'

    @staticmethod
    def getShapeIter():
        if False:
            for i in range(10):
                print('nop')
        return tshape_set_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return False
    add_shapes = add_shapes_set
    sub_shapes = sub_shapes_set
    mult_shapes = mult_shapes_set
    floordiv_shapes = floordiv_shapes_set
    truediv_shapes = truediv_shapes_set
    olddiv_shapes = olddiv_shapes_set
    mod_shapes = mod_shapes_set
    divmod_shapes = divmod_shapes_set
    pow_shapes = pow_shapes_set
    bitor_shapes = bitor_shapes_set
    bitand_shapes = bitand_shapes_set
    bitxor_shapes = bitxor_shapes_set
    lshift_shapes = lshift_shapes_set
    rshift_shapes = rshift_shapes_set
    matmult_shapes = matmult_shapes_set

    def getComparisonLtShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return operation_result_unknown
tshape_set = ShapeTypeSet()

class ShapeTypeSetIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_set.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'setiterator' if python_version < 768 else 'set_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return False
tshape_set_iterator = ShapeTypeSetIterator()

class ShapeTypeFrozenset(ShapeContainerImmutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = frozenset()

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'frozenset'

    @staticmethod
    def getShapeIter():
        if False:
            while True:
                i = 10
        return tshape_set_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return False
    add_shapes = add_shapes_frozenset
    sub_shapes = sub_shapes_frozenset
    mult_shapes = mult_shapes_frozenset
    floordiv_shapes = floordiv_shapes_frozenset
    truediv_shapes = truediv_shapes_frozenset
    olddiv_shapes = olddiv_shapes_frozenset
    mod_shapes = mod_shapes_frozenset
    divmod_shapes = divmod_shapes_frozenset
    pow_shapes = pow_shapes_frozenset
    bitor_shapes = bitor_shapes_frozenset
    bitand_shapes = bitand_shapes_frozenset
    bitxor_shapes = bitxor_shapes_frozenset
    lshift_shapes = lshift_shapes_frozenset
    rshift_shapes = rshift_shapes_frozenset
    matmult_shapes = matmult_shapes_frozenset
tshape_frozenset = ShapeTypeFrozenset()
_the_empty_dict = {}

class ShapeTypeDict(ShapeContainerMutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = _the_empty_dict

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'dict'

    @staticmethod
    def getShapeIter():
        if False:
            return 10
        return tshape_dict_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return False
    add_shapes = add_shapes_dict
    sub_shapes = sub_shapes_dict
    mult_shapes = mult_shapes_dict
    floordiv_shapes = floordiv_shapes_dict
    truediv_shapes = truediv_shapes_dict
    olddiv_shapes = olddiv_shapes_dict
    mod_shapes = mod_shapes_dict
    divmod_shapes = divmod_shapes_dict
    pow_shapes = pow_shapes_dict
    bitor_shapes = bitor_shapes_dict
    bitand_shapes = bitand_shapes_dict
    bitxor_shapes = bitxor_shapes_dict
    lshift_shapes = lshift_shapes_dict
    rshift_shapes = rshift_shapes_dict
    matmult_shapes = matmult_shapes_dict
    ibitor_shapes = ibitor_shapes_dict

    def getComparisonLtShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return operation_result_unknown

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(_the_empty_dict, attribute_name)
tshape_dict = ShapeTypeDict()

class ShapeTypeDictIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_dict.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'dictionary-keyiterator' if python_version < 768 else 'dictkey_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            for i in range(10):
                print('nop')
        return False
tshape_dict_iterator = ShapeTypeDictIterator()

class ShapeTypeStr(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = 'a'

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'str'
    helper_code = 'STR' if python_version < 768 else 'UNICODE'

    @staticmethod
    def hasShapeSlotIter():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def hasShapeSlotLen():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotContains():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotInt():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotLong():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotFloat():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotComplex():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotHash():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def getShapeIter():
        if False:
            print('Hello World!')
        return tshape_str_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            for i in range(10):
                print('nop')
        return True
    add_shapes = add_shapes_str
    sub_shapes = sub_shapes_str
    mult_shapes = mult_shapes_str
    floordiv_shapes = floordiv_shapes_str
    truediv_shapes = truediv_shapes_str
    olddiv_shapes = olddiv_shapes_str
    mod_shapes = mod_shapes_str
    divmod_shapes = divmod_shapes_str
    pow_shapes = pow_shapes_str
    bitor_shapes = bitor_shapes_str
    bitand_shapes = bitand_shapes_str
    bitxor_shapes = bitxor_shapes_str
    lshift_shapes = lshift_shapes_str
    rshift_shapes = rshift_shapes_str
    matmult_shapes = matmult_shapes_str

    def getComparisonLtShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape is tshape_str:
            return operation_result_bool_noescape
        if right_shape is tshape_str_derived:
            return operation_result_unknown
        if right_shape is tshape_bytearray:
            if python_version < 768:
                return operation_result_bool_noescape
            else:
                return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr('a', attribute_name)
tshape_str = ShapeTypeStr()

class TypeShapeStrDerived(ShapeTypeUnknown):
    __slots__ = ()
tshape_str_derived = TypeShapeStrDerived()

class ShapeTypeStrIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_str.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'iterator' if python_version < 768 else 'str_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            while True:
                i = 10
        return False
tshape_str_iterator = ShapeTypeStrIterator()
if python_version < 768:

    class ShapeTypeUnicode(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
        __slots__ = ()
        typical_value = the_empty_unicode

        @staticmethod
        def getTypeName():
            if False:
                while True:
                    i = 10
            return 'unicode'
        helper_code = 'UNICODE'

        @staticmethod
        def hasShapeSlotIter():
            if False:
                i = 10
                return i + 15
            return True

        @staticmethod
        def hasShapeSlotLen():
            if False:
                return 10
            return True

        @staticmethod
        def hasShapeSlotContains():
            if False:
                return 10
            return True

        @staticmethod
        def hasShapeSlotInt():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotLong():
            if False:
                i = 10
                return i + 15
            return True

        @staticmethod
        def hasShapeSlotFloat():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotComplex():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotHash():
            if False:
                i = 10
                return i + 15
            return True

        @staticmethod
        def getShapeIter():
            if False:
                return 10
            return tshape_unicode_iterator

        @staticmethod
        def hasShapeIndexLookup():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeTrustedAttributes():
            if False:
                while True:
                    i = 10
            return True
        add_shapes = add_shapes_unicode
        sub_shapes = sub_shapes_unicode
        mult_shapes = mult_shapes_unicode
        floordiv_shapes = floordiv_shapes_unicode
        truediv_shapes = truediv_shapes_unicode
        olddiv_shapes = olddiv_shapes_unicode
        mod_shapes = mod_shapes_unicode
        divmod_shapes = divmod_shapes_unicode
        pow_shapes = pow_shapes_unicode
        bitor_shapes = bitor_shapes_unicode
        bitand_shapes = bitand_shapes_unicode
        bitxor_shapes = bitxor_shapes_unicode
        lshift_shapes = lshift_shapes_unicode
        rshift_shapes = rshift_shapes_unicode
        matmult_shapes = matmult_shapes_unicode

        def getComparisonLtShape(self, right_shape):
            if False:
                for i in range(10):
                    print('nop')
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape is tshape_unicode:
                return operation_result_bool_noescape
            if right_shape is tshape_unicode_derived:
                return operation_result_unknown
            return _getComparisonLtShapeGeneric(self, right_shape)

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                print('Hello World!')
            return hasattr(the_empty_unicode, attribute_name)
    tshape_unicode = ShapeTypeUnicode()

    class ShapeTypeUnicodeDerived(ShapeTypeUnknown):
        __slots__ = ()
    tshape_unicode_derived = ShapeTypeUnicodeDerived()

    class ShapeTypeUnicodeIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
        __slots__ = ()
        typical_value = iter(tshape_unicode.typical_value)

        @staticmethod
        def getTypeName():
            if False:
                print('Hello World!')
            return 'iterator'

        @staticmethod
        def hasShapeIndexLookup():
            if False:
                while True:
                    i = 10
            return False
    tshape_unicode_iterator = ShapeTypeUnicodeIterator()
else:
    tshape_unicode = tshape_str
    tshape_unicode_iterator = tshape_str_iterator
    tshape_unicode_derived = tshape_str_derived
if python_version < 768:

    class ShapeTypeStrOrUnicode(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
        __slots__ = ()

        @staticmethod
        def emitAlternatives(emit):
            if False:
                i = 10
                return i + 15
            emit(tshape_str)
            emit(tshape_unicode)

        @staticmethod
        def hasShapeSlotIter():
            if False:
                while True:
                    i = 10
            return True

        @staticmethod
        def hasShapeSlotLen():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotContains():
            if False:
                for i in range(10):
                    print('nop')
            return True

        @staticmethod
        def hasShapeSlotInt():
            if False:
                for i in range(10):
                    print('nop')
            return True

        @staticmethod
        def hasShapeSlotLong():
            if False:
                for i in range(10):
                    print('nop')
            return True

        @staticmethod
        def hasShapeSlotFloat():
            if False:
                for i in range(10):
                    print('nop')
            return True

        @staticmethod
        def hasShapeSlotComplex():
            if False:
                i = 10
                return i + 15
            return True

        @staticmethod
        def hasShapeSlotHash():
            if False:
                return 10
            return True

        @staticmethod
        def hasShapeIndexLookup():
            if False:
                while True:
                    i = 10
            return True

        @staticmethod
        def hasShapeTrustedAttributes():
            if False:
                while True:
                    i = 10
            return True
        add_shapes = add_shapes_strorunicode
        sub_shapes = sub_shapes_strorunicode
        mult_shapes = mult_shapes_strorunicode
        bitor_shapes = bitor_shapes_strorunicode
        bitand_shapes = bitand_shapes_strorunicode
        bitxor_shapes = bitxor_shapes_strorunicode
        lshift_shapes = lshift_shapes_strorunicode
        rshift_shapes = rshift_shapes_strorunicode
        matmult_shapes = matmult_shapes_strorunicode

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                for i in range(10):
                    print('nop')
            return hasattr('a', attribute_name) and hasattr(the_empty_unicode, attribute_name)
    tshape_str_or_unicode = ShapeTypeStrOrUnicode()

    class ShapeTypeStrOrUnicodeDerived(ShapeTypeUnknown):
        __slots__ = ()
    tshape_str_or_unicode_derived = ShapeTypeStrOrUnicodeDerived()
else:
    tshape_str_or_unicode = tshape_str
    tshape_str_or_unicode_derived = tshape_str_derived
if python_version >= 768:

    class ShapeTypeBytes(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
        __slots__ = ()
        typical_value = b'b'

        @staticmethod
        def getTypeName():
            if False:
                print('Hello World!')
            return 'bytes'
        helper_code = 'BYTES'

        @staticmethod
        def hasShapeSlotIter():
            if False:
                i = 10
                return i + 15
            return True

        @staticmethod
        def hasShapeSlotLen():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotContains():
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def hasShapeSlotInt():
            if False:
                return 10
            return True

        @staticmethod
        def hasShapeSlotLong():
            if False:
                return 10
            return True

        @staticmethod
        def hasShapeSlotFloat():
            if False:
                for i in range(10):
                    print('nop')
            return True

        @staticmethod
        def hasShapeSlotHash():
            if False:
                while True:
                    i = 10
            return True

        @staticmethod
        def getShapeIter():
            if False:
                i = 10
                return i + 15
            return tshape_bytes_iterator

        @staticmethod
        def hasShapeIndexLookup():
            if False:
                while True:
                    i = 10
            return True

        @staticmethod
        def hasShapeTrustedAttributes():
            if False:
                i = 10
                return i + 15
            return True
        add_shapes = add_shapes_bytes
        sub_shapes = sub_shapes_bytes
        mult_shapes = mult_shapes_bytes
        floordiv_shapes = floordiv_shapes_bytes
        truediv_shapes = truediv_shapes_bytes
        olddiv_shapes = olddiv_shapes_bytes
        mod_shapes = mod_shapes_bytes
        divmod_shapes = divmod_shapes_bytes
        pow_shapes = pow_shapes_bytes
        bitor_shapes = bitor_shapes_bytes
        bitand_shapes = bitand_shapes_bytes
        bitxor_shapes = bitxor_shapes_bytes
        lshift_shapes = lshift_shapes_bytes
        rshift_shapes = rshift_shapes_bytes
        matmult_shapes = matmult_shapes_bytes

        def getComparisonLtShape(self, right_shape):
            if False:
                print('Hello World!')
            if right_shape is tshape_unknown:
                return operation_result_unknown
            if right_shape is tshape_bytes:
                return operation_result_bool_noescape
            if right_shape is tshape_bytes_derived:
                return operation_result_unknown
            return _getComparisonLtShapeGeneric(self, right_shape)

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                return 10
            return hasattr(b'b', attribute_name)
    tshape_bytes = ShapeTypeBytes()

    class TypeShapeBytesDerived(ShapeTypeUnknown):
        __slots__ = ()
    tshape_bytes_derived = TypeShapeBytesDerived()

    class TypeShapeBytesIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
        __slots__ = ()
        typical_value = iter(tshape_bytes.typical_value)

        @staticmethod
        def getTypeName():
            if False:
                return 10
            return 'bytes_iterator'
    tshape_bytes_iterator = TypeShapeBytesIterator()
else:
    tshape_bytes = None
    tshape_bytes_iterator = None
    tshape_bytes_derived = None
_the_typical_bytearray_value = bytearray(b'b')

class ShapeTypeBytearray(ShapeContainerMutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = _the_typical_bytearray_value

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'bytearray'

    @staticmethod
    def getShapeIter():
        if False:
            print('Hello World!')
        return tshape_bytearray_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return True
    add_shapes = add_shapes_bytearray
    sub_shapes = sub_shapes_bytearray
    mult_shapes = mult_shapes_bytearray
    floordiv_shapes = floordiv_shapes_bytearray
    truediv_shapes = truediv_shapes_bytearray
    olddiv_shapes = olddiv_shapes_bytearray
    mod_shapes = mod_shapes_bytearray
    divmod_shapes = divmod_shapes_bytearray
    pow_shapes = pow_shapes_bytearray
    bitor_shapes = bitor_shapes_bytearray
    bitand_shapes = bitand_shapes_bytearray
    bitxor_shapes = bitxor_shapes_bytearray
    lshift_shapes = lshift_shapes_bytearray
    rshift_shapes = rshift_shapes_bytearray
    matmult_shapes = matmult_shapes_bytearray

    def getComparisonLtShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape in (tshape_bytearray, tshape_bytes):
            return operation_result_bool_noescape
        if right_shape is tshape_str:
            if python_version < 768:
                return operation_result_bool_noescape
            else:
                return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr(_the_typical_bytearray_value, attribute_name)
tshape_bytearray = ShapeTypeBytearray()

class ShapeTypeBytearrayIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_bytearray.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'bytearray_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            i = 10
            return i + 15
        return False
tshape_bytearray_iterator = ShapeTypeBytearrayIterator()

class ShapeTypeEllipsis(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = Ellipsis

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'ellipsis'

    @staticmethod
    def hasShapeSlotHash():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(Ellipsis, attribute_name)
tshape_ellipsis = ShapeTypeEllipsis()
_the_typical_slice_value = slice(7)

class ShapeTypeSlice(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = _the_typical_slice_value

    @staticmethod
    def getTypeName():
        if False:
            i = 10
            return i + 15
        return 'slice'

    @staticmethod
    def hasShapeSlotHash():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        return hasattr(_the_typical_slice_value, attribute_name)
tshape_slice = ShapeTypeSlice()
_the_typical_xrange_value = xrange(1) if python_version < 768 else range(1)

class ShapeTypeXrange(ShapeContainerImmutableMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = _the_typical_xrange_value

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'xrange' if python_version < 768 else 'range'

    @staticmethod
    def getShapeIter():
        if False:
            i = 10
            return i + 15
        return tshape_xrange_iterator

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getComparisonLtShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if python_version < 768:
            if right_shape in (tshape_list, tshape_tuple):
                return operation_result_bool_elementbased
            if right_shape is tshape_xrange:
                return operation_result_bool_elementbased
        else:
            return operation_result_unknown
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(_the_typical_xrange_value, attribute_name)
tshape_xrange = ShapeTypeXrange()

class ShapeTypeXrangeIterator(ShapeIteratorMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = iter(tshape_xrange.typical_value)

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'rangeiterator' if python_version < 768 else 'range_iterator'

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            return 10
        return False
tshape_xrange_iterator = ShapeTypeXrangeIterator()

class ShapeTypeType(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = int

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'type'

    @staticmethod
    def hasShapeSlotHash():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getComparisonLtShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        if right_shape is tshape_type:
            return (tshape_unknown, ControlFlowDescriptionNoEscape)
        return _getComparisonLtShapeGeneric(self, right_shape)

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(int, attribute_name)
tshape_type = ShapeTypeType()

class ShapeTypeModule(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = __import__('sys')

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'module'

    @staticmethod
    def hasShapeModule():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotHash():
        if False:
            i = 10
            return i + 15
        return True
tshape_module = ShapeTypeModule()

class ShapeTypeFunction(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()

    @staticmethod
    def getTypeName():
        if False:
            return 10
        return 'function'

    @staticmethod
    def hasShapeSlotHash():
        if False:
            return 10
        return True
tshape_function = ShapeTypeFunction()

class ShapeTypeBuiltinModule(ShapeTypeModule):
    __slots__ = ()
    typical_value = __import__('_ctypes')
tshape_module_builtin = ShapeTypeBuiltinModule()

class ShapeTypeFile(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = __import__('sys').stdout

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'file'

    @staticmethod
    def hasShapeSlotIter():
        if False:
            return 10
        return True

    @staticmethod
    def hasShapeSlotNext():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotContains():
        if False:
            return 10
        return True

    @staticmethod
    def hasShapeSlotHash():
        if False:
            for i in range(10):
                print('nop')
        return True
tshape_file = ShapeTypeFile()

class ShapeTypeStaticmethod(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()

    @staticmethod
    def getTypeName():
        if False:
            print('Hello World!')
        return 'staticmethod'
tshape_staticmethod = ShapeTypeStaticmethod()

class ShapeTypeClassmethod(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()

    @staticmethod
    def getTypeName():
        if False:
            while True:
                i = 10
        return 'classmethod'
tshape_classmethod = ShapeTypeClassmethod()
operation_result_bool_noescape = (tshape_bool, ControlFlowDescriptionNoEscape)
operation_result_float_noescape = (tshape_float, ControlFlowDescriptionNoEscape)
operation_result_int_noescape = (tshape_int, ControlFlowDescriptionNoEscape)
operation_result_long_noescape = (tshape_long, ControlFlowDescriptionNoEscape)
operation_result_intorlong_noescape = (tshape_int_or_long, ControlFlowDescriptionNoEscape)
operation_result_complex_noescape = (tshape_complex, ControlFlowDescriptionNoEscape)
operation_result_tuple_noescape = (tshape_tuple, ControlFlowDescriptionNoEscape)
operation_result_list_noescape = (tshape_list, ControlFlowDescriptionNoEscape)
operation_result_set_noescape = (tshape_set, ControlFlowDescriptionNoEscape)
operation_result_frozenset_noescape = (tshape_frozenset, ControlFlowDescriptionNoEscape)
operation_result_str_noescape = (tshape_str, ControlFlowDescriptionNoEscape)
operation_result_unicode_noescape = (tshape_unicode, ControlFlowDescriptionNoEscape)
operation_result_strorunicode_noescape = (tshape_str_or_unicode, ControlFlowDescriptionNoEscape)
operation_result_bytes_noescape = (tshape_bytes, ControlFlowDescriptionNoEscape)
operation_result_bytearray_noescape = (tshape_bytearray, ControlFlowDescriptionNoEscape)
operation_result_dict_noescape = (tshape_dict, ControlFlowDescriptionNoEscape)
operation_result_dict_valueerror = (tshape_dict, ControlFlowDescriptionValueErrorNoEscape)
operation_result_bool_elementbased = (tshape_bool, ControlFlowDescriptionElementBasedEscape)
operation_result_unorderable_comparison = (tshape_unknown, ControlFlowDescriptionComparisonUnorderable)
operation_result_unsupported_add = (tshape_unknown, ControlFlowDescriptionAddUnsupported)
operation_result_unsupported_sub = (tshape_unknown, ControlFlowDescriptionSubUnsupported)
operation_result_unsupported_mul = (tshape_unknown, ControlFlowDescriptionMulUnsupported)
operation_result_unsupported_floordiv = (tshape_unknown, ControlFlowDescriptionFloorDivUnsupported)
operation_result_unsupported_truediv = (tshape_unknown, ControlFlowDescriptionTrueDivUnsupported)
operation_result_unsupported_olddiv = (tshape_unknown, ControlFlowDescriptionOldDivUnsupported)
operation_result_unsupported_mod = (tshape_unknown, ControlFlowDescriptionModUnsupported)
operation_result_unsupported_divmod = (tshape_unknown, ControlFlowDescriptionDivmodUnsupported)
operation_result_unsupported_pow = (tshape_unknown, ControlFlowDescriptionPowUnsupported)
operation_result_unsupported_bitor = (tshape_unknown, ControlFlowDescriptionBitorUnsupported)
operation_result_unsupported_bitand = (tshape_unknown, ControlFlowDescriptionBitandUnsupported)
operation_result_unsupported_bitxor = (tshape_unknown, ControlFlowDescriptionBitxorUnsupported)
operation_result_unsupported_lshift = (tshape_unknown, ControlFlowDescriptionLshiftUnsupported)
operation_result_unsupported_rshift = (tshape_unknown, ControlFlowDescriptionRshiftUnsupported)
operation_result_unsupported_matmult = (tshape_unknown, ControlFlowDescriptionMatmultUnsupported)
operation_result_zerodiv_int = (tshape_int, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_zerodiv_long = (tshape_long, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_zerodiv_intorlong = (tshape_int_or_long, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_zerodiv_float = (tshape_float, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_zerodiv_complex = (tshape_complex, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_zerodiv_tuple = (tshape_tuple, ControlFlowDescriptionZeroDivisionNoEscape)
operation_result_valueerror_intorlong = (tshape_int_or_long, ControlFlowDescriptionValueErrorNoEscape)
operation_result_valueerror_long = (tshape_long, ControlFlowDescriptionValueErrorNoEscape)
operation_result_str_formaterror = (tshape_str, ControlFlowDescriptionFormatError)
operation_result_unicode_formaterror = (tshape_unicode, ControlFlowDescriptionFormatError)
operation_result_bytes_formaterror = (tshape_bytes, ControlFlowDescriptionFormatError)
operation_result_bytearray_formaterror = (tshape_bytearray, ControlFlowDescriptionFormatError)

def _rejectEverything(shapes, operation_unsupported):
    if False:
        i = 10
        return i + 15
    shapes.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_unsupported, tshape_long: operation_unsupported, tshape_int_or_long: operation_unsupported, tshape_bool: operation_unsupported, tshape_long: operation_unsupported, tshape_float: operation_unsupported, tshape_complex: operation_unsupported, tshape_str: operation_unsupported, tshape_bytes: operation_unsupported, tshape_bytearray: operation_unsupported, tshape_unicode: operation_unsupported, tshape_tuple: operation_unsupported, tshape_list: operation_unsupported, tshape_set: operation_unsupported, tshape_frozenset: operation_unsupported, tshape_dict: operation_unsupported, tshape_type: operation_unsupported, tshape_none: operation_unsupported})
_rejectEverything(add_shapes_none, operation_result_unsupported_add)
_rejectEverything(sub_shapes_none, operation_result_unsupported_sub)
_rejectEverything(mult_shapes_none, operation_result_unsupported_mul)
_rejectEverything(floordiv_shapes_none, operation_result_unsupported_floordiv)
_rejectEverything(truediv_shapes_none, operation_result_unsupported_truediv)
_rejectEverything(olddiv_shapes_none, operation_result_unsupported_olddiv)
_rejectEverything(mod_shapes_none, operation_result_unsupported_mod)
_rejectEverything(divmod_shapes_none, operation_result_unsupported_divmod)
_rejectEverything(pow_shapes_none, operation_result_unsupported_pow)
_rejectEverything(bitor_shapes_none, operation_result_unsupported_bitor)
_rejectEverything(bitand_shapes_none, operation_result_unsupported_bitand)
_rejectEverything(bitxor_shapes_none, operation_result_unsupported_bitxor)
_rejectEverything(lshift_shapes_none, operation_result_unsupported_lshift)
_rejectEverything(rshift_shapes_none, operation_result_unsupported_rshift)
_rejectEverything(matmult_shapes_none, operation_result_unsupported_rshift)

def cloneWithUnsupportedChange(op_shapes, operation_result_unsupported):
    if False:
        while True:
            i = 10
    r = {}
    for (key, value) in op_shapes.items():
        if value[1].getExceptionExit() is TypeError:
            value = operation_result_unsupported
        r[key] = value
    return r
add_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_intorlong_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_int_noescape, tshape_long: operation_result_long_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_bool.update(cloneWithUnsupportedChange(add_shapes_bool, operation_result_unsupported_sub))
mult_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_int_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_int_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_str_noescape, tshape_bytes: operation_result_bytes_noescape, tshape_bytearray: operation_result_bytearray_noescape, tshape_unicode: operation_result_unicode_noescape, tshape_tuple: operation_result_tuple_noescape, tshape_list: operation_result_list_noescape, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
floordiv_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_int, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_floordiv, tshape_bytes: operation_result_unsupported_floordiv, tshape_bytearray: operation_result_unsupported_floordiv, tshape_unicode: operation_result_unsupported_floordiv, tshape_tuple: operation_result_unsupported_floordiv, tshape_list: operation_result_unsupported_floordiv, tshape_set: operation_result_unsupported_floordiv, tshape_frozenset: operation_result_unsupported_floordiv, tshape_dict: operation_result_unsupported_floordiv, tshape_type: operation_result_unsupported_floordiv, tshape_none: operation_result_unsupported_floordiv})
truediv_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_float, tshape_long: operation_result_zerodiv_float, tshape_int_or_long: operation_result_zerodiv_float, tshape_bool: operation_result_zerodiv_float, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_truediv, tshape_bytes: operation_result_unsupported_truediv, tshape_bytearray: operation_result_unsupported_truediv, tshape_unicode: operation_result_unsupported_truediv, tshape_tuple: operation_result_unsupported_truediv, tshape_list: operation_result_unsupported_truediv, tshape_set: operation_result_unsupported_truediv, tshape_frozenset: operation_result_unsupported_truediv, tshape_dict: operation_result_unsupported_truediv, tshape_type: operation_result_unsupported_truediv, tshape_none: operation_result_unsupported_truediv})
olddiv_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_int, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_olddiv, tshape_bytes: operation_result_unsupported_olddiv, tshape_bytearray: operation_result_unsupported_olddiv, tshape_unicode: operation_result_unsupported_olddiv, tshape_tuple: operation_result_unsupported_olddiv, tshape_list: operation_result_unsupported_olddiv, tshape_set: operation_result_unsupported_olddiv, tshape_dict: operation_result_unsupported_olddiv, tshape_type: operation_result_unsupported_olddiv, tshape_none: operation_result_unsupported_olddiv})
mod_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_int, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex if python_version < 768 else operation_result_unsupported_mod, tshape_str: operation_result_unsupported_mod, tshape_bytes: operation_result_unsupported_mod, tshape_bytearray: operation_result_unsupported_mod, tshape_unicode: operation_result_unsupported_mod, tshape_tuple: operation_result_unsupported_mod, tshape_list: operation_result_unsupported_mod, tshape_set: operation_result_unsupported_mod, tshape_frozenset: operation_result_unsupported_mod, tshape_dict: operation_result_unsupported_mod, tshape_type: operation_result_unsupported_mod, tshape_none: operation_result_unsupported_mod})
divmod_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_tuple, tshape_long: operation_result_zerodiv_tuple, tshape_int_or_long: operation_result_zerodiv_tuple, tshape_bool: operation_result_zerodiv_tuple, tshape_float: operation_result_zerodiv_tuple, tshape_complex: operation_result_zerodiv_tuple, tshape_str: operation_result_unsupported_divmod, tshape_bytes: operation_result_unsupported_divmod, tshape_bytearray: operation_result_unsupported_divmod, tshape_unicode: operation_result_unsupported_divmod, tshape_tuple: operation_result_unsupported_divmod, tshape_list: operation_result_unsupported_divmod, tshape_set: operation_result_unsupported_divmod, tshape_frozenset: operation_result_unsupported_divmod, tshape_dict: operation_result_unsupported_divmod, tshape_type: operation_result_unsupported_divmod, tshape_none: operation_result_unsupported_divmod})
pow_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unknown, tshape_long: operation_result_unknown, tshape_int_or_long: operation_result_unknown, tshape_bool: operation_result_int_noescape, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_pow, tshape_bytes: operation_result_unsupported_pow, tshape_bytearray: operation_result_unsupported_pow, tshape_unicode: operation_result_unsupported_pow, tshape_tuple: operation_result_unsupported_pow, tshape_list: operation_result_unsupported_pow, tshape_set: operation_result_unsupported_pow, tshape_frozenset: operation_result_unsupported_pow, tshape_dict: operation_result_unsupported_pow, tshape_type: operation_result_unsupported_pow, tshape_none: operation_result_unsupported_pow})
bitor_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_int_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_bool_noescape, tshape_float: operation_result_unsupported_bitor, tshape_complex: operation_result_unsupported_bitor, tshape_str: operation_result_unsupported_bitor, tshape_bytes: operation_result_unsupported_bitor, tshape_bytearray: operation_result_unsupported_bitor, tshape_unicode: operation_result_unsupported_bitor, tshape_tuple: operation_result_unsupported_bitor, tshape_list: operation_result_unsupported_bitor, tshape_set: operation_result_unsupported_bitor, tshape_frozenset: operation_result_unsupported_bitor, tshape_dict: operation_result_unsupported_bitor, tshape_type: operation_result_unsupported_bitor, tshape_none: operation_result_unsupported_bitor})
bitand_shapes_bool.update(cloneWithUnsupportedChange(bitor_shapes_bool, operation_result_unsupported_bitand))
bitxor_shapes_bool.update(cloneWithUnsupportedChange(bitor_shapes_bool, operation_result_unsupported_bitand))
lshift_shapes_bool.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_valueerror_intorlong, tshape_long: operation_result_valueerror_long, tshape_int_or_long: operation_result_valueerror_intorlong, tshape_bool: operation_result_valueerror_intorlong, tshape_float: operation_result_unsupported_lshift, tshape_complex: operation_result_unsupported_lshift, tshape_str: operation_result_unsupported_lshift, tshape_bytes: operation_result_unsupported_lshift, tshape_bytearray: operation_result_unsupported_lshift, tshape_unicode: operation_result_unsupported_lshift, tshape_tuple: operation_result_unsupported_lshift, tshape_list: operation_result_unsupported_lshift, tshape_set: operation_result_unsupported_lshift, tshape_frozenset: operation_result_unsupported_lshift, tshape_dict: operation_result_unsupported_lshift, tshape_type: operation_result_unsupported_lshift, tshape_none: operation_result_unsupported_lshift})
rshift_shapes_bool.update(cloneWithUnsupportedChange(lshift_shapes_bool, operation_result_unsupported_rshift))
add_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_intorlong_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_intorlong_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_int.update(cloneWithUnsupportedChange(add_shapes_int, operation_result_unsupported_sub))
mult_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_intorlong_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_int_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_str_noescape, tshape_bytes: operation_result_bytes_noescape, tshape_bytearray: operation_result_bytearray_noescape, tshape_unicode: operation_result_unicode_noescape, tshape_tuple: operation_result_tuple_noescape, tshape_list: operation_result_list_noescape, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
floordiv_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_intorlong, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_floordiv, tshape_bytes: operation_result_unsupported_floordiv, tshape_bytearray: operation_result_unsupported_floordiv, tshape_unicode: operation_result_unsupported_floordiv, tshape_tuple: operation_result_unsupported_floordiv, tshape_list: operation_result_unsupported_floordiv, tshape_set: operation_result_unsupported_floordiv, tshape_frozenset: operation_result_unsupported_floordiv, tshape_dict: operation_result_unsupported_floordiv, tshape_type: operation_result_unsupported_floordiv, tshape_none: operation_result_unsupported_floordiv})
truediv_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_float, tshape_long: operation_result_zerodiv_float, tshape_int_or_long: operation_result_zerodiv_float, tshape_bool: operation_result_zerodiv_float, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_truediv, tshape_bytes: operation_result_unsupported_truediv, tshape_bytearray: operation_result_unsupported_truediv, tshape_unicode: operation_result_unsupported_truediv, tshape_tuple: operation_result_unsupported_truediv, tshape_list: operation_result_unsupported_truediv, tshape_set: operation_result_unsupported_truediv, tshape_frozenset: operation_result_unsupported_truediv, tshape_dict: operation_result_unsupported_truediv, tshape_type: operation_result_unsupported_truediv, tshape_none: operation_result_unsupported_truediv})
olddiv_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_intorlong, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_olddiv, tshape_bytes: operation_result_unsupported_olddiv, tshape_bytearray: operation_result_unsupported_olddiv, tshape_unicode: operation_result_unsupported_olddiv, tshape_tuple: operation_result_unsupported_olddiv, tshape_list: operation_result_unsupported_olddiv, tshape_set: operation_result_unsupported_olddiv, tshape_dict: operation_result_unsupported_olddiv, tshape_type: operation_result_unsupported_olddiv, tshape_none: operation_result_unsupported_olddiv})
mod_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_intorlong, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_intorlong, tshape_bool: operation_result_zerodiv_int, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex if python_version < 768 else operation_result_unsupported_mod, tshape_str: operation_result_unsupported_mod, tshape_bytes: operation_result_unsupported_mod, tshape_bytearray: operation_result_unsupported_mod, tshape_unicode: operation_result_unsupported_mod, tshape_tuple: operation_result_unsupported_mod, tshape_list: operation_result_unsupported_mod, tshape_set: operation_result_unsupported_mod, tshape_frozenset: operation_result_unsupported_mod, tshape_dict: operation_result_unsupported_mod, tshape_type: operation_result_unsupported_mod, tshape_none: operation_result_unsupported_mod})
pow_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unknown, tshape_long: operation_result_unknown, tshape_int_or_long: operation_result_unknown, tshape_bool: operation_result_int_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_pow, tshape_bytes: operation_result_unsupported_pow, tshape_bytearray: operation_result_unsupported_pow, tshape_unicode: operation_result_unsupported_pow, tshape_tuple: operation_result_unsupported_pow, tshape_list: operation_result_unsupported_pow, tshape_set: operation_result_unsupported_pow, tshape_frozenset: operation_result_unsupported_pow, tshape_dict: operation_result_unsupported_pow, tshape_type: operation_result_unsupported_pow, tshape_none: operation_result_unsupported_pow})
bitor_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_int_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_intorlong_noescape, tshape_bool: operation_result_int_noescape, tshape_float: operation_result_unsupported_bitor, tshape_complex: operation_result_unsupported_bitor, tshape_str: operation_result_unsupported_bitor, tshape_bytes: operation_result_unsupported_bitor, tshape_bytearray: operation_result_unsupported_bitor, tshape_unicode: operation_result_unsupported_bitor, tshape_tuple: operation_result_unsupported_bitor, tshape_list: operation_result_unsupported_bitor, tshape_set: operation_result_unsupported_bitor, tshape_frozenset: operation_result_unsupported_bitor, tshape_dict: operation_result_unsupported_bitor, tshape_type: operation_result_unsupported_bitor, tshape_none: operation_result_unsupported_bitor})
bitand_shapes_int.update(cloneWithUnsupportedChange(bitor_shapes_int, operation_result_unsupported_bitand))
bitxor_shapes_int.update(cloneWithUnsupportedChange(bitor_shapes_int, operation_result_unsupported_bitand))
lshift_shapes_int.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_valueerror_intorlong, tshape_long: operation_result_valueerror_long, tshape_int_or_long: operation_result_valueerror_intorlong, tshape_bool: operation_result_valueerror_intorlong, tshape_float: operation_result_unsupported_lshift, tshape_complex: operation_result_unsupported_lshift, tshape_str: operation_result_unsupported_lshift, tshape_bytes: operation_result_unsupported_lshift, tshape_bytearray: operation_result_unsupported_lshift, tshape_unicode: operation_result_unsupported_lshift, tshape_tuple: operation_result_unsupported_lshift, tshape_list: operation_result_unsupported_lshift, tshape_set: operation_result_unsupported_lshift, tshape_frozenset: operation_result_unsupported_lshift, tshape_dict: operation_result_unsupported_lshift, tshape_type: operation_result_unsupported_lshift, tshape_none: operation_result_unsupported_lshift})
rshift_shapes_int.update(cloneWithUnsupportedChange(lshift_shapes_int, operation_result_unsupported_rshift))
add_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_long_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_long_noescape, tshape_bool: operation_result_long_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_long_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_long_noescape, tshape_bool: operation_result_long_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_sub, tshape_bytes: operation_result_unsupported_sub, tshape_bytearray: operation_result_unsupported_sub, tshape_unicode: operation_result_unsupported_sub, tshape_tuple: operation_result_unsupported_sub, tshape_list: operation_result_unsupported_sub, tshape_set: operation_result_unsupported_sub, tshape_frozenset: operation_result_unsupported_sub, tshape_dict: operation_result_unsupported_sub, tshape_type: operation_result_unsupported_sub, tshape_none: operation_result_unsupported_sub})
mult_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_long_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_long_noescape, tshape_bool: operation_result_long_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_str_noescape, tshape_bytes: operation_result_bytes_noescape, tshape_bytearray: operation_result_bytearray_noescape, tshape_unicode: operation_result_unicode_noescape, tshape_tuple: operation_result_tuple_noescape, tshape_list: operation_result_list_noescape, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
floordiv_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_long, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_long, tshape_bool: operation_result_zerodiv_long, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_floordiv, tshape_bytes: operation_result_unsupported_floordiv, tshape_bytearray: operation_result_unsupported_floordiv, tshape_unicode: operation_result_unsupported_floordiv, tshape_tuple: operation_result_unsupported_floordiv, tshape_list: operation_result_unsupported_floordiv, tshape_set: operation_result_unsupported_floordiv, tshape_frozenset: operation_result_unsupported_floordiv, tshape_dict: operation_result_unsupported_floordiv, tshape_type: operation_result_unsupported_floordiv, tshape_none: operation_result_unsupported_floordiv})
olddiv_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_long, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_long, tshape_bool: operation_result_zerodiv_long, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_olddiv, tshape_bytes: operation_result_unsupported_olddiv, tshape_bytearray: operation_result_unsupported_olddiv, tshape_unicode: operation_result_unsupported_olddiv, tshape_tuple: operation_result_unsupported_olddiv, tshape_list: operation_result_unsupported_olddiv, tshape_set: operation_result_unsupported_olddiv, tshape_dict: operation_result_unsupported_olddiv, tshape_type: operation_result_unsupported_olddiv, tshape_none: operation_result_unsupported_olddiv})
mod_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_long, tshape_long: operation_result_zerodiv_long, tshape_int_or_long: operation_result_zerodiv_long, tshape_bool: operation_result_zerodiv_long, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex if python_version < 768 else operation_result_unsupported_mod, tshape_str: operation_result_unsupported_mod, tshape_bytes: operation_result_unsupported_mod, tshape_bytearray: operation_result_unsupported_mod, tshape_unicode: operation_result_unsupported_mod, tshape_tuple: operation_result_unsupported_mod, tshape_list: operation_result_unsupported_mod, tshape_set: operation_result_unsupported_mod, tshape_frozenset: operation_result_unsupported_mod, tshape_dict: operation_result_unsupported_mod, tshape_type: operation_result_unsupported_mod, tshape_none: operation_result_unsupported_mod})
pow_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unknown, tshape_long: operation_result_unknown, tshape_int_or_long: operation_result_unknown, tshape_bool: operation_result_long_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_pow, tshape_bytes: operation_result_unsupported_pow, tshape_bytearray: operation_result_unsupported_pow, tshape_unicode: operation_result_unsupported_pow, tshape_tuple: operation_result_unsupported_pow, tshape_list: operation_result_unsupported_pow, tshape_set: operation_result_unsupported_pow, tshape_frozenset: operation_result_unsupported_pow, tshape_dict: operation_result_unsupported_pow, tshape_type: operation_result_unsupported_pow, tshape_none: operation_result_unsupported_pow})
bitor_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_long_noescape, tshape_long: operation_result_long_noescape, tshape_int_or_long: operation_result_long_noescape, tshape_bool: operation_result_long_noescape, tshape_float: operation_result_unsupported_bitor, tshape_complex: operation_result_unsupported_bitor, tshape_str: operation_result_unsupported_bitor, tshape_bytes: operation_result_unsupported_bitor, tshape_bytearray: operation_result_unsupported_bitor, tshape_unicode: operation_result_unsupported_bitor, tshape_tuple: operation_result_unsupported_bitor, tshape_list: operation_result_unsupported_bitor, tshape_set: operation_result_unsupported_bitor, tshape_frozenset: operation_result_unsupported_bitor, tshape_dict: operation_result_unsupported_bitor, tshape_type: operation_result_unsupported_bitor, tshape_none: operation_result_unsupported_bitor})
bitand_shapes_long.update(cloneWithUnsupportedChange(bitor_shapes_long, operation_result_unsupported_bitand))
bitxor_shapes_long.update(cloneWithUnsupportedChange(bitor_shapes_long, operation_result_unsupported_bitand))
lshift_shapes_long.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_valueerror_long, tshape_long: operation_result_valueerror_long, tshape_int_or_long: operation_result_valueerror_long, tshape_bool: operation_result_valueerror_long, tshape_float: operation_result_unsupported_lshift, tshape_complex: operation_result_unsupported_lshift, tshape_str: operation_result_unsupported_lshift, tshape_bytes: operation_result_unsupported_lshift, tshape_bytearray: operation_result_unsupported_lshift, tshape_unicode: operation_result_unsupported_lshift, tshape_tuple: operation_result_unsupported_lshift, tshape_list: operation_result_unsupported_lshift, tshape_set: operation_result_unsupported_lshift, tshape_frozenset: operation_result_unsupported_lshift, tshape_dict: operation_result_unsupported_lshift, tshape_type: operation_result_unsupported_lshift, tshape_none: operation_result_unsupported_lshift})
rshift_shapes_long.update(cloneWithUnsupportedChange(lshift_shapes_long, operation_result_unsupported_rshift))

def mergeIntOrLong(op_shapes_int, op_shapes_long):
    if False:
        for i in range(10):
            print('nop')
    r = {}
    for (key, value) in op_shapes_int.items():
        value2 = op_shapes_long[key]
        if value is value2:
            r[key] = value
        elif value[0] is tshape_int_or_long and value2[0] is tshape_long:
            assert value[1] is value2[1]
            r[key] = value
        elif value[0] is tshape_int and value2[0] is tshape_long:
            assert value[1] is value2[1]
            if value[1] is operation_result_intorlong_noescape[1]:
                r[key] = operation_result_intorlong_noescape
            elif value[1] is operation_result_zerodiv_intorlong[1]:
                r[key] = operation_result_zerodiv_intorlong
            else:
                assert False
        else:
            assert False, (key, '->', value, '!=', value2)
    return r
add_shapes_intorlong.update(mergeIntOrLong(add_shapes_int, add_shapes_long))
sub_shapes_intorlong.update(mergeIntOrLong(sub_shapes_int, sub_shapes_long))
mult_shapes_intorlong.update(mergeIntOrLong(mult_shapes_int, mult_shapes_long))
floordiv_shapes_intorlong.update(mergeIntOrLong(floordiv_shapes_int, floordiv_shapes_long))
truediv_shapes_intorlong.update(mergeIntOrLong(truediv_shapes_int, truediv_shapes_long))
olddiv_shapes_intorlong.update(mergeIntOrLong(olddiv_shapes_int, olddiv_shapes_long))
mod_shapes_intorlong.update(mergeIntOrLong(mod_shapes_int, mod_shapes_long))
divmod_shapes_intorlong.update(mergeIntOrLong(divmod_shapes_int, divmod_shapes_long))
pow_shapes_intorlong.update(mergeIntOrLong(pow_shapes_int, pow_shapes_long))
lshift_shapes_intorlong.update(mergeIntOrLong(lshift_shapes_int, lshift_shapes_long))
rshift_shapes_intorlong.update(mergeIntOrLong(rshift_shapes_int, rshift_shapes_long))
bitor_shapes_intorlong.update(mergeIntOrLong(bitor_shapes_int, bitor_shapes_long))
bitand_shapes_intorlong.update(mergeIntOrLong(bitand_shapes_int, bitand_shapes_long))
bitxor_shapes_intorlong.update(mergeIntOrLong(bitxor_shapes_int, bitxor_shapes_long))
add_shapes_float.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_float_noescape, tshape_long: operation_result_float_noescape, tshape_int_or_long: operation_result_float_noescape, tshape_bool: operation_result_float_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_float.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_float_noescape, tshape_long: operation_result_float_noescape, tshape_int_or_long: operation_result_float_noescape, tshape_bool: operation_result_float_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_sub, tshape_bytes: operation_result_unsupported_sub, tshape_bytearray: operation_result_unsupported_sub, tshape_unicode: operation_result_unsupported_sub, tshape_tuple: operation_result_unsupported_sub, tshape_list: operation_result_unsupported_sub, tshape_set: operation_result_unsupported_sub, tshape_frozenset: operation_result_unsupported_sub, tshape_dict: operation_result_unsupported_sub, tshape_type: operation_result_unsupported_sub, tshape_none: operation_result_unsupported_sub})
mult_shapes_float.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_float_noescape, tshape_long: operation_result_float_noescape, tshape_int_or_long: operation_result_float_noescape, tshape_bool: operation_result_float_noescape, tshape_float: operation_result_float_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
floordiv_shapes_float.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_float, tshape_long: operation_result_zerodiv_float, tshape_int_or_long: operation_result_zerodiv_float, tshape_bool: operation_result_zerodiv_float, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_floordiv, tshape_bytes: operation_result_unsupported_floordiv, tshape_bytearray: operation_result_unsupported_floordiv, tshape_unicode: operation_result_unsupported_floordiv, tshape_tuple: operation_result_unsupported_floordiv, tshape_list: operation_result_unsupported_floordiv, tshape_set: operation_result_unsupported_floordiv, tshape_frozenset: operation_result_unsupported_floordiv, tshape_dict: operation_result_unsupported_floordiv, tshape_type: operation_result_unsupported_floordiv, tshape_none: operation_result_unsupported_floordiv})
truediv_shapes_float.update(cloneWithUnsupportedChange(floordiv_shapes_float, operation_result_unsupported_truediv))
olddiv_shapes_float.update(cloneWithUnsupportedChange(floordiv_shapes_float, operation_result_unsupported_olddiv))
mod_shapes_float.update(cloneWithUnsupportedChange(floordiv_shapes_float, operation_result_unsupported_mod))
pow_shapes_float.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_float, tshape_long: operation_result_zerodiv_float, tshape_int_or_long: operation_result_zerodiv_float, tshape_bool: operation_result_float_noescape, tshape_float: operation_result_zerodiv_float, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_pow, tshape_bytes: operation_result_unsupported_pow, tshape_bytearray: operation_result_unsupported_pow, tshape_unicode: operation_result_unsupported_pow, tshape_tuple: operation_result_unsupported_pow, tshape_list: operation_result_unsupported_pow, tshape_set: operation_result_unsupported_pow, tshape_frozenset: operation_result_unsupported_pow, tshape_dict: operation_result_unsupported_pow, tshape_type: operation_result_unsupported_pow, tshape_none: operation_result_unsupported_pow})
add_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_complex_noescape, tshape_long: operation_result_complex_noescape, tshape_int_or_long: operation_result_complex_noescape, tshape_bool: operation_result_complex_noescape, tshape_float: operation_result_complex_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_complex_noescape, tshape_long: operation_result_complex_noescape, tshape_int_or_long: operation_result_complex_noescape, tshape_bool: operation_result_complex_noescape, tshape_float: operation_result_complex_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_sub, tshape_bytes: operation_result_unsupported_sub, tshape_bytearray: operation_result_unsupported_sub, tshape_unicode: operation_result_unsupported_sub, tshape_tuple: operation_result_unsupported_sub, tshape_list: operation_result_unsupported_sub, tshape_set: operation_result_unsupported_sub, tshape_frozenset: operation_result_unsupported_sub, tshape_dict: operation_result_unsupported_sub, tshape_type: operation_result_unsupported_sub, tshape_none: operation_result_unsupported_sub})
mult_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_complex_noescape, tshape_long: operation_result_complex_noescape, tshape_int_or_long: operation_result_complex_noescape, tshape_bool: operation_result_complex_noescape, tshape_float: operation_result_complex_noescape, tshape_complex: operation_result_complex_noescape, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
truediv_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_complex, tshape_long: operation_result_zerodiv_complex, tshape_int_or_long: operation_result_zerodiv_complex, tshape_bool: operation_result_zerodiv_complex, tshape_float: operation_result_zerodiv_complex, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_truediv, tshape_bytes: operation_result_unsupported_truediv, tshape_bytearray: operation_result_unsupported_truediv, tshape_unicode: operation_result_unsupported_truediv, tshape_tuple: operation_result_unsupported_truediv, tshape_list: operation_result_unsupported_truediv, tshape_set: operation_result_unsupported_truediv, tshape_frozenset: operation_result_unsupported_truediv, tshape_dict: operation_result_unsupported_truediv, tshape_type: operation_result_unsupported_truediv, tshape_none: operation_result_unsupported_truediv})
if python_version < 768:
    floordiv_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_complex, tshape_long: operation_result_zerodiv_complex, tshape_int_or_long: operation_result_zerodiv_complex, tshape_bool: operation_result_zerodiv_complex, tshape_float: operation_result_zerodiv_complex, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_floordiv, tshape_bytes: operation_result_unsupported_floordiv, tshape_bytearray: operation_result_unsupported_floordiv, tshape_unicode: operation_result_unsupported_floordiv, tshape_tuple: operation_result_unsupported_floordiv, tshape_list: operation_result_unsupported_floordiv, tshape_set: operation_result_unsupported_floordiv, tshape_frozenset: operation_result_unsupported_floordiv, tshape_dict: operation_result_unsupported_floordiv, tshape_type: operation_result_unsupported_floordiv, tshape_none: operation_result_unsupported_floordiv})
olddiv_shapes_complex.update(cloneWithUnsupportedChange(truediv_shapes_complex, operation_result_unsupported_olddiv))
mod_shapes_complex.update(cloneWithUnsupportedChange(truediv_shapes_complex, operation_result_unsupported_mod))
pow_shapes_complex.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_zerodiv_complex, tshape_long: operation_result_zerodiv_complex, tshape_int_or_long: operation_result_zerodiv_complex, tshape_bool: operation_result_complex_noescape, tshape_float: operation_result_zerodiv_complex, tshape_complex: operation_result_zerodiv_complex, tshape_str: operation_result_unsupported_pow, tshape_bytes: operation_result_unsupported_pow, tshape_bytearray: operation_result_unsupported_pow, tshape_unicode: operation_result_unsupported_pow, tshape_tuple: operation_result_unsupported_pow, tshape_list: operation_result_unsupported_pow, tshape_set: operation_result_unsupported_pow, tshape_frozenset: operation_result_unsupported_pow, tshape_dict: operation_result_unsupported_pow, tshape_type: operation_result_unsupported_pow, tshape_none: operation_result_unsupported_pow})
add_shapes_tuple.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_complex: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_tuple_noescape, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
mult_shapes_tuple.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_tuple_noescape, tshape_long: operation_result_tuple_noescape, tshape_int_or_long: operation_result_tuple_noescape, tshape_bool: operation_result_tuple_noescape, tshape_float: operation_result_unsupported_mul, tshape_complex: operation_result_unsupported_mul, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
add_shapes_list.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_complex: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_list_noescape, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
iadd_shapes_list.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_complex: operation_result_unsupported_add, tshape_str: operation_result_list_noescape, tshape_bytes: operation_result_list_noescape, tshape_bytearray: operation_result_list_noescape, tshape_unicode: operation_result_list_noescape, tshape_tuple: operation_result_list_noescape, tshape_list: operation_result_list_noescape, tshape_set: operation_result_list_noescape, tshape_frozenset: operation_result_list_noescape, tshape_dict: operation_result_list_noescape, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
nothing_multiplicants = (tshape_none, tshape_set, tshape_dict, tshape_type, tshape_list_iterator, tshape_dict_iterator, tshape_set_iterator, tshape_tuple_iterator)

def updateNonMultiplicants(op_shapes):
    if False:
        while True:
            i = 10
    for shape in nothing_multiplicants:
        op_shapes[shape] = operation_result_unsupported_mul
sequence_non_multiplicants = (tshape_float, tshape_str, tshape_bytes, tshape_bytearray, tshape_unicode, tshape_tuple, tshape_list, tshape_set, tshape_frozenset, tshape_dict)

def updateSequenceNonMultiplicants(op_shapes):
    if False:
        i = 10
        return i + 15
    updateNonMultiplicants(op_shapes)
    for shape in sequence_non_multiplicants:
        op_shapes[shape] = operation_result_unsupported_mul
mult_shapes_list.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_list_noescape, tshape_long: operation_result_list_noescape, tshape_int_or_long: operation_result_list_noescape, tshape_bool: operation_result_list_noescape})
updateSequenceNonMultiplicants(mult_shapes_list)
add_shapes_set.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_set.update(sub_shapes_none)
sub_shapes_set[tshape_set] = operation_result_set_noescape
sub_shapes_set[tshape_frozenset] = operation_result_set_noescape
bitor_shapes_set.update(bitor_shapes_none)
bitor_shapes_set[tshape_set] = operation_result_set_noescape
bitor_shapes_set[tshape_frozenset] = operation_result_set_noescape
bitand_shapes_set.update(cloneWithUnsupportedChange(bitor_shapes_set, operation_result_unsupported_bitand))
bitxor_shapes_set.update(cloneWithUnsupportedChange(bitor_shapes_set, operation_result_unsupported_bitxor))
add_shapes_frozenset.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
sub_shapes_frozenset.update(sub_shapes_set)
sub_shapes_frozenset[tshape_set] = operation_result_frozenset_noescape
sub_shapes_frozenset[tshape_frozenset] = operation_result_frozenset_noescape
bitor_shapes_frozenset.update(bitor_shapes_none)
bitor_shapes_frozenset[tshape_set] = operation_result_frozenset_noescape
bitor_shapes_frozenset[tshape_frozenset] = operation_result_frozenset_noescape
bitand_shapes_frozenset.update(cloneWithUnsupportedChange(bitor_shapes_frozenset, operation_result_unsupported_bitand))
bitxor_shapes_frozenset.update(cloneWithUnsupportedChange(bitor_shapes_frozenset, operation_result_unsupported_bitxor))
add_shapes_dict.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
add_shapes_str.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_str_noescape, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_bytearray_noescape if python_version < 768 else operation_result_unsupported_add, tshape_unicode: operation_result_unicode_noescape, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
mult_shapes_str.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_str_noescape, tshape_long: operation_result_str_noescape, tshape_int_or_long: operation_result_str_noescape, tshape_bool: operation_result_str_noescape, tshape_float: operation_result_unsupported_mul, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
mod_shapes_str.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_str_formaterror, tshape_long: operation_result_str_formaterror, tshape_int_or_long: operation_result_str_formaterror, tshape_bool: operation_result_str_formaterror, tshape_float: operation_result_str_formaterror, tshape_str: operation_result_str_formaterror, tshape_bytes: operation_result_str_formaterror, tshape_bytearray: operation_result_str_formaterror, tshape_unicode: operation_result_str_formaterror, tshape_tuple: operation_result_str_formaterror, tshape_list: operation_result_str_formaterror, tshape_set: operation_result_str_formaterror, tshape_frozenset: operation_result_str_formaterror, tshape_dict: operation_result_str_formaterror, tshape_type: operation_result_str_formaterror, tshape_none: operation_result_str_formaterror})
add_shapes_bytes.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_unsupported_add, tshape_bytes: operation_result_bytes_noescape, tshape_bytearray: operation_result_bytearray_noescape, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
mult_shapes_bytes.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_bytes_noescape, tshape_long: operation_result_bytes_noescape, tshape_int_or_long: operation_result_bytes_noescape, tshape_bool: operation_result_bytes_noescape, tshape_float: operation_result_unsupported_mul, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
if python_version < 848:
    operation_result_350_bytes_mod_noescape = operation_result_unsupported_mod
else:
    operation_result_350_bytes_mod_noescape = operation_result_bytes_formaterror
mod_shapes_bytes.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_350_bytes_mod_noescape, tshape_bool: operation_result_350_bytes_mod_noescape, tshape_float: operation_result_350_bytes_mod_noescape, tshape_bytes: operation_result_350_bytes_mod_noescape, tshape_bytearray: operation_result_350_bytes_mod_noescape, tshape_unicode: operation_result_350_bytes_mod_noescape, tshape_tuple: operation_result_350_bytes_mod_noescape, tshape_list: operation_result_350_bytes_mod_noescape, tshape_set: operation_result_350_bytes_mod_noescape, tshape_frozenset: operation_result_350_bytes_mod_noescape, tshape_dict: operation_result_350_bytes_mod_noescape, tshape_type: operation_result_350_bytes_mod_noescape, tshape_none: operation_result_350_bytes_mod_noescape})
add_shapes_bytearray.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_bytearray_noescape if python_version < 768 else operation_result_unsupported_add, tshape_bytes: operation_result_bytearray_noescape, tshape_bytearray: operation_result_bytearray_noescape, tshape_unicode: operation_result_unsupported_add, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
mult_shapes_bytearray.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_bytearray_noescape, tshape_long: operation_result_bytearray_noescape, tshape_int_or_long: operation_result_bytearray_noescape, tshape_bool: operation_result_bytearray_noescape, tshape_float: operation_result_unsupported_mul, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
if python_version < 848:
    operation_result_350_bytearray_mod_noescape = operation_result_unsupported_mod
else:
    operation_result_350_bytearray_mod_noescape = operation_result_bytearray_formaterror
mod_shapes_bytearray.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_350_bytearray_mod_noescape, tshape_bool: operation_result_350_bytearray_mod_noescape, tshape_float: operation_result_350_bytearray_mod_noescape, tshape_bytes: operation_result_350_bytearray_mod_noescape, tshape_bytearray: operation_result_350_bytearray_mod_noescape, tshape_unicode: operation_result_350_bytearray_mod_noescape, tshape_tuple: operation_result_350_bytearray_mod_noescape, tshape_list: operation_result_350_bytearray_mod_noescape, tshape_set: operation_result_350_bytearray_mod_noescape, tshape_frozenset: operation_result_350_bytes_mod_noescape, tshape_dict: operation_result_350_bytearray_mod_noescape, tshape_type: operation_result_350_bytearray_mod_noescape, tshape_none: operation_result_350_bytearray_mod_noescape})
add_shapes_unicode.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unsupported_add, tshape_long: operation_result_unsupported_add, tshape_int_or_long: operation_result_unsupported_add, tshape_bool: operation_result_unsupported_add, tshape_float: operation_result_unsupported_add, tshape_str: operation_result_unicode_noescape, tshape_bytes: operation_result_unsupported_add, tshape_bytearray: operation_result_unsupported_add, tshape_unicode: operation_result_unicode_noescape, tshape_tuple: operation_result_unsupported_add, tshape_list: operation_result_unsupported_add, tshape_set: operation_result_unsupported_add, tshape_frozenset: operation_result_unsupported_add, tshape_dict: operation_result_unsupported_add, tshape_type: operation_result_unsupported_add, tshape_none: operation_result_unsupported_add})
mult_shapes_unicode.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unicode_noescape, tshape_long: operation_result_unicode_noescape, tshape_int_or_long: operation_result_unicode_noescape, tshape_bool: operation_result_unicode_noescape, tshape_float: operation_result_unsupported_mul, tshape_str: operation_result_unsupported_mul, tshape_bytes: operation_result_unsupported_mul, tshape_bytearray: operation_result_unsupported_mul, tshape_unicode: operation_result_unsupported_mul, tshape_tuple: operation_result_unsupported_mul, tshape_list: operation_result_unsupported_mul, tshape_set: operation_result_unsupported_mul, tshape_frozenset: operation_result_unsupported_mul, tshape_dict: operation_result_unsupported_mul, tshape_type: operation_result_unsupported_mul, tshape_none: operation_result_unsupported_mul})
mod_shapes_unicode.update({tshape_unknown: operation_result_unknown, tshape_long_derived: operation_result_unknown, tshape_int_or_long_derived: operation_result_unknown, tshape_float_derived: operation_result_unknown, tshape_str_derived: operation_result_unknown, tshape_unicode_derived: operation_result_unknown, tshape_bytes_derived: operation_result_unknown, tshape_int: operation_result_unicode_formaterror, tshape_long: operation_result_unicode_formaterror, tshape_int_or_long: operation_result_unicode_formaterror, tshape_bool: operation_result_unicode_formaterror, tshape_float: operation_result_unicode_formaterror, tshape_str: operation_result_unicode_formaterror, tshape_bytes: operation_result_unicode_formaterror, tshape_bytearray: operation_result_unicode_formaterror, tshape_unicode: operation_result_unicode_formaterror, tshape_tuple: operation_result_unicode_formaterror, tshape_list: operation_result_unicode_formaterror, tshape_set: operation_result_unicode_formaterror, tshape_frozenset: operation_result_unicode_formaterror, tshape_dict: operation_result_unicode_formaterror, tshape_type: operation_result_unicode_formaterror, tshape_none: operation_result_unicode_formaterror})

def mergeStrOrUnicode(op_shapes_str, op_shapes_unicode):
    if False:
        for i in range(10):
            print('nop')
    r = {}
    for (key, value) in op_shapes_str.items():
        value2 = op_shapes_unicode[key]
        if value is value2:
            r[key] = value
        elif value[0] is tshape_str_or_unicode and value2[0] is tshape_unicode:
            assert value[1] is value2[1]
            r[key] = value
        elif value[0] is tshape_str and value2[0] is tshape_unicode:
            assert value[1] is value2[1] in (operation_result_strorunicode_noescape[1], ControlFlowDescriptionFormatError), (value, value2)
            r[key] = operation_result_strorunicode_noescape
        elif key == tshape_bytearray:
            r[key] = operation_result_unknown
        else:
            assert False, (key, '->', value, '!=', value2)
    return r
add_shapes_strorunicode.update(mergeStrOrUnicode(add_shapes_str, add_shapes_unicode))
sub_shapes_strorunicode.update(mergeStrOrUnicode(sub_shapes_str, sub_shapes_unicode))
mult_shapes_strorunicode.update(mergeStrOrUnicode(mult_shapes_str, mult_shapes_unicode))
floordiv_shapes_strorunicode.update(mergeStrOrUnicode(floordiv_shapes_str, floordiv_shapes_unicode))
truediv_shapes_strorunicode.update(mergeStrOrUnicode(truediv_shapes_str, truediv_shapes_unicode))
olddiv_shapes_strorunicode.update(mergeStrOrUnicode(olddiv_shapes_str, olddiv_shapes_unicode))
mod_shapes_strorunicode.update(mergeStrOrUnicode(mod_shapes_str, mod_shapes_unicode))
divmod_shapes_strorunicode.update(mergeStrOrUnicode(divmod_shapes_str, divmod_shapes_unicode))
pow_shapes_strorunicode.update(mergeStrOrUnicode(pow_shapes_str, pow_shapes_unicode))
lshift_shapes_strorunicode.update(mergeStrOrUnicode(lshift_shapes_str, lshift_shapes_unicode))
rshift_shapes_strorunicode.update(mergeStrOrUnicode(rshift_shapes_str, rshift_shapes_unicode))
bitor_shapes_strorunicode.update(mergeStrOrUnicode(bitor_shapes_str, bitor_shapes_unicode))
bitand_shapes_strorunicode.update(mergeStrOrUnicode(bitand_shapes_str, bitand_shapes_unicode))
bitxor_shapes_strorunicode.update(mergeStrOrUnicode(bitxor_shapes_str, bitxor_shapes_unicode))
if python_version >= 912:
    bitor_shapes_dict[tshape_dict] = operation_result_dict_noescape
    ibitor_shapes_dict[tshape_dict] = operation_result_dict_noescape
    ibitor_shapes_dict[tshape_tuple] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_list] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_set] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_frozenset] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_str] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_bytes] = operation_result_dict_valueerror
    ibitor_shapes_dict[tshape_bytearray] = operation_result_dict_valueerror

class ShapeTypeBuiltinExceptionClass(ShapeNotContainerMixin, ShapeNotNumberMixin, ShapeBase):
    __slots__ = ()
    typical_value = None
tshape_exception_class = ShapeTypeBuiltinExceptionClass()