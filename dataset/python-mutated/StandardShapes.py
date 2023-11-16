""" Standard shapes that commonly appear. """
from abc import abstractmethod
from nuitka.code_generation.c_types.CTypePyObjectPointers import CTypePyObjectPtr
from nuitka.code_generation.Reports import onMissingOperation
from nuitka.utils.SlotMetaClasses import getMetaClassBase
from .ControlFlowDescriptions import ControlFlowDescriptionFullEscape
from .ShapeMixins import ShapeIteratorMixin

class ShapeBase(getMetaClassBase('Shape', require_slots=True)):
    __slots__ = ()

    def __repr__(self):
        if False:
            return 10
        return '<%s %s %s>' % (self.__class__.__name__, self.getTypeName(), self.helper_code)

    @staticmethod
    def getTypeName():
        if False:
            for i in range(10):
                print('nop')
        return None
    helper_code = 'OBJECT'

    @staticmethod
    def getCType():
        if False:
            print('Hello World!')
        return CTypePyObjectPtr

    @staticmethod
    def getShapeIter():
        if False:
            for i in range(10):
                print('nop')
        return tshape_unknown

    @staticmethod
    def hasShapeIndexLookup():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeModule():
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def hasShapeSlotBytes():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeSlotComplex():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeSlotBool():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeSlotAbs():
        if False:
            return 10
        return None

    @staticmethod
    def hasShapeSlotLen():
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def hasShapeSlotInt():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeSlotLong():
        if False:
            return 10
        return None

    @staticmethod
    def hasShapeSlotFloat():
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def hasShapeSlotIter():
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def hasShapeSlotNext():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeSlotNextCode():
        if False:
            return 10
        return None

    @staticmethod
    def hasShapeSlotContains():
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def hasShapeSlotHash():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def isShapeIterator():
        if False:
            while True:
                i = 10
        return None
    add_shapes = {}

    def getOperationBinaryAddShape(self, right_shape):
        if False:
            print('Hello World!')
        result = self.add_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryAddLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Add', self, right_shape)
            return operation_result_unknown
    iadd_shapes = {}

    def getOperationInplaceAddShape(self, right_shape):
        if False:
            while True:
                i = 10
        'Inplace add operation shape, for overload.'
        if self.iadd_shapes:
            result = self.iadd_shapes.get(right_shape)
            if result is not None:
                return result
            else:
                right_shape_type = type(right_shape)
                if right_shape_type is ShapeLoopCompleteAlternative:
                    return right_shape.getOperationBinaryAddLShape(self)
                if right_shape_type is ShapeLoopInitialAlternative:
                    return operation_result_unknown
                onMissingOperation('IAdd', self, right_shape)
                return operation_result_unknown
        else:
            return self.getOperationBinaryAddShape(right_shape)
    sub_shapes = {}

    def getOperationBinarySubShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.sub_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinarySubLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Sub', self, right_shape)
            return operation_result_unknown
    mult_shapes = {}

    def getOperationBinaryMultShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.mult_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryMultLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Mult', self, right_shape)
            return operation_result_unknown
    floordiv_shapes = {}

    def getOperationBinaryFloorDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.floordiv_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryFloorDivLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('FloorDiv', self, right_shape)
            return operation_result_unknown
    olddiv_shapes = {}

    def getOperationBinaryOldDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.olddiv_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryOldDivLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('OldDiv', self, right_shape)
            return operation_result_unknown
    truediv_shapes = {}

    def getOperationBinaryTrueDivShape(self, right_shape):
        if False:
            return 10
        result = self.truediv_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryTrueDivLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('TrueDiv', self, right_shape)
            return operation_result_unknown
    mod_shapes = {}

    def getOperationBinaryModShape(self, right_shape):
        if False:
            print('Hello World!')
        result = self.mod_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryModLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Mod', self, right_shape)
            return operation_result_unknown
    divmod_shapes = {}

    def getOperationBinaryDivmodShape(self, right_shape):
        if False:
            return 10
        result = self.divmod_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryDivmodLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Divmod', self, right_shape)
            return operation_result_unknown
    pow_shapes = {}

    def getOperationBinaryPowShape(self, right_shape):
        if False:
            print('Hello World!')
        result = self.pow_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryPowLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('Pow', self, right_shape)
            return operation_result_unknown
    lshift_shapes = {}

    def getOperationBinaryLShiftShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.lshift_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryLShiftLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('LShift', self, right_shape)
            return operation_result_unknown
    rshift_shapes = {}

    def getOperationBinaryRShiftShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.rshift_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryRShiftLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('RShift', self, right_shape)
            return operation_result_unknown
    bitor_shapes = {}

    def getOperationBinaryBitOrShape(self, right_shape):
        if False:
            return 10
        result = self.bitor_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryBitOrLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('BitOr', self, right_shape)
            return operation_result_unknown
    bitand_shapes = {}

    def getOperationBinaryBitAndShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        result = self.bitand_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryBitAndLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('BitAnd', self, right_shape)
            return operation_result_unknown
    bitxor_shapes = {}

    def getOperationBinaryBitXorShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.bitxor_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryBitXorLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('BitXor', self, right_shape)
            return operation_result_unknown
    ibitor_shapes = {}

    def getOperationInplaceBitOrShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        'Inplace bitor operation shape, for overload.'
        if self.ibitor_shapes:
            result = self.ibitor_shapes.get(right_shape)
            if result is not None:
                return result
            else:
                right_shape_type = type(right_shape)
                if right_shape_type is ShapeLoopCompleteAlternative:
                    return right_shape.getOperationBinaryBitOrLShape(self)
                if right_shape_type is ShapeLoopInitialAlternative:
                    return operation_result_unknown
                onMissingOperation('IBitOr', self, right_shape)
                return operation_result_unknown
        else:
            return self.getOperationBinaryBitOrShape(right_shape)
    matmult_shapes = {}

    def getOperationBinaryMatMultShape(self, right_shape):
        if False:
            while True:
                i = 10
        result = self.matmult_shapes.get(right_shape)
        if result is not None:
            return result
        else:
            right_shape_type = type(right_shape)
            if right_shape_type is ShapeLoopCompleteAlternative:
                return right_shape.getOperationBinaryBitMatMultLShape(self)
            if right_shape_type is ShapeLoopInitialAlternative:
                return operation_result_unknown
            onMissingOperation('MatMult', self, right_shape)
            return operation_result_unknown

    def getComparisonLtShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        onMissingOperation('Lt', self, right_shape)
        return operation_result_unknown

    def getComparisonLteShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return self.getComparisonLtShape(right_shape)

    def getComparisonGtShape(self, right_shape):
        if False:
            print('Hello World!')
        return self.getComparisonLtShape(right_shape)

    def getComparisonGteShape(self, right_shape):
        if False:
            return 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonEqShape(self, right_shape):
        if False:
            while True:
                i = 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonNeqShape(self, right_shape):
        if False:
            print('Hello World!')
        return self.getComparisonLtShape(right_shape)

    @abstractmethod
    def getOperationUnaryReprEscape(self):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return None

    def emitAlternatives(self, emit):
        if False:
            print('Hello World!')
        emit(self)

class ShapeTypeUnknown(ShapeBase):
    __slots__ = ()

    @staticmethod
    def getOperationBinaryAddShape(right_shape):
        if False:
            for i in range(10):
                print('nop')
        return operation_result_unknown

    @staticmethod
    def getOperationBinarySubShape(right_shape):
        if False:
            while True:
                i = 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryMultShape(right_shape):
        if False:
            i = 10
            return i + 15
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryFloorDivShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryOldDivShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryTrueDivShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryModShape(right_shape):
        if False:
            print('Hello World!')
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryDivmodShape(right_shape):
        if False:
            print('Hello World!')
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryPowShape(right_shape):
        if False:
            i = 10
            return i + 15
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryLShiftShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryRShiftShape(right_shape):
        if False:
            i = 10
            return i + 15
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryBitOrShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryBitAndShape(right_shape):
        if False:
            for i in range(10):
                print('nop')
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryBitXorShape(right_shape):
        if False:
            print('Hello World!')
        return operation_result_unknown

    @staticmethod
    def getOperationBinaryMatMultShape(right_shape):
        if False:
            i = 10
            return i + 15
        return operation_result_unknown

    @staticmethod
    def getComparisonLtShape(right_shape):
        if False:
            return 10
        return operation_result_unknown

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionFullEscape
tshape_unknown = ShapeTypeUnknown()

class ShapeTypeUninitialized(ShapeTypeUnknown):
    __slots__ = ()
tshape_uninitialized = ShapeTypeUninitialized()

class ValueShapeBase(object):
    __slots__ = ()

    def hasShapeSlotLen(self):
        if False:
            print('Hello World!')
        return self.getTypeShape().hasShapeSlotLen()

class ValueShapeUnknown(ValueShapeBase):
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_unknown

    @staticmethod
    def isConstant():
        if False:
            return 10
        "Can't say if it's constant, we don't know anything."
        return None
vshape_unknown = ValueShapeUnknown()

class ShapeLargeConstantValue(object):
    __slots__ = ('shape', 'size')

    def __init__(self, size, shape):
        if False:
            print('Hello World!')
        self.size = size
        self.shape = shape

    def getTypeShape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shape

    @staticmethod
    def isConstant():
        if False:
            print('Hello World!')
        return True

    def hasShapeSlotLen(self):
        if False:
            return 10
        return self.shape.hasShapeSlotLen()

class ShapeLargeConstantValuePredictable(ShapeLargeConstantValue):
    __slots__ = ('predictor',)

    def __init__(self, size, predictor, shape):
        if False:
            for i in range(10):
                print('nop')
        ShapeLargeConstantValue.__init__(self, size, shape)
        self.predictor = predictor

class ShapeIterator(ShapeBase, ShapeIteratorMixin):
    """Iterator created by iter with 2 arguments, TODO: could be way more specific."""
    __slots__ = ()

    @staticmethod
    def isShapeIterator():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeSlotBool():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeSlotLen():
        if False:
            for i in range(10):
                print('nop')
        return None

    @staticmethod
    def hasShapeSlotInt():
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def hasShapeSlotLong():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def hasShapeSlotFloat():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def getShapeIter():
        if False:
            for i in range(10):
                print('nop')
        return tshape_iterator

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionFullEscape
tshape_iterator = ShapeIterator()

class ShapeLoopInitialAlternative(ShapeBase):
    """Merge of loop wrap around with loop start value.

    Happens at the start of loop blocks. This is for loop closed SSA, to
    make it clear, that the entered value, can be anything really, and
    will only later be clarified.

    They will start out with just one previous, and later be updated with
    all of the variable versions at loop continue times.
    """
    __slots__ = ('type_shapes',)

    def __init__(self, shapes):
        if False:
            i = 10
            return i + 15
        self.type_shapes = shapes

    def emitAlternatives(self, emit):
        if False:
            return 10
        for type_shape in self.type_shapes:
            type_shape.emitAlternatives(emit)

    def _collectInitialShape(self, operation):
        if False:
            return 10
        result = set()
        for type_shape in self.type_shapes:
            try:
                (entry, _description) = operation(type_shape)
            except TypeError:
                assert False, type_shape
            if entry is tshape_unknown:
                return tshape_unknown
            result.add(entry)
        return ShapeLoopInitialAlternative(result)

    def getOperationBinaryAddShape(self, right_shape):
        if False:
            print('Hello World!')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryAddShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationInplaceAddShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationInplaceAddShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinarySubShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinarySubShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryMultShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryMultShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryFloorDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryFloorDivShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryOldDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryOldDivShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryTrueDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryTrueDivShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryModShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryModShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryDivmodShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryDivmodShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryPowShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryPowShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryLShiftShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryLShiftShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryRShiftShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryRShiftShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryBitOrShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryBitOrShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryBitAndShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryBitAndShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryBitXorShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryBitXorShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getOperationBinaryMatMultShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getOperationBinaryMatMultShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getComparisonLtShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        else:
            return (self._collectInitialShape(operation=lambda left_shape: left_shape.getComparisonLtShape(right_shape)), ControlFlowDescriptionFullEscape)

    def getComparisonLteShape(self, right_shape):
        if False:
            while True:
                i = 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonGtShape(self, right_shape):
        if False:
            while True:
                i = 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonGteShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        return self.getComparisonLtShape(right_shape)

    def getComparisonEqShape(self, right_shape):
        if False:
            while True:
                i = 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonNeqShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return self.getComparisonLtShape(right_shape)

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionFullEscape

class ShapeLoopCompleteAlternative(ShapeBase):
    """Merge of loop wrap around with loop start value.

    Happens at the start of loop blocks. This is for loop closed SSA, to
    make it clear, that the entered value, can be one of multiple types,
    but only those.

    They will start out with just one previous, and later be updated with
    all of the variable versions at loop continue times.
    """
    __slots__ = ('type_shapes',)

    def __init__(self, shapes):
        if False:
            return 10
        self.type_shapes = shapes

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return 27

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self.__class__ is not other.__class__:
            return False
        return self.type_shapes == other.type_shapes

    def emitAlternatives(self, emit):
        if False:
            for i in range(10):
                print('nop')
        for type_shape in self.type_shapes:
            type_shape.emitAlternatives(emit)

    def _collectShapeOperation(self, operation):
        if False:
            print('Hello World!')
        result = None
        escape_description = None
        single = True
        for type_shape in self.type_shapes:
            (entry, description) = operation(type_shape)
            if entry is tshape_unknown:
                return operation_result_unknown
            if single:
                if result is None:
                    result = entry
                    escape_description = description
                elif result is not entry:
                    single = False
                    result = set((result, entry))
                    escape_description = set((escape_description, description))
            else:
                result.add(entry)
                escape_description.add(description)
        if single:
            assert result is not None
            return (result, escape_description)
        else:
            if len(escape_description) > 1:
                if ControlFlowDescriptionFullEscape in escape_description:
                    escape_description = ControlFlowDescriptionFullEscape
                else:
                    assert False
            else:
                (escape_description,) = escape_description
            return (ShapeLoopCompleteAlternative(result), escape_description)

    def getOperationBinaryAddShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryAddShape(right_shape))

    def getOperationInplaceAddShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationInplaceAddShape(right_shape))

    def getOperationBinarySubShape(self, right_shape):
        if False:
            print('Hello World!')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinarySubShape(right_shape))

    def getOperationBinaryMultShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryMultShape(right_shape))

    def getOperationBinaryFloorDivShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryFloorDivShape(right_shape))

    def getOperationBinaryOldDivShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryOldDivShape(right_shape))

    def getOperationBinaryTrueDivShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryTrueDivShape(right_shape))

    def getOperationBinaryModShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryModShape(right_shape))

    def getOperationBinaryDivmodShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryDivmodShape(right_shape))

    def getOperationBinaryPowShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryPowShape(right_shape))

    def getOperationBinaryLShiftShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryLShiftShape(right_shape))

    def getOperationBinaryRShiftShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryRShiftShape(right_shape))

    def getOperationBinaryBitOrShape(self, right_shape):
        if False:
            return 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryBitOrShape(right_shape))

    def getOperationBinaryBitAndShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryBitAndShape(right_shape))

    def getOperationBinaryBitXorShape(self, right_shape):
        if False:
            while True:
                i = 10
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryBitXorShape(right_shape))

    def getOperationBinaryMatMultShape(self, right_shape):
        if False:
            for i in range(10):
                print('nop')
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getOperationBinaryMatMultShape(right_shape))

    def getOperationBinaryAddLShape(self, left_shape):
        if False:
            i = 10
            return i + 15
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryAddShape)

    def getOperationBinarySubLShape(self, left_shape):
        if False:
            while True:
                i = 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinarySubShape)

    def getOperationBinaryMultLShape(self, left_shape):
        if False:
            i = 10
            return i + 15
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryMultShape)

    def getOperationBinaryFloorDivLShape(self, left_shape):
        if False:
            print('Hello World!')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryFloorDivShape)

    def getOperationBinaryOldDivLShape(self, left_shape):
        if False:
            return 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryOldDivShape)

    def getOperationBinaryTrueDivLShape(self, left_shape):
        if False:
            while True:
                i = 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryTrueDivShape)

    def getOperationBinaryModLShape(self, left_shape):
        if False:
            for i in range(10):
                print('nop')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryModShape)

    def getOperationBinaryDivmodLShape(self, left_shape):
        if False:
            for i in range(10):
                print('nop')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryDivmodShape)

    def getOperationBinaryPowLShape(self, left_shape):
        if False:
            i = 10
            return i + 15
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryPowShape)

    def getOperationBinaryLShiftLShape(self, left_shape):
        if False:
            i = 10
            return i + 15
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryLShiftShape)

    def getOperationBinaryRShiftLShape(self, left_shape):
        if False:
            for i in range(10):
                print('nop')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryRShiftShape)

    def getOperationBinaryBitOrLShape(self, left_shape):
        if False:
            print('Hello World!')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryBitOrShape)

    def getOperationBinaryBitAndLShape(self, left_shape):
        if False:
            return 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryBitAndShape)

    def getOperationBinaryBitXorLShape(self, left_shape):
        if False:
            return 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryBitXorShape)

    def getOperationBinaryMatMultLShape(self, left_shape):
        if False:
            return 10
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getOperationBinaryMatMultShape)

    def getComparisonLtShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        if right_shape is tshape_unknown:
            return operation_result_unknown
        return self._collectShapeOperation(operation=lambda left_shape: left_shape.getComparisonLtShape(right_shape))

    def getComparisonLtLShape(self, left_shape):
        if False:
            for i in range(10):
                print('nop')
        assert left_shape is not tshape_unknown
        return self._collectShapeOperation(operation=left_shape.getComparisonLtShape)

    def getComparisonLteShape(self, right_shape):
        if False:
            print('Hello World!')
        return self.getComparisonLtShape(right_shape)

    def getComparisonGtShape(self, right_shape):
        if False:
            while True:
                i = 10
        return self.getComparisonLtShape(right_shape)

    def getComparisonGteShape(self, right_shape):
        if False:
            i = 10
            return i + 15
        return self.getComparisonLtShape(right_shape)

    def getComparisonEqShape(self, right_shape):
        if False:
            print('Hello World!')
        return self.getComparisonLtShape(right_shape)

    def getComparisonNeqShape(self, right_shape):
        if False:
            return 10
        return self.getComparisonLtShape(right_shape)

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            print('Hello World!')
        return ControlFlowDescriptionFullEscape

    def _delegatedCheck(self, check):
        if False:
            return 10
        result = None
        for type_shape in self.type_shapes:
            r = check(type_shape)
            if r is None:
                return None
            elif r is True:
                if result is False:
                    return None
                elif result is None:
                    result = True
            elif r is False:
                if result is True:
                    return None
                elif result is None:
                    result = False
        return result

    def hasShapeSlotBool(self):
        if False:
            return 10
        return self._delegatedCheck(lambda x: x.hasShapeSlotBool())

    def hasShapeSlotLen(self):
        if False:
            return 10
        return self._delegatedCheck(lambda x: x.hasShapeSlotLen())

    def hasShapeSlotIter(self):
        if False:
            print('Hello World!')
        return self._delegatedCheck(lambda x: x.hasShapeSlotIter())

    def hasShapeSlotNext(self):
        if False:
            while True:
                i = 10
        return self._delegatedCheck(lambda x: x.hasShapeSlotNext())

    def hasShapeSlotContains(self):
        if False:
            for i in range(10):
                print('nop')
        return self._delegatedCheck(lambda x: x.hasShapeSlotContains())

    def hasShapeSlotInt(self):
        if False:
            i = 10
            return i + 15
        return self._delegatedCheck(lambda x: x.hasShapeSlotInt())

    def hasShapeSlotLong(self):
        if False:
            i = 10
            return i + 15
        return self._delegatedCheck(lambda x: x.hasShapeSlotLong())

    def hasShapeSlotFloat(self):
        if False:
            for i in range(10):
                print('nop')
        return self._delegatedCheck(lambda x: x.hasShapeSlotFloat())

    def hasShapeSlotComplex(self):
        if False:
            i = 10
            return i + 15
        return self._delegatedCheck(lambda x: x.hasShapeSlotComplex())

    def hasShapeSlotBytes(self):
        if False:
            return 10
        return self._delegatedCheck(lambda x: x.hasShapeSlotBytes())

    def hasShapeModule(self):
        if False:
            while True:
                i = 10
        return self._delegatedCheck(lambda x: x.hasShapeModule())
tshape_unknown_loop = ShapeLoopCompleteAlternative(shapes=(tshape_unknown,))
operation_result_unknown = (tshape_unknown, ControlFlowDescriptionFullEscape)