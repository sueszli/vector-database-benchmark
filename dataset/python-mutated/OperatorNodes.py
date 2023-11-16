""" Nodes for unary and binary operations.

No short-circuit involved, boolean 'not' is an unary operation like '-' is,
no real difference.
"""
import copy
import math
from abc import abstractmethod
from nuitka import PythonOperators
from nuitka.Errors import NuitkaAssumptionError
from nuitka.PythonVersions import python_version
from .ChildrenHavingMixins import ChildrenHavingLeftRightMixin
from .ExpressionBases import ExpressionBase
from .NodeMakingHelpers import makeRaiseExceptionReplacementExpressionFromInstance, wrapExpressionWithSideEffects
from .shapes.BuiltinTypeShapes import tshape_bool, tshape_int_or_long
from .shapes.StandardShapes import ShapeLargeConstantValue, ShapeLargeConstantValuePredictable, tshape_unknown, vshape_unknown

class ExpressionPropertiesFromTypeShapeMixin(object):
    """Given a self.type_shape, this can derive default properties from there."""
    __slots__ = ()

    def isKnownToBeHashable(self):
        if False:
            i = 10
            return i + 15
        return self.type_shape.hasShapeSlotHash()

class ExpressionOperationBinaryBase(ExpressionPropertiesFromTypeShapeMixin, ChildrenHavingLeftRightMixin, ExpressionBase):
    """Base class for all binary operation expression."""
    __slots__ = ('type_shape', 'escape_desc', 'inplace_suspect', 'shape')
    named_children = ('left', 'right')
    nice_children = tuple((child_name + ' operand' for child_name in named_children))

    def __init__(self, left, right, source_ref):
        if False:
            print('Hello World!')
        ChildrenHavingLeftRightMixin.__init__(self, left=left, right=right)
        ExpressionBase.__init__(self, source_ref)
        self.type_shape = tshape_unknown
        self.escape_desc = None
        self.inplace_suspect = False
        self.shape = vshape_unknown

    @staticmethod
    def isExpressionOperationBinary():
        if False:
            while True:
                i = 10
        return True

    def getOperator(self):
        if False:
            i = 10
            return i + 15
        return self.operator

    def markAsInplaceSuspect(self):
        if False:
            i = 10
            return i + 15
        self.inplace_suspect = True

    def removeMarkAsInplaceSuspect(self):
        if False:
            i = 10
            return i + 15
        self.inplace_suspect = False

    def isInplaceSuspect(self):
        if False:
            print('Hello World!')
        return self.inplace_suspect

    def getOperands(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.subnode_left, self.subnode_right)

    def mayRaiseExceptionOperation(self):
        if False:
            i = 10
            return i + 15
        return self.escape_desc.getExceptionExit() is not None

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.escape_desc is None or self.escape_desc.getExceptionExit() is not None or self.subnode_left.mayRaiseException(exception_type) or self.subnode_right.mayRaiseException(exception_type)

    def getTypeShape(self):
        if False:
            i = 10
            return i + 15
        return self.type_shape

    @abstractmethod
    def _getOperationShape(self, left_shape, right_shape):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def canCreateUnsupportedException(left_shape, right_shape):
        if False:
            return 10
        return hasattr(left_shape, 'typical_value') and hasattr(right_shape, 'typical_value')

    def createUnsupportedException(self, left_shape, right_shape):
        if False:
            print('Hello World!')
        left = left_shape.typical_value
        right = right_shape.typical_value
        try:
            self.simulator(left, right)
        except TypeError as e:
            return e
        except Exception as e:
            raise NuitkaAssumptionError('Unexpected exception type doing operation simulation', self.operator, self.simulator, left_shape, right_shape, repr(left), repr(right), e, '!=')
        else:
            raise NuitkaAssumptionError('Unexpected no-exception doing operation simulation', self.operator, self.simulator, left_shape, right_shape, repr(left), repr(right))

    @staticmethod
    def _isTooLarge():
        if False:
            return 10
        return False

    def _simulateOperation(self, trace_collection):
        if False:
            print('Hello World!')
        left_value = self.subnode_left.getCompileTimeConstant()
        right_value = self.subnode_right.getCompileTimeConstant()
        if self.subnode_left.isMutable():
            left_value = copy.copy(left_value)
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.simulator(left_value, right_value), description="Operator '%s' with constant arguments." % self.operator)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.shape is not None and self.shape.isConstant():
            return (self, None, None)
        left = self.subnode_left
        left_shape = left.getTypeShape()
        right = self.subnode_right
        right_shape = right.getTypeShape()
        (self.type_shape, self.escape_desc) = self._getOperationShape(left_shape, right_shape)
        if left.isCompileTimeConstant() and right.isCompileTimeConstant():
            if not self._isTooLarge():
                return self._simulateOperation(trace_collection)
        exception_raise_exit = self.escape_desc.getExceptionExit()
        if exception_raise_exit is not None:
            trace_collection.onExceptionRaiseExit(exception_raise_exit)
            if self.escape_desc.isUnsupported() and self.canCreateUnsupportedException(left_shape, right_shape):
                result = wrapExpressionWithSideEffects(new_node=makeRaiseExceptionReplacementExpressionFromInstance(expression=self, exception=self.createUnsupportedException(left_shape, right_shape)), old_node=self, side_effects=(left, right))
                return (result, 'new_raise', "Replaced operator '%s' with %s %s arguments that cannot work." % (self.operator, left_shape, right_shape))
        if self.escape_desc.isValueEscaping():
            trace_collection.removeKnowledge(left)
            trace_collection.removeKnowledge(right)
        if self.escape_desc.isControlFlowEscape():
            trace_collection.onControlFlowEscape(self)
        return (self, None, None)

    @staticmethod
    def canPredictIterationValues():
        if False:
            return 10
        return False

class ExpressionOperationAddMixin(object):
    __slots__ = ()

    def getValueShape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shape

    def _isTooLarge(self):
        if False:
            return 10
        if self.subnode_left.isKnownToBeIterable(None) and self.subnode_right.isKnownToBeIterable(None):
            size = self.subnode_left.getIterationLength() + self.subnode_right.getIterationLength()
            self.shape = ShapeLargeConstantValuePredictable(size=size, predictor=None, shape=self.subnode_left.getTypeShape())
            return size > 256
        else:
            return False

class ExpressionOperationBinaryAdd(ExpressionOperationAddMixin, ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_ADD'

    def __init__(self, left, right, source_ref):
        if False:
            while True:
                i = 10
        ExpressionOperationBinaryBase.__init__(self, left=left, right=right, source_ref=source_ref)
    operator = 'Add'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            return 10
        return left_shape.getOperationBinaryAddShape(right_shape)

class ExpressionOperationBinarySub(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_SUB'
    operator = 'Sub'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationBinarySubShape(right_shape)

class ExpressionOperationMultMixin(object):
    __slots__ = ()

    def getValueShape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shape

    def _isTooLarge(self):
        if False:
            print('Hello World!')
        if self.subnode_right.isNumberConstant():
            iter_length = self.subnode_left.getIterationLength()
            if iter_length is not None:
                size = iter_length * self.subnode_right.getCompileTimeConstant()
                if size > 256:
                    self.shape = ShapeLargeConstantValuePredictable(size=size, predictor=None, shape=self.subnode_left.getTypeShape())
                    return True
            if self.subnode_left.isNumberConstant():
                if self.subnode_left.isIndexConstant() and self.subnode_right.isIndexConstant():
                    left_value = self.subnode_left.getCompileTimeConstant()
                    if left_value != 0:
                        right_value = self.subnode_right.getCompileTimeConstant()
                        if right_value != 0:
                            if math.log10(abs(left_value)) + math.log10(abs(right_value)) > 20:
                                self.shape = ShapeLargeConstantValue(size=None, shape=tshape_int_or_long)
                                return True
        elif self.subnode_left.isNumberConstant():
            iter_length = self.subnode_right.getIterationLength()
            if iter_length is not None:
                left_value = self.subnode_left.getCompileTimeConstant()
                size = iter_length * left_value
                if iter_length * left_value > 256:
                    self.shape = ShapeLargeConstantValuePredictable(size=size, predictor=None, shape=self.subnode_right.getTypeShape())
                    return True
        return False

class ExpressionOperationBinaryMult(ExpressionOperationMultMixin, ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_MULT'
    operator = 'Mult'
    simulator = PythonOperators.binary_operator_functions[operator]

    def __init__(self, left, right, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionOperationBinaryBase.__init__(self, left=left, right=right, source_ref=source_ref)

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return left_shape.getOperationBinaryMultShape(right_shape)

    def getIterationLength(self):
        if False:
            while True:
                i = 10
        left_length = self.subnode_left.getIterationLength()
        if left_length is not None:
            right_value = self.subnode_right.getIntegerValue()
            if right_value is not None:
                return left_length * right_value
        right_length = self.subnode_right.getIterationLength()
        if right_length is not None:
            left_value = self.subnode_left.getIntegerValue()
            if left_value is not None:
                return right_length * left_value
        return ExpressionOperationBinaryBase.getIterationLength(self)

    def extractSideEffects(self):
        if False:
            i = 10
            return i + 15
        left_length = self.subnode_left.getIterationLength()
        if left_length is not None:
            right_value = self.subnode_right.getIntegerValue()
            if right_value is not None:
                return self.subnode_left.extractSideEffects() + self.subnode_right.extractSideEffects()
        right_length = self.subnode_right.getIterationLength()
        if right_length is not None:
            left_value = self.subnode_left.getIntegerValue()
            if left_value is not None:
                return self.subnode_left.extractSideEffects() + self.subnode_right.extractSideEffects()
        return ExpressionOperationBinaryBase.extractSideEffects(self)

class ExpressionOperationBinaryFloorDiv(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_FLOOR_DIV'
    operator = 'FloorDiv'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryFloorDivShape(right_shape)
if python_version < 768:

    class ExpressionOperationBinaryOldDiv(ExpressionOperationBinaryBase):
        kind = 'EXPRESSION_OPERATION_BINARY_OLD_DIV'
        operator = 'OldDiv'
        simulator = PythonOperators.binary_operator_functions[operator]

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                while True:
                    i = 10
            return left_shape.getOperationBinaryOldDivShape(right_shape)

class ExpressionOperationBinaryTrueDiv(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_TRUE_DIV'
    operator = 'TrueDiv'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            return 10
        return left_shape.getOperationBinaryTrueDivShape(right_shape)

class ExpressionOperationBinaryMod(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_MOD'
    operator = 'Mod'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            while True:
                i = 10
        return left_shape.getOperationBinaryModShape(right_shape)

class ExpressionOperationBinaryDivmod(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_DIVMOD'
    operator = 'Divmod'
    simulator = PythonOperators.binary_operator_functions[operator]

    def __init__(self, left, right, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionOperationBinaryBase.__init__(self, left=left, right=right, source_ref=source_ref)

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            while True:
                i = 10
        return left_shape.getOperationBinaryDivmodShape(right_shape)

class ExpressionOperationPowMixin(object):
    __slots__ = ()

    def getValueShape(self):
        if False:
            print('Hello World!')
        return self.shape

    def _isTooLarge(self):
        if False:
            for i in range(10):
                print('nop')
        if self.subnode_right.isIndexConstant():
            left_value = abs(self.subnode_left.getCompileTimeConstant())
            if left_value in (0, 1):
                return False
            if self.subnode_left.isIndexConstant():
                right_value = self.subnode_right.getCompileTimeConstant()
                if right_value <= 1:
                    return False
                if math.log10(left_value) * right_value > 20:
                    self.shape = ShapeLargeConstantValue(size=None, shape=tshape_int_or_long)
                    return True
        return False

class ExpressionOperationBinaryPow(ExpressionOperationPowMixin, ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_POW'
    operator = 'Pow'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryPowShape(right_shape)

class ExpressionOperationLshiftMixin(object):
    __slots__ = ()

    def getValueShape(self):
        if False:
            i = 10
            return i + 15
        return self.shape

    def _isTooLarge(self):
        if False:
            return 10
        if self.subnode_right.isNumberConstant():
            if self.subnode_left.isNumberConstant():
                left_value = self.subnode_left.getCompileTimeConstant()
                if left_value != 0:
                    right_value = self.subnode_right.getCompileTimeConstant()
                    if right_value > 64:
                        self.shape = ShapeLargeConstantValue(size=None, shape=tshape_int_or_long)
                        return True
        return False

class ExpressionOperationBinaryLshift(ExpressionOperationLshiftMixin, ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_LSHIFT'
    operator = 'LShift'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryLShiftShape(right_shape)

class ExpressionOperationBinaryRshift(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_RSHIFT'
    operator = 'RShift'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return left_shape.getOperationBinaryRShiftShape(right_shape)

class ExpressionOperationBinaryBitOr(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_BIT_OR'
    operator = 'BitOr'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return left_shape.getOperationBinaryBitOrShape(right_shape)

class ExpressionOperationBinaryBitAnd(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_BIT_AND'
    operator = 'BitAnd'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            while True:
                i = 10
        return left_shape.getOperationBinaryBitAndShape(right_shape)

class ExpressionOperationBinaryBitXor(ExpressionOperationBinaryBase):
    kind = 'EXPRESSION_OPERATION_BINARY_BIT_XOR'
    operator = 'BitXor'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationBinaryBitXorShape(right_shape)
if python_version >= 848:

    class ExpressionOperationBinaryMatMult(ExpressionOperationBinaryBase):
        kind = 'EXPRESSION_OPERATION_BINARY_MAT_MULT'
        operator = 'MatMult'
        simulator = PythonOperators.binary_operator_functions[operator]

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                for i in range(10):
                    print('nop')
            return left_shape.getOperationBinaryMatMultShape(right_shape)
_operator2binary_operation_node_class = {'Add': ExpressionOperationBinaryAdd, 'Sub': ExpressionOperationBinarySub, 'Mult': ExpressionOperationBinaryMult, 'FloorDiv': ExpressionOperationBinaryFloorDiv, 'TrueDiv': ExpressionOperationBinaryTrueDiv, 'Mod': ExpressionOperationBinaryMod, 'Pow': ExpressionOperationBinaryPow, 'LShift': ExpressionOperationBinaryLshift, 'RShift': ExpressionOperationBinaryRshift, 'BitOr': ExpressionOperationBinaryBitOr, 'BitAnd': ExpressionOperationBinaryBitAnd, 'BitXor': ExpressionOperationBinaryBitXor}
if python_version < 768:
    _operator2binary_operation_node_class['OldDiv'] = ExpressionOperationBinaryOldDiv
if python_version >= 848:
    _operator2binary_operation_node_class['MatMult'] = ExpressionOperationBinaryMatMult

def makeBinaryOperationNode(operator, left, right, source_ref):
    if False:
        print('Hello World!')
    node_class = _operator2binary_operation_node_class[operator]
    return node_class(left=left, right=right, source_ref=source_ref)

class ExpressionOperationBinaryInplaceBase(ExpressionOperationBinaryBase):
    """Base class for all inplace operations."""

    def __init__(self, left, right, source_ref):
        if False:
            print('Hello World!')
        ExpressionOperationBinaryBase.__init__(self, left=left, right=right, source_ref=source_ref)
        self.inplace_suspect = True

    @staticmethod
    def isExpressionOperationInplace():
        if False:
            i = 10
            return i + 15
        return True

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        if self.shape is not None and self.shape.isConstant():
            return (self, None, None)
        left = self.subnode_left
        left_shape = left.getTypeShape()
        right = self.subnode_right
        right_shape = right.getTypeShape()
        (self.type_shape, self.escape_desc) = self._getOperationShape(left_shape, right_shape)
        if left.isCompileTimeConstant() and right.isCompileTimeConstant():
            if not self._isTooLarge():
                return self._simulateOperation(trace_collection)
        exception_raise_exit = self.escape_desc.getExceptionExit()
        if exception_raise_exit is not None:
            trace_collection.onExceptionRaiseExit(exception_raise_exit)
            if self.escape_desc.isUnsupported() and self.canCreateUnsupportedException(left_shape, right_shape):
                result = wrapExpressionWithSideEffects(new_node=makeRaiseExceptionReplacementExpressionFromInstance(expression=self, exception=self.createUnsupportedException(left_shape, right_shape)), old_node=self, side_effects=(left, right))
                return (result, 'new_raise', "Replaced inplace-operator '%s' with %s %s arguments that cannot work." % (self.operator, left_shape, right_shape))
        if self.escape_desc.isValueEscaping():
            trace_collection.removeKnowledge(left)
            trace_collection.removeKnowledge(right)
        if self.escape_desc.isControlFlowEscape():
            trace_collection.onControlFlowEscape(self)
        if left_shape is tshape_bool:
            result = makeBinaryOperationNode(self.operator[1:], left, right, self.source_ref)
            return trace_collection.computedExpressionResult(result, 'new_expression', "Lowered inplace-operator '%s' to binary operation." % self.operator)
        return (self, None, None)

class ExpressionOperationInplaceAdd(ExpressionOperationAddMixin, ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_ADD'
    operator = 'IAdd'
    simulator = PythonOperators.binary_operator_functions[operator]

    def __init__(self, left, right, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionOperationBinaryInplaceBase.__init__(self, left=left, right=right, source_ref=source_ref)

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationInplaceAddShape(right_shape)

class ExpressionOperationInplaceSub(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_SUB'
    operator = 'ISub'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationBinarySubShape(right_shape)

class ExpressionOperationInplaceMult(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_MULT'
    operator = 'IMult'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryMultShape(right_shape)

class ExpressionOperationInplaceFloorDiv(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_FLOOR_DIV'
    operator = 'IFloorDiv'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationBinaryFloorDivShape(right_shape)
if python_version < 768:

    class ExpressionOperationInplaceOldDiv(ExpressionOperationBinaryInplaceBase):
        kind = 'EXPRESSION_OPERATION_INPLACE_OLD_DIV'
        operator = 'IOldDiv'
        simulator = PythonOperators.binary_operator_functions[operator]

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                i = 10
                return i + 15
            return left_shape.getOperationBinaryOldDivShape(right_shape)

class ExpressionOperationInplaceTrueDiv(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_TRUE_DIV'
    operator = 'ITrueDiv'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return left_shape.getOperationBinaryTrueDivShape(right_shape)

class ExpressionOperationInplaceMod(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_MOD'
    operator = 'IMod'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            for i in range(10):
                print('nop')
        return left_shape.getOperationBinaryModShape(right_shape)

class ExpressionOperationInplacePow(ExpressionOperationPowMixin, ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_POW'
    operator = 'IPow'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryPowShape(right_shape)

class ExpressionOperationInplaceLshift(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_LSHIFT'
    operator = 'ILShift'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            return 10
        return left_shape.getOperationBinaryLShiftShape(right_shape)

class ExpressionOperationInplaceRshift(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_RSHIFT'
    operator = 'IRShift'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            print('Hello World!')
        return left_shape.getOperationBinaryRShiftShape(right_shape)

class ExpressionOperationInplaceBitOr(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_BIT_OR'
    operator = 'IBitOr'
    simulator = PythonOperators.binary_operator_functions[operator]
    if python_version < 912:

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                return 10
            return left_shape.getOperationBinaryBitOrShape(right_shape)
    else:

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                return 10
            return left_shape.getOperationInplaceBitOrShape(right_shape)

class ExpressionOperationInplaceBitAnd(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_BIT_AND'
    operator = 'IBitAnd'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            return 10
        return left_shape.getOperationBinaryBitAndShape(right_shape)

class ExpressionOperationInplaceBitXor(ExpressionOperationBinaryInplaceBase):
    kind = 'EXPRESSION_OPERATION_INPLACE_BIT_XOR'
    operator = 'IBitXor'
    simulator = PythonOperators.binary_operator_functions[operator]

    @staticmethod
    def _getOperationShape(left_shape, right_shape):
        if False:
            i = 10
            return i + 15
        return left_shape.getOperationBinaryBitXorShape(right_shape)
if python_version >= 848:

    class ExpressionOperationInplaceMatMult(ExpressionOperationBinaryInplaceBase):
        kind = 'EXPRESSION_OPERATION_INPLACE_MAT_MULT'
        operator = 'IMatMult'
        simulator = PythonOperators.binary_operator_functions[operator]

        @staticmethod
        def _getOperationShape(left_shape, right_shape):
            if False:
                i = 10
                return i + 15
            return left_shape.getOperationBinaryMatMultShape(right_shape)
_operator2binary_inplace_node_class = {'IAdd': ExpressionOperationInplaceAdd, 'ISub': ExpressionOperationInplaceSub, 'IMult': ExpressionOperationInplaceMult, 'IFloorDiv': ExpressionOperationInplaceFloorDiv, 'ITrueDiv': ExpressionOperationInplaceTrueDiv, 'IMod': ExpressionOperationInplaceMod, 'IPow': ExpressionOperationInplacePow, 'ILShift': ExpressionOperationInplaceLshift, 'IRShift': ExpressionOperationInplaceRshift, 'IBitOr': ExpressionOperationInplaceBitOr, 'IBitAnd': ExpressionOperationInplaceBitAnd, 'IBitXor': ExpressionOperationInplaceBitXor}
if python_version < 768:
    _operator2binary_inplace_node_class['IOldDiv'] = ExpressionOperationInplaceOldDiv
if python_version >= 848:
    _operator2binary_inplace_node_class['IMatMult'] = ExpressionOperationInplaceMatMult

def makeExpressionOperationBinaryInplace(operator, left, right, source_ref):
    if False:
        while True:
            i = 10
    node_class = _operator2binary_inplace_node_class[operator]
    return node_class(left=left, right=right, source_ref=source_ref)