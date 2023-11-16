""" Nodes for unary operations.

Some of these come from built-ins, e.g. abs, some from syntax, and repr from both.
"""
from nuitka import PythonOperators
from .ChildrenHavingMixins import ChildHavingOperandMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin, ExpressionStrOrUnicodeDerivedShapeMixin

class ExpressionOperationUnaryBase(ChildHavingOperandMixin, ExpressionBase):
    named_children = ('operand',)
    __slots__ = ('operator', 'simulator')

    def __init__(self, operand, source_ref):
        if False:
            while True:
                i = 10
        ChildHavingOperandMixin.__init__(self, operand=operand)
        ExpressionBase.__init__(self, source_ref)

    def getOperator(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operator

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        operator = self.getOperator()
        operand = self.subnode_operand
        if operand.isCompileTimeConstant():
            operand_value = operand.getCompileTimeConstant()
            return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.simulator(operand_value), description="Operator '%s' with constant argument." % operator)
        else:
            trace_collection.onExceptionRaiseExit(BaseException)
            trace_collection.onControlFlowEscape(self)
            return (self, None, None)

    @staticmethod
    def isExpressionOperationUnary():
        if False:
            return 10
        return True

class ExpressionOperationUnaryRepr(ExpressionStrOrUnicodeDerivedShapeMixin, ExpressionOperationUnaryBase):
    """Python unary operator `x` and repr built-in."""
    kind = 'EXPRESSION_OPERATION_UNARY_REPR'
    operator = 'Repr'
    __slots__ = ('escape_desc',)

    def __init__(self, operand, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)
        self.escape_desc = None

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        (result, self.escape_desc) = self.subnode_operand.computeExpressionOperationRepr(repr_node=self, trace_collection=trace_collection)
        return result

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.escape_desc is None or self.escape_desc.getExceptionExit() is not None or self.subnode_operand.mayRaiseException(exception_type)

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        operand = self.subnode_operand
        if operand.mayHaveSideEffects():
            return True
        return self.escape_desc is None or self.escape_desc.isControlFlowEscape()

class ExpressionOperationUnarySub(ExpressionOperationUnaryBase):
    """Python unary operator -"""
    kind = 'EXPRESSION_OPERATION_UNARY_SUB'
    operator = 'USub'
    simulator = PythonOperators.unary_operator_functions[operator]

    def __init__(self, operand, source_ref):
        if False:
            return 10
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)

class ExpressionOperationUnaryAdd(ExpressionOperationUnaryBase):
    """Python unary operator +"""
    kind = 'EXPRESSION_OPERATION_UNARY_ADD'
    operator = 'UAdd'
    simulator = PythonOperators.unary_operator_functions[operator]

    def __init__(self, operand, source_ref):
        if False:
            return 10
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)

class ExpressionOperationUnaryInvert(ExpressionOperationUnaryBase):
    """Python unary operator ~"""
    kind = 'EXPRESSION_OPERATION_UNARY_INVERT'
    operator = 'Invert'
    simulator = PythonOperators.unary_operator_functions[operator]

    def __init__(self, operand, source_ref):
        if False:
            print('Hello World!')
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)

class ExpressionOperationNot(ExpressionBoolShapeExactMixin, ExpressionOperationUnaryBase):
    kind = 'EXPRESSION_OPERATION_NOT'
    operator = 'Not'
    simulator = PythonOperators.unary_operator_functions[operator]

    def __init__(self, operand, source_ref):
        if False:
            while True:
                i = 10
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return self.subnode_operand.computeExpressionOperationNot(not_node=self, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_operand.mayRaiseException(exception_type) or self.subnode_operand.mayRaiseExceptionBool(exception_type)

    def getTruthValue(self):
        if False:
            i = 10
            return i + 15
        result = self.subnode_operand.getTruthValue()
        return None if result is None else not result

    def mayHaveSideEffects(self):
        if False:
            i = 10
            return i + 15
        operand = self.subnode_operand
        if operand.mayHaveSideEffects():
            return True
        return operand.mayHaveSideEffectsBool()

    def mayHaveSideEffectsBool(self):
        if False:
            return 10
        return self.subnode_operand.mayHaveSideEffectsBool()

    def extractSideEffects(self):
        if False:
            while True:
                i = 10
        operand = self.subnode_operand
        if operand.isExpressionMakeSequence():
            return operand.extractSideEffects()
        if operand.isExpressionMakeDict():
            return operand.extractSideEffects()
        return (self,)

class ExpressionOperationUnaryAbs(ExpressionOperationUnaryBase):
    kind = 'EXPRESSION_OPERATION_UNARY_ABS'
    operator = 'Abs'
    simulator = PythonOperators.unary_operator_functions[operator]

    def __init__(self, operand, source_ref):
        if False:
            print('Hello World!')
        ExpressionOperationUnaryBase.__init__(self, operand=operand, source_ref=source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_operand.computeExpressionAbs(abs_node=self, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        operand = self.subnode_operand
        if operand.mayRaiseException(exception_type):
            return True
        return operand.mayRaiseExceptionAbs(exception_type)

    def mayHaveSideEffects(self):
        if False:
            for i in range(10):
                print('nop')
        operand = self.subnode_operand
        if operand.mayHaveSideEffects():
            return True
        return operand.mayHaveSideEffectsAbs()

def makeExpressionOperationUnary(operator, operand, source_ref):
    if False:
        return 10
    if operator == 'Repr':
        unary_class = ExpressionOperationUnaryRepr
    elif operator == 'USub':
        unary_class = ExpressionOperationUnarySub
    elif operator == 'UAdd':
        unary_class = ExpressionOperationUnaryAdd
    elif operator == 'Invert':
        unary_class = ExpressionOperationUnaryInvert
    else:
        assert False, operand
    if operand.isCompileTimeConstant():
        try:
            constant = unary_class.simulator(operand.getCompileTimeConstant())
        except Exception:
            pass
        else:
            return makeConstantRefNode(constant=constant, source_ref=source_ref, user_provided=getattr(operand, 'user_provided', False))
    return unary_class(operand=operand, source_ref=source_ref)