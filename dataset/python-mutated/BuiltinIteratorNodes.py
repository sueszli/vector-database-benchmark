""" Built-in iterator nodes.

These play a role in for loops, and in unpacking. They can something be
predicted to succeed or fail, in which case, code can become less complex.

The length of things is an important optimization issue for these to be
good.
"""
from nuitka.PythonVersions import python_version
from .BuiltinLenNodes import ExpressionBuiltinLen
from .ExpressionBases import ExpressionBuiltinSingleArgBase
from .ExpressionBasesGenerated import ExpressionBuiltinIter2Base
from .NodeMakingHelpers import makeRaiseExceptionReplacementStatement, makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue, wrapExpressionWithSideEffects
from .shapes.StandardShapes import tshape_iterator
from .StatementBasesGenerated import StatementSpecialUnpackCheckBase, StatementSpecialUnpackCheckFromIteratedBase
from .VariableRefNodes import ExpressionTempVariableRef

class ExpressionBuiltinIter1(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_ITER1'
    simulator = iter

    @staticmethod
    def isExpressionBuiltinIter1():
        if False:
            while True:
                i = 10
        return True

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_value.computeExpressionIter1(iter_node=self, trace_collection=trace_collection)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            return 10
        return (self, 'new_builtin', 'Eliminated useless iterator creation.')

    def getTypeShape(self):
        if False:
            while True:
                i = 10
        return self.subnode_value.getTypeShape().getShapeIter()

    def computeExpressionNext1(self, next_node, trace_collection):
        if False:
            i = 10
            return i + 15
        value = self.subnode_value
        if value.isKnownToBeIterableAtMin(1) and value.canPredictIterationValues():
            result = wrapExpressionWithSideEffects(new_node=value.getIterationValue(0), old_node=value, side_effects=value.getIterationValueRange(1, None))
            return (False, (result, 'new_expression', "Predicted 'next' value from iteration."))
        self.onContentEscapes(trace_collection)
        if value.mayHaveSideEffectsNext():
            trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (True, (next_node, None, None))

    def isKnownToBeIterable(self, count):
        if False:
            for i in range(10):
                print('nop')
        if count is None:
            return True
        iter_length = self.subnode_value.getIterationLength()
        return iter_length == count

    def isKnownToBeIterableAtMin(self, count):
        if False:
            i = 10
            return i + 15
        assert type(count) is int
        iter_length = self.subnode_value.getIterationMinLength()
        return iter_length is not None and count <= iter_length

    def getIterationLength(self):
        if False:
            return 10
        return self.subnode_value.getIterationLength()

    def canPredictIterationValues(self):
        if False:
            print('Hello World!')
        return self.subnode_value.canPredictIterationValues()

    def getIterationValue(self, element_index):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.getIterationValue(element_index)

    def getIterationHandle(self):
        if False:
            return 10
        return self.subnode_value.getIterationHandle()

    def extractSideEffects(self):
        if False:
            i = 10
            return i + 15
        value = self.subnode_value
        if value.isCompileTimeConstant() and value.isKnownToBeIterable(None):
            return ()
        else:
            return (self,)

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        value = self.subnode_value
        if value.isCompileTimeConstant():
            return not value.isKnownToBeIterable(None)
        return True

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        value = self.subnode_value
        if value.mayRaiseException(exception_type):
            return True
        if value.isKnownToBeIterable(None):
            return False
        return True

    def mayRaiseExceptionOperation(self):
        if False:
            while True:
                i = 10
        value = self.subnode_value
        return value.isKnownToBeIterable(None) is not True

    def onRelease(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        pass

class ExpressionBuiltinIterForUnpack(ExpressionBuiltinIter1):
    kind = 'EXPRESSION_BUILTIN_ITER_FOR_UNPACK'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        result = self.subnode_value.computeExpressionIter1(iter_node=self, trace_collection=trace_collection)
        result_node = result[0]
        if result_node.isExpressionRaiseException() and result_node.subnode_exception_type.isExpressionBuiltinExceptionRef() and (result_node.subnode_exception_type.getExceptionName() == 'TypeError'):
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template='cannot unpack non-iterable %s object', operation='iter', original_node=self, value_node=self.subnode_value)
        return result

    @staticmethod
    def simulator(value):
        if False:
            print('Hello World!')
        try:
            return iter(value)
        except TypeError:
            raise TypeError('cannot unpack non-iterable %s object' % type(value).__name__)

class StatementSpecialUnpackCheckFromIterated(StatementSpecialUnpackCheckFromIteratedBase):
    kind = 'STATEMENT_SPECIAL_UNPACK_CHECK_FROM_ITERATED'
    named_children = ('iterated_length',)
    node_attributes = ('count',)
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.subnode_iterated_length.isCompileTimeConstant():
            iterated_length_value = self.subnode_iterated_length.getCompileTimeConstant()
            if iterated_length_value <= self.count:
                return (None, 'new_statements', lambda : 'Determined iteration length check to be always true, because %d <= %d.' % (iterated_length_value, self.count))
            else:
                result = makeRaiseExceptionReplacementStatement(statement=self, exception_type='ValueError', exception_value='too many values to unpack' if python_version < 768 else 'too many values to unpack (expected %d)' % self.count)
                trace_collection.onExceptionRaiseExit(TypeError)
                return (result, 'new_raise', 'Determined iteration end check to always raise.')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

def makeStatementSpecialUnpackCheckFromIterated(tmp_iterated_variable, count, source_ref):
    if False:
        return 10
    return StatementSpecialUnpackCheckFromIterated(iterated_length=ExpressionBuiltinLen(ExpressionTempVariableRef(variable=tmp_iterated_variable, source_ref=source_ref), source_ref=source_ref), count=count, source_ref=source_ref)

class StatementSpecialUnpackCheck(StatementSpecialUnpackCheckBase):
    kind = 'STATEMENT_SPECIAL_UNPACK_CHECK'
    named_children = ('iterator',)
    node_attributes = ('count',)
    auto_compute_handling = 'operation'

    def getCount(self):
        if False:
            print('Hello World!')
        return self.count

    def computeStatementOperation(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        iterator = self.subnode_iterator
        if iterator.isExpressionTempVariableRef():
            iteration_source_node = iterator.variable_trace.getIterationSourceNode()
            if iteration_source_node is not None:
                if iteration_source_node.parent.isStatementAssignmentVariableIterator():
                    iterator_assign_node = iteration_source_node.parent
                    if iterator_assign_node.tmp_iterated_variable is not None:
                        result = makeStatementSpecialUnpackCheckFromIterated(tmp_iterated_variable=iterator_assign_node.tmp_iterated_variable, count=self.count, source_ref=self.source_ref)
                        return trace_collection.computedStatementResult(result, change_tags='new_statements', change_desc=lambda : "Iterator check of changed to iterated size check using '%s'." % iterator_assign_node.tmp_iterated_variable.getName())
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            for i in range(10):
                print('nop')
        return 'iteration check statement'

class ExpressionBuiltinIter2(ExpressionBuiltinIter2Base):
    kind = 'EXPRESSION_BUILTIN_ITER2'
    named_children = ('callable_arg', 'sentinel')
    auto_compute_handling = 'final'

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_iterator

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (self, 'new_builtin', 'Eliminated useless iterator creation.')

class ExpressionAsyncIter(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_ASYNC_ITER'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        value = self.subnode_value
        return value.computeExpressionAsyncIter(iter_node=self, trace_collection=trace_collection)

    def isKnownToBeIterable(self, count):
        if False:
            i = 10
            return i + 15
        if count is None:
            return True
        return None

    def getIterationLength(self):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_value.getIterationLength()

    def extractSideEffects(self):
        if False:
            while True:
                i = 10
        if self.subnode_value.isCompileTimeConstant():
            return ()
        else:
            return (self,)

    def mayHaveSideEffects(self):
        if False:
            return 10
        if self.subnode_value.isCompileTimeConstant():
            return self.subnode_value.isKnownToBeIterable(None)
        return True

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        value = self.subnode_value
        if value.mayRaiseException(exception_type):
            return True
        if value.isKnownToBeIterable(None):
            return False
        return True

    def isKnownToBeIterableAtMin(self, count):
        if False:
            i = 10
            return i + 15
        assert type(count) is int
        iter_length = self.subnode_value.getIterationMinLength()
        return iter_length is not None and iter_length < count

    def onRelease(self, trace_collection):
        if False:
            while True:
                i = 10
        pass

class ExpressionAsyncNext(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_ASYNC_NEXT'

    def __init__(self, value, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionBuiltinSingleArgBase.__init__(self, value=value, source_ref=source_ref)

    def computeExpression(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)