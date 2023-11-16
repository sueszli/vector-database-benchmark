""" Subscript node.

Subscripts are important when working with lists and dictionaries. Tracking
them can allow to achieve more compact code, or predict results at compile time.

There is be a method "computeExpressionSubscript" to aid predicting them in the
other nodes.
"""
from nuitka.PythonVersions import python_version
from .ChildrenHavingMixins import ChildrenHavingExpressionSubscriptMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin
from .NodeBases import SideEffectsFromChildrenMixin
from .NodeMakingHelpers import makeRaiseExceptionExpressionFromTemplate, wrapExpressionWithNodeSideEffects
from .StatementBasesGenerated import StatementAssignmentSubscriptBase, StatementDelSubscriptBase

class StatementAssignmentSubscript(StatementAssignmentSubscriptBase):
    kind = 'STATEMENT_ASSIGNMENT_SUBSCRIPT'
    named_children = ('source', 'subscribed', 'subscript')
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            while True:
                i = 10
        return self.subnode_subscribed.computeExpressionSetSubscript(set_node=self, subscript=self.subnode_subscript, value_node=self.subnode_source, trace_collection=trace_collection)

    @staticmethod
    def getStatementNiceName():
        if False:
            print('Hello World!')
        return 'subscript assignment statement'

class StatementDelSubscript(StatementDelSubscriptBase):
    kind = 'STATEMENT_DEL_SUBSCRIPT'
    named_children = ('subscribed', 'subscript')
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            print('Hello World!')
        return self.subnode_subscribed.computeExpressionDelSubscript(del_node=self, subscript=self.subnode_subscript, trace_collection=trace_collection)

    @staticmethod
    def getStatementNiceName():
        if False:
            i = 10
            return i + 15
        return 'subscript del statement'

class ExpressionSubscriptLookup(ChildrenHavingExpressionSubscriptMixin, ExpressionBase):
    kind = 'EXPRESSION_SUBSCRIPT_LOOKUP'
    named_children = ('expression', 'subscript')

    def __init__(self, expression, subscript, source_ref):
        if False:
            return 10
        ChildrenHavingExpressionSubscriptMixin.__init__(self, expression=expression, subscript=subscript)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        return self.subnode_expression.computeExpressionSubscript(lookup_node=self, subscript=self.subnode_subscript, trace_collection=trace_collection)

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            print('Hello World!')
        return None

def makeExpressionSubscriptLookup(expression, subscript, source_ref):
    if False:
        while True:
            i = 10
    return ExpressionSubscriptLookup(expression=expression, subscript=subscript, source_ref=source_ref)

def makeExpressionIndexLookup(expression, index_value, source_ref):
    if False:
        return 10
    return makeExpressionSubscriptLookup(expression=expression, subscript=makeConstantRefNode(constant=index_value, source_ref=source_ref, user_provided=True), source_ref=source_ref)

def hasSubscript(value, subscript):
    if False:
        i = 10
        return i + 15
    'Check if a value has a subscript.'
    try:
        value[subscript]
    except Exception:
        return False
    else:
        return True

class ExpressionSubscriptCheck(ExpressionBoolShapeExactMixin, SideEffectsFromChildrenMixin, ChildrenHavingExpressionSubscriptMixin, ExpressionBase):
    kind = 'EXPRESSION_SUBSCRIPT_CHECK'
    named_children = ('expression', 'subscript')

    def __init__(self, expression, subscript, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingExpressionSubscriptMixin.__init__(self, expression=expression, subscript=subscript)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        source = self.subnode_expression
        subscript = self.subnode_subscript
        if source.isCompileTimeConstant() and subscript.isCompileTimeConstant():
            (result, tags, change_desc) = trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : hasSubscript(source.getCompileTimeConstant(), subscript.getCompileTimeConstant()), description='Subscript check has been pre-computed.')
            result = wrapExpressionWithNodeSideEffects(new_node=result, old_node=source)
            return (result, tags, change_desc)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        return False

class ExpressionSubscriptLookupForUnpack(ExpressionSubscriptLookup):
    kind = 'EXPRESSION_SUBSCRIPT_LOOKUP_FOR_UNPACK'
    __slots__ = ('expected',)

    def __init__(self, expression, subscript, expected, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionSubscriptLookup.__init__(self, expression=expression, subscript=subscript, source_ref=source_ref)
        self.expected = expected

    def computeExpression(self, trace_collection):
        if False:
            return 10
        result = self.subnode_expression.computeExpressionSubscript(lookup_node=self, subscript=self.subnode_subscript, trace_collection=trace_collection)
        result_node = result[0]
        if result_node.isExpressionRaiseException() and result_node.subnode_exception_type.isExpressionBuiltinExceptionRef() and (result_node.subnode_exception_type.getExceptionName() == 'IndexError'):
            if python_version >= 864:
                return (makeRaiseExceptionExpressionFromTemplate(exception_type='ValueError', template='not enough values to unpack (expected %d, got %d)', template_args=(makeConstantRefNode(constant=self.expected, source_ref=self.source_ref), self.subnode_subscript), source_ref=self.source_ref), 'new_raise', 'Raising for unpack too short iterator.')
            else:
                return (makeRaiseExceptionExpressionFromTemplate(exception_type='ValueError', template='need more than %d value to unpack', template_args=self.subnode_subscript, source_ref=self.source_ref), 'new_raise', 'Raising for unpack too short iterator.')
        return result