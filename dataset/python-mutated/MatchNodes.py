""" Nodes for match statement for Python3.10+ """
from .ChildrenHavingMixins import ChildHavingExpressionMixin
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionTupleShapeExactMixin

class ExpressionMatchArgs(ExpressionTupleShapeExactMixin, ChildHavingExpressionMixin, ExpressionBase):
    kind = 'EXPRESSION_MATCH_ARGS'
    named_children = ('expression',)
    __slots__ = ('max_allowed',)

    def __init__(self, expression, max_allowed, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildHavingExpressionMixin.__init__(self, expression=expression)
        ExpressionBase.__init__(self, source_ref)
        self.max_allowed = max_allowed

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)