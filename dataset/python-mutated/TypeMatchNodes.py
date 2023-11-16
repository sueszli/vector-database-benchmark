"""Dedicated nodes used for the 3.10 matching

Not usable with older Python as it depends on type flags not present.
"""
from .ChildrenHavingMixins import ChildHavingValueMixin
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin
from .NodeBases import SideEffectsFromChildrenMixin

class ExpressionMatchTypeCheckBase(ExpressionBoolShapeExactMixin, SideEffectsFromChildrenMixin, ChildHavingValueMixin, ExpressionBase):
    named_children = ('value',)

    def __init__(self, value, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return self.subnode_value.mayRaiseException(exception_type)

class ExpressionMatchTypeCheckSequence(ExpressionMatchTypeCheckBase):
    kind = 'EXPRESSION_MATCH_TYPE_CHECK_SEQUENCE'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

class ExpressionMatchTypeCheckMapping(ExpressionMatchTypeCheckBase):
    kind = 'EXPRESSION_MATCH_TYPE_CHECK_MAPPING'

    def computeExpression(self, trace_collection):
        if False:
            return 10
        return (self, None, None)