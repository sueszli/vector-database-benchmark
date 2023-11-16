""" Node the calls to the 'hash' built-in.

This is a specific thing, which must be calculated at run time, but we can
predict things about its type, and the fact that it won't raise an exception
for some types, so it is still useful. Also calls to it can be accelerated
slightly.
"""
from .ChildrenHavingMixins import ChildHavingValueMixin
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionIntShapeExactMixin

class ExpressionBuiltinHash(ExpressionIntShapeExactMixin, ChildHavingValueMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_HASH'
    named_children = ('value',)

    def __init__(self, value, source_ref):
        if False:
            while True:
                i = 10
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        value = self.subnode_value
        if not value.isKnownToBeHashable():
            trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.mayRaiseException(exception_type) or not self.subnode_value.isKnownToBeHashable()

    def mayRaiseExceptionOperation(self):
        if False:
            for i in range(10):
                print('nop')
        return not self.subnode_value.isKnownToBeHashable()