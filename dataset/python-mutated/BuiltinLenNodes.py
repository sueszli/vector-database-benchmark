""" Built-in iterator nodes.

These play a role in for loops, and in unpacking. They can something be
predicted to succeed or fail, in which case, code can become less complex.

The length of things is an important optimization issue for these to be
good.
"""
from nuitka.specs import BuiltinParameterSpecs
from .ExpressionBases import ExpressionBuiltinSingleArgBase
from .ExpressionShapeMixins import ExpressionIntOrLongExactMixin

class ExpressionBuiltinLen(ExpressionIntOrLongExactMixin, ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_LEN'
    builtin_spec = BuiltinParameterSpecs.builtin_len_spec

    def getIntegerValue(self):
        if False:
            while True:
                i = 10
        value = self.subnode_value
        if value.hasShapeSlotLen():
            return value.getIterationLength()
        else:
            return None

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        return self.subnode_value.computeExpressionLen(len_node=self, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        value = self.subnode_value
        if value.mayRaiseException(exception_type):
            return True
        return not value.getTypeShape().hasShapeSlotLen()