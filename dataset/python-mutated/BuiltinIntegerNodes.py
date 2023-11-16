""" Node for the calls to the 'int' and 'long' (Python2) built-ins.

These are divided into variants for one and two arguments and they have a
common base class, because most of the behavior is the same there. The ones
with 2 arguments only work on strings, and give errors otherwise, the ones
with one argument, use slots, "__int__" and "__long__", so what they do does
largely depend on the arguments slot.
"""
from nuitka.__past__ import long
from nuitka.specs import BuiltinParameterSpecs
from .ChildrenHavingMixins import ChildHavingValueMixin, ChildrenHavingValueOptionalBaseMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase, ExpressionSpecBasedComputationMixin
from .ExpressionShapeMixins import ExpressionIntOrLongExactMixin, ExpressionLongShapeExactMixin
from .shapes.BuiltinTypeShapes import tshape_int_or_long_derived, tshape_long_derived

class ExpressionBuiltinInt1(ChildHavingValueMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_INT1'
    named_children = ('value',)

    def __init__(self, value, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_int_or_long_derived

    def computeExpression(self, trace_collection):
        if False:
            return 10
        return self.subnode_value.computeExpressionInt(int_node=self, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_value.mayRaiseExceptionInt(exception_type)

class ExpressionBuiltinIntLong2Base(ExpressionSpecBasedComputationMixin, ChildrenHavingValueOptionalBaseMixin, ExpressionBase):
    named_children = ('value|optional', 'base')
    try:
        int(base=2)
    except TypeError:
        base_only_value = False
    else:
        base_only_value = True
    builtin = int

    def __init__(self, value, base, source_ref):
        if False:
            i = 10
            return i + 15
        if value is None and self.base_only_value:
            value = makeConstantRefNode(constant='0', source_ref=source_ref, user_provided=True)
        ChildrenHavingValueOptionalBaseMixin.__init__(self, value=value, base=base)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        value = self.subnode_value
        base = self.subnode_base
        if value is None:
            if base is not None:
                if not self.base_only_value:
                    return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.builtin(base=2), description='%s built-in call with only base argument' % self.builtin.__name__)
            given_values = ()
        else:
            given_values = (value, base)
        return self.computeBuiltinSpec(trace_collection=trace_collection, given_values=given_values)

class ExpressionBuiltinInt2(ExpressionIntOrLongExactMixin, ExpressionBuiltinIntLong2Base):
    kind = 'EXPRESSION_BUILTIN_INT2'
    builtin_spec = BuiltinParameterSpecs.builtin_int_spec
    builtin = int

class ExpressionBuiltinLong1(ChildHavingValueMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_LONG1'
    named_children = ('value',)

    def __init__(self, value, source_ref):
        if False:
            while True:
                i = 10
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_long_derived

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return self.subnode_value.computeExpressionLong(long_node=self, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.mayRaiseExceptionLong(exception_type)

class ExpressionBuiltinLong2(ExpressionLongShapeExactMixin, ExpressionBuiltinIntLong2Base):
    kind = 'EXPRESSION_BUILTIN_LONG2'
    builtin_spec = BuiltinParameterSpecs.builtin_long_spec
    builtin = long