""" Node dedicated to "".join() code pattern.

This is used for Python 3.6 fstrings re-formulation and has pretty direct
code alternative to actually looking up that method from the empty string
object, so it got a dedicated node, also to perform optimizations specific
to this.
"""
from .ChildrenHavingMixins import ChildHavingValuesTupleMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionStrOrUnicodeExactMixin

class ExpressionStringConcatenation(ExpressionStrOrUnicodeExactMixin, ChildHavingValuesTupleMixin, ExpressionBase):
    kind = 'EXPRESSION_STRING_CONCATENATION'
    named_children = ('values|tuple+setter',)

    def __init__(self, values, source_ref):
        if False:
            return 10
        assert values
        ChildHavingValuesTupleMixin.__init__(self, values=values)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        streaks = []
        start = None
        values = self.subnode_values
        for (count, value) in enumerate(values):
            if value.isCompileTimeConstant() and value.hasShapeStrOrUnicodeExact():
                if start is None:
                    start = count
            else:
                if start is not None:
                    if start != count - 1:
                        streaks.append((start, count))
                start = None
        if start is not None:
            if start != len(values) - 1:
                streaks.append((start, len(values)))
        if streaks:
            values = list(values)
            for streak in reversed(streaks):
                new_element = makeConstantRefNode(constant=''.join((value.getCompileTimeConstant() for value in values[streak[0]:streak[1]])), source_ref=values[streak[0]].source_ref, user_provided=True)
                values[streak[0]:streak[1]] = (new_element,)
            if len(values) > 1:
                self.setChildValues(tuple(values))
                return (self, 'new_constant', 'Partially combined strings for concatenation')
        if len(values) == 1 and values[0].hasShapeStrOrUnicodeExact():
            return (values[0], 'new_constant', 'Removed strings concatenation of one value.')
        return (self, None, None)