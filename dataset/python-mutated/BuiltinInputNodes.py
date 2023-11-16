""" Node the calls to the 'input' built-in.

This has a specific result value, which can be useful to know, but mostly
we want to apply hacks for redirected error output only, and still have
the "input" function being usable.
"""
from .ExpressionBasesGenerated import ExpressionBuiltinInputBase
from .ExpressionShapeMixins import ExpressionStrShapeExactMixin

class ExpressionBuiltinInput(ExpressionStrShapeExactMixin, ExpressionBuiltinInputBase):
    kind = 'EXPRESSION_BUILTIN_INPUT'
    named_children = ('prompt|optional',)
    auto_compute_handling = 'final'

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return True