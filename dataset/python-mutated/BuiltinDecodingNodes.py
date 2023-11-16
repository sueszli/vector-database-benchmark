""" Built-in ord/chr nodes

These are good for optimizations, as they give a very well known result. In the case of
'chr', it's one of 256 strings, and in case of 'ord' it's one of 256 numbers, so these can
answer quite a few questions at compile time.

"""
from nuitka.specs import BuiltinParameterSpecs
from .ExpressionBases import ExpressionBuiltinSingleArgBase

class ExpressionBuiltinOrd(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_ORD'
    builtin_spec = BuiltinParameterSpecs.builtin_ord_spec

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionBuiltinChr(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_CHR'
    builtin_spec = BuiltinParameterSpecs.builtin_chr_spec

    def isKnownToBeIterable(self, count):
        if False:
            for i in range(10):
                print('nop')
        if self.mayRaiseException(BaseException):
            return None
        return count is None or count == 1