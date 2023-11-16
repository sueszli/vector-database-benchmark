""" Built-in staticmethod/classmethod nodes

These are good for optimizations, as they give a very well known result, changing
only the way a class member is being called. Being able to avoid going through a
C call to the built-ins resulting wrapper, will speed up things.
"""
from .ExpressionBasesGenerated import ExpressionBuiltinClassmethodBase, ExpressionBuiltinStaticmethodBase
from .shapes.BuiltinTypeShapes import tshape_classmethod, tshape_staticmethod

class BuiltinStaticmethodClassmethodMixin(object):
    __slots__ = ()
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return True

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_value.mayRaiseException(exception_type)

    def mayHaveSideEffect(self):
        if False:
            while True:
                i = 10
        return self.subnode_value.mayHaveSideEffect()

    def extractSideEffects(self):
        if False:
            return 10
        return self.subnode_value.extractSideEffects()

class ExpressionBuiltinStaticmethod(BuiltinStaticmethodClassmethodMixin, ExpressionBuiltinStaticmethodBase):
    kind = 'EXPRESSION_BUILTIN_STATICMETHOD'
    named_children = ('value',)

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_staticmethod

class ExpressionBuiltinClassmethod(BuiltinStaticmethodClassmethodMixin, ExpressionBuiltinClassmethodBase):
    kind = 'EXPRESSION_BUILTIN_CLASSMETHOD'
    named_children = ('value',)

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_classmethod