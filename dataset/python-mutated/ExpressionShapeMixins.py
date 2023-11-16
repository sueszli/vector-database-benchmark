"""Mixins for expressions that have specific shapes.

Providing derived implementation, such that e.g. for a given shape, shortcuts
are automatically implemented.
"""
from abc import abstractmethod
from nuitka.Constants import the_empty_bytearray, the_empty_dict, the_empty_frozenset, the_empty_list, the_empty_set, the_empty_slice, the_empty_tuple, the_empty_unicode
from .NodeMakingHelpers import makeConstantReplacementNode, makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue
from .shapes.BuiltinTypeShapes import tshape_bool, tshape_bytearray, tshape_bytes, tshape_complex, tshape_dict, tshape_ellipsis, tshape_float, tshape_frozenset, tshape_int, tshape_int_or_long, tshape_list, tshape_long, tshape_none, tshape_set, tshape_slice, tshape_str, tshape_str_derived, tshape_str_or_unicode, tshape_str_or_unicode_derived, tshape_tuple, tshape_type, tshape_unicode, tshape_unicode_derived

class ExpressionSpecificDerivedMixinBase(object):
    """Mixin that provides all shapes exactly false overloads.

    This is to be used as a base class for specific or derived shape
    mixins, such that they automatically provide false for all other exact
    shape checks except the one they care about.
    """
    __slots__ = ()

    @staticmethod
    def hasShapeNoneExact():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeBoolExact():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeDictionaryExact():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeListExact():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeSetExact():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeFrozensetExact():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeTupleExact():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeStrExact():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeUnicodeExact():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeStrOrUnicodeExact():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def hasShapeBytesExact():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeBytearrayExact():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeFloatExact():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def hasShapeComplexExact():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeIntExact():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeLongExact():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeSliceExact():
        if False:
            return 10
        return False

class ExpressionSpecificExactMixinBase(ExpressionSpecificDerivedMixinBase):
    """Mixin that provides attribute knowledge for exact type shapes.

    This is to be used as a base class for specific shape mixins,
    such that they automatically provide false for all other exact
    shape checks except the one they care about.
    """
    __slots__ = ()

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            i = 10
            return i + 15
        return True

    @abstractmethod
    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return True

    @abstractmethod
    def getKnownAttributeValue(self, attribute_name):
        if False:
            while True:
                i = 10
        'Can be used as isKnownToHaveAttribute is True'

    def mayRaiseExceptionAttributeLookup(self, exception_type, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return not self.isKnownToHaveAttribute(attribute_name)

    @staticmethod
    def mayRaiseExceptionBool(exception_type):
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def mayHaveSideEffectsBool():
        if False:
            print('Hello World!')
        return False

class ExpressionNonIterableTypeShapeMixin(object):
    """Mixin for nodes known to not be iterable."""
    __slots__ = ()

    @staticmethod
    def getIterationLength():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isKnownToBeIterableAtMin(count):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def canPredictIterationValues():
        if False:
            while True:
                i = 10
        return False

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            while True:
                i = 10
        shape = self.getTypeShape()
        assert shape.hasShapeSlotIter() is False
        trace_collection.onExceptionRaiseExit(BaseException)
        return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="'%s' object is not iterable", operation='iter', original_node=iter_node, value_node=self)

class ExpressionIterableTypeShapeMixin(object):
    """Mixin for nodes known to not be iterable."""
    __slots__ = ()

    def isKnownToBeIterable(self, count):
        if False:
            print('Hello World!')
        return count is None or self.getIterationLength() == count

    def isKnownToBeIterableAtMin(self, count):
        if False:
            while True:
                i = 10
        length = self.getIterationLength()
        return length is not None and length >= count

    def canPredictIterationValues(self):
        if False:
            i = 10
            return i + 15
        return self.isKnownToBeIterable(None)

class ExpressionDictShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact dictionary shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_dict

    @staticmethod
    def hasShapeDictionaryExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        return hasattr(the_empty_dict, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(the_empty_dict, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return False

    def extractUnhashableNodeType(self):
        if False:
            while True:
                i = 10
        return makeConstantReplacementNode(constant=dict, node=self, user_provided=False)

    @staticmethod
    def getExpressionDictInConstant(value):
        if False:
            while True:
                i = 10
        return None

class ExpressionListShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact list shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_list

    @staticmethod
    def hasShapeListExact():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(the_empty_list, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(the_empty_list, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return False

    def extractUnhashableNodeType(self):
        if False:
            while True:
                i = 10
        return makeConstantReplacementNode(constant=list, node=self, user_provided=False)

class ExpressionFrozensetShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact frozenset shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_frozenset

    @staticmethod
    def hasShapeListExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(the_empty_frozenset, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(the_empty_frozenset, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

class ExpressionSetShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact set shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_set

    @staticmethod
    def hasShapeSetExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(the_empty_set, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(the_empty_set, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            while True:
                i = 10
        return False

    def extractUnhashableNodeType(self):
        if False:
            while True:
                i = 10
        return makeConstantReplacementNode(constant=set, node=self, user_provided=False)

class ExpressionTupleShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact tuple shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_tuple

    @staticmethod
    def hasShapeTupleExact():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(the_empty_tuple, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(the_empty_tuple, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return None

class ExpressionBoolShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact bool shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_bool

    @staticmethod
    def hasShapeBoolExact():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(False, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(False, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            while True:
                i = 10
        return True

class ExpressionStrShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact str shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_str

    @staticmethod
    def hasShapeStrExact():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeStrOrUnicodeExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr('', attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            print('Hello World!')
        return getattr('', attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

class ExpressionBytesShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact bytes shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_bytes

    @staticmethod
    def hasShapeBytesExact():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        return hasattr(b'', attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(b'', attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

class ExpressionBytearrayShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact bytearray shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_bytearray

    @staticmethod
    def hasShapeBytearrayExact():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr(the_empty_bytearray, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            return 10
        return getattr(the_empty_bytearray, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return False

    def extractUnhashableNodeType(self):
        if False:
            for i in range(10):
                print('nop')
        return makeConstantReplacementNode(constant=bytearray, node=self, user_provided=False)

class ExpressionUnicodeShapeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact unicode shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_unicode

    @staticmethod
    def hasShapeUnicodeExact():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeStrOrUnicodeExact():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr(the_empty_unicode, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(the_empty_unicode, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True
if str is not bytes:
    ExpressionStrOrUnicodeExactMixin = ExpressionStrShapeExactMixin
else:

    class ExpressionStrOrUnicodeExactMixin(ExpressionIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
        """Mixin for nodes with str_or_unicode shape."""
        __slots__ = ()

        @staticmethod
        def getTypeShape():
            if False:
                print('Hello World!')
            return tshape_str_or_unicode

        @staticmethod
        def hasShapeStrOrUnicodeExact():
            if False:
                return 10
            return True

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                i = 10
                return i + 15
            return hasattr('', attribute_name) and hasattr(the_empty_unicode, attribute_name)

        @staticmethod
        def getKnownAttributeValue(attribute_name):
            if False:
                i = 10
                return i + 15
            return getattr('', attribute_name)

        @staticmethod
        def isKnownToBeHashable():
            if False:
                print('Hello World!')
            return True

class ExpressionFloatShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact float shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_float

    @staticmethod
    def hasShapeFloatExact():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr(0.0, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            while True:
                i = 10
        return getattr(0.0, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

class ExpressionIntShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact int shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_int

    @staticmethod
    def hasShapeIntExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(0, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            print('Hello World!')
        return getattr(0, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return True

class ExpressionLongShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact long shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            while True:
                i = 10
        return tshape_long

    @staticmethod
    def hasShapeLongExact():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(tshape_long.typical_value, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            print('Hello World!')
        return getattr(tshape_long.typical_value, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return True
if str is not bytes:
    ExpressionIntOrLongExactMixin = ExpressionIntShapeExactMixin
else:

    class ExpressionIntOrLongExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
        """Mixin for nodes with int_or_long shape."""
        __slots__ = ()

        @staticmethod
        def getTypeShape():
            if False:
                i = 10
                return i + 15
            return tshape_int_or_long

        @staticmethod
        def isKnownToHaveAttribute(attribute_name):
            if False:
                return 10
            return hasattr(0, attribute_name) and hasattr(tshape_long.typical_value, attribute_name)

        @staticmethod
        def getKnownAttributeValue(attribute_name):
            if False:
                for i in range(10):
                    print('nop')
            return getattr(0, attribute_name)

        @staticmethod
        def isKnownToBeHashable():
            if False:
                for i in range(10):
                    print('nop')
            return True

class ExpressionEllipsisShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact ellipsis shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_ellipsis

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            while True:
                i = 10
        return hasattr(Ellipsis, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(Ellipsis, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def getTruthValue():
        if False:
            i = 10
            return i + 15
        'Return known truth value.'
        return True

class ExpressionNoneShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact None shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_none

    @staticmethod
    def hasShapeNoneExact():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            i = 10
            return i + 15
        return hasattr(None, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(None, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return True

    @staticmethod
    def getTruthValue():
        if False:
            return 10
        'Return known truth value.'
        return False

class ExpressionComplexShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact complex shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_complex

    @staticmethod
    def hasShapeComplexExact():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        if attribute_name in ('imag', 'real'):
            return False
        return hasattr(0j, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(0j, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return True

class ExpressionSliceShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact complex shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_slice

    @staticmethod
    def hasShapeSliceExact():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            print('Hello World!')
        return hasattr(the_empty_slice, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(the_empty_slice, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return False

class ExpressionTypeShapeExactMixin(ExpressionNonIterableTypeShapeMixin, ExpressionSpecificExactMixinBase):
    """Mixin for nodes with exact 'type' shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_type

    @staticmethod
    def hasShapeTypeExact():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(type, attribute_name)

    @staticmethod
    def getKnownAttributeValue(attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(type, attribute_name)

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionStrDerivedShapeMixin(ExpressionSpecificDerivedMixinBase):
    """Mixin for nodes with str derived shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_str_derived

class ExpressionUnicodeDerivedShapeMixin(ExpressionSpecificDerivedMixinBase):
    """Mixin for nodes with unicode derived shape."""
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_unicode_derived
if str is not bytes:
    ExpressionStrOrUnicodeDerivedShapeMixin = ExpressionUnicodeDerivedShapeMixin
else:

    class ExpressionStrOrUnicodeDerivedShapeMixin(ExpressionSpecificDerivedMixinBase):
        """Mixin for nodes with str or unicode derived shape."""
        __slots__ = ()

        @staticmethod
        def getTypeShape():
            if False:
                while True:
                    i = 10
            return tshape_str_or_unicode_derived