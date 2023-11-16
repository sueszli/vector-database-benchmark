""" Mixins to use for composing type shapes.

"""
from .ControlFlowDescriptions import ControlFlowDescriptionElementBasedEscape, ControlFlowDescriptionNoEscape

class ShapeContainerMixin(object):
    """Mixin that defines the common container shape functions."""
    __slots__ = ()

    @staticmethod
    def hasShapeSlotBool():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotLen():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotContains():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotIter():
        if False:
            return 10
        return True

    @staticmethod
    def hasShapeSlotNext():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def hasShapeModule():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            for i in range(10):
                print('nop')
        return ControlFlowDescriptionElementBasedEscape

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            while True:
                i = 10
        return True

class ShapeContainerMutableMixin(ShapeContainerMixin):
    __slots__ = ()

    @staticmethod
    def hasShapeSlotHash():
        if False:
            for i in range(10):
                print('nop')
        return False

class ShapeContainerImmutableMixin(ShapeContainerMixin):
    __slots__ = ()

    @staticmethod
    def hasShapeSlotHash():
        if False:
            while True:
                i = 10
        return True

class ShapeNotContainerMixin(object):
    __slots__ = ()

    @staticmethod
    def hasShapeSlotBool():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotLen():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeSlotIter():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeSlotNext():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeSlotContains():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeModule():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            for i in range(10):
                print('nop')
        return ControlFlowDescriptionNoEscape

class ShapeNotNumberMixin(object):
    """Mixin that defines the number slots to be set."""
    __slots__ = ()

    @staticmethod
    def hasShapeSlotBool():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotAbs():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def hasShapeSlotInt():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def hasShapeSlotLong():
        if False:
            return 10
        return False

    @staticmethod
    def hasShapeSlotFloat():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeSlotComplex():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def hasShapeModule():
        if False:
            for i in range(10):
                print('nop')
        return False

class ShapeNumberMixin(object):
    """Mixin that defines the number slots to be set."""
    __slots__ = ()

    @staticmethod
    def hasShapeSlotBool():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotAbs():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotInt():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotLong():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def hasShapeSlotFloat():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def hasShapeSlotComplex():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def hasShapeSlotHash():
        if False:
            return 10
        return True

    @staticmethod
    def hasShapeModule():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            return 10
        return True

    @staticmethod
    def getOperationUnaryReprEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionNoEscape

class ShapeIteratorMixin(ShapeNotContainerMixin):
    __slots__ = ()

    @staticmethod
    def isShapeIterator():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def getIteratedShape():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def hasShapeSlotIter():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotNext():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasShapeSlotNextCode():
        if False:
            i = 10
            return i + 15
        'Does next execute code, i.e. control flow escaped.\n\n        For most known iterators that is not the case, only the generic\n        tshape_iterator needs to say "do not know", aka None.\n        '
        return False

    @staticmethod
    def hasShapeSlotContains():
        if False:
            return 10
        return True

    @staticmethod
    def hasShapeSlotHash():
        if False:
            for i in range(10):
                print('nop')
        return True