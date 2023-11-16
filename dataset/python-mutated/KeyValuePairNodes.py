""" Dictionary pairs, for use in dictionary building, calls, etc.

These represent the a=b part, as they occur in calls, and dictionary
values, but they do not form a dictionary. As a sequence, they can
have order.

"""
from abc import abstractmethod
from nuitka.PythonVersions import python_version
from .BuiltinHashNodes import ExpressionBuiltinHash
from .ChildrenHavingMixins import ChildHavingValueMixin, ChildrenHavingKeyValueMixin, ChildrenHavingValueKeyMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .NodeBases import SideEffectsFromChildrenMixin

class ExpressionKeyValuePairMixin(object):
    __slots__ = ()

    @staticmethod
    def isExpressionKeyValuePair():
        if False:
            while True:
                i = 10
        return True

    @abstractmethod
    def mayKeyRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def mayValueRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def getKeyNode(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def getValueNode(self):
        if False:
            while True:
                i = 10
        pass

class ExpressionKeyValuePairNonConstantMixin(ExpressionKeyValuePairMixin):
    __slots__ = ()

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        key = self.subnode_key
        hashable = key.isKnownToBeHashable()
        if not hashable:
            trace_collection.onExceptionRaiseExit(TypeError)
        if hashable is False:
            pass
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        key = self.subnode_key
        return key.mayRaiseException(exception_type) or key.isKnownToBeHashable() is not True or self.subnode_value.mayRaiseException(exception_type)

    def isKeyKnownToBeHashable(self):
        if False:
            print('Hello World!')
        return self.subnode_key.isKnownToBeHashable()

    def extractSideEffects(self):
        if False:
            return 10
        if self.subnode_key.isKnownToBeHashable() is True:
            key_part = self.subnode_key.extractSideEffects()
        else:
            key_part = (ExpressionBuiltinHash(value=self.subnode_key, source_ref=self.subnode_key.source_ref),)
        if python_version < 848:
            return self.subnode_value.extractSideEffects() + key_part
        else:
            return key_part + self.subnode_value.extractSideEffects()

    def onContentEscapes(self, trace_collection):
        if False:
            i = 10
            return i + 15
        self.subnode_key.onContentEscapes(trace_collection)
        self.subnode_value.onContentEscapes(trace_collection)

    def isCompileTimeConstant(self):
        if False:
            return 10
        return self.subnode_value.isCompileTimeConstant() and self.subnode_key.isCompileTimeConstant()

    def isKeyExpressionConstantStrRef(self):
        if False:
            i = 10
            return i + 15
        return self.subnode_key.isKeyExpressionConstantStrRef()

    def getKeyCompileTimeConstant(self):
        if False:
            while True:
                i = 10
        return self.subnode_key.getCompileTimeConstant()

    def getValueCompileTimeConstant(self):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_value.getCompileTimeConstant()

    def mayKeyRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_key.mayRaiseException(exception_type)

    def mayValueRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.mayRaiseException(exception_type)

    def getKeyNode(self):
        if False:
            print('Hello World!')
        return self.subnode_key

    def getValueNode(self):
        if False:
            while True:
                i = 10
        return self.subnode_value

    def getCompatibleSourceReference(self):
        if False:
            return 10
        return self.subnode_value.getCompatibleSourceReference()

class ExpressionKeyValuePairOld(ExpressionKeyValuePairNonConstantMixin, SideEffectsFromChildrenMixin, ChildrenHavingValueKeyMixin, ExpressionBase):
    kind = 'EXPRESSION_KEY_VALUE_PAIR_OLD'
    python_version_spec = '< 0x350'
    named_children = ('value', 'key')

    def __init__(self, value, key, source_ref):
        if False:
            return 10
        ChildrenHavingValueKeyMixin.__init__(self, value=value, key=key)
        ExpressionBase.__init__(self, source_ref)

class ExpressionKeyValuePairNew(ExpressionKeyValuePairNonConstantMixin, SideEffectsFromChildrenMixin, ChildrenHavingKeyValueMixin, ExpressionBase):
    kind = 'EXPRESSION_KEY_VALUE_PAIR_NEW'
    python_version_spec = '>= 0x350'
    named_children = ('key', 'value')

    def __init__(self, key, value, source_ref):
        if False:
            return 10
        ChildrenHavingKeyValueMixin.__init__(self, key=key, value=value)
        ExpressionBase.__init__(self, source_ref)
if python_version < 848:
    ExpressionKeyValuePair = ExpressionKeyValuePairOld
else:
    ExpressionKeyValuePair = ExpressionKeyValuePairNew

class ExpressionKeyValuePairConstantKey(ExpressionKeyValuePairMixin, SideEffectsFromChildrenMixin, ChildHavingValueMixin, ExpressionBase):
    kind = 'EXPRESSION_KEY_VALUE_PAIR_CONSTANT_KEY'
    named_children = ('value',)
    __slots__ = ('key',)

    def __init__(self, key, value, source_ref):
        if False:
            i = 10
            return i + 15
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)
        self.key = key

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'key': self.key}

    @staticmethod
    def isKeyKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_value.mayRaiseException(exception_type)

    def extractSideEffects(self):
        if False:
            while True:
                i = 10
        return self.subnode_value.extractSideEffects()

    def onContentEscapes(self, trace_collection):
        if False:
            return 10
        self.subnode_value.onContentEscapes(trace_collection)

    def isCompileTimeConstant(self):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.isCompileTimeConstant()

    def isKeyExpressionConstantStrRef(self):
        if False:
            i = 10
            return i + 15
        return type(self.key) is str

    def getKeyCompileTimeConstant(self):
        if False:
            print('Hello World!')
        return self.key

    def getValueCompileTimeConstant(self):
        if False:
            while True:
                i = 10
        return self.subnode_value.getCompileTimeConstant()

    @staticmethod
    def mayKeyRaiseException(exception_type):
        if False:
            print('Hello World!')
        return False

    def mayValueRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return self.subnode_value.mayRaiseException(exception_type)

    def getKeyNode(self):
        if False:
            return 10
        return makeConstantRefNode(constant=self.key, source_ref=self.source_ref)

    def getValueNode(self):
        if False:
            while True:
                i = 10
        return self.subnode_value

    def getCompatibleSourceReference(self):
        if False:
            print('Hello World!')
        return self.subnode_value.getCompatibleSourceReference()

class ExpressionKeyValuePairConstantKeyValue(ExpressionKeyValuePairMixin, ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_KEY_VALUE_PAIR_CONSTANT_KEY_VALUE'
    __slots__ = ('key', 'value')

    def __init__(self, key, value, source_ref):
        if False:
            return 10
        self.key = key
        self.value = value
        ExpressionBase.__init__(self, source_ref)

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.key
        del self.value

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'key': self.key, 'value': self.value}

    @staticmethod
    def isKeyKnownToBeHashable():
        if False:
            while True:
                i = 10
        return True

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    @staticmethod
    def isCompileTimeConstant():
        if False:
            i = 10
            return i + 15
        return True

    def isKeyExpressionConstantStrRef(self):
        if False:
            return 10
        return type(self.key) is str

    def getKeyCompileTimeConstant(self):
        if False:
            print('Hello World!')
        return self.key

    def getValueCompileTimeConstant(self):
        if False:
            i = 10
            return i + 15
        return self.value

    @staticmethod
    def mayKeyRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

    def mayValueRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_value.mayRaiseException(exception_type)

    def getKeyNode(self):
        if False:
            print('Hello World!')
        return makeConstantRefNode(constant=self.key, source_ref=self.source_ref, user_provided=True)

    def getValueNode(self):
        if False:
            print('Hello World!')
        return makeConstantRefNode(constant=self.value, source_ref=self.source_ref, user_provided=True)

def makeExpressionPairs(keys, values):
    if False:
        return 10
    assert len(keys) == len(values)
    return tuple((makeExpressionKeyValuePair(key=key, value=value) for (key, value) in zip(keys, values)))

def makeExpressionKeyValuePair(key, value):
    if False:
        while True:
            i = 10
    if key.isCompileTimeConstant() and key.isKnownToBeHashable():
        return makeExpressionKeyValuePairConstantKey(key=key.getCompileTimeConstant(), value=value)
    else:
        return ExpressionKeyValuePair(key=key, value=value, source_ref=value.getSourceReference())

def makeExpressionKeyValuePairConstantKey(key, value):
    if False:
        return 10
    if value.isCompileTimeConstant():
        return ExpressionKeyValuePairConstantKeyValue(key=key, value=value.getCompileTimeConstant(), source_ref=value.getSourceReference())
    else:
        return ExpressionKeyValuePairConstantKey(key=key, value=value, source_ref=value.getSourceReference())

def makeKeyValuePairExpressionsFromKwArgs(pairs):
    if False:
        i = 10
        return i + 15
    return tuple((makeExpressionKeyValuePairConstantKey(key=key, value=value) for (key, value) in pairs))