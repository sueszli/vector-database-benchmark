""" Nodes that build and operate on dictionaries.

The "pair" is a sub-structure of the dictionary, representing a key/value pair
that is the child of the dictionary creation.

"""
from nuitka import Constants
from nuitka.specs.BuiltinDictOperationSpecs import dict_fromkeys_spec
from nuitka.specs.BuiltinParameterSpecs import extractBuiltinArgs
from .AttributeNodes import makeExpressionAttributeLookup
from .BuiltinOperationNodeBasesGenerated import ExpressionDictOperationClearBase, ExpressionDictOperationCopyBase, ExpressionDictOperationFromkeys2Base, ExpressionDictOperationFromkeys3Base, ExpressionDictOperationGet2Base, ExpressionDictOperationGet3Base, ExpressionDictOperationHaskeyBase, ExpressionDictOperationItemsBase, ExpressionDictOperationIteritemsBase, ExpressionDictOperationIterkeysBase, ExpressionDictOperationItervaluesBase, ExpressionDictOperationKeysBase, ExpressionDictOperationPop2Base, ExpressionDictOperationPop3Base, ExpressionDictOperationPopitemBase, ExpressionDictOperationSetdefault2Base, ExpressionDictOperationSetdefault3Base, ExpressionDictOperationUpdate2Base, ExpressionDictOperationUpdate3Base, ExpressionDictOperationValuesBase, ExpressionDictOperationViewitemsBase, ExpressionDictOperationViewkeysBase, ExpressionDictOperationViewvaluesBase
from .ChildrenHavingMixins import ChildrenExpressionDictOperationItemMixin, ChildrenExpressionDictOperationUpdatePairsMixin, ChildrenExpressionMakeDictMixin, ChildrenHavingKeyDictArgMixin
from .ConstantRefNodes import ExpressionConstantDictEmptyRef, makeConstantRefNode
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin, ExpressionDictShapeExactMixin
from .NodeBases import SideEffectsFromChildrenMixin
from .NodeMakingHelpers import makeConstantReplacementNode, makeRaiseExceptionExpressionFromTemplate, makeRaiseExceptionReplacementExpression, makeStatementOnlyNodesFromExpressions, wrapExpressionWithSideEffects
from .shapes.StandardShapes import tshape_iterator
from .StatementBasesGenerated import StatementDictOperationRemoveBase, StatementDictOperationSetBase, StatementDictOperationSetKeyValueBase, StatementDictOperationUpdateBase

def makeExpressionMakeDict(pairs, source_ref):
    if False:
        for i in range(10):
            print('nop')
    if pairs:
        return ExpressionMakeDict(pairs, source_ref)
    else:
        return ExpressionConstantDictEmptyRef(user_provided=False, source_ref=source_ref)

def makeExpressionMakeDictOrConstant(pairs, user_provided, source_ref):
    if False:
        i = 10
        return i + 15
    for pair in pairs:
        if not pair.isCompileTimeConstant() or pair.isKeyKnownToBeHashable() is not True:
            result = makeExpressionMakeDict(pairs, source_ref)
            break
    else:
        result = makeConstantRefNode(constant=Constants.createConstantDict(keys=[pair.getKeyCompileTimeConstant() for pair in pairs], values=[pair.getValueCompileTimeConstant() for pair in pairs]), user_provided=user_provided, source_ref=source_ref)
    if pairs:
        result.setCompatibleSourceReference(source_ref=pairs[-1].getCompatibleSourceReference())
    return result

class ExpressionMakeDictMixin(object):
    __slots__ = ()

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        for pair in self.subnode_pairs:
            if pair.mayRaiseException(exception_type):
                return True
        return False

    def isKnownToBeIterable(self, count):
        if False:
            for i in range(10):
                print('nop')
        return count is None or count == len(self.subnode_pairs)

    def getIterationLength(self):
        if False:
            return 10
        pair_count = len(self.subnode_pairs)
        if pair_count > 1:
            return None
        else:
            return pair_count

    @staticmethod
    def getIterationMinLength():
        if False:
            while True:
                i = 10
        return 1

    @staticmethod
    def canPredictIterationValues():
        if False:
            print('Hello World!')
        return True

    def getIterationValue(self, count):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_pairs[count].getKeyNode()

    def isMappingWithConstantStringKeys(self):
        if False:
            i = 10
            return i + 15
        return all((pair.isKeyExpressionConstantStrRef() for pair in self.subnode_pairs))

    def getMappingStringKeyPairs(self):
        if False:
            for i in range(10):
                print('nop')
        return [(pair.getKeyCompileTimeConstant(), pair.getValueNode()) for pair in self.subnode_pairs]

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        expressions = []
        for pair in self.subnode_pairs:
            expressions.extend(pair.extractSideEffects())
        result = makeStatementOnlyNodesFromExpressions(expressions=expressions)
        del self.parent
        return (result, 'new_statements', 'Removed sequence creation for unused sequence.')

    @staticmethod
    def computeExpressionIter1(iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (iter_node, None, None)

    def onContentEscapes(self, trace_collection):
        if False:
            print('Hello World!')
        for pair in self.subnode_pairs:
            pair.onContentEscapes(trace_collection)

class ExpressionMakeDict(ExpressionDictShapeExactMixin, SideEffectsFromChildrenMixin, ExpressionMakeDictMixin, ChildrenExpressionMakeDictMixin, ExpressionBase):
    kind = 'EXPRESSION_MAKE_DICT'
    named_children = ('pairs|tuple',)

    def __init__(self, pairs, source_ref):
        if False:
            while True:
                i = 10
        assert pairs
        ChildrenExpressionMakeDictMixin.__init__(self, pairs=pairs)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        pairs = self.subnode_pairs
        is_constant = True
        for pair in pairs:
            if pair.isKeyKnownToBeHashable() is False:
                key = pair.subnode_key
                side_effects = []
                for pair2 in pairs:
                    side_effects.extend(pair2.extractSideEffects())
                    if pair2 is pair:
                        break
                result = makeRaiseExceptionExpressionFromTemplate(exception_type='TypeError', template="unhashable type: '%s'", template_args=makeExpressionAttributeLookup(expression=key.extractUnhashableNodeType(), attribute_name='__name__', source_ref=key.source_ref), source_ref=key.source_ref)
                result = wrapExpressionWithSideEffects(side_effects=side_effects, old_node=key, new_node=result)
                return (result, 'new_raise', 'Dictionary key is known to not be hashable.')
            if is_constant:
                if not pair.isCompileTimeConstant():
                    is_constant = False
        if not is_constant:
            return (self, None, None)
        constant_value = Constants.createConstantDict(keys=[pair.getKeyCompileTimeConstant() for pair in pairs], values=[pair.getValueCompileTimeConstant() for pair in pairs])
        new_node = makeConstantReplacementNode(constant=constant_value, node=self, user_provided=True)
        return (new_node, 'new_constant', 'Created dictionary found to be constant.')

    @staticmethod
    def getTruthValue():
        if False:
            i = 10
            return i + 15
        return True

class StatementDictOperationSetMixin(object):
    __slots__ = ()

    def computeStatementOperation(self, trace_collection):
        if False:
            print('Hello World!')
        key = self.subnode_key
        if not key.isKnownToBeHashable():
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        key = self.subnode_key
        if not key.isKnownToBeHashable():
            return True
        if key.mayRaiseException(exception_type):
            return True
        value = self.subnode_value
        if value.mayRaiseException(exception_type):
            return True
        return False

    def mayRaiseExceptionOperation(self):
        if False:
            return 10
        return not self.subnode_key.isKnownToBeHashable()

class StatementDictOperationSet(StatementDictOperationSetMixin, StatementDictOperationSetBase):
    kind = 'STATEMENT_DICT_OPERATION_SET'
    named_children = ('value', 'dict_arg', 'key')
    auto_compute_handling = 'operation'

class StatementDictOperationSetKeyValue(StatementDictOperationSetMixin, StatementDictOperationSetKeyValueBase):
    kind = 'STATEMENT_DICT_OPERATION_SET_KEY_VALUE'
    named_children = ('value', 'dict_arg', 'key')
    auto_compute_handling = 'operation'

class StatementDictOperationRemove(StatementDictOperationRemoveBase):
    kind = 'STATEMENT_DICT_OPERATION_REMOVE'
    named_children = ('dict_arg', 'key')
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        key = self.subnode_key
        if not key.isKnownToBeHashable():
            return True
        if key.mayRaiseException(exception_type):
            return True
        return True

class ExpressionDictOperationPop2(ExpressionDictOperationPop2Base):
    """This operation represents d.pop(key), i.e. default None."""
    kind = 'EXPRESSION_DICT_OPERATION_POP2'
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, source_ref):
        if False:
            while True:
                i = 10
        ExpressionDictOperationPop2Base.__init__(self, dict_arg=dict_arg, key=key, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.pop', side_effects=(dict_arg, key))
        trace_collection.removeKnowledge(dict_arg)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            print('Hello World!')
        return True

class ExpressionDictOperationPop3(ExpressionDictOperationPop3Base):
    """This operation represents d.pop(key, default)."""
    kind = 'EXPRESSION_DICT_OPERATION_POP3'
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, default, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionDictOperationPop3Base.__init__(self, dict_arg=dict_arg, key=key, default=default, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.pop', side_effects=(dict_arg, key, self.subnode_default))
        trace_collection.removeKnowledge(dict_arg)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        if self.known_hashable_key is None:
            return True
        else:
            return self.subnode_dict_arg.mayRaiseException(exception_type) or self.subnode_key.mayRaiseException(exception_type) or self.subnode_default.mayRaiseException(exception_type)

class ExpressionDictOperationPopitem(ExpressionDictOperationPopitemBase):
    """This operation represents d.popitem()."""
    kind = 'EXPRESSION_DICT_OPERATION_POPITEM'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        dict_arg = self.subnode_dict_arg
        trace_collection.removeKnowledge(dict_arg)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return True

class ExpressionDictOperationSetdefault2(ExpressionDictOperationSetdefault2Base):
    """This operation represents d.setdefault(key), i.e. default None."""
    kind = 'EXPRESSION_DICT_OPERATION_SETDEFAULT2'
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, source_ref):
        if False:
            return 10
        ExpressionDictOperationSetdefault2Base.__init__(self, dict_arg=dict_arg, key=key, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.setdefault', side_effects=(dict_arg, key))
        trace_collection.removeKnowledge(dict_arg)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        if self.known_hashable_key is not True:
            return True
        else:
            return self.subnode_dict_arg.mayRaiseException(exception_type) or self.subnode_key.mayRaiseException(exception_type)

class ExpressionDictOperationSetdefault3(ExpressionDictOperationSetdefault3Base):
    """This operation represents d.setdefault(key, default)."""
    kind = 'EXPRESSION_DICT_OPERATION_SETDEFAULT3'
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, default, source_ref):
        if False:
            print('Hello World!')
        ExpressionDictOperationSetdefault3Base.__init__(self, dict_arg=dict_arg, key=key, default=default, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.setdefault', side_effects=(dict_arg, key, self.subnode_default))
        trace_collection.removeKnowledge(dict_arg)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        if self.known_hashable_key is not True:
            return True
        else:
            return self.subnode_dict_arg.mayRaiseException(exception_type) or self.subnode_key.mayRaiseException(exception_type) or self.subnode_default.mayRaiseException(exception_type)

class ExpressionDictOperationItem(ChildrenExpressionDictOperationItemMixin, ExpressionBase):
    """This operation represents d[key] with an exception for missing key."""
    kind = 'EXPRESSION_DICT_OPERATION_ITEM'
    named_children = ('dict_arg', 'key')

    def __init__(self, dict_arg, key, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenExpressionDictOperationItemMixin.__init__(self, dict_arg=dict_arg, key=key)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if dict_arg.isCompileTimeConstant() and key.isCompileTimeConstant():
            return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.getCompileTimeConstant()[dict_arg.getCompileTimeConstant()[key.getCompileTimeConstant()]], user_provided=dict_arg.user_provided, description='Dictionary item lookup with constant key.')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionDictOperationGet2(ExpressionDictOperationGet2Base):
    """This operation represents d.get(key) with no exception for missing key but None default."""
    kind = 'EXPRESSION_DICT_OPERATION_GET2'
    named_children = ('dict_arg', 'key')
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, source_ref):
        if False:
            print('Hello World!')
        ExpressionDictOperationGet2Base.__init__(self, dict_arg=dict_arg, key=key, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = self.subnode_key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.get', side_effects=(dict_arg, key))
        if dict_arg.isCompileTimeConstant() and key.isCompileTimeConstant():
            result = wrapExpressionWithSideEffects(new_node=makeConstantReplacementNode(constant=dict_arg.getCompileTimeConstant().get(key.getCompileTimeConstant()), node=self, user_provided=dict_arg.user_provided), old_node=self, side_effects=(dict_arg, key))
            return (result, 'new_expression', "Compile time computed 'dict.get' on constant argument.")
        if self.known_hashable_key is None:
            trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        if self.known_hashable_key is None:
            return True
        else:
            return self.subnode_dict_arg.mayRaiseException(exception_type) or self.subnode_key.mayRaiseException(exception_type)

    def mayHaveSideEffects(self):
        if False:
            while True:
                i = 10
        if self.known_hashable_key is None:
            return True
        else:
            return self.subnode_dict_arg.mayHaveSideEffects() or self.subnode_key.mayHaveSideEffects()

    def extractSideEffects(self):
        if False:
            return 10
        if self.known_hashable_key is None:
            return (self,)
        else:
            return self.subnode_dict_arg.extractSideEffects() + self.subnode_key.extractSideEffects()

class ExpressionDictOperationGet3(ExpressionDictOperationGet3Base):
    """This operation represents d.get(key, default) with no exception for missing key but default value."""
    kind = 'EXPRESSION_DICT_OPERATION_GET3'
    named_children = ('dict_arg', 'key', 'default')
    __slots__ = ('known_hashable_key',)

    def __init__(self, dict_arg, key, default, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionDictOperationGet3Base.__init__(self, dict_arg=dict_arg, key=key, default=default, source_ref=source_ref)
        self.known_hashable_key = None

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        dict_arg = self.subnode_dict_arg
        key = self.subnode_key
        if self.known_hashable_key is None:
            self.known_hashable_key = key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=key, operation='dict.get', side_effects=(dict_arg, key, self.subnode_default))
        if dict_arg.isCompileTimeConstant() and key.isCompileTimeConstant():
            dict_value = dict_arg.getCompileTimeConstant()
            key_value = key.getCompileTimeConstant()
            if key_value in dict_value:
                result = wrapExpressionWithSideEffects(new_node=makeConstantReplacementNode(constant=dict_value[key_value], node=self, user_provided=dict_arg.user_provided), old_node=self, side_effects=(dict_arg, key, self.subnode_default))
                description = "Compile time computed 'dict.get' on constant argument to not use default."
            else:
                result = wrapExpressionWithSideEffects(new_node=self.subnode_default, old_node=self, side_effects=(dict_arg, key))
                description = "Compile time computed 'dict.get' on constant argument to use default."
            return (result, 'new_expression', description)
        if self.known_hashable_key is None:
            trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        if self.known_hashable_key is None:
            return True
        else:
            return self.subnode_dict_arg.mayRaiseException(exception_type) or self.subnode_key.mayRaiseException(exception_type) or self.subnode_default.mayRaiseException(exception_type)

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        if self.known_hashable_key is None:
            return True
        else:
            return self.subnode_dict_arg.mayHaveSideEffects() or self.subnode_key.mayHaveSideEffects() or self.subnode_default.mayHaveSideEffects()

    def extractSideEffects(self):
        if False:
            print('Hello World!')
        if self.known_hashable_key is None:
            return (self,)
        else:
            return self.subnode_dict_arg.extractSideEffects() + self.subnode_key.extractSideEffects() + self.subnode_defaults.extractSideEffects()

class ExpressionDictOperationCopy(SideEffectsFromChildrenMixin, ExpressionDictOperationCopyBase):
    kind = 'EXPRESSION_DICT_OPERATION_COPY'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        dict_arg = self.subnode_dict_arg
        if dict_arg.isCompileTimeConstant():
            result = makeConstantReplacementNode(constant=dict_arg.getCompileTimeConstant().copy(), node=self, user_provided=dict_arg.user_provided)
            return (result, 'new_expression', "Compile time computed 'dict.copy' on constant argument.")
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationClear(ExpressionDictOperationClearBase):
    kind = 'EXPRESSION_DICT_OPERATION_CLEAR'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationKeys(SideEffectsFromChildrenMixin, ExpressionDictOperationKeysBase):
    kind = 'EXPRESSION_DICT_OPERATION_KEYS'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        dict_arg = self.subnode_dict_arg
        if dict_arg.isCompileTimeConstant():
            result = makeConstantReplacementNode(constant=dict_arg.getCompileTimeConstant().keys(), node=self, user_provided=dict_arg.user_provided)
            return (result, 'new_expression', "Compile time computed 'dict.keys' on constant argument.")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationViewkeys(SideEffectsFromChildrenMixin, ExpressionDictOperationViewkeysBase):
    kind = 'EXPRESSION_DICT_OPERATION_VIEWKEYS'

    def computeExpression(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationIterkeys(SideEffectsFromChildrenMixin, ExpressionDictOperationIterkeysBase):
    kind = 'EXPRESSION_DICT_OPERATION_ITERKEYS'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationValues(SideEffectsFromChildrenMixin, ExpressionDictOperationValuesBase):
    kind = 'EXPRESSION_DICT_OPERATION_VALUES'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        dict_arg = self.subnode_dict_arg
        if dict_arg.isCompileTimeConstant():
            result = makeConstantReplacementNode(constant=dict_arg.getCompileTimeConstant().values(), node=self, user_provided=dict_arg.user_provided)
            return (result, 'new_expression', "Compile time computed 'dict.values' on constant argument.")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationViewvalues(SideEffectsFromChildrenMixin, ExpressionDictOperationViewvaluesBase):
    kind = 'EXPRESSION_DICT_OPERATION_VIEWVALUES'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationItervalues(SideEffectsFromChildrenMixin, ExpressionDictOperationItervaluesBase):
    kind = 'EXPRESSION_DICT_OPERATION_ITERVALUES'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            return 10
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationItems(SideEffectsFromChildrenMixin, ExpressionDictOperationItemsBase):
    kind = 'EXPRESSION_DICT_OPERATION_ITEMS'

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        dict_arg = self.subnode_dict_arg
        if dict_arg.isCompileTimeConstant():
            result = makeConstantReplacementNode(constant=dict_arg.getCompileTimeConstant().items(), node=self, user_provided=dict_arg.user_provided)
            return (result, 'new_expression', "Compile time computed 'dict.items' on constant argument.")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationIteritems(SideEffectsFromChildrenMixin, ExpressionDictOperationIteritemsBase):
    kind = 'EXPRESSION_DICT_OPERATION_ITERITEMS'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationViewitems(SideEffectsFromChildrenMixin, ExpressionDictOperationViewitemsBase):
    kind = 'EXPRESSION_DICT_OPERATION_VIEWITEMS'

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_iterator

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_dict_arg.mayRaiseException(exception_type)

class ExpressionDictOperationUpdate2(ExpressionDictOperationUpdate2Base):
    """This operation represents d.update(iterable)."""
    kind = 'EXPRESSION_DICT_OPERATION_UPDATE2'

    def __init__(self, dict_arg, iterable, source_ref):
        if False:
            return 10
        if type(iterable) is tuple:
            (iterable,) = iterable
        ExpressionDictOperationUpdate2Base.__init__(self, dict_arg=dict_arg, iterable=iterable, source_ref=source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        trace_collection.removeKnowledge(self.subnode_iterable)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return True

def makeExpressionDictOperationUpdate3(dict_arg, iterable, pairs, source_ref):
    if False:
        print('Hello World!')
    if type(iterable) is tuple:
        if not iterable:
            iterable = None
        else:
            (iterable,) = iterable
    if iterable is not None:
        return ExpressionDictOperationUpdate3(dict_arg, iterable, pairs, source_ref)
    else:
        return ExpressionDictOperationUpdatePairs(dict_arg, pairs, source_ref)

class ExpressionDictOperationUpdate3(ExpressionDictOperationUpdate3Base):
    """This operation represents d.update(iterable, **pairs)."""
    kind = 'EXPRESSION_DICT_OPERATION_UPDATE3'

    def __init__(self, dict_arg, iterable, pairs, source_ref):
        if False:
            print('Hello World!')
        ExpressionDictOperationUpdate3Base.__init__(self, dict_arg=dict_arg, iterable=iterable, pairs=pairs, source_ref=source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        trace_collection.removeKnowledge(self.subnode_iterable)
        for pair in self.subnode_pairs:
            trace_collection.removeKnowledge(pair)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        return True

class ExpressionDictOperationUpdatePairs(ChildrenExpressionDictOperationUpdatePairsMixin, ExpressionBase):
    """This operation represents d.update(iterable, **pairs)."""
    kind = 'EXPRESSION_DICT_OPERATION_UPDATE_PAIRS'
    named_children = ('dict_arg', 'pairs|tuple')

    def __init__(self, dict_arg, pairs, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenExpressionDictOperationUpdatePairsMixin.__init__(self, dict_arg=dict_arg, pairs=pairs)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        for pair in self.subnode_pairs:
            trace_collection.removeKnowledge(pair)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return True

class StatementDictOperationUpdate(StatementDictOperationUpdateBase):
    """Update dict value.

    This is mainly used for re-formulations, where a dictionary
    update will be performed on what is known not to be a
    general mapping.
    """
    kind = 'STATEMENT_DICT_OPERATION_UPDATE'
    named_children = ('dict_arg', 'value')
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.removeKnowledge(self.subnode_dict_arg)
        trace_collection.removeKnowledge(self.subnode_value)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

def makeUnhashableExceptionReplacementExpression(node, key, side_effects, operation):
    if False:
        for i in range(10):
            print('nop')
    unhashable_type_name = key.extractUnhashableNodeType().getCompileTimeConstant().__name__
    result = makeRaiseExceptionReplacementExpression(expression=node, exception_type='TypeError', exception_value="unhashable type: '%s'" % unhashable_type_name)
    result = wrapExpressionWithSideEffects(side_effects=side_effects, old_node=node, new_node=result)
    return (result, 'new_raise', "Dictionary operation '%s' with key of type '%s' that is known to not be hashable." % (operation, unhashable_type_name))

class ExpressionDictOperationInNotInUncertainMixin(object):
    __slots__ = ()

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        if self.known_hashable_key is None:
            self.known_hashable_key = self.subnode_key.isKnownToBeHashable()
            if self.known_hashable_key is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeUnhashableExceptionReplacementExpression(node=self, key=self.subnode_key, operation='in (dict)', side_effects=self.getVisitableNodes())
        if self.known_hashable_key is None:
            trace_collection.onExceptionRaiseExit(BaseException)
        if self.subnode_key.isCompileTimeConstant():
            truth_value = self.subnode_dict_arg.getExpressionDictInConstant(self.subnode_key.getCompileTimeConstant())
            if truth_value is not None:
                if 'NOT' in self.kind:
                    truth_value = not truth_value
                result = wrapExpressionWithSideEffects(new_node=makeConstantReplacementNode(constant=truth_value, node=self, user_provided=True), old_node=self, side_effects=self.getVisitableNodes())
                return (result, 'new_constant', "Predicted dict 'in' truth value")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_key.mayRaiseException(exception_type) or self.subnode_dict_arg.mayRaiseException(exception_type) or self.known_hashable_key is not True

    def mayHaveSideEffects(self):
        if False:
            print('Hello World!')
        return self.mayRaiseException(BaseException)

    def extractSideEffects(self):
        if False:
            i = 10
            return i + 15
        if self.known_hashable_key is not True:
            return (self,)
        else:
            result = []
            for child in self.getVisitableNodes():
                result.extend(child.extractSideEffects())
            return tuple(result)

class ExpressionDictOperationInNotInUncertainBase(ExpressionDictOperationInNotInUncertainMixin, ExpressionBoolShapeExactMixin, ChildrenHavingKeyDictArgMixin, ExpressionBase):
    named_children = ('key', 'dict_arg')
    __slots__ = ('known_hashable_key',)

    def __init__(self, key, dict_arg, source_ref):
        if False:
            return 10
        ChildrenHavingKeyDictArgMixin.__init__(self, key=key, dict_arg=dict_arg)
        ExpressionBase.__init__(self, source_ref)
        self.known_hashable_key = None

class ExpressionDictOperationIn(ExpressionDictOperationInNotInUncertainBase):
    kind = 'EXPRESSION_DICT_OPERATION_IN'

class ExpressionDictOperationNotIn(ExpressionDictOperationInNotInUncertainBase):
    kind = 'EXPRESSION_DICT_OPERATION_NOT_IN'

class ExpressionDictOperationHaskey(ExpressionDictOperationInNotInUncertainMixin, ExpressionDictOperationHaskeyBase):
    kind = 'EXPRESSION_DICT_OPERATION_HASKEY'
    named_children = ('dict_arg', 'key')
    __slots__ = ('known_hashable_key',)

    def __init__(self, key, dict_arg, source_ref):
        if False:
            while True:
                i = 10
        ExpressionDictOperationHaskeyBase.__init__(self, key=key, dict_arg=dict_arg, source_ref=source_ref)
        self.known_hashable_key = None

class ExpressionDictOperationFromkeys2(ExpressionDictOperationFromkeys2Base):
    kind = 'EXPRESSION_DICT_OPERATION_FROMKEYS2'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.mayRaiseExceptionOperation():
            trace_collection.onExceptionRaiseExit(BaseException)
        if self.subnode_iterable.isCompileTimeConstant():
            if None is not self.subnode_iterable.getIterationLength() < 256:
                return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : dict.fromkeys(self.subnode_iterable.getCompileTimeConstant()), description="Computed 'dict.fromkeys' with constant value.")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_iterable.mayRaiseException(exception_type) or self.mayRaiseExceptionOperation()

    def mayRaiseExceptionOperation(self):
        if False:
            return 10
        return None is not self.subnode_iterable.getIterationLength() < 256

class ExpressionDictOperationFromkeys3(ExpressionDictOperationFromkeys3Base):
    kind = 'EXPRESSION_DICT_OPERATION_FROMKEYS3'

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        if self.mayRaiseExceptionOperation():
            trace_collection.onExceptionRaiseExit(BaseException)
        if self.subnode_iterable.isCompileTimeConstant() and self.subnode_value.isCompileTimeConstant():
            if None is not self.subnode_iterable.getIterationLength() < 256:
                return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : dict.fromkeys(self.subnode_iterable.getCompileTimeConstant(), self.subnode_value.getCompileTimeConstant()), description="Computed 'dict.fromkeys' with constant values.")
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return self.subnode_iterable.mayRaiseException(exception_type) or self.subnode_value.mayRaiseException(exception_type) or self.mayRaiseExceptionOperation()

    def mayRaiseExceptionOperation(self):
        if False:
            while True:
                i = 10
        return None is not self.subnode_iterable.getIterationLength() < 256

class ExpressionDictOperationFromkeysRef(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_DICT_OPERATION_FROMKEYS_REF'

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(BaseException)

        def wrapExpressionDictOperationFromkeys(iterable, value, source_ref):
            if False:
                i = 10
                return i + 15
            if value is None:
                return ExpressionDictOperationFromkeys2(iterable=iterable, source_ref=source_ref)
            else:
                return ExpressionDictOperationFromkeys3(iterable=iterable, value=value, source_ref=source_ref)
        result = extractBuiltinArgs(node=call_node, builtin_class=wrapExpressionDictOperationFromkeys, builtin_spec=dict_fromkeys_spec)
        return trace_collection.computedExpressionResult(expression=result, change_tags='new_expression', change_desc="Call to 'dict.fromkeys' recognized.")