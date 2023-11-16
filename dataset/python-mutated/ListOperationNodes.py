""" Nodes that build and operate on lists."""
from .ChildrenHavingMixins import ChildHavingListArgMixin, ChildrenExpressionListOperationExtendMixin, ChildrenHavingListArgIndexItemMixin, ChildrenHavingListArgIndexMixin, ChildrenHavingListArgKeyMixin, ChildrenHavingListArgKeyOptionalReverseMixin, ChildrenHavingListArgValueMixin, ChildrenHavingListArgValueStartMixin, ChildrenHavingListArgValueStartStopMixin
from .ExpressionBases import ExpressionBase
from .ExpressionBasesGenerated import ExpressionListOperationAppendBase, ExpressionListOperationClearBase, ExpressionListOperationCountBase, ExpressionListOperationReverseBase
from .ExpressionShapeMixins import ExpressionIntOrLongExactMixin
from .NodeBases import SideEffectsFromChildrenMixin

class ExpressionListOperationAppend(ExpressionListOperationAppendBase):
    """This operation represents l.append(object)."""
    kind = 'EXPRESSION_LIST_OPERATION_APPEND'
    named_children = ('list_arg', 'item')
    auto_compute_handling = 'no_raise'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.removeKnowledge(self.subnode_list_arg)
        self.subnode_item.onContentEscapes(trace_collection)
        return (self, None, None)

class ExpressionListOperationClear(ExpressionListOperationClearBase):
    """This operation represents l.clear()."""
    kind = 'EXPRESSION_LIST_OPERATION_CLEAR'
    named_children = ('list_arg',)
    auto_compute_handling = 'no_raise'

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.removeKnowledge(self.subnode_list_arg)
        return (self, None, None)

class ExpressionListOperationCopy(SideEffectsFromChildrenMixin, ChildHavingListArgMixin, ExpressionBase):
    """This operation represents l.copy()."""
    kind = 'EXPRESSION_LIST_OPERATION_COPY'
    named_children = ('list_arg',)

    def __init__(self, list_arg, source_ref):
        if False:
            i = 10
            return i + 15
        ChildHavingListArgMixin.__init__(self, list_arg=list_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        self.subnode_list_arg.onContentEscapes(trace_collection)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionListOperationCount(SideEffectsFromChildrenMixin, ExpressionIntOrLongExactMixin, ExpressionListOperationCountBase):
    """This operation represents l.count()."""
    kind = 'EXPRESSION_LIST_OPERATION_COUNT'
    named_children = ('list_arg', 'value')
    auto_compute_handling = 'final,no_raise'

class ExpressionListOperationExtend(ChildrenExpressionListOperationExtendMixin, ExpressionBase):
    kind = 'EXPRESSION_LIST_OPERATION_EXTEND'
    named_children = ('list_arg', 'value')

    def __init__(self, list_arg, value, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenExpressionListOperationExtendMixin.__init__(self, list_arg=list_arg, value=value)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.removeKnowledge(self.subnode_list_arg)
        self.subnode_value.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionListOperationExtendForUnpack(ExpressionListOperationExtend):
    kind = 'EXPRESSION_LIST_OPERATION_EXTEND_FOR_UNPACK'

class ExpressionListOperationIndex2(ExpressionIntOrLongExactMixin, ChildrenHavingListArgValueMixin, ExpressionBase):
    """This operation represents l.index(value)."""
    kind = 'EXPRESSION_LIST_OPERATION_INDEX2'
    named_children = ('list_arg', 'value')

    def __init__(self, list_arg, value, source_ref):
        if False:
            return 10
        ChildrenHavingListArgValueMixin.__init__(self, list_arg=list_arg, value=value)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        self.subnode_value.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return True

class ExpressionListOperationIndex3(ExpressionIntOrLongExactMixin, ChildrenHavingListArgValueStartMixin, ExpressionBase):
    """This operation represents l.index(value, start)."""
    kind = 'EXPRESSION_LIST_OPERATION_INDEX3'
    named_children = ('list_arg', 'value', 'start')

    def __init__(self, list_arg, value, start, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingListArgValueStartMixin.__init__(self, list_arg=list_arg, value=value, start=start)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        self.subnode_value.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            while True:
                i = 10
        return True

class ExpressionListOperationIndex4(ExpressionIntOrLongExactMixin, ChildrenHavingListArgValueStartStopMixin, ExpressionBase):
    """This operation represents l.index(value, start, stop)."""
    kind = 'EXPRESSION_LIST_OPERATION_INDEX4'
    named_children = ('list_arg', 'value', 'start', 'stop')

    def __init__(self, list_arg, value, start, stop, source_ref):
        if False:
            return 10
        ChildrenHavingListArgValueStartStopMixin.__init__(self, list_arg=list_arg, value=value, start=start, stop=stop)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        self.subnode_value.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            i = 10
            return i + 15
        return True

class ExpressionListOperationInsert(ChildrenHavingListArgIndexItemMixin, ExpressionBase):
    """This operation represents l.insert(index, item)."""
    kind = 'EXPRESSION_LIST_OPERATION_INSERT'
    named_children = ('list_arg', 'index', 'item')

    def __init__(self, list_arg, index, item, source_ref):
        if False:
            return 10
        ChildrenHavingListArgIndexItemMixin.__init__(self, list_arg=list_arg, index=index, item=item)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.removeKnowledge(self.subnode_list_arg)
        self.subnode_item.onContentEscapes(trace_collection)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_list_arg.mayRaiseException(exception_type)

    def mayRaiseExceptionOperation(self):
        if False:
            i = 10
            return i + 15
        return self.subnode_item.isExpressionConstantRef() and self.subnode_item.isIndexConstant()

class ExpressionListOperationPop1(ChildHavingListArgMixin, ExpressionBase):
    """This operation represents l.pop()."""
    kind = 'EXPRESSION_LIST_OPERATION_POP1'
    named_children = ('list_arg',)

    def __init__(self, list_arg, source_ref):
        if False:
            i = 10
            return i + 15
        ChildHavingListArgMixin.__init__(self, list_arg=list_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        non_empty = self.subnode_list_arg.isKnownToBeIterableAtMin(1)
        trace_collection.removeKnowledge(self.subnode_list_arg)
        if not non_empty:
            trace_collection.onExceptionRaiseExit(IndexError)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            print('Hello World!')
        return True

class ExpressionListOperationPop2(ChildrenHavingListArgIndexMixin, ExpressionBase):
    """This operation represents l.pop(index)."""
    kind = 'EXPRESSION_LIST_OPERATION_POP2'
    named_children = ('list_arg', 'index')

    def __init__(self, list_arg, index, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingListArgIndexMixin.__init__(self, list_arg=list_arg, index=index)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.removeKnowledge(self.subnode_list_arg)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return True

class ExpressionListOperationRemove(ChildrenHavingListArgValueMixin, ExpressionBase):
    """This operation represents l.remove(value)."""
    kind = 'EXPRESSION_LIST_OPERATION_REMOVE'
    named_children = ('list_arg', 'value')

    def __init__(self, list_arg, value, source_ref):
        if False:
            while True:
                i = 10
        ChildrenHavingListArgValueMixin.__init__(self, list_arg=list_arg, value=value)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.removeKnowledge(self.subnode_list_arg)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            print('Hello World!')
        return True

class ExpressionListOperationReverse(ExpressionListOperationReverseBase):
    """This operation represents l.reverse()."""
    kind = 'EXPRESSION_LIST_OPERATION_REVERSE'
    named_children = ('list_arg',)
    auto_compute_handling = 'no_raise'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.removeKnowledge(self.subnode_list_arg)
        return (self, None, None)

class ExpressionListOperationSort1(ChildHavingListArgMixin, ExpressionBase):
    """This operation represents l.sort()."""
    kind = 'EXPRESSION_LIST_OPERATION_SORT1'
    named_children = ('list_arg',)

    def __init__(self, list_arg, source_ref):
        if False:
            return 10
        ChildHavingListArgMixin.__init__(self, list_arg=list_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.removeKnowledge(self.subnode_list_arg)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            while True:
                i = 10
        return True

class ExpressionListOperationSort2(ChildrenHavingListArgKeyMixin, ExpressionBase):
    """This operation represents l.sort(key)."""
    kind = 'EXPRESSION_LIST_OPERATION_SORT2'
    named_children = ('list_arg', 'key')

    def __init__(self, list_arg, key, source_ref):
        if False:
            print('Hello World!')
        ChildrenHavingListArgKeyMixin.__init__(self, list_arg=list_arg, key=key)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.removeKnowledge(self.subnode_list_arg)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return True

class ExpressionListOperationSort3(ChildrenHavingListArgKeyOptionalReverseMixin, ExpressionBase):
    """This operation represents l.sort(key, reversed)."""
    kind = 'EXPRESSION_LIST_OPERATION_SORT3'
    named_children = ('list_arg', 'key|optional', 'reverse')

    def __init__(self, list_arg, key, reverse, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenHavingListArgKeyOptionalReverseMixin.__init__(self, list_arg=list_arg, key=key, reverse=reverse)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.removeKnowledge(self.subnode_list_arg)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            while True:
                i = 10
        return True