""" Node for the calls to the 'next' built-in and unpacking special next.

    The unpacking next has only special that it raises a different exception
    text, explaining things about its context.
"""
from .ChildrenHavingMixins import ChildrenHavingIteratorDefaultMixin
from .ExpressionBases import ExpressionBase, ExpressionBuiltinSingleArgBase

class ExpressionBuiltinNext1(ExpressionBuiltinSingleArgBase):
    __slots__ = ('may_raise',)
    kind = 'EXPRESSION_BUILTIN_NEXT1'

    def __init__(self, value, source_ref):
        if False:
            print('Hello World!')
        ExpressionBuiltinSingleArgBase.__init__(self, value=value, source_ref=source_ref)
        self.may_raise = True

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        (self.may_raise, result) = self.subnode_value.computeExpressionNext1(next_node=self, trace_collection=trace_collection)
        return result

    def mayRaiseExceptionOperation(self):
        if False:
            i = 10
            return i + 15
        return self.may_raise

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.may_raise or self.subnode_value.mayRaiseException(exception_type)

class ExpressionSpecialUnpack(ExpressionBuiltinNext1):
    __slots__ = ('count', 'expected', 'starred')
    kind = 'EXPRESSION_SPECIAL_UNPACK'

    def __init__(self, value, count, expected, starred, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionBuiltinNext1.__init__(self, value=value, source_ref=source_ref)
        self.count = int(count)
        self.expected = int(expected)
        self.starred = starred

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        result = ExpressionBuiltinNext1.getDetails(self)
        result['count'] = self.getCount()
        result['expected'] = self.getExpected()
        result['starred'] = self.getStarred()
        return result

    def getCount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.count

    def getExpected(self):
        if False:
            print('Hello World!')
        return self.expected

    def getStarred(self):
        if False:
            for i in range(10):
                print('nop')
        return self.starred

class ExpressionBuiltinNext2(ChildrenHavingIteratorDefaultMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_NEXT2'
    named_children = ('iterator', 'default')

    def __init__(self, iterator, default, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingIteratorDefaultMixin.__init__(self, iterator=iterator, default=default)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)