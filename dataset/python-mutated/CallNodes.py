""" Call node

Function calls and generally calling expressions are the same thing. This is
very important, because it allows to predict most things, and avoid expensive
operations like parameter parsing at run time.

There will be a method "computeExpressionCall" to aid predicting them in other
nodes.
"""
from .ChildrenHavingMixins import ChildrenExpressionCallEmptyMixin, ChildrenExpressionCallKeywordsOnlyMixin, ChildrenExpressionCallMixin, ChildrenExpressionCallNoKeywordsMixin
from .ExpressionBases import ExpressionBase

class ExpressionCallMixin(object):
    __slots__ = ()

    @staticmethod
    def isExpressionCall():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            print('Hello World!')
        return True

class ExpressionCall(ExpressionCallMixin, ChildrenExpressionCallMixin, ExpressionBase):
    kind = 'EXPRESSION_CALL'
    named_children = ('called|setter', 'args', 'kwargs')

    def __init__(self, called, args, kwargs, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenExpressionCallMixin.__init__(self, called=called, args=args, kwargs=kwargs)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        called = self.subnode_called
        return called.computeExpressionCall(call_node=self, call_args=self.subnode_args, call_kw=self.subnode_kwargs, trace_collection=trace_collection)

    def extractSideEffectsPreCall(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.subnode_args
        kwargs = self.subnode_kwargs
        return args.extractSideEffects() + kwargs.extractSideEffects()

    def onContentEscapes(self, trace_collection):
        if False:
            print('Hello World!')
        self.subnode_called.onContentEscapes(trace_collection)
        self.subnode_args.onContentEscapes(trace_collection)
        self.subnode_kwargs.onContentEscapes(trace_collection)

class ExpressionCallNoKeywords(ExpressionCallMixin, ChildrenExpressionCallNoKeywordsMixin, ExpressionBase):
    kind = 'EXPRESSION_CALL_NO_KEYWORDS'
    named_children = ('called|setter', 'args')
    subnode_kwargs = None

    def __init__(self, called, args, source_ref):
        if False:
            return 10
        ChildrenExpressionCallNoKeywordsMixin.__init__(self, called=called, args=args)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            print('Hello World!')
        called = self.subnode_called
        return called.computeExpressionCall(call_node=self, call_args=self.subnode_args, call_kw=None, trace_collection=trace_collection)

    def extractSideEffectsPreCall(self):
        if False:
            i = 10
            return i + 15
        args = self.subnode_args
        return args.extractSideEffects()

    def onContentEscapes(self, trace_collection):
        if False:
            while True:
                i = 10
        self.subnode_called.onContentEscapes(trace_collection)
        self.subnode_args.onContentEscapes(trace_collection)

class ExpressionCallKeywordsOnly(ExpressionCallMixin, ChildrenExpressionCallKeywordsOnlyMixin, ExpressionBase):
    kind = 'EXPRESSION_CALL_KEYWORDS_ONLY'
    named_children = ('called|setter', 'kwargs')
    subnode_args = None

    def __init__(self, called, kwargs, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionCallKeywordsOnlyMixin.__init__(self, called=called, kwargs=kwargs)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        called = self.subnode_called
        return called.computeExpressionCall(call_node=self, call_args=None, call_kw=self.subnode_kwargs, trace_collection=trace_collection)

    def extractSideEffectsPreCall(self):
        if False:
            print('Hello World!')
        kwargs = self.subnode_kwargs
        return kwargs.extractSideEffects()

    def onContentEscapes(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        self.subnode_called.onContentEscapes(trace_collection)
        self.subnode_kwargs.onContentEscapes(trace_collection)

class ExpressionCallEmpty(ExpressionCallMixin, ChildrenExpressionCallEmptyMixin, ExpressionBase):
    kind = 'EXPRESSION_CALL_EMPTY'
    named_children = ('called|setter',)
    subnode_args = None
    subnode_kwargs = None

    def __init__(self, called, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenExpressionCallEmptyMixin.__init__(self, called=called)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        called = self.subnode_called
        return called.computeExpressionCall(call_node=self, call_args=None, call_kw=None, trace_collection=trace_collection)

    @staticmethod
    def extractSideEffectsPreCall():
        if False:
            print('Hello World!')
        return ()

    def onContentEscapes(self, trace_collection):
        if False:
            while True:
                i = 10
        self.subnode_called.onContentEscapes(trace_collection)

def makeExpressionCall(called, args, kw, source_ref):
    if False:
        print('Hello World!')
    'Make the most simple call node possible.\n\n    By avoiding the more complex classes, we can achieve that there is\n    less work to do for analysis.\n    '
    has_kw = kw is not None and (not kw.isExpressionConstantDictEmptyRef())
    has_args = args is not None and (not args.isExpressionConstantTupleEmptyRef())
    if has_kw:
        if has_args:
            return ExpressionCall(called, args, kw, source_ref)
        else:
            return ExpressionCallKeywordsOnly(called, kw, source_ref)
    elif has_args:
        return ExpressionCallNoKeywords(called, args, source_ref)
    else:
        return ExpressionCallEmpty(called, source_ref)