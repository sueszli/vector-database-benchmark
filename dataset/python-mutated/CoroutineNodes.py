""" Nodes for coroutine objects and their creations.

Coroutines are turned into normal functions that create generator objects,
whose implementation lives here. The creation itself also lives here.

"""
from .ChildrenHavingMixins import ChildHavingCoroutineRefMixin, ChildHavingExpressionMixin
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .FunctionNodes import ExpressionFunctionEntryPointBase

class ExpressionMakeCoroutineObject(ExpressionNoSideEffectsMixin, ChildHavingCoroutineRefMixin, ExpressionBase):
    kind = 'EXPRESSION_MAKE_COROUTINE_OBJECT'
    named_children = ('coroutine_ref',)
    __slots__ = ('variable_closure_traces',)

    def __init__(self, coroutine_ref, source_ref):
        if False:
            for i in range(10):
                print('nop')
        assert coroutine_ref.getFunctionBody().isExpressionCoroutineObjectBody()
        ChildHavingCoroutineRefMixin.__init__(self, coroutine_ref=coroutine_ref)
        ExpressionBase.__init__(self, source_ref)
        self.variable_closure_traces = None

    def getDetailsForDisplay(self):
        if False:
            while True:
                i = 10
        return {'coroutine': self.subnode_coroutine_ref.getFunctionBody().getCodeName()}

    def computeExpression(self, trace_collection):
        if False:
            return 10
        self.variable_closure_traces = []
        for closure_variable in self.subnode_coroutine_ref.getFunctionBody().getClosureVariables():
            trace = trace_collection.getVariableCurrentTrace(closure_variable)
            trace.addNameUsage()
            self.variable_closure_traces.append((closure_variable, trace))
        return (self, None, None)

    def getClosureVariableVersions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_closure_traces

class ExpressionCoroutineObjectBody(ExpressionFunctionEntryPointBase):
    kind = 'EXPRESSION_COROUTINE_OBJECT_BODY'
    __slots__ = ('qualname_setup', 'needs_generator_return_exit')

    def __init__(self, provider, name, code_object, flags, auto_release, source_ref):
        if False:
            return 10
        ExpressionFunctionEntryPointBase.__init__(self, provider=provider, name=name, code_object=code_object, code_prefix='coroutine', flags=flags, auto_release=auto_release, source_ref=source_ref)
        self.needs_generator_return_exit = False
        self.qualname_setup = None

    def getFunctionName(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def markAsNeedsGeneratorReturnHandling(self, value):
        if False:
            return 10
        self.needs_generator_return_exit = max(self.needs_generator_return_exit, value)

    def needsGeneratorReturnHandling(self):
        if False:
            while True:
                i = 10
        return self.needs_generator_return_exit == 2

    def needsGeneratorReturnExit(self):
        if False:
            print('Hello World!')
        return bool(self.needs_generator_return_exit)

    @staticmethod
    def needsCreation():
        if False:
            return 10
        return False

    @staticmethod
    def isUnoptimized():
        if False:
            return 10
        return False

class ExpressionAsyncWait(ChildHavingExpressionMixin, ExpressionBase):
    kind = 'EXPRESSION_ASYNC_WAIT'
    named_children = ('expression',)
    __slots__ = ('exception_preserving',)

    def __init__(self, expression, source_ref):
        if False:
            i = 10
            return i + 15
        ChildHavingExpressionMixin.__init__(self, expression=expression)
        ExpressionBase.__init__(self, source_ref)
        self.exception_preserving = False

    @staticmethod
    def isExpressionAsyncWait():
        if False:
            while True:
                i = 10
        return True

    def computeExpression(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionAsyncWaitEnter(ExpressionAsyncWait):
    kind = 'EXPRESSION_ASYNC_WAIT_ENTER'

class ExpressionAsyncWaitExit(ExpressionAsyncWait):
    kind = 'EXPRESSION_ASYNC_WAIT_EXIT'