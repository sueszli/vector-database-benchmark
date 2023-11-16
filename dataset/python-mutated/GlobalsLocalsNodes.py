""" Globals/locals/single arg dir nodes

These nodes give access to variables, highly problematic, because using them,
the code may change or access anything about them, so nothing can be trusted
anymore, if we start to not know where their value goes.

The "dir()" call without arguments is reformulated to locals or globals calls.
"""
from .DictionaryNodes import makeExpressionMakeDict
from .ExpressionBases import ExpressionBase, ExpressionBuiltinSingleArgBase, ExpressionNoSideEffectsMixin
from .KeyValuePairNodes import makeKeyValuePairExpressionsFromKwArgs
from .VariableRefNodes import ExpressionTempVariableRef, ExpressionVariableRef

class ExpressionBuiltinGlobals(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_GLOBALS'

    def __init__(self, source_ref):
        if False:
            while True:
                i = 10
        ExpressionBase.__init__(self, source_ref)

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

class ExpressionBuiltinLocalsBase(ExpressionNoSideEffectsMixin, ExpressionBase):
    __slots__ = ('variable_traces', 'locals_scope')

    def __init__(self, locals_scope, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionBase.__init__(self, source_ref)
        self.variable_traces = None
        self.locals_scope = locals_scope

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.locals_scope
        del self.variable_traces

    def getVariableTraces(self):
        if False:
            print('Hello World!')
        return self.variable_traces

    def getLocalsScope(self):
        if False:
            print('Hello World!')
        return self.locals_scope

class ExpressionBuiltinLocalsUpdated(ExpressionBuiltinLocalsBase):
    kind = 'EXPRESSION_BUILTIN_LOCALS_UPDATED'

    def __init__(self, locals_scope, source_ref):
        if False:
            return 10
        ExpressionBuiltinLocalsBase.__init__(self, locals_scope=locals_scope, source_ref=source_ref)
        assert locals_scope is not None

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        self.variable_traces = trace_collection.onLocalsUsage(self.locals_scope)
        return (self, None, None)

class ExpressionBuiltinLocalsRef(ExpressionBuiltinLocalsBase):
    kind = 'EXPRESSION_BUILTIN_LOCALS_REF'

    def __init__(self, locals_scope, source_ref):
        if False:
            print('Hello World!')
        ExpressionBuiltinLocalsBase.__init__(self, locals_scope=locals_scope, source_ref=source_ref)

    def getLocalsScope(self):
        if False:
            return 10
        return self.locals_scope

    def isFinalUseOfLocals(self):
        if False:
            print('Hello World!')
        return self.parent.isStatementReturn()

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        if self.locals_scope.isMarkedForPropagation():
            result = makeExpressionMakeDict(pairs=makeKeyValuePairExpressionsFromKwArgs(pairs=((variable_name, ExpressionTempVariableRef(variable=variable, source_ref=self.source_ref)) for (variable_name, variable) in self.locals_scope.getPropagationVariables().items())), source_ref=self.source_ref)
            new_result = result.computeExpressionRaw(trace_collection)
            assert new_result[0] is result
            self.finalize()
            return (result, 'new_expression', 'Propagated locals dictionary reference.')
        if not self.isFinalUseOfLocals():
            trace_collection.onLocalsUsage(locals_scope=self.locals_scope)
        return (self, None, None)

class ExpressionBuiltinLocalsCopy(ExpressionBuiltinLocalsBase):
    kind = 'EXPRESSION_BUILTIN_LOCALS_COPY'

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        self.variable_traces = trace_collection.onLocalsUsage(locals_scope=self.locals_scope)
        for (variable, variable_trace) in self.variable_traces:
            if not variable_trace.mustHaveValue() and (not variable_trace.mustNotHaveValue()):
                return (self, None, None)
            if variable_trace.getNameUsageCount() > 1:
                return (self, None, None)
        pairs = makeKeyValuePairExpressionsFromKwArgs(((variable.getName(), ExpressionVariableRef(variable=variable, source_ref=self.source_ref)) for (variable, variable_trace) in self.variable_traces if variable_trace.mustHaveValue()))

        def _sorted(pairs):
            if False:
                return 10
            names = [variable.getName() for variable in self.locals_scope.getProvidedVariables()]
            return tuple(sorted(pairs, key=lambda pair: names.index(pair.getKeyCompileTimeConstant())))
        result = makeExpressionMakeDict(pairs=_sorted(pairs), source_ref=self.source_ref)
        return (result, 'new_expression', 'Statically predicted locals dictionary.')

class ExpressionBuiltinDir1(ExpressionBuiltinSingleArgBase):
    kind = 'EXPRESSION_BUILTIN_DIR1'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)