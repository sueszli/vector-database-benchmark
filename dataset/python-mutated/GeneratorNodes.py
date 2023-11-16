""" Nodes for generator objects and their creations.

Generators are turned into normal functions that create generator objects,
whose implementation lives here. The creation itself also lives here.

"""
from nuitka.PythonVersions import python_version
from .ChildrenHavingMixins import ChildHavingGeneratorRefMixin
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .FunctionNodes import ExpressionFunctionEntryPointBase
from .IndicatorMixins import MarkUnoptimizedFunctionIndicatorMixin
from .ReturnNodes import StatementReturn, StatementReturnNone

class ExpressionMakeGeneratorObject(ExpressionNoSideEffectsMixin, ChildHavingGeneratorRefMixin, ExpressionBase):
    kind = 'EXPRESSION_MAKE_GENERATOR_OBJECT'
    named_children = ('generator_ref',)
    __slots__ = ('variable_closure_traces',)

    def __init__(self, generator_ref, source_ref):
        if False:
            for i in range(10):
                print('nop')
        assert generator_ref.getFunctionBody().isExpressionGeneratorObjectBody(), generator_ref
        ChildHavingGeneratorRefMixin.__init__(self, generator_ref=generator_ref)
        ExpressionBase.__init__(self, source_ref)
        self.variable_closure_traces = None

    def getCodeObject(self):
        if False:
            return 10
        return self.code_object

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        self.variable_closure_traces = []
        for closure_variable in self.subnode_generator_ref.getFunctionBody().getClosureVariables():
            trace = trace_collection.getVariableCurrentTrace(closure_variable)
            trace.addNameUsage()
            self.variable_closure_traces.append((closure_variable, trace))
        return (self, None, None)

    def getClosureVariableVersions(self):
        if False:
            return 10
        return self.variable_closure_traces

class ExpressionGeneratorObjectBody(MarkUnoptimizedFunctionIndicatorMixin, ExpressionFunctionEntryPointBase):
    kind = 'EXPRESSION_GENERATOR_OBJECT_BODY'
    __slots__ = ('unoptimized_locals', 'unqualified_exec', 'needs_generator_return_exit', 'qualname_provider')
    if python_version >= 832:
        __slots__ += ('qualname_setup',)

    def __init__(self, provider, name, code_object, flags, auto_release, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionFunctionEntryPointBase.__init__(self, provider=provider, name=name, code_object=code_object, code_prefix='genexpr' if name == '<genexpr>' else 'genobj', flags=flags, auto_release=auto_release, source_ref=source_ref)
        MarkUnoptimizedFunctionIndicatorMixin.__init__(self, flags)
        self.needs_generator_return_exit = False
        self.trace_collection = None
        if python_version >= 832:
            self.qualname_setup = None

    def getFunctionName(self):
        if False:
            return 10
        return self.name

    def markAsNeedsGeneratorReturnHandling(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.needs_generator_return_exit = max(self.needs_generator_return_exit, value)

    def needsGeneratorReturnHandling(self):
        if False:
            while True:
                i = 10
        return self.needs_generator_return_exit == 2

    def needsGeneratorReturnExit(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.needs_generator_return_exit)

    @staticmethod
    def needsCreation():
        if False:
            for i in range(10):
                print('nop')
        return False

    def getConstantReturnValue(self):
        if False:
            while True:
                i = 10
        'Special function that checks if code generation allows to use common C code.'
        body = self.subnode_body
        if body is None:
            return (True, None)
        return (False, False)

class StatementGeneratorReturn(StatementReturn):
    kind = 'STATEMENT_GENERATOR_RETURN'

    def __init__(self, expression, source_ref):
        if False:
            while True:
                i = 10
        StatementReturn.__init__(self, expression=expression, source_ref=source_ref)

    @staticmethod
    def isStatementGeneratorReturn():
        if False:
            return 10
        return True

    def computeStatement(self, trace_collection):
        if False:
            i = 10
            return i + 15
        expression = trace_collection.onExpression(self.subnode_expression)
        if expression.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        if expression.willRaiseAnyException():
            from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode
            result = makeStatementExpressionOnlyReplacementNode(expression=expression, node=self)
            return (result, 'new_raise', 'Return statement raises in returned expression, removed return.')
        trace_collection.onFunctionReturn()
        if expression.isExpressionConstantNoneRef():
            result = StatementGeneratorReturnNone(source_ref=self.source_ref)
            return (result, 'new_statements', 'Generator return value is always None.')
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'generator return statement'

class StatementGeneratorReturnNone(StatementReturnNone):
    kind = 'STATEMENT_GENERATOR_RETURN_NONE'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            while True:
                i = 10
        StatementReturnNone.__init__(self, source_ref=source_ref)

    @staticmethod
    def isStatementGeneratorReturn():
        if False:
            return 10
        return True

    @staticmethod
    def getStatementNiceName():
        if False:
            i = 10
            return i + 15
        return 'generator return statement'