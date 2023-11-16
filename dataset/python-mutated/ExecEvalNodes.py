""" Nodes concern with exec and eval builtins.

These are the dynamic codes, and as such rather difficult. We would like
to eliminate or limit their impact as much as possible, but it's difficult
to do.
"""
from .ChildrenHavingMixins import ChildrenExpressionBuiltinCompileMixin, ChildrenExpressionBuiltinEvalMixin, ChildrenExpressionBuiltinExecfileMixin, ChildrenExpressionBuiltinExecMixin
from .ExpressionBases import ExpressionBase
from .StatementBasesGenerated import StatementExecBase, StatementLocalsDictSyncBase

class ExpressionBuiltinEval(ChildrenExpressionBuiltinEvalMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_EVAL'
    named_children = ('source_code', 'globals_arg', 'locals_arg')

    def __init__(self, source_code, globals_arg, locals_arg, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionBuiltinEvalMixin.__init__(self, source_code=source_code, globals_arg=globals_arg, locals_arg=locals_arg)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

class ExpressionBuiltinExec(ChildrenExpressionBuiltinExecMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_EXEC'
    python_version_spec = '>= 0x300'
    named_children = ('source_code', 'globals_arg', 'locals_arg', 'closure|optional')

    def __init__(self, source_code, globals_arg, locals_arg, closure, source_ref):
        if False:
            while True:
                i = 10
        ChildrenExpressionBuiltinExecMixin.__init__(self, source_code=source_code, globals_arg=globals_arg, locals_arg=locals_arg, closure=closure)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            return 10
        return (statement, None, None)

class ExpressionBuiltinExecfile(ChildrenExpressionBuiltinExecfileMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_EXECFILE'
    python_version_spec = '< 0x300'
    named_children = ('source_code', 'globals_arg', 'locals_arg')
    __slots__ = ('in_class_body',)

    def __init__(self, in_class_body, source_code, globals_arg, locals_arg, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionBuiltinExecfileMixin.__init__(self, source_code=source_code, globals_arg=globals_arg, locals_arg=locals_arg)
        ExpressionBase.__init__(self, source_ref)
        self.in_class_body = in_class_body

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        return {'in_class_body': self.in_class_body}

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            return 10
        if self.in_class_body:
            result = StatementExec(source_code=self.subnode_source_code, globals_arg=self.subnode_globals_arg, locals_arg=self.subnode_locals_arg, source_ref=self.source_ref)
            del self.parent
            return (result, 'new_statements', "Changed 'execfile' with unused result to 'exec' on class level.")
        else:
            return (statement, None, None)

class StatementExec(StatementExecBase):
    kind = 'STATEMENT_EXEC'
    named_children = ('source_code', 'globals_arg|auto_none', 'locals_arg|auto_none')
    auto_compute_handling = 'operation'
    python_version_spec = '< 0x300'

    def computeStatementOperation(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class StatementLocalsDictSync(StatementLocalsDictSyncBase):
    kind = 'STATEMENT_LOCALS_DICT_SYNC'
    named_children = ('locals_arg',)
    node_attributes = ('locals_scope',)
    auto_compute_handling = 'operation,post_init'
    __slots__ = ('previous_traces', 'variable_traces')

    def postInitNode(self):
        if False:
            while True:
                i = 10
        self.previous_traces = None
        self.variable_traces = None

    def getPreviousVariablesTraces(self):
        if False:
            while True:
                i = 10
        return self.previous_traces

    def computeStatementOperation(self, trace_collection):
        if False:
            return 10
        provider = self.getParentVariableProvider()
        if provider.isCompiledPythonModule():
            return (None, 'new_statements', 'Removed sync back to locals without locals.')
        self.previous_traces = trace_collection.onLocalsUsage(self.locals_scope)
        if not self.previous_traces:
            return (None, 'new_statements', 'Removed sync back to locals without locals.')
        trace_collection.removeAllKnowledge()
        self.variable_traces = trace_collection.onLocalsUsage(self.locals_scope)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionBuiltinCompile(ChildrenExpressionBuiltinCompileMixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_COMPILE'
    named_children = ('source', 'filename', 'mode', 'flags|optional', 'dont_inherit|optional', 'optimize|optional')

    def __init__(self, source_code, filename, mode, flags, dont_inherit, optimize, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenExpressionBuiltinCompileMixin.__init__(self, source=source_code, filename=filename, mode=mode, flags=flags, dont_inherit=dont_inherit, optimize=optimize)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)