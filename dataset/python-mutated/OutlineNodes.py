""" Outline nodes.

We use them for re-formulations and for in-lining of code. They are expressions
that get their value from return statements in their code body. They do not
own anything by themselves. It's just a way of having try/finally for the
expressions, or multiple returns, without running in a too different context.
"""
from .ChildrenHavingMixins import ChildHavingBodyOptionalMixin
from .ConstantRefNodes import makeConstantRefNode
from .ExceptionNodes import ExpressionRaiseException
from .ExpressionBases import ExpressionBase
from .FunctionNodes import ExpressionFunctionBodyBase
from .LocalsScopes import getLocalsDictHandle

class ExpressionOutlineBody(ChildHavingBodyOptionalMixin, ExpressionBase):
    """Outlined expression code.

    This is for a call to a piece of code to be executed in a specific
    context. It contains an exclusively owned function body, that has
    no other references, and can be considered part of the calling
    context.

    It must return a value, to use as expression value.
    """
    kind = 'EXPRESSION_OUTLINE_BODY'
    named_children = ('body|optional+setter',)
    __slots__ = ('provider', 'name', 'temp_scope')

    @staticmethod
    def isExpressionOutlineBody():
        if False:
            i = 10
            return i + 15
        return True

    def __init__(self, provider, name, source_ref, body=None):
        if False:
            return 10
        assert name != ''
        ChildHavingBodyOptionalMixin.__init__(self, body=body)
        ExpressionBase.__init__(self, source_ref)
        self.provider = provider
        self.name = name
        self.temp_scope = None
        self.parent = provider

    def getDetails(self):
        if False:
            return 10
        return {'provider': self.provider, 'name': self.name}

    def getOutlineTempScope(self):
        if False:
            print('Hello World!')
        if self.temp_scope is None:
            self.temp_scope = self.provider.allocateTempScope(self.name)
        return self.temp_scope

    def allocateTempVariable(self, temp_scope, name, temp_type):
        if False:
            while True:
                i = 10
        if temp_scope is None:
            temp_scope = self.getOutlineTempScope()
        return self.provider.allocateTempVariable(temp_scope=temp_scope, name=name, temp_type=temp_type)

    def allocateTempScope(self, name):
        if False:
            return 10
        return self.provider.allocateTempScope(name=self.name + '$' + name)

    def getContainingClassDictCreation(self):
        if False:
            print('Hello World!')
        return self.getParentVariableProvider().getContainingClassDictCreation()

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        owning_module = self.getParentModule()
        from nuitka.ModuleRegistry import addUsedModule
        addUsedModule(module=owning_module, using_module=None, usage_tag='outline', reason='Owning module', source_ref=self.source_ref)
        abort_context = trace_collection.makeAbortStackContext(catch_breaks=False, catch_continues=False, catch_returns=True, catch_exceptions=False)
        with abort_context:
            body = self.subnode_body
            result = body.computeStatementsSequence(trace_collection=trace_collection)
            if result is not body:
                self.setChildBody(result)
                body = result
            return_collections = trace_collection.getFunctionReturnCollections()
        if return_collections:
            trace_collection.mergeMultipleBranches(return_collections)
        first_statement = body.subnode_statements[0]
        if first_statement.isStatementReturnConstant():
            return (makeConstantRefNode(constant=first_statement.getConstant(), source_ref=first_statement.source_ref), 'new_expression', "Outline '%s' is now simple return, use directly." % self.name)
        if first_statement.isStatementReturn():
            return (first_statement.subnode_expression, 'new_expression', "Outline '%s' is now simple return, use directly." % self.name)
        if first_statement.isStatementRaiseException():
            result = ExpressionRaiseException(exception_type=first_statement.subnode_exception_type, exception_value=first_statement.subnode_exception_value, source_ref=first_statement.getSourceReference())
            return (result, 'new_expression', 'Outline is now exception raise, use directly.')
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_body.mayRaiseException(exception_type)

    def willRaiseAnyException(self):
        if False:
            print('Hello World!')
        return self.subnode_body.willRaiseAnyException()

    def getEntryPoint(self):
        if False:
            return 10
        'Entry point for code.\n\n        Normally ourselves. Only outlines will refer to their parent which\n        technically owns them.\n\n        '
        return self.provider.getEntryPoint()

    def getCodeName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.provider.getCodeName()

class ExpressionOutlineFunctionBase(ExpressionFunctionBodyBase):
    """Outlined function code.

    This is for a call to a function to be called in-line to be executed
    in a specific context. It contains an exclusively owned function body,
    that has no other references, and can be considered part of the calling
    context.

    As normal function it must return a value, to use as expression value,
    but we know we only exist once.

    Once this has no frame, it can be changed to a mere outline expression.
    """
    __slots__ = ('temp_scope', 'locals_scope')

    def __init__(self, provider, name, body, code_prefix, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionFunctionBodyBase.__init__(self, provider=provider, name=name, body=body, code_prefix=code_prefix, flags=None, source_ref=source_ref)
        self.temp_scope = None
        self.locals_scope = None

    @staticmethod
    def isExpressionOutlineFunctionBase():
        if False:
            return 10
        return True

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'name': self.name, 'provider': self.provider}

    def getDetailsForDisplay(self):
        if False:
            print('Hello World!')
        return {'name': self.name, 'provider': self.provider.getCodeName()}

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        trace_collection.addOutlineFunction(self)
        abort_context = trace_collection.makeAbortStackContext(catch_breaks=False, catch_continues=False, catch_returns=True, catch_exceptions=False)
        with abort_context:
            body = self.subnode_body
            result = body.computeStatementsSequence(trace_collection=trace_collection)
            if result is not body:
                self.setChildBody(result)
                body = result
            return_collections = trace_collection.getFunctionReturnCollections()
        if return_collections:
            trace_collection.mergeMultipleBranches(return_collections)
        first_statement = body.subnode_statements[0]
        if first_statement.isStatementReturnConstant():
            return (makeConstantRefNode(constant=first_statement.getConstant(), source_ref=first_statement.source_ref), 'new_expression', "Outline function '%s' is now simple return, use directly." % self.name)
        if first_statement.isStatementReturn():
            return (first_statement.subnode_expression, 'new_expression', "Outline function '%s' is now simple return, use directly." % self.name)
        if first_statement.isStatementRaiseException():
            result = ExpressionRaiseException(exception_type=first_statement.subnode_exception_type, exception_value=first_statement.subnode_exception_value, source_ref=first_statement.getSourceReference())
            return (result, 'new_expression', 'Outline function is now exception raise, use directly.')
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_body.mayRaiseException(exception_type)

    def willRaiseAnyException(self):
        if False:
            return 10
        return self.subnode_body.willRaiseAnyException()

    def getTraceCollection(self):
        if False:
            for i in range(10):
                print('nop')
        return self.provider.getTraceCollection()

    def getOutlineTempScope(self):
        if False:
            i = 10
            return i + 15
        if self.temp_scope is None:
            self.temp_scope = self.provider.allocateTempScope(self.name)
        return self.temp_scope

    def allocateTempVariable(self, temp_scope, name, temp_type):
        if False:
            return 10
        if temp_scope is None:
            temp_scope = self.getOutlineTempScope()
        return self.provider.allocateTempVariable(temp_scope=temp_scope, name=name, temp_type=temp_type)

    def allocateTempScope(self, name):
        if False:
            return 10
        return self.provider.allocateTempScope(name=self.name + '$' + name)

    def getEntryPoint(self):
        if False:
            return 10
        'Entry point for code.\n\n        Normally ourselves. Only outlines will refer to their parent which\n        technically owns them.\n\n        '
        return self.provider.getEntryPoint()

    def getClosureVariable(self, variable_name):
        if False:
            return 10
        return self.provider.getVariableForReference(variable_name=variable_name)

    def getLocalsScope(self):
        if False:
            while True:
                i = 10
        return self.locals_scope

    def isEarlyClosure(self):
        if False:
            print('Hello World!')
        return self.provider.isEarlyClosure()

    def isUnoptimized(self):
        if False:
            return 10
        return self.provider.isUnoptimized()

class ExpressionOutlineFunction(ExpressionOutlineFunctionBase):
    kind = 'EXPRESSION_OUTLINE_FUNCTION'
    __slots__ = ('locals_scope',)

    def __init__(self, provider, name, source_ref, body=None):
        if False:
            for i in range(10):
                print('nop')
        ExpressionOutlineFunctionBase.__init__(self, provider=provider, name=name, code_prefix='outline', body=body, source_ref=source_ref)
        self.locals_scope = getLocalsDictHandle('locals_%s' % self.getCodeName(), 'python_function', self)

    def getChildQualname(self, function_name):
        if False:
            for i in range(10):
                print('nop')
        return self.provider.getChildQualname(function_name)