""" Nodes related to raising and making exceptions.

"""
from .ChildrenHavingMixins import ChildrenHavingExceptionTypeExceptionValueMixin
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .ExpressionBasesGenerated import ExpressionBuiltinMakeExceptionBase, ExpressionBuiltinMakeExceptionImportErrorBase
from .NodeBases import StatementBase
from .StatementBasesGenerated import StatementRaiseExceptionBase

class StatementRaiseExceptionMixin(object):
    __slots__ = ()

    @staticmethod
    def isStatementAborting():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isStatementRaiseException():
        if False:
            return 10
        return True

    @staticmethod
    def willRaiseAnyException():
        if False:
            i = 10
            return i + 15
        return True

class StatementRaiseException(StatementRaiseExceptionMixin, StatementRaiseExceptionBase):
    kind = 'STATEMENT_RAISE_EXCEPTION'
    named_children = ('exception_type', 'exception_value|optional', 'exception_trace|optional', 'exception_cause|optional')
    auto_compute_handling = 'post_init,operation'
    __slots__ = ('reraise_finally',)

    def postInitNode(self):
        if False:
            print('Hello World!')
        self.reraise_finally = False

    def computeStatementOperation(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def needsFrame():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'exception raise statement'

class StatementRaiseExceptionImplicit(StatementRaiseException):
    kind = 'STATEMENT_RAISE_EXCEPTION_IMPLICIT'

    @staticmethod
    def getStatementNiceName():
        if False:
            print('Hello World!')
        return 'implicit exception raise statement'

class StatementReraiseException(StatementRaiseExceptionMixin, StatementBase):
    kind = 'STATEMENT_RERAISE_EXCEPTION'

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent

    def computeStatement(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def needsFrame():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def getStatementNiceName():
        if False:
            print('Hello World!')
        return 'exception re-raise statement'

class ExpressionRaiseException(ChildrenHavingExceptionTypeExceptionValueMixin, ExpressionBase):
    """This node type is only produced via optimization.

    CPython only knows exception raising as a statement, but often the raising
    of exceptions can be predicted to occur as part of an expression, which it
    replaces then.
    """
    kind = 'EXPRESSION_RAISE_EXCEPTION'
    named_children = ('exception_type', 'exception_value')

    def __init__(self, exception_type, exception_value, source_ref):
        if False:
            return 10
        ChildrenHavingExceptionTypeExceptionValueMixin.__init__(self, exception_type=exception_type, exception_value=exception_value)
        ExpressionBase.__init__(self, source_ref)

    @staticmethod
    def willRaiseAnyException():
        if False:
            i = 10
            return i + 15
        return True

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            print('Hello World!')
        result = self.asStatement()
        del self.parent
        return (result, 'new_raise', 'Propagated implicit raise expression to raise statement.')

    def asStatement(self):
        if False:
            return 10
        return StatementRaiseExceptionImplicit(exception_type=self.subnode_exception_type, exception_value=self.subnode_exception_value, exception_trace=None, exception_cause=None, source_ref=self.source_ref)

class ExpressionBuiltinMakeException(ExpressionBuiltinMakeExceptionBase):
    kind = 'EXPRESSION_BUILTIN_MAKE_EXCEPTION'
    named_children = ('args|tuple',)
    __slots__ = ('exception_name',)
    auto_compute_handling = 'final,no_raise'

    def __init__(self, exception_name, args, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionBuiltinMakeExceptionBase.__init__(self, args, source_ref=source_ref)
        self.exception_name = exception_name

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'exception_name': self.exception_name}

    def getExceptionName(self):
        if False:
            while True:
                i = 10
        return self.exception_name

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        for arg in self.subnode_args:
            if arg.mayRaiseException(exception_type):
                return True
        return False

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            return 10
        return False

class ExpressionBuiltinMakeExceptionImportError(ExpressionBuiltinMakeExceptionImportErrorBase):
    """Python3 ImportError dedicated node with extra arguments."""
    kind = 'EXPRESSION_BUILTIN_MAKE_EXCEPTION_IMPORT_ERROR'
    named_children = ('args|tuple', 'name|optional', 'path|optional')
    __slots__ = ('exception_name',)
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def getExceptionName():
        if False:
            print('Hello World!')
        return 'ImportError'

    def computeExpression(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        for arg in self.subnode_args:
            if arg.mayRaiseException(exception_type):
                return True
        return False

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionCaughtMixin(ExpressionNoSideEffectsMixin):
    """Common things for all caught exception references."""
    __slots__ = ()

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

class ExpressionCaughtExceptionTypeRef(ExpressionCaughtMixin, ExpressionBase):
    kind = 'EXPRESSION_CAUGHT_EXCEPTION_TYPE_REF'

    def __init__(self, source_ref):
        if False:
            print('Hello World!')
        ExpressionBase.__init__(self, source_ref)

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

class ExpressionCaughtExceptionValueRef(ExpressionCaughtMixin, ExpressionBase):
    kind = 'EXPRESSION_CAUGHT_EXCEPTION_VALUE_REF'

    def __init__(self, source_ref):
        if False:
            return 10
        ExpressionBase.__init__(self, source_ref)

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)

class ExpressionCaughtExceptionTracebackRef(ExpressionCaughtMixin, ExpressionBase):
    kind = 'EXPRESSION_CAUGHT_EXCEPTION_TRACEBACK_REF'

    def __init__(self, source_ref):
        if False:
            print('Hello World!')
        ExpressionBase.__init__(self, source_ref)

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)