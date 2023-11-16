""" Return node

This one exits functions. The only other exit is the default exit of functions with 'None' value, if no return is done.
"""
from abc import abstractmethod
from .NodeBases import StatementBase
from .StatementBasesGenerated import StatementReturnBase

class StatementReturnMixin(object):
    __slots__ = ()

    @staticmethod
    def isStatementReturn():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def mayReturn():
        if False:
            return 10
        return True

    @staticmethod
    def isStatementAborting():
        if False:
            print('Hello World!')
        return True

class StatementReturn(StatementReturnMixin, StatementReturnBase):
    kind = 'STATEMENT_RETURN'
    named_children = ('expression',)
    nice_children = ('return value',)

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_expression.mayRaiseException(exception_type)

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        expression = trace_collection.onExpression(self.subnode_expression)
        if expression.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        if expression.willRaiseAnyException():
            from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode
            result = makeStatementExpressionOnlyReplacementNode(expression=expression, node=self)
            return (result, 'new_raise', 'Return statement raises in returned expression, removed return.')
        trace_collection.onFunctionReturn()
        if expression.isExpressionConstantRef():
            result = makeStatementReturnConstant(constant=expression.getCompileTimeConstant(), source_ref=self.source_ref)
            del self.parent
            return (result, 'new_statements', 'Return value is constant.')
        return (self, None, None)

class StatementReturnConstantBase(StatementReturnMixin, StatementBase):
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            return 10
        StatementBase.__init__(self, source_ref=source_ref)

    @staticmethod
    def isStatementReturnConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return False

    def computeStatement(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onFunctionReturn()
        return (self, None, None)

    @abstractmethod
    def getConstant(self):
        if False:
            print('Hello World!')
        'The returned constant value.'

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'return statement'

class StatementReturnNone(StatementReturnConstantBase):
    kind = 'STATEMENT_RETURN_NONE'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            i = 10
            return i + 15
        StatementReturnConstantBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent

    def getConstant(self):
        if False:
            for i in range(10):
                print('nop')
        return None

class StatementReturnFalse(StatementReturnConstantBase):
    kind = 'STATEMENT_RETURN_FALSE'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            return 10
        StatementReturnConstantBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            return 10
        del self.parent

    def getConstant(self):
        if False:
            i = 10
            return i + 15
        return False

class StatementReturnTrue(StatementReturnConstantBase):
    kind = 'STATEMENT_RETURN_TRUE'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        StatementReturnConstantBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def getConstant(self):
        if False:
            i = 10
            return i + 15
        return True

class StatementReturnConstant(StatementReturnConstantBase):
    kind = 'STATEMENT_RETURN_CONSTANT'
    __slots__ = ('constant',)

    def __init__(self, constant, source_ref):
        if False:
            return 10
        StatementReturnConstantBase.__init__(self, source_ref=source_ref)
        self.constant = constant

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent
        del self.constant

    def getConstant(self):
        if False:
            return 10
        return self.constant

    def getDetails(self):
        if False:
            return 10
        return {'constant': self.constant}

class StatementReturnReturnedValue(StatementBase):
    kind = 'STATEMENT_RETURN_RETURNED_VALUE'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            print('Hello World!')
        StatementBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent

    @staticmethod
    def isStatementReturnReturnedValue():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isStatementReturn():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isStatementAborting():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def mayReturn():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

    def computeStatement(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onFunctionReturn()
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            i = 10
            return i + 15
        return 'rereturn statement'

def makeStatementReturnConstant(constant, source_ref):
    if False:
        print('Hello World!')
    if constant is None:
        return StatementReturnNone(source_ref=source_ref)
    elif constant is True:
        return StatementReturnTrue(source_ref=source_ref)
    elif constant is False:
        return StatementReturnFalse(source_ref=source_ref)
    else:
        return StatementReturnConstant(constant=constant, source_ref=source_ref)

def makeStatementReturn(expression, source_ref):
    if False:
        return 10
    'Create the best return statement variant.'
    if expression is None:
        return StatementReturnNone(source_ref=source_ref)
    elif expression.isCompileTimeConstant():
        return makeStatementReturnConstant(constant=expression.getCompileTimeConstant(), source_ref=source_ref)
    else:
        return StatementReturn(expression=expression, source_ref=source_ref)