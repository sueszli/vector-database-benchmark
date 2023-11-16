""" Nodes for named variable reference, assignment, and deletion

x = ...
del x
... = x

Variable name references might be in a class context, and then it
is unclear what this really will become. These nodes are used in
the early tree building phase, but never reach optimization phase
or even code generation.
"""
from .ExpressionBases import ExpressionBase
from .NodeBases import StatementBase
from .StatementBasesGenerated import StatementAssignmentVariableNameBase

class StatementAssignmentVariableName(StatementAssignmentVariableNameBase):
    """Precursor of StatementAssignmentVariable used during tree building phase"""
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_NAME'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('provider', 'variable_name')
    auto_compute_handling = 'post_init'

    def postInitNode(self):
        if False:
            print('Hello World!')
        assert not self.provider.isExpressionOutlineBody(), self.source_ref

    def getVariableName(self):
        if False:
            while True:
                i = 10
        return self.variable_name

    def computeStatement(self, trace_collection):
        if False:
            i = 10
            return i + 15
        assert False

    @staticmethod
    def getStatementNiceName():
        if False:
            for i in range(10):
                print('nop')
        return 'variable assignment statement'

class StatementDelVariableName(StatementBase):
    """Precursor of StatementDelVariable used during tree building phase"""
    kind = 'STATEMENT_DEL_VARIABLE_NAME'
    __slots__ = ('variable_name', 'provider', 'tolerant')

    def __init__(self, provider, variable_name, tolerant, source_ref):
        if False:
            return 10
        StatementBase.__init__(self, source_ref=source_ref)
        self.variable_name = variable_name
        self.provider = provider
        self.tolerant = tolerant

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent
        del self.provider

    def getDetails(self):
        if False:
            return 10
        return {'variable_name': self.variable_name, 'provider': self.provider, 'tolerant': self.tolerant}

    def getVariableName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_name

    def computeStatement(self, trace_collection):
        if False:
            i = 10
            return i + 15
        assert False

class ExpressionVariableNameRef(ExpressionBase):
    """These are used before the actual variable object is known from VariableClosure."""
    kind = 'EXPRESSION_VARIABLE_NAME_REF'
    __slots__ = ('variable_name', 'provider')

    def __init__(self, provider, variable_name, source_ref):
        if False:
            return 10
        assert not provider.isExpressionOutlineBody(), source_ref
        ExpressionBase.__init__(self, source_ref)
        self.variable_name = variable_name
        self.provider = provider

    def finalize(self):
        if False:
            return 10
        del self.parent
        del self.provider

    @staticmethod
    def isExpressionVariableNameRef():
        if False:
            while True:
                i = 10
        return True

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'variable_name': self.variable_name, 'provider': self.provider}

    def getVariableName(self):
        if False:
            print('Hello World!')
        return self.variable_name

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    @staticmethod
    def needsFallback():
        if False:
            return 10
        return True

class ExpressionVariableLocalNameRef(ExpressionVariableNameRef):
    """These are used before the actual variable object is known from VariableClosure.

    The special thing about this as opposed to ExpressionVariableNameRef is that
    these must remain local names and cannot fallback to outside scopes. This is
    used for "__annotations__".

    """
    kind = 'EXPRESSION_VARIABLE_LOCAL_NAME_REF'

    @staticmethod
    def needsFallback():
        if False:
            i = 10
            return i + 15
        return False