""" Nodes for variable release

These refer to resolved variable objects.

"""
from nuitka.ModuleRegistry import getOwnerFromCodeName
from .NodeBases import StatementBase

class StatementReleaseVariableBase(StatementBase):
    """Releasing a variable.

    Just release the value, which of course is not to be used afterwards.

    Typical code: Function exit user variables, try/finally release of temporary
    variables.
    """
    __slots__ = ('variable', 'variable_trace')

    def __init__(self, variable, source_ref):
        if False:
            return 10
        StatementBase.__init__(self, source_ref=source_ref)
        self.variable = variable
        self.variable_trace = None

    @staticmethod
    def isStatementReleaseVariable():
        if False:
            while True:
                i = 10
        return True

    def finalize(self):
        if False:
            print('Hello World!')
        del self.variable
        del self.variable_trace
        del self.parent

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'variable': self.variable}

    def getDetailsForDisplay(self):
        if False:
            i = 10
            return i + 15
        return {'variable_name': self.variable.getName(), 'owner': self.variable.getOwner().getCodeName()}

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            return 10
        assert cls is makeStatementReleaseVariable, cls
        owner = getOwnerFromCodeName(args['owner'])
        assert owner is not None, args['owner']
        variable = owner.getProvidedVariable(args['variable_name'])
        return cls(variable=variable, source_ref=source_ref)

    def getVariable(self):
        if False:
            print('Hello World!')
        return self.variable

    def getVariableTrace(self):
        if False:
            while True:
                i = 10
        return self.variable_trace

    def setVariable(self, variable):
        if False:
            i = 10
            return i + 15
        self.variable = variable

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        self.variable_trace = trace_collection.getVariableCurrentTrace(self.variable)
        if self.variable_trace.mustNotHaveValue():
            return (None, 'new_statements', 'Uninitialized %s is not released.' % self.variable.getDescription())
        escape_desc = self.variable_trace.getReleaseEscape()
        assert escape_desc is not None, self.variable_trace
        if escape_desc.isControlFlowEscape():
            trace_collection.onControlFlowEscape(self)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            while True:
                i = 10
        return False

class StatementReleaseVariableTemp(StatementReleaseVariableBase):
    kind = 'STATEMENT_RELEASE_VARIABLE_TEMP'

class StatementReleaseVariableLocal(StatementReleaseVariableBase):
    kind = 'STATEMENT_RELEASE_VARIABLE_LOCAL'

class StatementReleaseVariableParameter(StatementReleaseVariableLocal):
    kind = 'STATEMENT_RELEASE_VARIABLE_PARAMETER'

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        if self.variable.getOwner().isAutoReleaseVariable(self.variable):
            return (None, 'new_statements', "Original parameter variable value of '%s' is not released." % self.variable.getName())
        return StatementReleaseVariableLocal.computeStatement(self, trace_collection)

def makeStatementReleaseVariable(variable, source_ref):
    if False:
        for i in range(10):
            print('nop')
    if variable.isTempVariable():
        return StatementReleaseVariableTemp(variable=variable, source_ref=source_ref)
    elif variable.isParameterVariable():
        return StatementReleaseVariableParameter(variable=variable, source_ref=source_ref)
    else:
        return StatementReleaseVariableLocal(variable=variable, source_ref=source_ref)

def makeStatementsReleaseVariables(variables, source_ref):
    if False:
        return 10
    return tuple((makeStatementReleaseVariable(variable=variable, source_ref=source_ref) for variable in variables))