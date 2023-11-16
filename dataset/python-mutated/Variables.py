""" Variables link the storage and use of a Python variable together.

Different kinds of variables represent different scopes and owners types,
and their links between each other, i.e. references as in closure or
module variable references.

"""
from abc import abstractmethod
from nuitka.__past__ import iterItems
from nuitka.nodes.shapes.BuiltinTypeShapes import tshape_dict
from nuitka.nodes.shapes.StandardShapes import tshape_unknown
from nuitka.utils import Utils
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
from nuitka.utils.SlotMetaClasses import getMetaClassBase
complete = False

class Variable(getMetaClassBase('Variable', require_slots=True)):
    __slots__ = ('variable_name', 'owner', 'version_number', 'shared_users', 'traces', 'users', 'writers')

    @counted_init
    def __init__(self, owner, variable_name):
        if False:
            i = 10
            return i + 15
        assert type(variable_name) is str, variable_name
        assert type(owner) not in (tuple, list), owner
        self.variable_name = variable_name
        self.owner = owner
        self.version_number = 0
        self.shared_users = False
        self.traces = set()
        self.users = None
        self.writers = None
    if isCountingInstances():
        __del__ = counted_del()

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.users
        del self.writers
        del self.traces
        del self.owner

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "<%s '%s' of '%s'>" % (self.__class__.__name__, self.variable_name, self.owner.getName())

    @abstractmethod
    def getVariableType(self):
        if False:
            while True:
                i = 10
        pass

    def getDescription(self):
        if False:
            print('Hello World!')
        return "variable '%s'" % self.variable_name

    def getName(self):
        if False:
            return 10
        return self.variable_name

    def getOwner(self):
        if False:
            return 10
        return self.owner

    def getEntryPoint(self):
        if False:
            while True:
                i = 10
        return self.owner.getEntryPoint()

    def getCodeName(self):
        if False:
            i = 10
            return i + 15
        var_name = self.variable_name
        var_name = var_name.replace('.', '$')
        var_name = Utils.encodeNonAscii(var_name)
        return var_name

    def allocateTargetNumber(self):
        if False:
            return 10
        self.version_number += 1
        return self.version_number

    @staticmethod
    def isLocalVariable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isParameterVariable():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isModuleVariable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isIncompleteModuleVariable():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isTempVariable():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isTempVariableBool():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isLocalsDictVariable():
        if False:
            for i in range(10):
                print('nop')
        return False

    def addVariableUser(self, user):
        if False:
            print('Hello World!')
        if user is not self.owner:
            self.shared_users = True
            if user.isExpressionGeneratorObjectBody() or user.isExpressionCoroutineObjectBody() or user.isExpressionAsyncgenObjectBody():
                if self.owner is user.getParentVariableProvider():
                    return
            _variables_in_shared_scopes.add(self)

    def isSharedTechnically(self):
        if False:
            return 10
        if not self.shared_users:
            return False
        if not self.users:
            return False
        owner = self.owner.getEntryPoint()
        for user in self.users:
            user = user.getEntryPoint()
            while user is not owner and (user.isExpressionFunctionBody() and (not user.needsCreation()) or user.isExpressionClassBodyBase()):
                user = user.getParentVariableProvider()
            if user is not owner:
                return True
        return False

    def addTrace(self, variable_trace):
        if False:
            i = 10
            return i + 15
        self.traces.add(variable_trace)

    def removeTrace(self, variable_trace):
        if False:
            return 10
        self.traces.remove(variable_trace)

    def getTraces(self):
        if False:
            print('Hello World!')
        'For debugging only'
        return self.traces

    def updateUsageState(self):
        if False:
            i = 10
            return i + 15
        writers = set()
        users = set()
        for trace in self.traces:
            owner = trace.owner
            users.add(owner)
            if trace.isAssignTrace():
                writers.add(owner)
            elif trace.isDeletedTrace() and owner is not self.owner:
                writers.add(owner)
        self.writers = writers
        self.users = users

    def hasAccessesOutsideOf(self, provider):
        if False:
            while True:
                i = 10
        if not self.owner.locals_scope.complete:
            return None
        elif self.users is None:
            return False
        elif provider in self.users:
            return len(self.users) > 1
        else:
            return bool(self.users)

    def hasWritersOutsideOf(self, provider):
        if False:
            for i in range(10):
                print('nop')
        if not self.owner.locals_scope.complete:
            return None
        elif self.writers is None:
            return False
        elif provider in self.writers:
            return len(self.writers) > 1
        else:
            return bool(self.writers)

    def getMatchingAssignTrace(self, assign_node):
        if False:
            print('Hello World!')
        for trace in self.traces:
            if trace.isAssignTrace() and trace.getAssignNode() is assign_node:
                return trace
        return None

    def getMatchingUnescapedAssignTrace(self, assign_node):
        if False:
            print('Hello World!')
        found = None
        for trace in self.traces:
            if trace.isAssignTrace() and trace.getAssignNode() is assign_node:
                found = trace
            if trace.isEscapeTrace():
                return None
        return found

    def getMatchingDelTrace(self, del_node):
        if False:
            return 10
        for trace in self.traces:
            if trace.isDeletedTrace() and trace.getDelNode() is del_node:
                return trace
        return None

    def getTypeShapes(self):
        if False:
            for i in range(10):
                print('nop')
        result = set()
        for trace in self.traces:
            if trace.isAssignTrace():
                result.add(trace.getAssignNode().getTypeShape())
            elif trace.isUnknownTrace():
                result.add(tshape_unknown)
            elif trace.isEscapeTrace():
                result.add(tshape_unknown)
            elif trace.isInitTrace():
                result.add(tshape_unknown)
            elif trace.isUnassignedTrace():
                pass
            elif trace.isMergeTrace():
                pass
            elif trace.isLoopTrace():
                trace.getTypeShape().emitAlternatives(result.add)
            else:
                assert False, trace
        return result

    @staticmethod
    def onControlFlowEscape(trace_collection):
        if False:
            print('Hello World!')
        'Mark the variable as escaped or unknown, or keep it depending on variable type.'

    def removeKnowledge(self, trace_collection):
        if False:
            print('Hello World!')
        'Remove knowledge for the variable marking as unknown or escaped.'
        trace_collection.markActiveVariableAsEscaped(self)

    def removeAllKnowledge(self, trace_collection):
        if False:
            print('Hello World!')
        'Remove all knowledge for the variable marking as unknown, or keep it depending on variable type.'
        trace_collection.markActiveVariableAsUnknown(self)

class LocalVariable(Variable):
    __slots__ = ()

    def __init__(self, owner, variable_name):
        if False:
            for i in range(10):
                print('nop')
        Variable.__init__(self, owner=owner, variable_name=variable_name)

    @staticmethod
    def isLocalVariable():
        if False:
            for i in range(10):
                print('nop')
        return True

    def initVariable(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        'Initialize variable in trace collection state.'
        return trace_collection.initVariableUninitialized(self)
    if str is not bytes:

        def onControlFlowEscape(self, trace_collection):
            if False:
                return 10
            if self.hasWritersOutsideOf(trace_collection.owner) is not False:
                trace_collection.markClosureVariableAsUnknown(self)
            elif self.hasAccessesOutsideOf(trace_collection.owner) is not False:
                trace_collection.markActiveVariableAsEscaped(self)
    else:

        def onControlFlowEscape(self, trace_collection):
            if False:
                return 10
            if self.hasAccessesOutsideOf(trace_collection.owner) is not False:
                trace_collection.markActiveVariableAsEscaped(self)

    @staticmethod
    def getVariableType():
        if False:
            i = 10
            return i + 15
        return 'object'

class ParameterVariable(LocalVariable):
    __slots__ = ()

    def __init__(self, owner, parameter_name):
        if False:
            while True:
                i = 10
        LocalVariable.__init__(self, owner=owner, variable_name=parameter_name)

    def getDescription(self):
        if False:
            return 10
        return "parameter variable '%s'" % self.variable_name

    @staticmethod
    def isParameterVariable():
        if False:
            i = 10
            return i + 15
        return True

    def initVariable(self, trace_collection):
        if False:
            print('Hello World!')
        'Initialize variable in trace collection state.'
        return trace_collection.initVariableInit(self)

class ModuleVariable(Variable):
    __slots__ = ()

    def __init__(self, module, variable_name):
        if False:
            while True:
                i = 10
        assert type(variable_name) is str, repr(variable_name)
        assert module.isCompiledPythonModule()
        Variable.__init__(self, owner=module, variable_name=variable_name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "<ModuleVariable '%s' of '%s'>" % (self.variable_name, self.owner.getFullName())

    def getDescription(self):
        if False:
            for i in range(10):
                print('nop')
        return "global variable '%s'" % self.variable_name

    @staticmethod
    def isModuleVariable():
        if False:
            for i in range(10):
                print('nop')
        return True

    def initVariable(self, trace_collection):
        if False:
            while True:
                i = 10
        'Initialize variable in trace collection state.'
        return trace_collection.initVariableModule(self)

    def onControlFlowEscape(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.markActiveVariableAsUnknown(self)

    def removeKnowledge(self, trace_collection):
        if False:
            return 10
        'Remove knowledge for the variable marking as unknown or escaped.'
        trace_collection.markActiveVariableAsUnknown(self)

    def isIncompleteModuleVariable(self):
        if False:
            print('Hello World!')
        return not self.owner.locals_scope.complete

    def hasDefiniteWrites(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.owner.locals_scope.complete:
            return None
        else:
            return bool(self.writers)

    def getModule(self):
        if False:
            for i in range(10):
                print('nop')
        return self.owner

    @staticmethod
    def getVariableType():
        if False:
            while True:
                i = 10
        return 'object'

class TempVariable(Variable):
    __slots__ = ('variable_type',)

    def __init__(self, owner, variable_name, variable_type):
        if False:
            for i in range(10):
                print('nop')
        Variable.__init__(self, owner=owner, variable_name=variable_name)
        self.variable_type = variable_type

    @staticmethod
    def isTempVariable():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getVariableType(self):
        if False:
            i = 10
            return i + 15
        return self.variable_type

    def isTempVariableBool(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_type == 'bool'

    def getDescription(self):
        if False:
            while True:
                i = 10
        return "temp variable '%s'" % self.variable_name

    def initVariable(self, trace_collection):
        if False:
            print('Hello World!')
        'Initialize variable in trace collection state.'
        return trace_collection.initVariableUninitialized(self)

    @staticmethod
    def removeAllKnowledge(trace_collection):
        if False:
            return 10
        'Remove all knowledge for the variable marking as unknown, or keep it depending on variable type.'

class LocalsDictVariable(Variable):
    __slots__ = ()

    def __init__(self, owner, variable_name):
        if False:
            return 10
        Variable.__init__(self, owner=owner, variable_name=variable_name)

    @staticmethod
    def isLocalsDictVariable():
        if False:
            return 10
        return True

    @staticmethod
    def getVariableType():
        if False:
            while True:
                i = 10
        return 'object'

    def initVariable(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Initialize variable in trace collection state.'
        if self.owner.getTypeShape() is tshape_dict:
            return trace_collection.initVariableUninitialized(self)
        else:
            return trace_collection.initVariableUnknown(self)

def updateVariablesFromCollection(old_collection, new_collection, source_ref):
    if False:
        print('Hello World!')
    touched_variables = set()
    loop_trace_removal = set()
    if old_collection is not None:
        for ((variable, _version), variable_trace) in iterItems(old_collection.getVariableTracesAll()):
            variable.removeTrace(variable_trace)
            touched_variables.add(variable)
            if variable_trace.isLoopTrace():
                loop_trace_removal.add(variable)
    if new_collection is not None:
        for ((variable, _version), variable_trace) in iterItems(new_collection.getVariableTracesAll()):
            variable.addTrace(variable_trace)
            touched_variables.add(variable)
            if variable_trace.isLoopTrace():
                if variable in loop_trace_removal:
                    loop_trace_removal.remove(variable)
        new_collection.variable_actives.clear()
        del new_collection.variable_actives
    for variable in touched_variables:
        variable.updateUsageState()
    if loop_trace_removal:
        if new_collection is not None:
            new_collection.signalChange('var_usage', source_ref, lambda : "Loop variable '%s' usage ceased." % ','.join((variable.getName() for variable in loop_trace_removal)))
_variables_in_shared_scopes = set()

def isSharedAmongScopes(variable):
    if False:
        print('Hello World!')
    return variable in _variables_in_shared_scopes

def releaseSharedScopeInformation(tree):
    if False:
        return 10
    assert tree.isCompiledPythonModule()
    global _variables_in_shared_scopes
    _variables_in_shared_scopes = set((variable for variable in _variables_in_shared_scopes if variable.getOwner().getParentModule() is not tree))