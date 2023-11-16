""" This module maintains the locals dict handles. """
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Errors import NuitkaOptimizationError
from nuitka.PythonVersions import python_version
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
from nuitka.Variables import LocalsDictVariable, LocalVariable
from .shapes.BuiltinTypeShapes import tshape_dict
from .shapes.StandardShapes import tshape_unknown
locals_dict_handles = {}

def getLocalsDictType(kind):
    if False:
        return 10
    if kind == 'python2_function_exec':
        locals_scope = LocalsDictExecHandle
    elif kind == 'python_function':
        locals_scope = LocalsDictFunctionHandle
    elif kind == 'python3_class':
        locals_scope = LocalsMappingHandle
    elif kind == 'python2_class':
        locals_scope = LocalsDictHandle
    elif kind == 'module_dict':
        locals_scope = GlobalsDictHandle
    else:
        assert False, kind
    return locals_scope

def getLocalsDictHandle(locals_name, kind, owner):
    if False:
        i = 10
        return i + 15
    if locals_name in locals_dict_handles:
        raise NuitkaOptimizationError('duplicate locals name', locals_name, kind, owner.getFullName(), owner.getCompileTimeFilename(), locals_dict_handles[locals_name].owner.getFullName(), locals_dict_handles[locals_name].owner.getCompileTimeFilename())
    locals_dict_handles[locals_name] = getLocalsDictType(kind)(locals_name=locals_name, owner=owner)
    return locals_dict_handles[locals_name]

class LocalsDictHandleBase(object):
    __slots__ = ('locals_name', 'variables', 'local_variables', 'providing', 'mark_for_propagation', 'prevented_propagation', 'propagation', 'owner', 'complete')

    @counted_init
    def __init__(self, locals_name, owner):
        if False:
            while True:
                i = 10
        self.locals_name = locals_name
        self.owner = owner
        self.variables = {}
        self.local_variables = {}
        self.providing = OrderedDict()
        self.mark_for_propagation = False
        self.propagation = None
        self.complete = False
    if isCountingInstances():
        __del__ = counted_del()

    def __repr__(self):
        if False:
            return 10
        return '<%s of %s>' % (self.__class__.__name__, self.locals_name)

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.locals_name

    def makeClone(self, new_owner):
        if False:
            while True:
                i = 10
        count = 1
        while 1:
            locals_name = self.locals_name + '_inline_%d' % count
            if locals_name not in locals_dict_handles:
                break
            count += 1
        result = self.__class__(locals_name=locals_name, owner=new_owner)
        variable_translation = {}
        for (variable_name, variable) in self.variables.items():
            new_variable = variable.makeClone(new_owner=new_owner)
            variable_translation[variable] = new_variable
            result.variables[variable_name] = new_variable
        for (variable_name, variable) in self.local_variables.items():
            new_variable = variable.makeClone(new_owner=new_owner)
            variable_translation[variable] = new_variable
            result.local_variables[variable_name] = new_variable
        result.providing = OrderedDict()
        for (variable_name, variable) in self.providing.items():
            if variable in variable_translation:
                new_variable = variable_translation[variable]
            else:
                new_variable = variable.makeClone(new_owner=new_owner)
                variable_translation[variable] = new_variable
            result.providing[variable_name] = new_variable
        return (result, variable_translation)

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_dict

    @staticmethod
    def hasShapeDictionaryExact():
        if False:
            while True:
                i = 10
        return True

    def getCodeName(self):
        if False:
            return 10
        return self.locals_name

    @staticmethod
    def isModuleScope():
        if False:
            return 10
        return False

    @staticmethod
    def isClassScope():
        if False:
            return 10
        return False

    @staticmethod
    def isFunctionScope():
        if False:
            return 10
        return False

    @staticmethod
    def isUnoptimizedFunctionScope():
        if False:
            return 10
        return False

    def getProvidedVariables(self):
        if False:
            while True:
                i = 10
        return self.providing.values()

    def registerProvidedVariable(self, variable):
        if False:
            print('Hello World!')
        variable_name = variable.getName()
        self.providing[variable_name] = variable

    def unregisterProvidedVariable(self, variable):
        if False:
            print('Hello World!')
        'Remove provided variable, e.g. because it became unused.'
        variable_name = variable.getName()
        if variable_name in self.providing:
            del self.providing[variable_name]
    registerClosureVariable = registerProvidedVariable
    unregisterClosureVariable = unregisterProvidedVariable

    def hasProvidedVariable(self, variable_name):
        if False:
            i = 10
            return i + 15
        'Test if a variable is provided.'
        return variable_name in self.providing

    def getProvidedVariable(self, variable_name):
        if False:
            i = 10
            return i + 15
        'Test if a variable is provided.'
        return self.providing[variable_name]

    def getLocalsRelevantVariables(self):
        if False:
            for i in range(10):
                print('nop')
        'The variables relevant to locals.'
        return self.providing.values()

    def getLocalsDictVariable(self, variable_name):
        if False:
            i = 10
            return i + 15
        if variable_name not in self.variables:
            result = LocalsDictVariable(owner=self, variable_name=variable_name)
            self.variables[variable_name] = result
        return self.variables[variable_name]

    def getLocalVariable(self, owner, variable_name):
        if False:
            while True:
                i = 10
        if variable_name not in self.local_variables:
            result = LocalVariable(owner=owner, variable_name=variable_name)
            self.local_variables[variable_name] = result
        return self.local_variables[variable_name]

    @staticmethod
    def preventLocalsDictPropagation():
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def isPreventedPropagation():
        if False:
            while True:
                i = 10
        return False

    def markForLocalsDictPropagation(self):
        if False:
            for i in range(10):
                print('nop')
        self.mark_for_propagation = True

    def isMarkedForPropagation(self):
        if False:
            print('Hello World!')
        return self.mark_for_propagation

    def allocateTempReplacementVariable(self, trace_collection, variable_name):
        if False:
            print('Hello World!')
        if self.propagation is None:
            self.propagation = OrderedDict()
        if variable_name not in self.propagation:
            provider = trace_collection.getOwner()
            self.propagation[variable_name] = provider.allocateTempVariable(temp_scope=None, name=self.getCodeName() + '_key_' + variable_name, temp_type='object')
        return self.propagation[variable_name]

    def getPropagationVariables(self):
        if False:
            print('Hello World!')
        if self.propagation is None:
            return ()
        return self.propagation

    def finalize(self):
        if False:
            i = 10
            return i + 15
        self.owner.locals_scope = None
        del self.owner
        del self.propagation
        del self.mark_for_propagation
        for variable in self.variables.values():
            variable.finalize()
        for variable in self.local_variables.values():
            variable.finalize()
        del self.variables
        del self.providing

    def markAsComplete(self, trace_collection):
        if False:
            i = 10
            return i + 15
        self.complete = True
        self._considerUnusedUserLocalVariables(trace_collection)
        self._considerPropagation(trace_collection)

    @staticmethod
    def _considerPropagation(trace_collection):
        if False:
            i = 10
            return i + 15
        'For overload by scope type. Check if this can be replaced.'

    def onPropagationComplete(self):
        if False:
            print('Hello World!')
        self.variables = {}
        self.mark_for_propagation = False

    def _considerUnusedUserLocalVariables(self, trace_collection):
        if False:
            return 10
        'Check scope for unused variables.'
        provided = self.getProvidedVariables()
        removals = []
        for variable in provided:
            if variable.isLocalVariable() and (not variable.isParameterVariable()) and (variable.getOwner() is self.owner):
                empty = trace_collection.hasEmptyTraces(variable)
                if empty:
                    removals.append(variable)
        for variable in removals:
            self.unregisterProvidedVariable(variable)
            trace_collection.signalChange('var_usage', self.owner.getSourceReference(), message="Remove unused local variable '%s'." % variable.getName())

class LocalsDictHandle(LocalsDictHandleBase):
    """Locals dict for a Python class with mere dict."""
    __slots__ = ()

    @staticmethod
    def isClassScope():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def getMappingValueShape(variable):
        if False:
            i = 10
            return i + 15
        return tshape_unknown

    def _considerPropagation(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if not self.variables:
            return
        for variable in self.variables.values():
            for variable_trace in variable.traces:
                if variable_trace.inhibitsClassScopeForwardPropagation():
                    return
        trace_collection.signalChange('var_usage', self.owner.getSourceReference(), message='Forward propagate locals dictionary.')
        self.markForLocalsDictPropagation()

class LocalsMappingHandle(LocalsDictHandle):
    """Locals dict of a Python3 class with a mapping."""
    __slots__ = ('type_shape',)
    if python_version >= 832:
        __slots__ += ('prevented_propagation',)

    def __init__(self, locals_name, owner):
        if False:
            print('Hello World!')
        LocalsDictHandle.__init__(self, locals_name=locals_name, owner=owner)
        self.type_shape = tshape_unknown
        if python_version >= 832:
            self.prevented_propagation = False

    def getTypeShape(self):
        if False:
            i = 10
            return i + 15
        return self.type_shape

    def setTypeShape(self, type_shape):
        if False:
            while True:
                i = 10
        self.type_shape = type_shape

    def hasShapeDictionaryExact(self):
        if False:
            return 10
        return self.type_shape is tshape_dict
    if python_version >= 832:

        def markAsComplete(self, trace_collection):
            if False:
                for i in range(10):
                    print('nop')
            if self.prevented_propagation:
                self.prevented_propagation = False
                return
            self.complete = True

        def preventLocalsDictPropagation(self):
            if False:
                i = 10
                return i + 15
            self.prevented_propagation = True

        def isPreventedPropagation(self):
            if False:
                i = 10
                return i + 15
            return self.prevented_propagation

    def _considerPropagation(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if not self.variables:
            return
        if self.type_shape is not tshape_dict:
            return
        for variable in self.variables.values():
            for variable_trace in variable.traces:
                if variable_trace.inhibitsClassScopeForwardPropagation():
                    return
        trace_collection.signalChange('var_usage', self.owner.getSourceReference(), message='Forward propagate locals dictionary.')
        self.markForLocalsDictPropagation()

    @staticmethod
    def isClassScope():
        if False:
            while True:
                i = 10
        return True

class LocalsDictExecHandle(LocalsDictHandleBase):
    """Locals dict of a Python2 function with an exec."""
    __slots__ = ('closure_variables',)

    def __init__(self, locals_name, owner):
        if False:
            while True:
                i = 10
        LocalsDictHandleBase.__init__(self, locals_name=locals_name, owner=owner)
        self.closure_variables = None

    @staticmethod
    def isFunctionScope():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isUnoptimizedFunctionScope():
        if False:
            return 10
        return True

    def getLocalsRelevantVariables(self):
        if False:
            print('Hello World!')
        if self.closure_variables is None:
            return self.providing.values()
        else:
            return [variable for variable in self.providing.values() if variable not in self.closure_variables]

    def registerClosureVariable(self, variable):
        if False:
            print('Hello World!')
        self.registerProvidedVariable(variable)
        if self.closure_variables is None:
            self.closure_variables = set()
        self.closure_variables.add(variable)

    def unregisterClosureVariable(self, variable):
        if False:
            print('Hello World!')
        self.unregisterProvidedVariable(variable)
        variable_name = variable.getName()
        if variable_name in self.providing:
            del self.providing[variable_name]

class LocalsDictFunctionHandle(LocalsDictHandleBase):
    """Locals dict of a Python3 function or Python2 function without an exec."""
    __slots__ = ()

    @staticmethod
    def isFunctionScope():
        if False:
            for i in range(10):
                print('nop')
        return True

class GlobalsDictHandle(LocalsDictHandleBase):
    __slots__ = ('escaped',)

    def __init__(self, locals_name, owner):
        if False:
            i = 10
            return i + 15
        LocalsDictHandleBase.__init__(self, locals_name=locals_name, owner=owner)
        self.escaped = False

    @staticmethod
    def isModuleScope():
        if False:
            print('Hello World!')
        return True

    def markAsEscaped(self):
        if False:
            for i in range(10):
                print('nop')
        self.escaped = True

    def isEscaped(self):
        if False:
            while True:
                i = 10
        return self.escaped