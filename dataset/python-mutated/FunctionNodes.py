""" Nodes for functions and their creations.

Lambdas are functions too. The functions are at the core of the language and
have their complexities.

Creating a CPython function object is an optional thing. Some things might
only be used to be called directly, while knowing exactly what it is. So
the "ExpressionFunctionCreation" might be used to provide that kind of
CPython reference, and may escape.

Coroutines and generators live in their dedicated module and share base
classes.
"""
import inspect
from nuitka import Options, Variables
from nuitka.Constants import isMutable
from nuitka.optimizations.TraceCollections import TraceCollectionPureFunction, withChangeIndicationsTo
from nuitka.PythonVersions import python_version
from nuitka.specs.ParameterSpecs import ParameterSpec, TooManyArguments, matchCall
from nuitka.Tracing import optimization_logger
from nuitka.tree.Extractions import updateVariableUsage
from nuitka.tree.TreeHelpers import makeDictCreationOrConstant2
from .ChildrenHavingMixins import ChildHavingBodyOptionalMixin, ChildrenHavingDefaultsTupleKwDefaultsOptionalAnnotationsOptionalFunctionRefMixin, ChildrenHavingFunctionValuesTupleMixin, ChildrenHavingKwDefaultsOptionalDefaultsTupleAnnotationsOptionalFunctionRefMixin
from .CodeObjectSpecs import CodeObjectSpec
from .ContainerMakingNodes import makeExpressionMakeTupleOrConstant
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .FutureSpecs import fromFlags
from .IndicatorMixins import EntryPointMixin, MarkUnoptimizedFunctionIndicatorMixin
from .LocalsScopes import getLocalsDictHandle
from .NodeBases import ClosureGiverNodeMixin, ClosureTakerMixin, SideEffectsFromChildrenMixin
from .NodeMakingHelpers import makeRaiseExceptionReplacementExpressionFromInstance, wrapExpressionWithSideEffects
from .shapes.BuiltinTypeShapes import tshape_function

class MaybeLocalVariableUsage(Exception):
    pass

class ExpressionFunctionBodyBase(ClosureTakerMixin, ClosureGiverNodeMixin, ChildHavingBodyOptionalMixin, ExpressionBase):
    __slots__ = ('provider', 'taken', 'name', 'code_prefix', 'code_name', 'uids', 'temp_variables', 'temp_scopes', 'preserver_id', 'flags')
    if python_version >= 832:
        __slots__ += ('qualname_provider',)
    if python_version >= 768:
        __slots__ += ('non_local_declarations',)
    named_children = ('body|optional+setter',)

    def __init__(self, provider, name, body, code_prefix, flags, source_ref):
        if False:
            while True:
                i = 10
        while provider.isExpressionOutlineBody():
            provider = provider.getParentVariableProvider()
        ChildHavingBodyOptionalMixin.__init__(self, body=body)
        ClosureTakerMixin.__init__(self, provider=provider)
        ClosureGiverNodeMixin.__init__(self, name=name, code_prefix=code_prefix)
        ExpressionBase.__init__(self, source_ref)
        self.flags = flags or None
        self.parent = provider
        if python_version >= 832:
            self.qualname_provider = provider
        if python_version >= 768:
            self.non_local_declarations = None

    @staticmethod
    def isExpressionFunctionBodyBase():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getEntryPoint(self):
        if False:
            i = 10
            return i + 15
        'Entry point for code.\n\n        Normally ourselves. Only outlines will refer to their parent which\n        technically owns them.\n\n        '
        return self

    def getContainingClassDictCreation(self):
        if False:
            return 10
        current = self
        while not current.isCompiledPythonModule():
            if current.isExpressionClassBodyBase():
                return current
            current = current.getParentVariableProvider()
        return None

    def hasFlag(self, flag):
        if False:
            for i in range(10):
                print('nop')
        return self.flags is not None and flag in self.flags

    def discardFlag(self, flag):
        if False:
            return 10
        if self.flags is not None:
            self.flags.discard(flag)

    @staticmethod
    def isEarlyClosure():
        if False:
            print('Hello World!')
        "Early closure taking means immediate binding of references.\n\n        Normally it's good to lookup name references immediately, but not for\n        functions. In case of a function body it is not allowed to do that,\n        because a later assignment needs to be queried first. Nodes need to\n        indicate via this if they would like to resolve references at the same\n        time as assignments.\n        "
        return False

    def getLocalsScope(self):
        if False:
            for i in range(10):
                print('nop')
        return self.locals_scope

    def hasVariableName(self, variable_name):
        if False:
            while True:
                i = 10
        return self.locals_scope.hasProvidedVariable(variable_name) or variable_name in self.temp_variables

    def getProvidedVariables(self):
        if False:
            i = 10
            return i + 15
        if self.locals_scope is not None:
            return self.locals_scope.getProvidedVariables()
        else:
            return ()

    def getLocalVariables(self):
        if False:
            i = 10
            return i + 15
        return [variable for variable in self.getProvidedVariables() if variable.isLocalVariable()]

    def getUserLocalVariables(self):
        if False:
            i = 10
            return i + 15
        return [variable for variable in self.getProvidedVariables() if variable.isLocalVariable() and (not variable.isParameterVariable()) if variable.getOwner() is self]

    def getOutlineLocalVariables(self):
        if False:
            i = 10
            return i + 15
        result = []
        outlines = self.getTraceCollection().getOutlineFunctions()
        if outlines is None:
            return result
        for outline in outlines:
            result.extend(outline.getUserLocalVariables())
        return result

    def removeClosureVariable(self, variable):
        if False:
            return 10
        assert not variable.isParameterVariable() or variable.getOwner() is not self
        self.locals_scope.unregisterClosureVariable(variable)
        self.taken.remove(variable)
        self.code_object.removeFreeVarname(variable.getName())

    def demoteClosureVariable(self, variable):
        if False:
            return 10
        assert variable.isLocalVariable()
        self.taken.remove(variable)
        assert variable.getOwner() is not self
        new_variable = Variables.LocalVariable(owner=self, variable_name=variable.getName())
        for variable_trace in variable.traces:
            if variable_trace.getOwner() is self:
                new_variable.addTrace(variable_trace)
        new_variable.updateUsageState()
        self.locals_scope.unregisterClosureVariable(variable)
        self.locals_scope.registerProvidedVariable(new_variable)
        updateVariableUsage(provider=self, old_variable=variable, new_variable=new_variable)

    def hasClosureVariable(self, variable):
        if False:
            return 10
        return variable in self.taken

    def getVariableForAssignment(self, variable_name):
        if False:
            print('Hello World!')
        if self.hasTakenVariable(variable_name):
            result = self.getTakenVariable(variable_name)
        else:
            result = self.getProvidedVariable(variable_name)
        return result

    def getVariableForReference(self, variable_name):
        if False:
            while True:
                i = 10
        if self.hasProvidedVariable(variable_name):
            result = self.getProvidedVariable(variable_name)
        else:
            result = self.getClosureVariable(variable_name=variable_name)
            if not result.isModuleVariable():
                self.locals_scope.registerClosureVariable(result)
            entry_point = self.getEntryPoint()
            if python_version < 768 and (not entry_point.isExpressionClassBodyBase()) and (not entry_point.isPythonMainModule()) and result.isModuleVariable() and entry_point.isUnoptimized():
                raise MaybeLocalVariableUsage
        return result

    def getVariableForClosure(self, variable_name):
        if False:
            return 10
        if self.hasProvidedVariable(variable_name):
            return self.getProvidedVariable(variable_name)
        return self.takeVariableForClosure(variable_name)

    def takeVariableForClosure(self, variable_name):
        if False:
            print('Hello World!')
        result = self.provider.getVariableForClosure(variable_name)
        self.taken.add(result)
        return result

    def createProvidedVariable(self, variable_name):
        if False:
            print('Hello World!')
        assert self.locals_scope, self
        return self.locals_scope.getLocalVariable(variable_name=variable_name, owner=self)

    def addNonlocalsDeclaration(self, names, user_provided, source_ref):
        if False:
            for i in range(10):
                print('nop')
        'Add a nonlocal declared name.\n\n        This happens during tree building, and is a Python3 only\n        feature. We remember the names for later use through the\n        function @consumeNonlocalDeclarations\n        '
        if self.non_local_declarations is None:
            self.non_local_declarations = []
        self.non_local_declarations.append((names, user_provided, source_ref))

    def consumeNonlocalDeclarations(self):
        if False:
            print('Hello World!')
        'Return the nonlocal declared names for this function.\n\n        There may not be any, which is why we assigned it to\n        None originally and now check and return empty tuple\n        in that case.\n        '
        result = self.non_local_declarations or ()
        self.non_local_declarations = None
        return result

    def getFunctionName(self):
        if False:
            while True:
                i = 10
        return self.name

    def getFunctionQualname(self):
        if False:
            print('Hello World!')
        'Function __qualname__ new in CPython3.3\n\n        Should contain some kind of full name descriptions for the closure to\n        recognize and will be used for outputs.\n        '
        function_name = self.getFunctionName()
        if python_version < 832:
            qualname_provider = self.getParentVariableProvider()
        else:
            qualname_provider = self.qualname_provider
        return qualname_provider.getChildQualname(function_name)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        assert False
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        body = self.subnode_body
        if body is None:
            return False
        else:
            return self.subnode_body.mayRaiseException(exception_type)

    def getFunctionInlineCost(self, values):
        if False:
            while True:
                i = 10
        "Cost of inlining this function with given arguments\n\n        Returns: None or integer values, None means don't do it.\n        "
        return None

    def optimizeUnusedClosureVariables(self):
        if False:
            return 10
        'Gets called once module is complete, to consider giving up on closure variables.'
        changed = False
        for closure_variable in self.getClosureVariables():
            if closure_variable.isParameterVariable() and self.isExpressionGeneratorObjectBody():
                continue
            empty = self.trace_collection.hasEmptyTraces(closure_variable)
            if empty:
                changed = True
                self.trace_collection.signalChange('var_usage', self.source_ref, message="Remove unused closure variable '%s'." % closure_variable.getName())
                self.removeClosureVariable(closure_variable)
        return changed

    def optimizeVariableReleases(self):
        if False:
            while True:
                i = 10
        for parameter_variable in self.getParameterVariablesWithManualRelease():
            read_only = self.trace_collection.hasReadOnlyTraces(parameter_variable)
            if read_only:
                self.trace_collection.signalChange('var_usage', self.source_ref, message="Schedule removal releases of unassigned parameter variable '%s'." % parameter_variable.getName())
                self.removeVariableReleases(parameter_variable)

class ExpressionFunctionEntryPointBase(EntryPointMixin, ExpressionFunctionBodyBase):
    __slots__ = ('trace_collection', 'code_object', 'locals_scope', 'auto_release')

    def __init__(self, provider, name, code_object, code_prefix, flags, auto_release, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionFunctionBodyBase.__init__(self, provider=provider, name=name, code_prefix=code_prefix, flags=flags, body=None, source_ref=source_ref)
        EntryPointMixin.__init__(self)
        self.code_object = code_object
        provider.getParentModule().addFunction(self)
        if flags is not None and 'has_exec' in flags:
            locals_kind = 'python2_function_exec'
        else:
            locals_kind = 'python_function'
        self.locals_scope = getLocalsDictHandle('locals_%s' % self.getCodeName(), locals_kind, self)
        self.auto_release = auto_release or None

    def getDetails(self):
        if False:
            print('Hello World!')
        result = ExpressionFunctionBodyBase.getDetails(self)
        result['auto_release'] = tuple(sorted(self.auto_release or ()))
        return result

    def getCodeObject(self):
        if False:
            i = 10
            return i + 15
        return self.code_object

    def getChildQualname(self, function_name):
        if False:
            while True:
                i = 10
        return self.getFunctionQualname() + '.<locals>.' + function_name

    def computeFunctionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        from nuitka.optimizations.TraceCollections import TraceCollectionFunction
        trace_collection = TraceCollectionFunction(parent=trace_collection, function_body=self)
        old_collection = self.setTraceCollection(trace_collection)
        self.computeFunction(trace_collection)
        trace_collection.updateVariablesFromCollection(old_collection, self.source_ref)

    def computeFunction(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        statements_sequence = self.subnode_body
        if statements_sequence is not None and self.isExpressionFunctionBody():
            if statements_sequence.subnode_statements[0].isStatementReturnNone():
                statements_sequence.finalize()
                self.setChildBody(None)
                statements_sequence = None
        if statements_sequence is not None:
            result = statements_sequence.computeStatementsSequence(trace_collection=trace_collection)
            if result is not statements_sequence:
                self.setChildBody(result)

    def removeVariableReleases(self, variable):
        if False:
            while True:
                i = 10
        assert variable in self.locals_scope.providing.values(), (self, variable)
        if self.auto_release is None:
            self.auto_release = set()
        self.auto_release.add(variable)

    def getParameterVariablesWithManualRelease(self):
        if False:
            print('Hello World!')
        'Return the list of parameter variables that have release statements.\n\n        These are for consideration if these can be dropped, and if so, they\n        are releases automatically by function code.\n        '
        return tuple((variable for variable in self.locals_scope.getProvidedVariables() if not self.auto_release or variable not in self.auto_release if variable.isParameterVariable() if variable.getOwner() is self))

    def isAutoReleaseVariable(self, variable):
        if False:
            for i in range(10):
                print('nop')
        'Is this variable to be automatically released.'
        return self.auto_release is not None and variable in self.auto_release

    def getFunctionVariablesWithAutoReleases(self):
        if False:
            print('Hello World!')
        'Return the list of function variables that should be released at exit.'
        if self.auto_release is None:
            return ()
        return tuple((variable for variable in self.locals_scope.getProvidedVariables() if variable in self.auto_release))

    @staticmethod
    def getConstantReturnValue():
        if False:
            return 10
        'Special function that checks if code generation allows to use common C code.\n\n        Notes:\n            This is only done for standard functions.\n\n        '
        return (False, False)

class ExpressionFunctionBody(ExpressionNoSideEffectsMixin, MarkUnoptimizedFunctionIndicatorMixin, ExpressionFunctionEntryPointBase):
    kind = 'EXPRESSION_FUNCTION_BODY'
    __slots__ = ('unoptimized_locals', 'unqualified_exec', 'doc', 'return_exception', 'needs_creation', 'needs_direct', 'cross_module_use', 'parameters')
    if python_version >= 832:
        __slots__ += ('qualname_setup',)

    def __init__(self, provider, name, code_object, doc, parameters, flags, auto_release, source_ref):
        if False:
            print('Hello World!')
        ExpressionFunctionEntryPointBase.__init__(self, provider=provider, name=name, code_object=code_object, code_prefix='function', flags=flags, auto_release=auto_release, source_ref=source_ref)
        MarkUnoptimizedFunctionIndicatorMixin.__init__(self, flags)
        self.doc = doc
        self.return_exception = False
        self.needs_creation = False
        self.needs_direct = False
        self.cross_module_use = False
        if python_version >= 832:
            self.qualname_setup = None
        self.parameters = parameters
        self.parameters.setOwner(self)
        for variable in self.parameters.getAllVariables():
            self.locals_scope.registerProvidedVariable(variable)

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'name': self.getFunctionName(), 'ref_name': self.getCodeName(), 'parameters': self.getParameters(), 'code_object': self.code_object, 'provider': self.provider.getCodeName(), 'doc': self.doc, 'flags': self.flags}

    def getDetailsForDisplay(self):
        if False:
            while True:
                i = 10
        result = {'name': self.getFunctionName(), 'provider': self.provider.getCodeName(), 'flags': self.flags}
        result.update(self.parameters.getDetails())
        if self.code_object:
            result.update(self.code_object.getDetails())
        if self.doc is not None:
            result['doc'] = self.doc
        return result

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            print('Hello World!')
        assert provider is not None
        parameter_spec_args = {}
        code_object_args = {}
        other_args = {}
        for (key, value) in args.items():
            if key.startswith('ps_'):
                parameter_spec_args[key] = value
            elif key.startswith('co_'):
                code_object_args[key] = value
            elif key == 'code_flags':
                code_object_args['future_spec'] = fromFlags(args['code_flags'])
            else:
                other_args[key] = value
        parameters = ParameterSpec(**parameter_spec_args)
        code_object = CodeObjectSpec(**code_object_args)
        if 'doc' not in other_args:
            other_args['doc'] = None
        return cls(provider=provider, parameters=parameters, code_object=code_object, source_ref=source_ref, **other_args)

    @staticmethod
    def isExpressionFunctionBody():
        if False:
            print('Hello World!')
        return True

    def getParent(self):
        if False:
            while True:
                i = 10
        assert False

    def getDoc(self):
        if False:
            for i in range(10):
                print('nop')
        return self.doc

    def getParameters(self):
        if False:
            while True:
                i = 10
        return self.parameters

    def needsCreation(self):
        if False:
            return 10
        return self.needs_creation

    def markAsNeedsCreation(self):
        if False:
            while True:
                i = 10
        self.needs_creation = True

    def needsDirectCall(self):
        if False:
            i = 10
            return i + 15
        return self.needs_direct

    def markAsDirectlyCalled(self):
        if False:
            print('Hello World!')
        self.needs_direct = True

    def isCrossModuleUsed(self):
        if False:
            while True:
                i = 10
        return self.cross_module_use

    def markAsCrossModuleUsed(self):
        if False:
            for i in range(10):
                print('nop')
        self.cross_module_use = True

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            return 10
        assert False, self

    @staticmethod
    def isCompileTimeConstant():
        if False:
            for i in range(10):
                print('nop')
        return False

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        body = self.subnode_body
        return body is not None and body.mayRaiseException(exception_type)

    def markAsExceptionReturnValue(self):
        if False:
            i = 10
            return i + 15
        self.return_exception = True

    def needsExceptionReturnValue(self):
        if False:
            while True:
                i = 10
        return self.return_exception

    def getConstantReturnValue(self):
        if False:
            for i in range(10):
                print('nop')
        'Special function that checks if code generation allows to use common C code.'
        body = self.subnode_body
        if body is None:
            return (True, None)
        first_statement = body.subnode_statements[0]
        if first_statement.isStatementReturnConstant():
            constant_value = first_statement.getConstant()
            if not isMutable(constant_value):
                return (True, constant_value)
            else:
                return (False, False)
        else:
            return (False, False)

class ExpressionFunctionPureBody(ExpressionFunctionBody):
    kind = 'EXPRESSION_FUNCTION_PURE_BODY'
    __slots__ = ('optimization_done',)

    def __init__(self, provider, name, code_object, doc, parameters, flags, auto_release, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionFunctionBody.__init__(self, provider=provider, name=name, code_object=code_object, doc=doc, parameters=parameters, flags=flags, auto_release=auto_release, source_ref=source_ref)
        self.optimization_done = False

    def computeFunctionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.optimization_done:
            for function_body in self.trace_collection.getUsedFunctions():
                trace_collection.onUsedFunction(function_body)
            return

        def mySignal(tag, source_ref, change_desc):
            if False:
                for i in range(10):
                    print('nop')
            if Options.is_verbose:
                optimization_logger.info('{source_ref} : {tags} : {message}'.format(source_ref=source_ref.getAsString(), tags=tag, message=change_desc() if inspect.isfunction(change_desc) else change_desc))
            tags.add(tag)
        tags = set()
        while 1:
            trace_collection = TraceCollectionPureFunction(function_body=self)
            old_collection = self.setTraceCollection(trace_collection)
            with withChangeIndicationsTo(mySignal):
                self.computeFunction(trace_collection)
            trace_collection.updateVariablesFromCollection(old_collection, self.source_ref)
            if tags:
                tags.clear()
            else:
                break
        self.optimization_done = True

class ExpressionFunctionPureInlineConstBody(ExpressionFunctionBody):
    kind = 'EXPRESSION_FUNCTION_PURE_INLINE_CONST_BODY'

    def getFunctionInlineCost(self, values):
        if False:
            return 10
        return 0

def makeExpressionFunctionCreation(function_ref, defaults, kw_defaults, annotations, source_ref):
    if False:
        while True:
            i = 10
    if kw_defaults is not None and kw_defaults.isExpressionConstantDictEmptyRef():
        kw_defaults = None
    assert function_ref.isExpressionFunctionRef()
    return ExpressionFunctionCreation(function_ref=function_ref, defaults=defaults, kw_defaults=kw_defaults, annotations=annotations, source_ref=source_ref)

class ExpressionFunctionCreationMixin(SideEffectsFromChildrenMixin):
    __slots__ = ()

    @staticmethod
    def isExpressionFunctionCreation():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getName(self):
        if False:
            return 10
        return self.subnode_function_ref.getName()

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_function

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        self.variable_closure_traces = []
        for closure_variable in self.subnode_function_ref.getFunctionBody().getClosureVariables():
            trace = trace_collection.getVariableCurrentTrace(closure_variable)
            trace.addNameUsage()
            self.variable_closure_traces.append((closure_variable, trace))
        kw_defaults = self.subnode_kw_defaults
        if kw_defaults is not None:
            kw_defaults.onContentEscapes(trace_collection)
        for default in self.subnode_defaults:
            default.onContentEscapes(trace_collection)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        for default in self.subnode_defaults:
            if default.mayRaiseException(exception_type):
                return True
        kw_defaults = self.subnode_kw_defaults
        if kw_defaults is not None and kw_defaults.mayRaiseException(exception_type):
            return True
        annotations = self.subnode_annotations
        if annotations is not None and annotations.mayRaiseException(exception_type):
            return True
        return False

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        if call_kw is not None and (not call_kw.isExpressionConstantDictEmptyRef()):
            return (call_node, None, None)
        if call_args is None:
            args_tuple = ()
        else:
            assert call_args.isExpressionConstantTupleRef() or call_args.isExpressionMakeTuple()
            args_tuple = call_args.getIterationValues()
        function_body = self.subnode_function_ref.getFunctionBody()
        call_spec = function_body.getParameters()
        try:
            args_dict = matchCall(func_name=self.getName(), args=call_spec.getArgumentNames(), kw_only_args=call_spec.getKwOnlyParameterNames(), star_list_arg=call_spec.getStarListArgumentName(), star_dict_arg=call_spec.getStarDictArgumentName(), star_list_single_arg=False, num_defaults=call_spec.getDefaultCount(), num_pos_only=call_spec.getPosOnlyParameterCount(), positional=args_tuple, pairs=())
            values = [args_dict[name] for name in call_spec.getParameterNames()]
            if None in values:
                return (call_node, None, None)
            if call_spec.getStarDictArgumentName():
                values[-1] = makeDictCreationOrConstant2(keys=[value[0] for value in values[-1]], values=[value[1] for value in values[-1]], source_ref=call_node.source_ref)
                star_list_offset = -2
            else:
                star_list_offset = -1
            if call_spec.getStarListArgumentName():
                values[star_list_offset] = makeExpressionMakeTupleOrConstant(elements=values[star_list_offset], user_provided=False, source_ref=call_node.source_ref)
            result = makeExpressionFunctionCall(function=self.makeClone(), values=values, source_ref=call_node.source_ref)
            return (result, 'new_statements', "Replaced call to created function body '%s' with direct function call." % self.getName())
        except TooManyArguments as e:
            result = wrapExpressionWithSideEffects(new_node=makeRaiseExceptionReplacementExpressionFromInstance(expression=call_node, exception=e.getRealException()), old_node=call_node, side_effects=call_node.extractSideEffectsPreCall())
            return (result, 'new_raise', "Replaced call to created function body '%s' to argument error" % self.getName())

    def getClosureVariableVersions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_closure_traces

class ExpressionFunctionCreationOld(ExpressionFunctionCreationMixin, ChildrenHavingKwDefaultsOptionalDefaultsTupleAnnotationsOptionalFunctionRefMixin, ExpressionBase):
    kind = 'EXPRESSION_FUNCTION_CREATION_OLD'
    python_version_spec = '< 0x340'
    kw_defaults_before_defaults = True
    named_children = ('kw_defaults|optional', 'defaults|tuple', 'annotations|optional', 'function_ref')
    __slots__ = ('variable_closure_traces',)

    def __init__(self, kw_defaults, defaults, annotations, function_ref, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingKwDefaultsOptionalDefaultsTupleAnnotationsOptionalFunctionRefMixin.__init__(self, kw_defaults=kw_defaults, defaults=defaults, annotations=annotations, function_ref=function_ref)
        ExpressionBase.__init__(self, source_ref)
        self.variable_closure_traces = None

class ExpressionFunctionCreation(ExpressionFunctionCreationMixin, ChildrenHavingDefaultsTupleKwDefaultsOptionalAnnotationsOptionalFunctionRefMixin, ExpressionBase):
    kind = 'EXPRESSION_FUNCTION_CREATION'
    python_version_spec = '>= 0x340'
    kw_defaults_before_defaults = False
    named_children = ('defaults|tuple', 'kw_defaults|optional', 'annotations|optional', 'function_ref')
    __slots__ = ('variable_closure_traces',)

    def __init__(self, defaults, kw_defaults, annotations, function_ref, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenHavingDefaultsTupleKwDefaultsOptionalAnnotationsOptionalFunctionRefMixin.__init__(self, kw_defaults=kw_defaults, defaults=defaults, annotations=annotations, function_ref=function_ref)
        ExpressionBase.__init__(self, source_ref)
        self.variable_closure_traces = None
if python_version < 832:
    ExpressionFunctionCreation = ExpressionFunctionCreationOld

class ExpressionFunctionRef(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_FUNCTION_REF'
    __slots__ = ('function_body', 'code_name')

    def __init__(self, source_ref, function_body=None, code_name=None):
        if False:
            while True:
                i = 10
        assert function_body is not None or code_name is not None
        assert code_name != 'None'
        ExpressionBase.__init__(self, source_ref)
        self.function_body = function_body
        self.code_name = code_name

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent
        del self.function_body

    def getName(self):
        if False:
            return 10
        return self.function_body.getName()

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'function_body': self.function_body}

    def getDetailsForDisplay(self):
        if False:
            while True:
                i = 10
        return {'code_name': self.getFunctionBody().getCodeName()}

    def getFunctionBody(self):
        if False:
            while True:
                i = 10
        if self.function_body is None:
            (module_code_name, _) = self.code_name.split('$$$', 1)
            from nuitka.ModuleRegistry import getModuleFromCodeName
            module = getModuleFromCodeName(module_code_name)
            self.function_body = module.getFunctionFromCodeName(self.code_name)
        return self.function_body

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onUsedFunction(self.getFunctionBody())
        return (self, None, None)

def makeExpressionFunctionCall(function, values, source_ref):
    if False:
        print('Hello World!')
    assert function.isExpressionFunctionCreation()
    return ExpressionFunctionCall(function=function, values=tuple(values), source_ref=source_ref)

class ExpressionFunctionCall(ChildrenHavingFunctionValuesTupleMixin, ExpressionBase):
    """Shared function call.

    This is for calling created function bodies with multiple users. Not
    clear if such a thing should exist. But what this will do is to have
    respect for the fact that there are multiple such calls.
    """
    kind = 'EXPRESSION_FUNCTION_CALL'
    __slots__ = ('variable_closure_traces',)
    named_children = ('function', 'values|tuple')

    def __init__(self, function, values, source_ref):
        if False:
            return 10
        ChildrenHavingFunctionValuesTupleMixin.__init__(self, function=function, values=values)
        ExpressionBase.__init__(self, source_ref)
        self.variable_closure_traces = None

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        function = self.subnode_function
        function_body = function.subnode_function_ref.getFunctionBody()
        if function_body.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        values = self.subnode_values
        cost = function_body.getFunctionInlineCost(values)
        if cost is not None and cost < 50:
            from nuitka.optimizations.FunctionInlining import convertFunctionCallToOutline
            result = convertFunctionCallToOutline(provider=self.getParentVariableProvider(), function_body=function_body, values=values, call_source_ref=self.source_ref)
            return (result, 'new_statements', lambda : "Function call to '%s' in-lined." % function_body.getCodeName())
        self.variable_closure_traces = []
        for closure_variable in function_body.getClosureVariables():
            trace = trace_collection.getVariableCurrentTrace(closure_variable)
            trace.addNameUsage()
            self.variable_closure_traces.append((closure_variable, trace))
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        function = self.subnode_function
        if function.subnode_function_ref.getFunctionBody().mayRaiseException(exception_type):
            return True
        values = self.subnode_values
        for value in values:
            if value.mayRaiseException(exception_type):
                return True
        return False

    def getClosureVariableVersions(self):
        if False:
            print('Hello World!')
        return self.variable_closure_traces