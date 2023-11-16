""" Module/Package nodes

The top of the tree. Packages are also modules. Modules are what hold a program
together and cross-module optimizations are the most difficult to tackle.
"""
import os
from nuitka import Options, Variables
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.importing.Importing import locateModule, makeModuleUsageAttempt
from nuitka.importing.Recursion import decideRecursion, recurseTo
from nuitka.ModuleRegistry import getModuleByName, getOwnerFromCodeName
from nuitka.optimizations.TraceCollections import TraceCollectionModule
from nuitka.Options import hasPythonFlagIsolated
from nuitka.PythonVersions import python_version
from nuitka.SourceCodeReferences import fromFilename
from nuitka.tree.SourceHandling import parsePyIFile, readSourceCodeFromFilename
from nuitka.utils.CStrings import encodePythonIdentifierToC
from nuitka.utils.ModuleNames import ModuleName
from .ChildrenHavingMixins import ModuleChildrenHavingBodyOptionalStatementsOrNoneFunctionsTupleMixin
from .FutureSpecs import fromFlags
from .IndicatorMixins import EntryPointMixin, MarkNeedsAnnotationsMixin
from .LocalsScopes import getLocalsDictHandle
from .NodeBases import ClosureGiverNodeMixin, NodeBase, extractKindAndArgsFromXML, fromXML

class PythonModuleBase(NodeBase):
    __slots__ = ('module_name', 'reason')

    def __init__(self, module_name, reason, source_ref):
        if False:
            i = 10
            return i + 15
        assert type(module_name) is ModuleName, module_name
        NodeBase.__init__(self, source_ref=source_ref)
        self.module_name = module_name
        self.reason = reason

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'module_name': self.module_name}

    def getFullName(self):
        if False:
            print('Hello World!')
        return self.module_name

    @staticmethod
    def isMainModule():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isTopModule():
        if False:
            while True:
                i = 10
        return False

    def attemptRecursion(self):
        if False:
            i = 10
            return i + 15
        package_name = self.module_name.getPackageName()
        if package_name is None:
            return ()
        package = getModuleByName(package_name)
        if package_name is not None and package is None:
            (_package_name, package_filename, package_module_kind, finding) = locateModule(module_name=package_name, parent_package=None, level=0)
            if python_version >= 768 and (not package_filename):
                return ()
            if package_name == 'uniconvertor.app.modules':
                return ()
            assert package_filename is not None, (package_name, finding)
            assert _package_name == package_name, (package_filename, _package_name, package_name)
            (decision, _reason) = decideRecursion(using_module_name=self.getFullName(), module_filename=package_filename, module_name=package_name, module_kind=package_module_kind)
            if decision is not None:
                package = recurseTo(module_name=package_name, module_filename=package_filename, module_kind=package_module_kind, source_ref=self.source_ref, reason='parent package', using_module_name=self.module_name)
        if package:
            from nuitka.ModuleRegistry import addUsedModule
            addUsedModule(package, using_module=self, usage_tag='package', reason="Containing package of '%s'." % self.getFullName(), source_ref=self.source_ref)

    def getCodeName(self):
        if False:
            i = 10
            return i + 15
        return None

    def getCompileTimeFilename(self):
        if False:
            i = 10
            return i + 15
        'The compile time filename for the module.\n\n        Returns:\n            Full path to module file at compile time.\n        Notes:\n            We are getting the absolute path here, since we do\n            not want to have to deal with resolving paths at\n            all.\n\n        '
        return os.path.abspath(self.source_ref.getFilename())

    def getCompileTimeDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        'The compile time directory for the module.\n\n        Returns:\n            Full path to module directory at compile time.\n        Notes:\n            For packages, we let the package directory be\n            the result, otherwise the containing directory\n            is the result.\n        Notes:\n            Use this to find files nearby a module, mainly\n            in plugin code.\n        '
        result = self.getCompileTimeFilename()
        if not os.path.isdir(result):
            result = os.path.dirname(result)
        return result

    def getRunTimeFilename(self):
        if False:
            i = 10
            return i + 15
        reference_mode = Options.getFileReferenceMode()
        if reference_mode == 'original':
            return self.getCompileTimeFilename()
        elif reference_mode == 'frozen':
            return '<frozen %s>' % self.getFullName()
        else:
            filename = self.getCompileTimeFilename()
            full_name = self.getFullName()
            result = os.path.basename(filename)
            current = filename
            levels = full_name.count('.')
            if self.isCompiledPythonPackage():
                levels += 1
            for _i in range(levels):
                current = os.path.dirname(current)
                result = os.path.join(os.path.basename(current), result)
            return result

class CompiledPythonModule(ModuleChildrenHavingBodyOptionalStatementsOrNoneFunctionsTupleMixin, ClosureGiverNodeMixin, MarkNeedsAnnotationsMixin, EntryPointMixin, PythonModuleBase):
    """Compiled Python Module"""
    kind = 'COMPILED_PYTHON_MODULE'
    __slots__ = ('is_top', 'name', 'code_prefix', 'code_name', 'uids', 'temp_variables', 'temp_scopes', 'preserver_id', 'needs_annotations_dict', 'trace_collection', 'mode', 'variables', 'active_functions', 'visited_functions', 'cross_used_functions', 'used_modules', 'future_spec', 'source_code', 'module_dict_name', 'locals_scope')
    named_children = ('body|statements_or_none+setter', 'functions|tuple+setter')

    def __init__(self, module_name, reason, is_top, mode, future_spec, source_ref):
        if False:
            for i in range(10):
                print('nop')
        PythonModuleBase.__init__(self, module_name=module_name, reason=reason, source_ref=source_ref)
        ClosureGiverNodeMixin.__init__(self, name=module_name.getBasename(), code_prefix='module')
        ModuleChildrenHavingBodyOptionalStatementsOrNoneFunctionsTupleMixin.__init__(self, body=None, functions=())
        MarkNeedsAnnotationsMixin.__init__(self)
        EntryPointMixin.__init__(self)
        self.is_top = is_top
        self.mode = mode
        self.variables = {}
        self.active_functions = OrderedSet()
        self.visited_functions = set()
        self.cross_used_functions = OrderedSet()
        self.used_modules = OrderedSet()
        self.future_spec = future_spec
        self.source_code = None
        self.module_dict_name = 'globals_%s' % (self.getCodeName(),)
        self.locals_scope = getLocalsDictHandle(self.module_dict_name, 'module_dict', self)
        self.used_modules = OrderedSet()

    @staticmethod
    def isCompiledPythonModule():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        return {'filename': self.source_ref.getFilename(), 'module_name': self.module_name}

    def getDetailsForDisplay(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.getDetails()
        if self.future_spec is not None:
            result['code_flags'] = ','.join(self.future_spec.asFlags())
        return result

    def getCompilationMode(self):
        if False:
            i = 10
            return i + 15
        return self.mode

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            i = 10
            return i + 15
        assert False

    def getFutureSpec(self):
        if False:
            return 10
        return self.future_spec

    def setFutureSpec(self, future_spec):
        if False:
            while True:
                i = 10
        self.future_spec = future_spec

    def isTopModule(self):
        if False:
            i = 10
            return i + 15
        return self.is_top

    def asGraph(self, graph, desc):
        if False:
            print('Hello World!')
        graph = graph.add_subgraph(name='cluster_%s' % desc, comment='Graph for %s' % self.getName())

        def makeTraceNodeName(variable, version, variable_trace):
            if False:
                print('Hello World!')
            return '%s/ %s %s %s' % (desc, variable.getName(), version, variable_trace.__class__.__name__)
        for function_body in self.active_functions:
            trace_collection = function_body.trace_collection
            node_names = {}
            for ((variable, version), variable_trace) in trace_collection.getVariableTracesAll().items():
                node_name = makeTraceNodeName(variable, version, variable_trace)
                node_names[variable_trace] = node_name
            for ((variable, version), variable_trace) in trace_collection.getVariableTracesAll().items():
                node_name = node_names[variable_trace]
                previous = variable_trace.getPrevious()
                attrs = {'style': 'filled'}
                if variable_trace.getUsageCount():
                    attrs['color'] = 'blue'
                else:
                    attrs['color'] = 'red'
                graph.add_node(node_name, **attrs)
                if type(previous) is tuple:
                    for prev_trace in previous:
                        graph.add_edge(node_names[prev_trace], node_name)
                        assert prev_trace is not variable_trace
                elif previous is not None:
                    assert previous is not variable_trace
                    graph.add_edge(node_names[previous], node_name)
        return graph

    def getSourceCode(self):
        if False:
            print('Hello World!')
        if self.source_code is not None:
            return self.source_code
        else:
            return readSourceCodeFromFilename(module_name=self.getFullName(), source_filename=self.getCompileTimeFilename())

    def setSourceCode(self, code):
        if False:
            for i in range(10):
                print('nop')
        self.source_code = code

    def getParent(self):
        if False:
            print('Hello World!')
        return None

    def getParentVariableProvider(self):
        if False:
            print('Hello World!')
        return None

    def hasVariableName(self, variable_name):
        if False:
            while True:
                i = 10
        return variable_name in self.variables or variable_name in self.temp_variables

    def getProvidedVariables(self):
        if False:
            i = 10
            return i + 15
        return self.variables.values()

    def getFilename(self):
        if False:
            print('Hello World!')
        return self.source_ref.getFilename()

    def getVariableForAssignment(self, variable_name):
        if False:
            i = 10
            return i + 15
        return self.getProvidedVariable(variable_name)

    def getVariableForReference(self, variable_name):
        if False:
            while True:
                i = 10
        return self.getProvidedVariable(variable_name)

    def getVariableForClosure(self, variable_name):
        if False:
            return 10
        return self.getProvidedVariable(variable_name=variable_name)

    def createProvidedVariable(self, variable_name):
        if False:
            i = 10
            return i + 15
        assert variable_name not in self.variables
        result = Variables.ModuleVariable(module=self, variable_name=variable_name)
        self.variables[variable_name] = result
        return result

    @staticmethod
    def getContainingClassDictCreation():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def isEarlyClosure():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getEntryPoint(self):
        if False:
            i = 10
            return i + 15
        return self

    def getCodeName(self):
        if False:
            i = 10
            return i + 15
        return encodePythonIdentifierToC(self.getFullName())

    @staticmethod
    def getChildQualname(function_name):
        if False:
            while True:
                i = 10
        return function_name

    def addFunction(self, function_body):
        if False:
            return 10
        functions = self.subnode_functions
        assert function_body not in functions
        functions += (function_body,)
        self.setChildFunctions(functions)

    def startTraversal(self):
        if False:
            i = 10
            return i + 15
        self.used_modules = None
        self.active_functions = OrderedSet()

    def restartTraversal(self):
        if False:
            return 10
        self.visited_functions = set()
        self.used_modules = None

    def getUsedModules(self):
        if False:
            i = 10
            return i + 15
        return self.trace_collection.getModuleUsageAttempts()

    def getUsedDistributions(self):
        if False:
            i = 10
            return i + 15
        return self.trace_collection.getUsedDistributions()

    def addUsedFunction(self, function_body):
        if False:
            while True:
                i = 10
        assert function_body in self.subnode_functions, function_body
        assert function_body.isExpressionFunctionBody() or function_body.isExpressionClassBodyBase() or function_body.isExpressionGeneratorObjectBody() or function_body.isExpressionCoroutineObjectBody() or function_body.isExpressionAsyncgenObjectBody()
        self.active_functions.add(function_body)
        result = function_body not in self.visited_functions
        self.visited_functions.add(function_body)
        return result

    def getUsedFunctions(self):
        if False:
            return 10
        return self.active_functions

    def getUnusedFunctions(self):
        if False:
            i = 10
            return i + 15
        for function in self.subnode_functions:
            if function not in self.active_functions:
                yield function

    def addCrossUsedFunction(self, function_body):
        if False:
            print('Hello World!')
        if function_body not in self.cross_used_functions:
            self.cross_used_functions.add(function_body)

    def getCrossUsedFunctions(self):
        if False:
            return 10
        return self.cross_used_functions

    def getFunctionFromCodeName(self, code_name):
        if False:
            print('Hello World!')
        for function in self.subnode_functions:
            if function.getCodeName() == code_name:
                return function

    def getOutputFilename(self):
        if False:
            return 10
        main_filename = self.getFilename()
        if main_filename.endswith('.py'):
            result = main_filename[:-3]
        elif main_filename.endswith('.pyw'):
            result = main_filename[:-4]
        else:
            result = main_filename
        return result.replace(')', '').replace('(', '')

    def computeModule(self):
        if False:
            print('Hello World!')
        self.restartTraversal()
        old_collection = self.trace_collection
        self.trace_collection = TraceCollectionModule(self, very_trusted_module_variables=old_collection.getVeryTrustedModuleVariables() if old_collection is not None else {})
        module_body = self.subnode_body
        if module_body is not None:
            result = module_body.computeStatementsSequence(trace_collection=self.trace_collection)
            if result is not module_body:
                self.setChildBody(result)
        self.attemptRecursion()
        very_trusted_module_variables = {}
        for module_variable in self.locals_scope.getLocalsRelevantVariables():
            very_trusted_node = self.trace_collection.getVariableCurrentTrace(module_variable).getAttributeNodeVeryTrusted()
            if very_trusted_node is not None:
                very_trusted_module_variables[module_variable] = very_trusted_node
        if self.trace_collection.updateVeryTrustedModuleVariables(very_trusted_module_variables):
            self.trace_collection.signalChange(tags='trusted_module_variables', message="Trusting module variable(s) '%s'" % ','.join((variable.getName() for variable in self.trace_collection.getVeryTrustedModuleVariables())), source_ref=self.source_ref)
        self.trace_collection.updateVariablesFromCollection(old_collection=old_collection, source_ref=self.source_ref)
        was_complete = not self.locals_scope.complete

        def markAsComplete(body, trace_collection):
            if False:
                i = 10
                return i + 15
            if body.locals_scope is not None:
                if body.locals_scope.isMarkedForPropagation():
                    body.locals_scope.onPropagationComplete()
                body.locals_scope.markAsComplete(trace_collection)

        def markEntryPointAsComplete(body):
            if False:
                print('Hello World!')
            markAsComplete(body, body.trace_collection)
            outline_bodies = body.trace_collection.getOutlineFunctions()
            if outline_bodies is not None:
                for outline_body in outline_bodies:
                    markAsComplete(outline_body, body.trace_collection)
            body.optimizeUnusedTempVariables()
        markEntryPointAsComplete(self)
        for function_body in self.getUsedFunctions():
            markEntryPointAsComplete(function_body)
            function_body.optimizeUnusedClosureVariables()
            function_body.optimizeVariableReleases()
        return was_complete

    def getTraceCollections(self):
        if False:
            i = 10
            return i + 15
        yield self.trace_collection
        for function in self.getUsedFunctions():
            yield function.trace_collection

    def isUnoptimized(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def getLocalVariables(self):
        if False:
            for i in range(10):
                print('nop')
        return ()

    def getUserLocalVariables(self):
        if False:
            print('Hello World!')
        return ()

    @staticmethod
    def getFunctionVariablesWithAutoReleases():
        if False:
            while True:
                i = 10
        'Return the list of function variables that should be released at exit.'
        return ()

    def getOutlineLocalVariables(self):
        if False:
            i = 10
            return i + 15
        outlines = self.getTraceCollection().getOutlineFunctions()
        if outlines is None:
            return ()
        result = []
        for outline in outlines:
            result.extend(outline.getUserLocalVariables())
        return result

    def hasClosureVariable(self, variable):
        if False:
            while True:
                i = 10
        return False

    def removeUserVariable(self, variable):
        if False:
            i = 10
            return i + 15
        outlines = self.getTraceCollection().getOutlineFunctions()
        for outline in outlines:
            user_locals = outline.getUserLocalVariables()
            if variable in user_locals:
                outline.removeUserVariable(variable)
                break

    def getLocalsScope(self):
        if False:
            print('Hello World!')
        return self.locals_scope

    def getRuntimePackageValue(self):
        if False:
            while True:
                i = 10
        if self.isCompiledPythonPackage():
            return self.getFullName().asString()
        value = self.getFullName().getPackageName()
        if value is not None:
            return value.asString()
        if self.isMainModule():
            if self.main_added:
                return ''
            else:
                return None
        else:
            return None

    def getRuntimeNameValue(self):
        if False:
            print('Hello World!')
        if self.isMainModule() and Options.hasPythonFlagPackageMode():
            return '__main__'
        elif self.module_name.isMultidistModuleName():
            return '__main__'
        else:
            return self.getFullName().asString()

class CompiledPythonPackage(CompiledPythonModule):
    kind = 'COMPILED_PYTHON_PACKAGE'

    def __init__(self, module_name, reason, is_top, mode, future_spec, source_ref):
        if False:
            for i in range(10):
                print('nop')
        CompiledPythonModule.__init__(self, module_name=module_name, reason=reason, is_top=is_top, mode=mode, future_spec=future_spec, source_ref=source_ref)

    def getOutputFilename(self):
        if False:
            i = 10
            return i + 15
        result = self.getFilename()
        if os.path.isdir(result):
            return result
        else:
            return os.path.dirname(result)

    @staticmethod
    def canHaveExternalImports():
        if False:
            print('Hello World!')
        return not hasPythonFlagIsolated()

def makeUncompiledPythonModule(module_name, reason, filename, bytecode, is_package, technical):
    if False:
        return 10
    source_ref = fromFilename(filename)
    if is_package:
        return UncompiledPythonPackage(module_name=module_name, reason=reason, bytecode=bytecode, filename=filename, technical=technical, source_ref=source_ref)
    else:
        return UncompiledPythonModule(module_name=module_name, reason=reason, bytecode=bytecode, filename=filename, technical=technical, source_ref=source_ref)

class UncompiledPythonModule(PythonModuleBase):
    """Uncompiled Python Module"""
    kind = 'UNCOMPILED_PYTHON_MODULE'
    __slots__ = ('bytecode', 'filename', 'technical', 'used_modules', 'distribution_names')

    def __init__(self, module_name, reason, bytecode, filename, technical, source_ref):
        if False:
            i = 10
            return i + 15
        PythonModuleBase.__init__(self, module_name=module_name, reason=reason, source_ref=source_ref)
        self.bytecode = bytecode
        self.filename = filename
        self.technical = technical
        self.used_modules = ()
        self.distribution_names = ()

    def finalize(self):
        if False:
            return 10
        del self.used_modules
        del self.bytecode

    @staticmethod
    def isUncompiledPythonModule():
        if False:
            print('Hello World!')
        return True

    def isTechnical(self):
        if False:
            return 10
        "Must be bytecode as it's used in CPython library initialization."
        return self.technical

    def getByteCode(self):
        if False:
            i = 10
            return i + 15
        return self.bytecode

    def getFilename(self):
        if False:
            i = 10
            return i + 15
        return self.filename

    def getUsedModules(self):
        if False:
            while True:
                i = 10
        return self.used_modules

    def setUsedModules(self, used_modules):
        if False:
            while True:
                i = 10
        self.used_modules = used_modules

    def getUsedDistributions(self):
        if False:
            return 10
        return self.distribution_names

    def setUsedDistributions(self, distribution_names):
        if False:
            print('Hello World!')
        self.distribution_names = distribution_names

    @staticmethod
    def startTraversal():
        if False:
            for i in range(10):
                print('nop')
        pass

class UncompiledPythonPackage(UncompiledPythonModule):
    kind = 'UNCOMPILED_PYTHON_PACKAGE'

class PythonMainModule(CompiledPythonModule):
    """Main module of a program, typically "__main__" but can be inside a package too."""
    kind = 'PYTHON_MAIN_MODULE'
    __slots__ = ('main_added', 'standard_library_modules')

    def __init__(self, module_name, main_added, mode, future_spec, source_ref):
        if False:
            print('Hello World!')
        assert not Options.shallMakeModule()
        self.main_added = main_added
        CompiledPythonModule.__init__(self, module_name=module_name, reason='main', is_top=True, mode=mode, future_spec=future_spec, source_ref=source_ref)
        self.standard_library_modules = ()

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'filename': self.source_ref.getFilename(), 'module_name': self.module_name, 'main_added': self.main_added, 'mode': self.mode}

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            i = 10
            return i + 15
        future_spec = fromFlags(args['code_flags'])
        result = cls(main_added=args['main_added'] == 'True', mode=args['mode'], module_name=ModuleName(args['module_name']), future_spec=future_spec, source_ref=source_ref)
        from nuitka.ModuleRegistry import addRootModule
        addRootModule(result)
        function_work = []
        for xml in args['functions']:
            (_kind, node_class, func_args, source_ref) = extractKindAndArgsFromXML(xml, source_ref)
            if 'provider' in func_args:
                func_args['provider'] = getOwnerFromCodeName(func_args['provider'])
            else:
                func_args['provider'] = result
            if 'flags' in args:
                func_args['flags'] = set(func_args['flags'].split(','))
            if 'doc' not in args:
                func_args['doc'] = None
            function = node_class.fromXML(source_ref=source_ref, **func_args)
            function_work.append((function, iter(iter(xml).next()).next()))
        for (function, xml) in function_work:
            function.setChildBody(fromXML(provider=function, xml=xml, source_ref=function.getSourceReference()))
        result.setChildBody(fromXML(provider=result, xml=args['body'][0], source_ref=source_ref))
        return result

    @staticmethod
    def isMainModule():
        if False:
            return 10
        return True

    def getOutputFilename(self):
        if False:
            return 10
        if self.main_added:
            return os.path.dirname(self.getFilename())
        else:
            return CompiledPythonModule.getOutputFilename(self)

    def getUsedModules(self):
        if False:
            return 10
        for used_module in CompiledPythonModule.getUsedModules(self):
            yield used_module
        for used_module in self.standard_library_modules:
            yield used_module

    def setStandardLibraryModules(self, early_module_names, stdlib_modules_names):
        if False:
            print('Hello World!')
        self.standard_library_modules = OrderedSet()
        for early_module_name in early_module_names + stdlib_modules_names:
            (_early_module_name, module_filename, module_kind, finding) = locateModule(module_name=early_module_name, parent_package=None, level=0)
            assert finding != 'not-found'
            self.standard_library_modules.add(makeModuleUsageAttempt(module_name=early_module_name, filename=module_filename, module_kind=module_kind, finding=finding, level=0, source_ref=self.source_ref, reason='stdlib'))

class PythonExtensionModule(PythonModuleBase):
    kind = 'PYTHON_EXTENSION_MODULE'
    __slots__ = ('used_modules', 'technical')
    avoid_duplicates = set()

    def __init__(self, module_name, reason, technical, source_ref):
        if False:
            print('Hello World!')
        PythonModuleBase.__init__(self, module_name=module_name, reason=reason, source_ref=source_ref)
        assert os.path.basename(source_ref.getFilename()) != '<frozen>'
        assert module_name != '__main__'
        assert self.getFullName() not in self.avoid_duplicates, self.getFullName()
        self.avoid_duplicates.add(self.getFullName())
        self.technical = technical
        self.used_modules = None

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.used_modules

    def getFilename(self):
        if False:
            for i in range(10):
                print('nop')
        return self.source_ref.getFilename()

    @staticmethod
    def startTraversal():
        if False:
            return 10
        pass

    def isTechnical(self):
        if False:
            return 10
        "Must be present as it's used in CPython library initialization."
        return self.technical

    def getPyIFilename(self):
        if False:
            while True:
                i = 10
        'Get Python type description filename.'
        path = self.getFilename()
        filename = os.path.basename(path)
        dirname = os.path.dirname(path)
        return os.path.join(dirname, filename.split('.')[0]) + '.pyi'

    def _readPyIFile(self):
        if False:
            i = 10
            return i + 15
        'Read the .pyi file if present and scan for dependencies.'
        if self.used_modules is None:
            pyi_filename = self.getPyIFilename()
            if os.path.exists(pyi_filename):
                pyi_deps = parsePyIFile(module_name=self.getFullName(), pyi_filename=pyi_filename)
                if 'typing' in pyi_deps:
                    pyi_deps.discard('typing')
                if '__future__' in pyi_deps:
                    pyi_deps.discard('__future__')
                if self.getFullName() in pyi_deps:
                    pyi_deps.discard(self.getFullName())
                if self.getFullName().getPackageName() in pyi_deps:
                    pyi_deps.discard(self.getFullName().getPackageName())
                self.used_modules = tuple(pyi_deps)
            else:
                self.used_modules = ()

    def getPyIModuleImportedNames(self):
        if False:
            for i in range(10):
                print('nop')
        self._readPyIFile()
        assert '.' not in self.used_modules, self
        return self.used_modules

    @staticmethod
    def getUsedModules():
        if False:
            for i in range(10):
                print('nop')
        return ()

    @staticmethod
    def getUsedDistributions():
        if False:
            i = 10
            return i + 15
        return {}

    def getParentModule(self):
        if False:
            i = 10
            return i + 15
        return self