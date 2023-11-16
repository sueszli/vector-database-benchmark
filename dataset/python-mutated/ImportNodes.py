""" Nodes related to importing modules or names.

Normally imports are mostly relatively static, but Nuitka also attempts to
cover the uses of "__import__" built-in and other import techniques, that
allow dynamic values.

If other optimizations make it possible to predict these, the compiler can go
deeper that what it normally could. The import expression node can lead to
modules being added. After optimization it will be asked about used modules.
"""
import sys
from nuitka.__past__ import long, unicode, xrange
from nuitka.code_generation.Reports import onMissingTrust
from nuitka.HardImportRegistry import addModuleSingleAttributeNodeFactory, hard_modules_aliases, hard_modules_limited, hard_modules_non_stdlib, hard_modules_stdlib, hard_modules_trust, isHardModule, isHardModuleWithoutSideEffect, trust_constant, trust_importable, trust_may_exist, trust_node, trust_node_factory, trust_undefined
from nuitka.importing.Importing import isPackageDir, locateModule, makeModuleUsageAttempt
from nuitka.importing.ImportResolving import resolveModuleName
from nuitka.importing.Recursion import decideRecursion
from nuitka.importing.StandardLibrary import isStandardLibraryPath
from nuitka.Options import isExperimental, isStandaloneMode, shallMakeModule, shallWarnUnusualCode
from nuitka.PythonVersions import python_version
from nuitka.specs.BuiltinParameterSpecs import BuiltinParameterSpec, extractBuiltinArgs
from nuitka.Tracing import unusual_logger
from nuitka.utils.ModuleNames import ModuleName
from .ChildrenHavingMixins import ChildHavingModuleMixin, ChildrenExpressionBuiltinImportMixin, ChildrenExpressionImportlibImportModuleCallMixin
from .ExpressionBases import ExpressionBase
from .ImportHardNodes import ExpressionImportHardBase, ExpressionImportModuleNameHardExists, ExpressionImportModuleNameHardExistsSpecificBase, ExpressionImportModuleNameHardMaybeExists
from .LocalsScopes import GlobalsDictHandle
from .NodeMakingHelpers import makeConstantReplacementNode, makeRaiseExceptionReplacementExpression, makeRaiseImportErrorReplacementExpression
from .shapes.BuiltinTypeShapes import tshape_module, tshape_module_builtin
from .StatementBasesGenerated import StatementImportStarBase

def makeExpressionImportModuleNameHard(module_name, import_name, module_guaranteed, source_ref):
    if False:
        return 10
    if hard_modules_trust[module_name].get(import_name) is None:
        return ExpressionImportModuleNameHardMaybeExists(module_name=module_name, import_name=import_name, module_guaranteed=module_guaranteed, source_ref=source_ref)
    else:
        return ExpressionImportModuleNameHardExists(module_name=module_name, import_name=import_name, module_guaranteed=module_guaranteed, source_ref=source_ref)

class ExpressionImportAllowanceMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.finding == 'not-found':
            self.allowed = False
        elif self.finding == 'built-in':
            self.allowed = True
        elif self.module_name in hard_modules_stdlib:
            self.allowed = True
        else:
            (self.allowed, _reason) = decideRecursion(using_module_name=None, module_filename=self.module_filename, module_name=self.module_name, module_kind=self.module_kind)
            if self.allowed is None and self.isExpressionImportModuleHard():
                self.allowed = True

class ExpressionImportModuleFixed(ExpressionBase):
    """Hard coded import names, that we know to exist."

    These created as result of builtin imports and "importlib.import_module" calls
    that were compile time resolved, and for known module names.
    """
    kind = 'EXPRESSION_IMPORT_MODULE_FIXED'
    __slots__ = ('module_name', 'value_name', 'found_module_name', 'found_module_filename', 'module_kind', 'finding')

    def __init__(self, module_name, value_name, source_ref):
        if False:
            while True:
                i = 10
        ExpressionBase.__init__(self, source_ref)
        self.module_name = ModuleName(module_name)
        self.value_name = ModuleName(value_name)
        self.finding = None
        (self.found_module_name, self.found_module_filename, self.module_kind, self.finding) = self._attemptFollow()

    def _attemptFollow(self):
        if False:
            i = 10
            return i + 15
        (found_module_name, found_module_filename, module_kind, finding) = locateModule(module_name=self.module_name, parent_package=None, level=0)
        if self.finding == 'not-found':
            while True:
                module_name = found_module_filename.getPackageName()
                if module_name is None:
                    break
                (found_module_name, found_module_filename, module_kind, finding) = locateModule(module_name=module_name, parent_package=None, level=0)
                if self.finding != 'not-found':
                    break
        return (found_module_name, found_module_filename, module_kind, finding)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent

    def getDetails(self):
        if False:
            return 10
        return {'module_name': self.module_name, 'value_name': self.value_name}

    def getModuleName(self):
        if False:
            print('Hello World!')
        return self.module_name

    def getValueName(self):
        if False:
            i = 10
            return i + 15
        return self.value_name

    @staticmethod
    def mayHaveSideEffects():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            while True:
                i = 10
        return True

    def getTypeShape(self):
        if False:
            print('Hello World!')
        if self.module_name in sys.builtin_module_names:
            return tshape_module_builtin
        else:
            return tshape_module

    def getModuleUsageAttempt(self):
        if False:
            print('Hello World!')
        return makeModuleUsageAttempt(module_name=self.found_module_name, filename=self.found_module_filename, finding=self.finding, module_kind=self.module_kind, level=0, source_ref=self.source_ref, reason='import')

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.onModuleUsageAttempt(self.getModuleUsageAttempt())
        return (self, None, None)

    def computeExpressionImportName(self, import_node, import_name, trace_collection):
        if False:
            print('Hello World!')
        return self.computeExpressionAttribute(lookup_node=import_node, attribute_name=import_name, trace_collection=trace_collection)

class ExpressionImportModuleBuiltin(ExpressionBase):
    """Hard coded import names, that we know to exist."

    These created as result of builtin imports and "importlib.import_module" calls
    that were compile time resolved, and for known module names.
    """
    kind = 'EXPRESSION_IMPORT_MODULE_BUILTIN'
    __slots__ = ('module_name', 'value_name', 'module_kind', 'builtin_module')

    def __init__(self, module_name, value_name, source_ref):
        if False:
            while True:
                i = 10
        ExpressionBase.__init__(self, source_ref)
        self.module_name = ModuleName(module_name)
        self.value_name = ModuleName(value_name)
        self.builtin_module = __import__(module_name.asString())
        (_module_name, _module_filename, _module_kind, _finding) = locateModule(module_name=self.module_name, parent_package=None, level=0)
        assert _module_name == self.module_name, _module_name
        assert _finding == 'built-in', _finding
        assert _module_kind is None, _module_kind

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_module_builtin

    def mayRaiseExceptionImportName(self, exception_type, import_name):
        if False:
            print('Hello World!')
        return not hasattr(self.builtin_module, import_name)

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'module_name': self.module_name, 'value_name': self.value_name}

    def getModuleName(self):
        if False:
            while True:
                i = 10
        return self.module_name

    def getValueName(self):
        if False:
            while True:
                i = 10
        return self.value_name

    @staticmethod
    def mayHaveSideEffects():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            while True:
                i = 10
        return True

    def getModuleUsageAttempt(self):
        if False:
            print('Hello World!')
        return makeModuleUsageAttempt(module_name=self.module_name, filename=None, finding='built-in', module_kind=None, level=0, source_ref=self.source_ref, reason='import')

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        if self.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.onModuleUsageAttempt(self.getModuleUsageAttempt())
        return (self, None, None)

    def computeExpressionImportName(self, import_node, import_name, trace_collection):
        if False:
            print('Hello World!')
        return self.computeExpressionAttribute(lookup_node=import_node, attribute_name=import_name, trace_collection=trace_collection)

class ExpressionImportModuleHard(ExpressionImportAllowanceMixin, ExpressionImportHardBase):
    """Hard coded import names, e.g. of "__future__"

    These are directly created for some Python mechanics, but also due to
    compile time optimization for imports of statically known modules.
    """
    kind = 'EXPRESSION_IMPORT_MODULE_HARD'
    __slots__ = ('module', 'allowed', 'guaranteed', 'value_name', 'is_package')

    def __init__(self, module_name, value_name, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionImportHardBase.__init__(self, module_name=module_name, source_ref=source_ref)
        ExpressionImportAllowanceMixin.__init__(self)
        self.value_name = value_name
        if self.finding != 'not-found' and isHardModuleWithoutSideEffect(self.module_name):
            __import__(self.module_name.asString())
            self.module = sys.modules[self.value_name]
            self.is_package = hasattr(self.module, '__path__')
        else:
            self.module = None
            self.is_package = None
        self.guaranteed = self.allowed and (not shallMakeModule() or self.module_name not in hard_modules_non_stdlib)

    @staticmethod
    def isExpressionImportModuleHard():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            while True:
                i = 10
        return True

    def finalize(self):
        if False:
            return 10
        del self.parent

    def getDetails(self):
        if False:
            return 10
        return {'module_name': self.module_name, 'value_name': self.value_name}

    def getModuleName(self):
        if False:
            print('Hello World!')
        return self.module_name

    def getValueName(self):
        if False:
            i = 10
            return i + 15
        return self.value_name

    def mayHaveSideEffects(self):
        if False:
            for i in range(10):
                print('nop')
        return self.module is None or not self.guaranteed

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return not self.allowed or self.mayHaveSideEffects()

    def getTypeShape(self):
        if False:
            while True:
                i = 10
        if self.module_name in sys.builtin_module_names:
            return tshape_module_builtin
        else:
            return tshape_module

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        if self.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.onModuleUsageAttempt(self.getModuleUsageAttempt())
        return (self, None, None)

    def computeExpressionImportName(self, import_node, import_name, trace_collection):
        if False:
            print('Hello World!')
        return self._computeExpressionAttribute(lookup_node=import_node, attribute_name=import_name, trace_collection=trace_collection, is_import=True)

    @staticmethod
    def _getImportNameErrorString(module, module_name, name):
        if False:
            i = 10
            return i + 15
        if python_version < 832:
            return 'cannot import name %s' % name
        if python_version < 880:
            return 'cannot import name %r' % name
        elif isStandaloneMode():
            return 'cannot import name %r from %r' % (name, module_name)
        else:
            return 'cannot import name %r from %r (%s)' % (name, module_name, module.__file__ if hasattr(module, '__file__') else 'unknown location')

    def _makeRaiseExceptionReplacementExpression(self, lookup_node, attribute_name, is_import):
        if False:
            i = 10
            return i + 15
        if is_import:
            return makeRaiseExceptionReplacementExpression(expression=lookup_node, exception_type='ImportError', exception_value=self._getImportNameErrorString(self.module, self.value_name, attribute_name))
        else:
            return makeRaiseExceptionReplacementExpression(expression=lookup_node, exception_type='AttributeError', exception_value=self._getImportNameErrorString(self.module, self.value_name, attribute_name))

    def _computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection, is_import):
        if False:
            return 10
        if self.module is not None and self.allowed:
            full_name = self.value_name.getChildNamed(attribute_name)
            full_name = ModuleName(hard_modules_aliases.get(full_name, full_name))
            if isHardModule(full_name):
                new_node = ExpressionImportModuleHard(module_name=full_name, value_name=full_name, source_ref=lookup_node.source_ref)
                return (new_node, 'new_expression', "Hard module '%s' submodule '%s' pre-computed." % (self.value_name, attribute_name))
            trust = hard_modules_trust[self.value_name].get(attribute_name, trust_undefined)
            if trust is trust_importable:
                trace_collection.onExceptionRaiseExit(BaseException)
            elif trust is trust_may_exist:
                trace_collection.onExceptionRaiseExit(BaseException)
            elif trust is not trust_undefined and (not hasattr(self.module, attribute_name)):
                trace_collection.onExceptionRaiseExit(ImportError)
                new_node = self._makeRaiseExceptionReplacementExpression(lookup_node=lookup_node, attribute_name=attribute_name, is_import=is_import)
                return (new_node, 'new_raise', "Hard module '%s' attribute missing '%s* pre-computed." % (self.value_name, attribute_name))
            elif trust is trust_undefined:
                if self.is_package and False:
                    full_name = self.value_name.getChildNamed(attribute_name)
                    (_sub_module_name, _sub_module_filename, finding) = locateModule(module_name=full_name, parent_package=None, level=0)
                    if finding != 'not-found':
                        result = makeExpressionImportModuleFixed(module_name=full_name, value_name=full_name, source_ref=lookup_node.getSourceReference())
                        return (result, 'new_expression', "Attribute lookup '%s* of hard module *%s* becomes hard module name import." % (self.value_name, attribute_name))
                trace_collection.onExceptionRaiseExit(ImportError)
                onMissingTrust("Hard module '%s' attribute '%s' missing trust config for existing value.", lookup_node.getSourceReference(), self.value_name, attribute_name)
            elif trust is trust_constant:
                assert hasattr(self.module, attribute_name), self
                return (makeConstantReplacementNode(constant=getattr(self.module, attribute_name), node=lookup_node, user_provided=True), 'new_constant', "Hard module '%s' imported '%s' pre-computed to constant value." % (self.value_name, attribute_name))
            elif trust is trust_node:
                trace_collection.onExceptionRaiseExit(ImportError)
                result = trust_node_factory[self.value_name, attribute_name](source_ref=lookup_node.source_ref)
                return (result, 'new_expression', "Attribute lookup '%s' of hard module '%s' becomes node '%s'." % (self.value_name, attribute_name, result.kind))
            else:
                result = makeExpressionImportModuleNameHard(module_name=self.value_name, import_name=attribute_name, module_guaranteed=self.guaranteed, source_ref=lookup_node.getSourceReference())
                return (result, 'new_expression', "Attribute lookup '%s' of hard module '%s' becomes hard module name import." % (self.value_name, attribute_name))
        else:
            trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            return 10
        return self._computeExpressionAttribute(lookup_node=lookup_node, attribute_name=attribute_name, trace_collection=trace_collection, is_import=False)

    def hasShapeTrustedAttributes(self):
        if False:
            i = 10
            return i + 15
        return True
importlib_import_module_spec = BuiltinParameterSpec('importlib.import_module', ('name', 'package'), default_count=1)

class ExpressionImportlibImportModuleRef(ExpressionImportModuleNameHardExistsSpecificBase):
    kind = 'EXPRESSION_IMPORTLIB_IMPORT_MODULE_REF'

    def __init__(self, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionImportModuleNameHardExistsSpecificBase.__init__(self, module_name='importlib', import_name='import_module', module_guaranteed=True, source_ref=source_ref)

    @staticmethod
    def getDetails():
        if False:
            return 10
        return {}

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        result = extractBuiltinArgs(node=call_node, builtin_class=ExpressionImportlibImportModuleCall, builtin_spec=importlib_import_module_spec)
        return (result, 'new_expression', "Call of 'importlib.import_module' recognized.")

def _getImportNameAsStr(value):
    if False:
        print('Hello World!')
    if value is None:
        result = None
    else:
        result = value.getCompileTimeConstant()
    if type(result) in (str, unicode):
        if str is not unicode and type(result) is unicode:
            result = str(result)
    return result

class ExpressionImportlibImportModuleCall(ChildrenExpressionImportlibImportModuleCallMixin, ExpressionBase):
    """Call to "importlib.import_module" """
    kind = 'EXPRESSION_IMPORTLIB_IMPORT_MODULE_CALL'
    named_children = ('name', 'package|optional')

    def __init__(self, name, package, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenExpressionImportlibImportModuleCallMixin.__init__(self, name=name, package=package)
        ExpressionBase.__init__(self, source_ref)

    @staticmethod
    def _resolveImportLibArgs(module_name, package_name):
        if False:
            print('Hello World!')
        if module_name.startswith('.'):
            if not package_name:
                return None
            level = 0
            for character in module_name:
                if character != '.':
                    break
                level += 1
            module_name = module_name[level:]
            dot = len(package_name)
            for _i in xrange(level, 1, -1):
                try:
                    dot = package_name.rindex('.', 0, dot)
                except ValueError:
                    return None
            package_name = package_name[:dot]
            if module_name == '':
                return package_name
            else:
                return '%s.%s' % (package_name, module_name)
        return module_name

    def computeExpression(self, trace_collection):
        if False:
            return 10
        module_name = self.subnode_name
        package_name = self.subnode_package
        if (package_name is None or package_name.isCompileTimeConstant()) and module_name.isCompileTimeConstant():
            imported_module_name = _getImportNameAsStr(module_name)
            imported_package_name = _getImportNameAsStr(package_name)
            if (imported_package_name is None or type(imported_package_name) is str) and type(imported_module_name) is str:
                resolved_module_name = self._resolveImportLibArgs(imported_module_name, imported_package_name)
                if resolved_module_name is not None:
                    trace_collection.onExceptionRaiseExit(BaseException)
                    result = makeExpressionImportModuleFixed(module_name=resolved_module_name, value_name=resolved_module_name, source_ref=self.source_ref)
                    return (result, 'new_expression', "Resolved importlib.import_module call to import of '%s'." % resolved_module_name)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)
addModuleSingleAttributeNodeFactory('importlib', 'import_module', ExpressionImportlibImportModuleRef)

class ExpressionBuiltinImport(ChildrenExpressionBuiltinImportMixin, ExpressionBase):
    __slots__ = ('follow_attempted', 'finding', 'used_modules')
    kind = 'EXPRESSION_BUILTIN_IMPORT'
    named_children = ('name|setter', 'globals_arg|optional', 'locals_arg|optional', 'fromlist|optional', 'level|optional')

    def __init__(self, name, globals_arg, locals_arg, fromlist, level, source_ref):
        if False:
            print('Hello World!')
        ChildrenExpressionBuiltinImportMixin.__init__(self, name=name, globals_arg=globals_arg, locals_arg=locals_arg, fromlist=fromlist, level=level)
        ExpressionBase.__init__(self, source_ref)
        self.follow_attempted = False
        self.used_modules = []
        self.finding = None

    def _attemptFollow(self, module_name):
        if False:
            print('Hello World!')
        parent_module = self.getParentModule()
        level = self.subnode_level
        if level is None:
            level = 0 if parent_module.getFutureSpec().isAbsoluteImport() else -1
        elif not level.isCompileTimeConstant():
            return
        else:
            level = level.getCompileTimeConstant()
        if level != 0:
            parent_package = parent_module.getFullName()
            if not parent_module.isCompiledPythonPackage():
                parent_package = parent_package.getPackageName()
        else:
            parent_package = None
        if type(level) not in (int, long):
            return None
        module_name_resolved = resolveModuleName(module_name)
        if module_name_resolved != module_name:
            module_name = module_name_resolved
            self.setChildName(makeConstantReplacementNode(constant=module_name.asString(), node=self.subnode_name, user_provided=True))
        module_name = ModuleName(module_name)
        (module_name_found, module_filename, module_kind, self.finding) = locateModule(module_name=ModuleName(module_name), parent_package=parent_package, level=level)
        self.used_modules = [makeModuleUsageAttempt(module_name=module_name_found, filename=module_filename, module_kind=module_kind, finding=self.finding, level=level, source_ref=self.source_ref, reason='import')]
        if self.finding != 'not-found':
            module_name = module_name_found
            import_list = self.subnode_fromlist
            if import_list is not None:
                if import_list.isCompileTimeConstant():
                    import_list = import_list.getCompileTimeConstant()
                if type(import_list) not in (tuple, list):
                    import_list = None
            if module_filename is not None and import_list and isPackageDir(module_filename):
                for import_item in import_list:
                    if import_item == '*':
                        continue
                    (name_import_module_name, name_import_module_filename, name_import_module_kind, name_import_finding) = locateModule(module_name=ModuleName(import_item), parent_package=module_name, level=1)
                    self.used_modules.append(makeModuleUsageAttempt(module_name=name_import_module_name, filename=name_import_module_filename, module_kind=name_import_module_kind, finding=name_import_finding, level=1, source_ref=self.source_ref, reason='import'))
            return module_filename
        else:
            while True:
                module_name = module_name.getPackageName()
                if not module_name:
                    break
                (found_module_name, module_filename, module_kind, finding) = locateModule(module_name=module_name, parent_package=parent_package, level=level)
                self.used_modules.append(makeModuleUsageAttempt(module_name=found_module_name, filename=module_filename, module_kind=module_kind, finding=finding, level=level, source_ref=self.source_ref, reason='import'))
                if finding != 'not-found':
                    break
            return None

    def _getImportedValueName(self, imported_module_name):
        if False:
            i = 10
            return i + 15
        from_list_truth = self.subnode_fromlist is not None and self.subnode_fromlist.getTruthValue()
        if from_list_truth is True:
            return imported_module_name
        else:
            return imported_module_name.getTopLevelPackageName()

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        if self.follow_attempted:
            if self.finding == 'not-found':
                trace_collection.onExceptionRaiseExit(BaseException)
            else:
                trace_collection.onExceptionRaiseExit(RuntimeError)
            for module_usage_attempt in self.used_modules:
                trace_collection.onModuleUsageAttempt(module_usage_attempt)
            return (self, None, None)
        if self.finding != 'built-in':
            trace_collection.onExceptionRaiseExit(BaseException)
        module_name = self.subnode_name
        if module_name.isCompileTimeConstant():
            imported_module_name = module_name.getCompileTimeConstant()
            module_filename = self._attemptFollow(module_name=imported_module_name)
            self.follow_attempted = True
            for module_usage_attempt in self.used_modules:
                trace_collection.onModuleUsageAttempt(module_usage_attempt)
            if type(imported_module_name) in (str, unicode):
                imported_module_name = resolveModuleName(imported_module_name)
                if self.finding == 'absolute' and isHardModule(imported_module_name):
                    if imported_module_name in hard_modules_non_stdlib or isStandardLibraryPath(module_filename):
                        result = ExpressionImportModuleHard(module_name=imported_module_name, value_name=self._getImportedValueName(imported_module_name), source_ref=self.source_ref)
                        return (result, 'new_expression', "Lowered import %s module '%s' to hard import." % ('hard import' if imported_module_name in hard_modules_non_stdlib else 'standard library', imported_module_name.asString()))
                    elif shallWarnUnusualCode():
                        unusual_logger.warning("%s: Standard library module '%s' used from outside path %r." % (self.source_ref.getAsString(), imported_module_name.asString(), self.module_filename))
                if self.finding == 'built-in':
                    result = makeExpressionImportModuleBuiltin(module_name=imported_module_name, value_name=self._getImportedValueName(imported_module_name), source_ref=self.source_ref)
                    return (result, 'new_expression', "Lowered import of built-in module '%s' to hard import." % imported_module_name.asString())
                if self.finding == 'not-found':
                    if imported_module_name in hard_modules_limited:
                        result = makeRaiseImportErrorReplacementExpression(expression=self, module_name=imported_module_name)
                        return (result, 'new_raise', "Lowered import of missing standard library module '%s' to hard import." % imported_module_name.asString())
                elif isStandaloneMode() and self.used_modules and isExperimental('standalone-imports'):
                    result = makeExpressionImportModuleFixed(module_name=self.used_modules[0].module_name, value_name=self._getImportedValueName(self.used_modules[0].module_name), source_ref=self.source_ref)
                    return (result, 'new_expression', "Lowered import of module '%s' to fixed import." % imported_module_name.asString())
            else:
                (new_node, change_tags, message) = trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : __import__(module_name.getCompileTimeConstant()), description="Replaced '__import__' call with non-string module name argument.")
                assert change_tags == 'new_raise', module_name
                return (new_node, change_tags, message)
        for module_usage_attempt in self.used_modules:
            trace_collection.onModuleUsageAttempt(module_usage_attempt)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.finding != 'built-in'

class StatementImportStar(StatementImportStarBase):
    kind = 'STATEMENT_IMPORT_STAR'
    named_children = ('module',)
    node_attributes = ('target_scope',)
    auto_compute_handling = 'post_init,operation'

    def postInitNode(self):
        if False:
            i = 10
            return i + 15
        if type(self.target_scope) is GlobalsDictHandle:
            self.target_scope.markAsEscaped()

    def getTargetDictScope(self):
        if False:
            return 10
        return self.target_scope

    def computeStatementOperation(self, trace_collection):
        if False:
            return 10
        trace_collection.onLocalsDictEscaped(self.target_scope)
        trace_collection.removeAllKnowledge()
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        return True

    @staticmethod
    def getStatementNiceName():
        if False:
            for i in range(10):
                print('nop')
        return 'star import statement'

class ExpressionImportName(ChildHavingModuleMixin, ExpressionBase):
    kind = 'EXPRESSION_IMPORT_NAME'
    named_children = ('module',)
    __slots__ = ('import_name', 'level')

    def __init__(self, module, import_name, level, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildHavingModuleMixin.__init__(self, module=module)
        ExpressionBase.__init__(self, source_ref)
        self.import_name = import_name
        self.level = level
        assert level is not None
        assert module is not None

    def getImportName(self):
        if False:
            i = 10
            return i + 15
        return self.import_name

    def getImportLevel(self):
        if False:
            i = 10
            return i + 15
        return self.level

    def getDetails(self):
        if False:
            return 10
        return {'import_name': self.import_name, 'level': self.level}

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return self.subnode_module.computeExpressionImportName(import_node=self, import_name=self.import_name, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_module.mayRaiseExceptionImportName(exception_type=exception_type, import_name=self.import_name)

def makeExpressionImportModuleFixed(module_name, value_name, source_ref):
    if False:
        i = 10
        return i + 15
    module_name = resolveModuleName(module_name)
    value_name = resolveModuleName(value_name)
    if isHardModule(module_name):
        return ExpressionImportModuleHard(module_name=module_name, value_name=value_name, source_ref=source_ref)
    else:
        return ExpressionImportModuleFixed(module_name=module_name, value_name=value_name, source_ref=source_ref)

def makeExpressionImportModuleBuiltin(module_name, value_name, source_ref):
    if False:
        return 10
    module_name = resolveModuleName(module_name)
    value_name = resolveModuleName(value_name)
    if isHardModule(module_name):
        return ExpressionImportModuleHard(module_name=module_name, value_name=value_name, source_ref=source_ref)
    else:
        return ExpressionImportModuleBuiltin(module_name=module_name, value_name=value_name, source_ref=source_ref)