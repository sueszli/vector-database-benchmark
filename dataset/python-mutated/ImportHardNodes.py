""" Nodes representing more trusted imports. """
from nuitka.importing.Importing import locateModule, makeModuleUsageAttempt
from nuitka.utils.ModuleNames import ModuleName
from .ExpressionBases import ExpressionBase

class ExpressionImportHardBase(ExpressionBase):
    __slots__ = ('module_name', 'finding', 'module_kind', 'module_filename')

    def __init__(self, module_name, source_ref):
        if False:
            while True:
                i = 10
        ExpressionBase.__init__(self, source_ref)
        self.module_name = ModuleName(module_name)
        (_module_name, self.module_filename, self.module_kind, self.finding) = locateModule(module_name=self.module_name, parent_package=None, level=0)
        assert self.finding != 'not-found', self.module_name
        assert _module_name == self.module_name, (self.module_name, _module_name)

    def getModuleUsageAttempt(self):
        if False:
            for i in range(10):
                print('nop')
        return makeModuleUsageAttempt(module_name=self.module_name, filename=self.module_filename, module_kind=self.module_kind, finding=self.finding, level=0, source_ref=self.source_ref, reason='import')

class ExpressionImportModuleNameHardBase(ExpressionImportHardBase):
    """Hard import names base class."""
    __slots__ = ('import_name', 'finding', 'module_filename', 'module_guaranteed')

    def __init__(self, module_name, import_name, module_guaranteed, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionImportHardBase.__init__(self, module_name=module_name, source_ref=source_ref)
        self.import_name = import_name
        self.module_guaranteed = module_guaranteed

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'module_name': self.module_name, 'import_name': self.import_name, 'module_guaranteed': self.module_guaranteed}

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            for i in range(10):
                print('nop')
        return True

    def finalize(self):
        if False:
            print('Hello World!')
        del self.parent

    def getModuleName(self):
        if False:
            i = 10
            return i + 15
        return self.module_name

    def getImportName(self):
        if False:
            print('Hello World!')
        return self.import_name

class ExpressionImportModuleNameHardMaybeExists(ExpressionImportModuleNameHardBase):
    """Hard coded import names, e.g. of "site.something"

    These are created for attributes of hard imported modules that are not know if
    they exist or not.
    """
    kind = 'EXPRESSION_IMPORT_MODULE_NAME_HARD_MAYBE_EXISTS'

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(AttributeError)
        trace_collection.onModuleUsageAttempt(self.getModuleUsageAttempt())
        return (self, None, None)

    @staticmethod
    def mayHaveSideEffects():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return True

class ExpressionImportModuleNameHardExists(ExpressionImportModuleNameHardBase):
    """Hard coded import names, e.g. of "sys.stdout"

    These are directly created for some Python mechanics.
    """
    kind = 'EXPRESSION_IMPORT_MODULE_NAME_HARD_EXISTS'

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if not self.module_guaranteed:
            trace_collection.onExceptionRaiseExit(ImportError)
        trace_collection.onModuleUsageAttempt(self.getModuleUsageAttempt())
        return (self, None, None)

    def mayHaveSideEffects(self):
        if False:
            i = 10
            return i + 15
        return not self.module_guaranteed

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return not self.module_guaranteed

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.computeExpressionCall(call_node=call_node, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)

class ExpressionImportModuleNameHardExistsSpecificBase(ExpressionImportModuleNameHardExists):
    """Base class for nodes that hard coded import names, e.g. of "importlib.import_module" name."""

    @staticmethod
    def getDetails():
        if False:
            print('Hello World!')
        return {}