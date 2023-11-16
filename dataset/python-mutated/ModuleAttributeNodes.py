""" Module/Package attribute nodes

The represent special values of the modules. The "__name__", "__package__",
"__file__", and "__spec__" values can all be highly dynamic and version
dependent

These nodes are intended to allow for as much compile time optimization as
possible, despite this difficulty. In some modes these node become constants
quickly, in others they will present boundaries for optimization.

"""
from nuitka import Options
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin

class ExpressionModuleAttributeBase(ExpressionBase):
    """Expression base class for module attributes.

    This
    """
    __slots__ = ('variable',)

    def __init__(self, variable, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionBase.__init__(self, source_ref)
        self.variable = variable

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent
        del self.variable

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'variable': self.variable}

    def getVariable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            while True:
                i = 10
        return False

class ExpressionModuleAttributeFileRef(ExpressionModuleAttributeBase):
    """Expression that represents accesses to __file__ of module.

    The __file__ is a static or dynamic value depending on the
    file reference mode. If it requests runtime, i.e. looks at
    where it is loaded from, then there is not a lot to be said
    about its value, otherwise it becomes a constant value
    quickly.
    """
    kind = 'EXPRESSION_MODULE_ATTRIBUTE_FILE_REF'

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        if Options.getFileReferenceMode() != 'runtime':
            result = makeConstantRefNode(constant=self.variable.getModule().getRunTimeFilename(), source_ref=self.source_ref)
            return (result, 'new_expression', "Using original '__file__' value.")
        return (self, None, None)

class ExpressionModuleAttributeNameRef(ExpressionModuleAttributeBase):
    """Expression that represents accesses to __name__ of module.

    For binaries this can be relatively well known, but modules
    living in a package, go by what loads them to ultimately
    determine their name.
    """
    kind = 'EXPRESSION_MODULE_ATTRIBUTE_NAME_REF'

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        if Options.getModuleNameMode() != 'runtime':
            result = makeConstantRefNode(constant=self.variable.getModule().getRuntimeNameValue(), source_ref=self.source_ref)
            return (result, 'new_expression', "Using constant '__name__' value.")
        return (self, None, None)

class ExpressionModuleAttributePackageRef(ExpressionModuleAttributeBase):
    """Expression that represents accesses to __package__ of module.

    For binaries this can be relatively well known, but modules
    living in a package, go by what loads them to ultimately
    determine their parent package.
    """
    kind = 'EXPRESSION_MODULE_ATTRIBUTE_PACKAGE_REF'

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        if Options.getModuleNameMode() != 'runtime':
            provider = self.variable.getModule()
            value = provider.getRuntimePackageValue()
            result = makeConstantRefNode(constant=value, source_ref=self.source_ref)
            return (result, 'new_expression', "Using constant '__package__' value.")
        return (self, None, None)

class ExpressionModuleAttributeLoaderRef(ExpressionModuleAttributeBase):
    """Expression that represents accesses to __loader__ of module.

    The loader of Nuitka is going to load the module, and there
    is not a whole lot to be said about it here, it is assumed
    to be largely ignored in user code.
    """
    kind = 'EXPRESSION_MODULE_ATTRIBUTE_LOADER_REF'

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

class ExpressionModuleAttributeSpecRef(ExpressionModuleAttributeBase):
    """Expression that represents accesses to __spec__ of module.

    The __spec__ is used by the loader mechanism and sometimes
    by code checking e.g. if something is a package. It exists
    only for modern Python. For the main program module, it's
    always None (it is also not really loaded in the same way
    as other code).
    """
    kind = 'EXPRESSION_MODULE_ATTRIBUTE_SPEC_REF'

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        if self.variable.getModule().isMainModule():
            result = makeConstantRefNode(constant=None, source_ref=self.source_ref)
            return (result, 'new_expression', "Using constant '__spec__' value for main module.")
        return (self, None, None)

class ExpressionNuitkaLoaderCreation(ExpressionNoSideEffectsMixin, ExpressionBase):
    __slots__ = ('provider',)
    kind = 'EXPRESSION_NUITKA_LOADER_CREATION'

    def __init__(self, provider, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionBase.__init__(self, source_ref)
        self.provider = provider

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        del self.parent
        del self.provider

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)