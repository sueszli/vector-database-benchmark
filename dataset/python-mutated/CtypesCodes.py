""" Code generation for ctypes module stuff. """
from .BuiltinCodes import getBuiltinCallViaSpecCode
from .ImportCodes import getImportModuleNameHardCode

def generateCtypesCdllCallCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    ctypes_cdll_class = context.allocateTempName('ctypes_cdll_class', unique=True)
    getImportModuleNameHardCode(to_name=ctypes_cdll_class, module_name='ctypes', import_name='CDLL', needs_check=False, emit=emit, context=context)
    getBuiltinCallViaSpecCode(spec=expression.spec, called_name=ctypes_cdll_class, to_name=to_name, expression=expression, emit=emit, context=context)