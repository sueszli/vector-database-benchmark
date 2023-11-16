""" Demotion of compiled modules to bytecode modules.

"""
import marshal
from nuitka.BytecodeCaching import writeImportedModulesNamesToCache
from nuitka.Bytecodes import compileSourceToBytecode
from nuitka.freezer.ImportDetection import detectEarlyImports
from nuitka.importing.ImportCache import isImportedModuleByName, replaceImportedModule
from nuitka.ModuleRegistry import replaceRootModule
from nuitka.nodes.ModuleNodes import makeUncompiledPythonModule
from nuitka.Options import isShowProgress, isStandaloneMode
from nuitka.plugins.Plugins import Plugins, isTriggerModule, replaceTriggerModule
from nuitka.Tracing import inclusion_logger

def demoteSourceCodeToBytecode(module_name, source_code, filename):
    if False:
        for i in range(10):
            print('nop')
    if isStandaloneMode():
        filename = module_name.asPath() + '.py'
    bytecode = compileSourceToBytecode(source_code, filename)
    bytecode = Plugins.onFrozenModuleBytecode(module_name=module_name, is_package=False, bytecode=bytecode)
    return marshal.dumps(bytecode)

def demoteCompiledModuleToBytecode(module):
    if False:
        return 10
    'Demote a compiled module to uncompiled (bytecode).'
    full_name = module.getFullName()
    filename = module.getCompileTimeFilename()
    if isShowProgress():
        inclusion_logger.info("Demoting module '%s' to bytecode from '%s'." % (full_name.asString(), filename))
    source_code = module.getSourceCode()
    bytecode = demoteSourceCodeToBytecode(module_name=full_name, source_code=source_code, filename=filename)
    uncompiled_module = makeUncompiledPythonModule(module_name=full_name, reason=module.reason, filename=filename, bytecode=bytecode, is_package=module.isCompiledPythonPackage(), technical=full_name in detectEarlyImports())
    used_modules = module.getUsedModules()
    uncompiled_module.setUsedModules(used_modules)
    distribution_names = module.getUsedDistributions()
    uncompiled_module.setUsedDistributions(distribution_names)
    module.finalize()
    if isImportedModuleByName(full_name):
        replaceImportedModule(old=module, new=uncompiled_module)
    replaceRootModule(old=module, new=uncompiled_module)
    if isTriggerModule(module):
        replaceTriggerModule(old=module, new=uncompiled_module)
    writeImportedModulesNamesToCache(module_name=full_name, source_code=source_code, used_modules=used_modules, distribution_names=distribution_names)