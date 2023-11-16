""" Recursion into other modules.

"""
import glob
import os
from nuitka import ModuleRegistry, Options
from nuitka.Errors import NuitkaForbiddenImportEncounter
from nuitka.freezer.ImportDetection import detectEarlyImports, detectStdlibAutoInclusionModules
from nuitka.importing import ImportCache, StandardLibrary
from nuitka.ModuleRegistry import addUsedModule, getRootTopModule
from nuitka.pgo.PGO import decideInclusionFromPGO
from nuitka.plugins.Plugins import Plugins
from nuitka.PythonVersions import python_version
from nuitka.Tracing import recursion_logger
from nuitka.utils.FileOperations import listDir
from nuitka.utils.Importing import getSharedLibrarySuffixes
from nuitka.utils.ModuleNames import ModuleName
from .Importing import getModuleNameAndKindFromFilename, isPackageDir, locateModule, warnAboutNotFoundImport

def _recurseTo(module_name, module_filename, module_kind, reason):
    if False:
        return 10
    from nuitka.tree import Building
    module = Building.buildModule(module_name=module_name, module_kind=module_kind, module_filename=module_filename, reason=reason, source_code=None, is_top=False, is_main=False, is_fake=False, hide_syntax_error=True)
    ImportCache.addImportedModule(module)
    return module

def recurseTo(module_name, module_filename, module_kind, source_ref, reason, using_module_name):
    if False:
        i = 10
        return i + 15
    try:
        module = ImportCache.getImportedModuleByNameAndPath(module_name, module_filename)
    except KeyError:
        module = None
    if module is None:
        Plugins.onModuleRecursion(module_filename=module_filename, module_name=module_name, module_kind=module_kind, using_module_name=using_module_name, source_ref=source_ref, reason=reason)
        module = _recurseTo(module_name=module_name, module_filename=module_filename, module_kind=module_kind, reason=reason)
    return module
_recursion_decision_cache = {}

def decideRecursion(using_module_name, module_filename, module_name, module_kind, extra_recursion=False):
    if False:
        while True:
            i = 10
    (package_part, _remainder) = module_name.splitModuleBasename()
    if package_part is not None:
        (_package_part, package_filename, package_module_kind, package_finding) = locateModule(module_name=package_part, parent_package=None, level=0)
        assert _package_part == package_part
        assert package_finding != 'not-found'
        (package_decision, package_reason) = decideRecursion(using_module_name=using_module_name, module_filename=package_filename, module_name=package_part, module_kind=package_module_kind, extra_recursion=extra_recursion)
        if package_decision is False:
            return (package_decision, package_reason)
    key = (using_module_name, module_filename, module_name, module_kind, extra_recursion)
    if key not in _recursion_decision_cache:
        _recursion_decision_cache[key] = _decideRecursion(using_module_name, module_filename, module_name, module_kind, extra_recursion)
        if _recursion_decision_cache[key][0]:
            Plugins.onModuleUsageLookAhead(module_name=module_name, module_filename=module_filename, module_kind=module_kind)
    return _recursion_decision_cache[key]

def _decideRecursion(using_module_name, module_filename, module_name, module_kind, extra_recursion):
    if False:
        i = 10
        return i + 15
    if module_name == '__main__':
        return (False, 'Main program is not followed to a second time.')
    if module_kind == 'extension' and (not Options.isStandaloneMode()):
        return (False, 'Extension modules cannot be inspected.')
    if module_name in detectEarlyImports():
        return (True, 'Technically required for CPython library startup.')
    if module_name in detectStdlibAutoInclusionModules():
        return (True, 'Including as part of the non-excluded parts of standard library.')
    if Options.hasPythonFlagPackageMode() and (not Options.shallMakeModule()) and (module_name.getBasename() == '__main__'):
        if module_name.getPackageName() == getRootTopModule().getRuntimePackageValue():
            return (False, 'Main program is already included in package mode.')
    (plugin_decision, deciding_plugins) = Plugins.onModuleEncounter(using_module_name=using_module_name, module_filename=module_filename, module_name=module_name, module_kind=module_kind)
    deciding_plugins = [deciding_plugin for deciding_plugin in deciding_plugins if deciding_plugin.plugin_name != 'anti-bloat']
    (no_case, reason) = module_name.matchesToShellPatterns(patterns=Options.getShallFollowInNoCase())
    if no_case:
        if plugin_decision and plugin_decision[0]:
            deciding_plugins[0].sysexit("Conflict between user and plugin decision for module '%s'." % module_name)
        return (False, 'Module %s instructed by user to not follow to.' % reason)
    (any_case, reason) = module_name.matchesToShellPatterns(patterns=Options.getShallFollowModules())
    if any_case:
        if plugin_decision and (not plugin_decision[0]) and deciding_plugins:
            deciding_plugins[0].sysexit("Conflict between user and plugin decision for module '%s'." % module_name)
        return (True, 'Module %s instructed by user to follow to.' % reason)
    if plugin_decision is not None:
        return plugin_decision
    if extra_recursion:
        return (True, 'Lives in user provided directory.')
    if module_kind == 'extension' and Options.isStandaloneMode():
        return (True, 'Extension module needed for standalone mode.')
    is_stdlib = StandardLibrary.isStandardLibraryPath(module_filename)
    if not is_stdlib or Options.shallFollowStandardLibrary():
        from nuitka.tree.Building import decideCompilationMode
        if decideCompilationMode(is_top=False, module_name=module_name, for_pgo=True) == 'compiled':
            pgo_decision = decideInclusionFromPGO(module_name=module_name, module_kind=module_kind)
            if pgo_decision is not None:
                return (pgo_decision, 'PGO based decision')
    if is_stdlib and (not Options.isStandaloneMode()) and (not Options.shallFollowStandardLibrary()):
        return (False, 'Not following into stdlib unless standalone or requested to follow into stdlib.')
    if Options.shallFollowAllImports():
        return (True, 'Instructed by user to follow to all modules.')
    if Options.shallFollowNoImports():
        return (None, 'Instructed by user to not follow at all.')
    return (None, 'Default behavior in non-standalone mode, not following without request.')

def isSameModulePath(path1, path2):
    if False:
        print('Hello World!')
    if os.path.basename(path1) == '__init__.py':
        path1 = os.path.dirname(path1)
    if os.path.basename(path2) == '__init__.py':
        path2 = os.path.dirname(path2)
    return os.path.abspath(path1) == os.path.abspath(path2)

def _addIncludedModule(module, package_only):
    if False:
        while True:
            i = 10
    if Options.isShowInclusion():
        recursion_logger.info("Included '%s' as '%s'." % (module.getFullName(), module))
    ImportCache.addImportedModule(module)
    if module.isCompiledPythonPackage() or module.isUncompiledPythonPackage():
        package_filename = module.getFilename()
        if os.path.isdir(package_filename):
            assert python_version >= 768
            package_dir = package_filename
        else:
            package_dir = os.path.dirname(package_filename)
            ModuleRegistry.addRootModule(module)
        if Options.isShowInclusion():
            recursion_logger.info("Package directory '%s'." % package_dir)
        if not package_only:
            for (sub_path, sub_filename) in listDir(package_dir):
                if sub_filename in ('__init__.py', '__pycache__'):
                    continue
                if isPackageDir(sub_path) and (not os.path.exists(sub_path + '.py')):
                    checkPluginSinglePath(sub_path, module_package=module.getFullName(), package_only=False)
                elif sub_path.endswith('.py'):
                    checkPluginSinglePath(sub_path, module_package=module.getFullName(), package_only=False)
    elif module.isCompiledPythonModule() or module.isUncompiledPythonModule():
        ModuleRegistry.addRootModule(module)
    elif module.isPythonExtensionModule():
        if Options.isStandaloneMode():
            ModuleRegistry.addRootModule(module)
    else:
        assert False, module

def checkPluginSinglePath(plugin_filename, module_package, package_only):
    if False:
        return 10
    plugin_filename = os.path.abspath(plugin_filename)
    if Options.isShowInclusion():
        recursion_logger.info("Checking detail plug-in path '%s' '%s':" % (plugin_filename, module_package))
    (module_name, module_kind) = getModuleNameAndKindFromFilename(plugin_filename)
    module_name = ModuleName.makeModuleNameInPackage(module_name, module_package)
    if module_kind == 'extension' and (not Options.isStandaloneMode()):
        recursion_logger.warning("Cannot include extension module '%s' unless using at least standalone mode, where they would be copied. In this mode, extension modules are not part of the compiled result, and therefore asking to include them makes no sense.\n" % module_name.asString())
    if module_kind is not None:
        (decision, decision_reason) = decideRecursion(using_module_name=None, module_filename=plugin_filename, module_name=module_name, module_kind=module_kind, extra_recursion=True)
        if decision:
            module = recurseTo(module_filename=plugin_filename, module_name=module_name, module_kind=module_kind, source_ref=None, reason='command line', using_module_name=None)
            if module:
                _addIncludedModule(module=module, package_only=package_only)
            else:
                recursion_logger.warning("Failed to include module from '%s'." % plugin_filename)
        else:
            recursion_logger.warning("Not allowed to include module '%s' due to '%s'." % (module_name, decision_reason))

def checkPluginPath(plugin_filename, module_package):
    if False:
        return 10
    if Options.isShowInclusion():
        recursion_logger.info("Checking top level inclusion path '%s' '%s'." % (plugin_filename, module_package))
    if os.path.isfile(plugin_filename) or isPackageDir(plugin_filename):
        checkPluginSinglePath(plugin_filename, module_package=module_package, package_only=False)
    elif os.path.isdir(plugin_filename):
        for (sub_path, sub_filename) in listDir(plugin_filename):
            assert sub_filename != '__init__.py'
            if isPackageDir(sub_path) or sub_path.endswith('.py'):
                checkPluginSinglePath(sub_path, module_package=None, package_only=False)
                continue
            for suffix in getSharedLibrarySuffixes():
                if sub_path.endswith(suffix):
                    checkPluginSinglePath(sub_path, module_package=None, package_only=False)
    else:
        recursion_logger.warning("Failed to include module from '%s'." % plugin_filename)

def checkPluginFilenamePattern(pattern):
    if False:
        print('Hello World!')
    if Options.isShowInclusion():
        recursion_logger.info("Checking plug-in pattern '%s':" % pattern)
    assert not os.path.isdir(pattern), pattern
    found = False
    for filename in glob.iglob(pattern):
        if filename.endswith('.pyc'):
            continue
        if not os.path.isfile(filename):
            continue
        found = True
        checkPluginSinglePath(filename, module_package=None, package_only=False)
    if not found:
        recursion_logger.warning("Didn't match any files against pattern '%s'." % pattern)

def considerUsedModules(module, pass_count):
    if False:
        print('Hello World!')
    if module.reason == 'stdlib':
        return
    for used_module in module.getUsedModules():
        if used_module.reason == 'stdlib':
            if pass_count == 1:
                continue
        elif pass_count == -1:
            continue
        if used_module.finding == 'not-found':
            warnAboutNotFoundImport(importing=module, source_ref=used_module.source_ref, module_name=used_module.module_name, level=used_module.level)
        if used_module.filename is None:
            continue
        try:
            (decision, decision_reason) = decideRecursion(using_module_name=module.getFullName(), module_filename=used_module.filename, module_name=used_module.module_name, module_kind=used_module.module_kind)
            if decision:
                new_module = recurseTo(module_name=used_module.module_name, module_filename=used_module.filename, module_kind=used_module.module_kind, source_ref=used_module.source_ref, reason=used_module.reason, using_module_name=module.module_name)
                addUsedModule(module=new_module, using_module=module, usage_tag=used_module.reason, reason=decision_reason, source_ref=used_module.source_ref)
        except NuitkaForbiddenImportEncounter as e:
            recursion_logger.sysexit("Error, forbidden import of '%s' (intending to avoid '%s') in module '%s' at '%s' encountered." % (e.args[0], e.args[1], module.getFullName(), used_module.source_ref.getAsString()))
    try:
        Plugins.considerImplicitImports(module=module)
    except NuitkaForbiddenImportEncounter as e:
        recursion_logger.sysexit("Error, forbidden import of '%s' (intending to avoid '%s') done implicitly by module '%s'." % (e.args[0], e.args[1], module.getFullName()))