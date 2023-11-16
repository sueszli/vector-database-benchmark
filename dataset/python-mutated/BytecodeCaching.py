""" Caching of compiled code.

Initially this deals with preserving compiled module state after bytecode demotion
such that it allows to restore it directly.
"""
import os
import sys
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.importing.Importing import locateModule, makeModuleUsageAttempt
from nuitka.plugins.Plugins import Plugins
from nuitka.utils.AppDirs import getCacheDir
from nuitka.utils.FileOperations import makePath
from nuitka.utils.Hashing import Hash, getStringHash
from nuitka.utils.Json import loadJsonFromFilename, writeJsonToFilename
from nuitka.utils.ModuleNames import ModuleName
from nuitka.Version import version_string

def getBytecodeCacheDir():
    if False:
        i = 10
        return i + 15
    module_cache_dir = os.path.join(getCacheDir(), 'module-cache')
    return module_cache_dir

def _getCacheFilename(module_name, extension):
    if False:
        while True:
            i = 10
    return os.path.join(getBytecodeCacheDir(), '%s.%s' % (module_name, extension))

def makeCacheName(module_name, source_code):
    if False:
        while True:
            i = 10
    module_config_hash = _getModuleConfigHash(module_name)
    return module_name.asString() + '@' + module_config_hash + '@' + getStringHash(source_code)

def hasCachedImportedModuleUsageAttempts(module_name, source_code, source_ref):
    if False:
        i = 10
        return i + 15
    result = getCachedImportedModuleUsageAttempts(module_name=module_name, source_code=source_code, source_ref=source_ref)
    return result is not None
_cache_format_version = 6

def getCachedImportedModuleUsageAttempts(module_name, source_code, source_ref):
    if False:
        return 10
    cache_name = makeCacheName(module_name, source_code)
    cache_filename = _getCacheFilename(cache_name, 'json')
    if not os.path.exists(cache_filename):
        return None
    data = loadJsonFromFilename(cache_filename)
    if data is None:
        return None
    if data.get('file_format_version') != _cache_format_version:
        return None
    if data['module_name'] != module_name:
        return None
    result = OrderedSet()
    for module_used in data['modules_used']:
        used_module_name = ModuleName(module_used['module_name'])
        if module_used['finding'] == 'relative':
            (_used_module_name, filename, module_kind, finding) = locateModule(module_name=used_module_name.getBasename(), parent_package=used_module_name.getPackageName(), level=1)
        else:
            (_used_module_name, filename, module_kind, finding) = locateModule(module_name=used_module_name, parent_package=None, level=0)
        if finding != module_used['finding'] or module_kind != module_used['module_kind']:
            assert module_name != 'email._header_value_parser', finding
            return None
        result.add(makeModuleUsageAttempt(module_name=used_module_name, filename=filename, finding=module_used['finding'], module_kind=module_used['module_kind'], level=0, source_ref=source_ref.atLineNumber(module_used['source_ref_line']), reason=module_used['reason']))
    for module_used in data['distribution_names']:
        pass
    return result

def writeImportedModulesNamesToCache(module_name, source_code, used_modules, distribution_names):
    if False:
        for i in range(10):
            print('nop')
    cache_name = makeCacheName(module_name, source_code)
    cache_filename = _getCacheFilename(cache_name, 'json')
    used_modules = [module.asDict() for module in used_modules]
    for module in used_modules:
        module['source_ref_line'] = module['source_ref'].getLineNumber()
        del module['source_ref']
    data = {'file_format_version': _cache_format_version, 'module_name': module_name.asString(), 'modules_used': used_modules, 'distribution_names': distribution_names}
    makePath(os.path.dirname(cache_filename))
    writeJsonToFilename(filename=cache_filename, contents=data)

def _getModuleConfigHash(full_name):
    if False:
        while True:
            i = 10
    'Calculate hash value for package packages importable for a module of this name.'
    hash_value = Hash()
    hash_value.updateFromValues(*Plugins.getCacheContributionValues(full_name))
    hash_value.updateFromValues(version_string, sys.version)
    return hash_value.asHexDigest()