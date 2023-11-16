""" Import cache.

This is not about caching the search of modules in the file system, but about
maintaining a cache of module trees built.

It can happen that modules become unused, and then dropped from active modules,
and then later active again, via another import, and in this case, we should
not start anew, but reuse what we already found out about it.
"""
import os
from nuitka.plugins.Plugins import Plugins
imported_modules = {}
imported_by_name = {}

def addImportedModule(imported_module):
    if False:
        for i in range(10):
            print('nop')
    module_filename = os.path.abspath(imported_module.getFilename())
    if os.path.basename(module_filename) == '__init__.py':
        module_filename = os.path.dirname(module_filename)
    key = (module_filename, imported_module.getFullName())
    if key in imported_modules:
        assert imported_module is imported_modules[key], key
    else:
        Plugins.onModuleDiscovered(imported_module)
    imported_modules[key] = imported_module
    imported_by_name[imported_module.getFullName()] = imported_module
    assert not imported_module.isMainModule()

def isImportedModuleByName(full_name):
    if False:
        print('Hello World!')
    return full_name in imported_by_name

def getImportedModuleByName(full_name):
    if False:
        for i in range(10):
            print('nop')
    return imported_by_name[full_name]

def getImportedModuleByNameAndPath(full_name, module_filename):
    if False:
        for i in range(10):
            print('nop')
    if module_filename is None:
        return getImportedModuleByName(full_name)
    module_filename = os.path.abspath(module_filename)
    if os.path.basename(module_filename) == '__init__.py':
        module_filename = os.path.dirname(module_filename)
    return imported_modules[module_filename, full_name]

def replaceImportedModule(old, new):
    if False:
        return 10
    for (key, value) in imported_by_name.items():
        if value == old:
            imported_by_name[key] = new
            break
    else:
        assert False
    for (key, value) in imported_modules.items():
        if value == old:
            imported_modules[key] = new
            break
    else:
        assert False