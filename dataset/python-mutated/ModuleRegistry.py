""" This to keep track of used modules.

    There is a set of root modules, which are user specified, and must be
    processed. As they go, they add more modules to active modules list
    and move done modules out of it.

    That process can be restarted and modules will be fetched back from
    the existing set of modules.
"""
import collections
import os
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.PythonVersions import python_version
root_modules = OrderedSet()
active_modules = OrderedSet()
active_modules_info = {}
ActiveModuleInfo = collections.namedtuple('ActiveModuleInfo', ('using_module', 'usage_tag', 'reason', 'source_ref'))
done_modules = set()

def addRootModule(module):
    if False:
        return 10
    root_modules.add(module)

def getRootModules():
    if False:
        for i in range(10):
            print('nop')
    return root_modules

def getRootTopModule():
    if False:
        print('Hello World!')
    top_module = next(iter(root_modules))
    assert top_module.isTopModule(), top_module
    return top_module

def hasRootModule(module_name):
    if False:
        print('Hello World!')
    for module in root_modules:
        if module.getFullName() == module_name:
            return True
    return False

def replaceRootModule(old, new):
    if False:
        return 10
    global root_modules
    new_root_modules = OrderedSet()
    for module in root_modules:
        new_root_modules.add(module if module is not old else new)
    assert len(root_modules) == len(new_root_modules)
    root_modules = new_root_modules

def getUncompiledModules():
    if False:
        i = 10
        return i + 15
    result = set()
    for module in getDoneModules():
        if module.isUncompiledPythonModule():
            result.add(module)
    return tuple(sorted(result, key=lambda module: module.getFullName()))

def getUncompiledTechnicalModules():
    if False:
        print('Hello World!')
    result = set()
    for module in getDoneModules():
        if module.isUncompiledPythonModule() and module.isTechnical():
            result.add(module)
    return tuple(sorted(result, key=lambda module: module.getFullName()))

def getUncompiledNonTechnicalModules():
    if False:
        print('Hello World!')
    result = set()
    for module in getDoneModules():
        if module.isUncompiledPythonModule():
            result.add(module)
    return tuple(sorted(result, key=lambda module: module.getFullName()))

def _normalizeModuleFilename(filename):
    if False:
        for i in range(10):
            print('nop')
    if python_version >= 768:
        filename = filename.replace('__pycache__', '')
        suffix = '.cpython-%d.pyc' % (python_version // 10)
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)] + '.py'
    elif filename.endswith('.pyc'):
        filename = filename[:-3] + '.py'
    if os.path.basename(filename) == '__init__.py':
        filename = os.path.dirname(filename)
    return filename

def startTraversal():
    if False:
        return 10
    global active_modules, done_modules, active_modules_info
    active_modules = OrderedSet(root_modules)
    active_modules_info = {}
    for root_module in root_modules:
        active_modules_info[root_module] = ActiveModuleInfo(using_module=None, usage_tag='root_module', reason='Root module', source_ref=None)
    done_modules = set()
    for active_module in active_modules:
        active_module.startTraversal()

def addUsedModule(module, using_module, usage_tag, reason, source_ref):
    if False:
        return 10
    if module not in done_modules and module not in active_modules:
        active_modules.add(module)
        active_modules_info[module] = ActiveModuleInfo(using_module=using_module, usage_tag=usage_tag, reason=reason, source_ref=source_ref)
        module.startTraversal()

def nextModule():
    if False:
        while True:
            i = 10
    if active_modules:
        result = active_modules.pop()
        done_modules.add(result)
        return result
    else:
        return None

def getRemainingModulesCount():
    if False:
        while True:
            i = 10
    return len(active_modules)

def getDoneModulesCount():
    if False:
        for i in range(10):
            print('nop')
    return len(done_modules)

def getDoneModules():
    if False:
        i = 10
        return i + 15
    return sorted(done_modules, key=lambda module: (module.getFullName(), module.kind))

def hasDoneModule(module_name):
    if False:
        return 10
    return any((module.getFullName() == module_name for module in done_modules))

def getModuleInclusionInfoByName(module_name):
    if False:
        print('Hello World!')
    for (module, info) in active_modules_info.items():
        if module.getFullName() == module_name:
            return info
    return None

def getModuleFromCodeName(code_name):
    if False:
        for i in range(10):
            print('nop')
    for module in root_modules:
        if module.getCodeName() == code_name:
            return module
    assert False, code_name

def getOwnerFromCodeName(code_name):
    if False:
        while True:
            i = 10
    if '$$$' in code_name:
        (module_code_name, _function_code_name) = code_name.split('$$$', 1)
        module = getModuleFromCodeName(module_code_name)
        return module.getFunctionFromCodeName(code_name)
    else:
        return getModuleFromCodeName(code_name)

def getModuleByName(module_name):
    if False:
        return 10
    for module in active_modules:
        if module.getFullName() == module_name:
            return module
    for module in done_modules:
        if module.getFullName() == module_name:
            return module
    return None
module_influencing_plugins = {}

def addModuleInfluencingCondition(module_name, plugin_name, condition, control_tags, result):
    if False:
        print('Hello World!')
    if module_name not in module_influencing_plugins:
        module_influencing_plugins[module_name] = OrderedSet()
    module_influencing_plugins[module_name].add((plugin_name, 'condition-used', (condition, tuple(control_tags), result)))

def getModuleInfluences(module_name):
    if False:
        for i in range(10):
            print('nop')
    return module_influencing_plugins.get(module_name, ())
module_timing_infos = {}
ModuleOptimizationTimingInfo = collections.namedtuple('ModuleOptimizationTimingInfo', ('pass_number', 'time_used'))

def addModuleOptimizationTimeInformation(module_name, pass_number, time_used):
    if False:
        print('Hello World!')
    module_timing_info = list(module_timing_infos.get(module_name, []))
    module_timing_info.append(ModuleOptimizationTimingInfo(pass_number=pass_number, time_used=time_used))
    module_timing_infos[module_name] = tuple(module_timing_info)

def getModuleOptimizationTimingInfos(module_name):
    if False:
        return 10
    return module_timing_infos.get(module_name, ())