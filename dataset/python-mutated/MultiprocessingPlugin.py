""" Standard plug-in to make multiprocessing and joblib work well.

On Windows, the multiprocessing modules forks new processes which then have
to start from scratch. This won't work if there is no "sys.executable" to
point to a "Python.exe" and won't use compiled code by default.

The issue applies to accelerated and standalone mode alike.

spell-checker: ignore joblib
"""
from nuitka import Options
from nuitka.ModuleRegistry import getModuleInclusionInfoByName, getRootTopModule
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonVersions import python_version
from nuitka.tree.SourceHandling import readSourceCodeFromFilename
from nuitka.utils.ModuleNames import ModuleName

class NuitkaPluginMultiprocessingWorkarounds(NuitkaPluginBase):
    """This is to make multiprocessing work with Nuitka and use compiled code.

    When running in accelerated mode, it's not good to fork a new Python
    instance to run other code, as that won't be accelerated. And when
    run in standalone mode, there may not even be a Python, but it's the
    same principle.

    So by default, this module is on and works around the behavior of the
    "multiprocessing.forking/multiprocessing.spawn/multiprocessing.manager"
    expectations.
    """
    plugin_name = 'multiprocessing'
    plugin_desc = "Required by Python's 'multiprocessing' module."

    @classmethod
    def isRelevant(cls):
        if False:
            print('Hello World!')
        return not Options.shallMakeModule()

    @staticmethod
    def isAlwaysEnabled():
        if False:
            return 10
        return True

    @staticmethod
    def createPreModuleLoadCode(module):
        if False:
            print('Hello World!')
        full_name = module.getFullName()
        if full_name == 'multiprocessing':
            code = 'import sys, os\nsys.frozen = 1\nargv0 = sys.argv[0]\nif sys.platform == "win32" and not os.path.exists(argv0) and not argv0.endswith(".exe"):\n    argv0 += ".exe"\n\nsys.executable = %s\nsys._base_executable = sys.executable\n' % ('__nuitka_binary_exe' if Options.isStandaloneMode() else 'argv0')
            return (code, 'Monkey patching "multiprocessing" load environment.')

    @staticmethod
    def createPostModuleLoadCode(module):
        if False:
            while True:
                i = 10
        full_name = module.getFullName()
        if full_name == 'multiprocessing':
            code = 'try:\n    from multiprocessing.forking import ForkingPickler\nexcept ImportError:\n    from multiprocessing.reduction import ForkingPickler\n\nclass C:\n   def f():\n       pass\n\ndef _reduce_compiled_method(m):\n    if m.im_self is None:\n        return getattr, (m.im_class, m.im_func.__name__)\n    else:\n        return getattr, (m.im_self, m.im_func.__name__)\n\nForkingPickler.register(type(C().f), _reduce_compiled_method)\nif str is bytes:\n    ForkingPickler.register(type(C.f), _reduce_compiled_method)\n'
            return (code, 'Monkey patching "multiprocessing" for compiled methods.')

    @staticmethod
    def createFakeModuleDependency(module):
        if False:
            for i in range(10):
                print('nop')
        full_name = module.getFullName()
        if full_name != 'multiprocessing':
            return
        root_module = getRootTopModule()
        module_name = ModuleName('__parents_main__')
        source_code = readSourceCodeFromFilename(module_name, root_module.getFilename())
        if python_version >= 832:
            source_code += '\ndef __nuitka_freeze_support():\n    import sys\n\n    # Not needed, and can crash from minor __file__ differences, depending on invocation\n    import multiprocessing.spawn\n    multiprocessing.spawn._fixup_main_from_path = lambda mod_name : None\n\n    # This is a variant of freeze_support that will work for multiprocessing and\n    # joblib equally well.\n    kwds = {}\n    args = []\n    for arg in sys.argv[2:]:\n        try:\n            name, value = arg.split(\'=\')\n        except ValueError:\n            name = "pipe_handle"\n            value = arg\n\n        if value == \'None\':\n            kwds[name] = None\n        else:\n            kwds[name] = int(value)\n\n    # Otherwise main module names will not work.\n    sys.modules["__main__"] = sys.modules["__parents_main__"]\n\n    multiprocessing.spawn.spawn_main(*args, **kwds)\n__nuitka_freeze_support()\n'
        else:
            source_code += '\n__import__("sys").modules["__main__"] = __import__("sys").modules[__name__]\n__import__("multiprocessing.forking").forking.freeze_support()'
        yield (module_name, source_code, root_module.getCompileTimeFilename(), 'Auto enable multiprocessing freeze support')

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            print('Hello World!')
        if module_name.hasNamespace('multiprocessing'):
            return (True, 'Multiprocessing plugin needs this to monkey patch it.')

    def decideCompilation(self, module_name):
        if False:
            while True:
                i = 10
        if module_name.hasNamespace('multiprocessing'):
            return 'bytecode'

    @staticmethod
    def getPreprocessorSymbols():
        if False:
            for i in range(10):
                print('nop')
        if getModuleInclusionInfoByName('__parents_main__'):
            return {'_NUITKA_PLUGIN_MULTIPROCESSING_ENABLED': '1'}