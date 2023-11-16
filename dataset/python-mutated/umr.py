"""User module reloader."""
import os
import sys
from spyder_kernels.customize.utils import path_is_library

class UserModuleReloader:
    """
    User Module Reloader (UMR) aims at deleting user modules
    to force Python to deeply reload them during import

    pathlist [list]: blacklist in terms of module path
    namelist [list]: blacklist in terms of module name
    """

    def __init__(self, namelist=None, pathlist=None):
        if False:
            while True:
                i = 10
        if namelist is None:
            namelist = []
        else:
            try:
                namelist = namelist.split(',')
            except Exception:
                namelist = []
        spy_modules = ['spyder_kernels']
        mpl_modules = ['matplotlib', 'tkinter', 'Tkinter']
        other_modules = ['pytorch', 'pythoncom', 'tensorflow']
        self.namelist = namelist + spy_modules + mpl_modules + other_modules
        self.pathlist = pathlist
        self.previous_modules = list(sys.modules.keys())
        enabled = os.environ.get('SPY_UMR_ENABLED', '')
        self.enabled = enabled.lower() == 'true'
        verbose = os.environ.get('SPY_UMR_VERBOSE', '')
        self.verbose = verbose.lower() == 'true'

    def is_module_reloadable(self, module, modname):
        if False:
            print('Hello World!')
        'Decide if a module is reloadable or not.'
        if path_is_library(getattr(module, '__file__', None), self.pathlist) or self.is_module_in_namelist(modname):
            return False
        else:
            return True

    def is_module_in_namelist(self, modname):
        if False:
            return 10
        'Decide if a module can be reloaded or not according to its name.'
        return set(modname.split('.')) & set(self.namelist)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Delete user modules to force Python to deeply reload them\n\n        Do not del modules which are considered as system modules, i.e.\n        modules installed in subdirectories of Python interpreter's binary\n        Do not del C modules\n        "
        modnames_to_reload = []
        for (modname, module) in list(sys.modules.items()):
            if modname not in self.previous_modules:
                if self.is_module_reloadable(module, modname):
                    modnames_to_reload.append(modname)
                    del sys.modules[modname]
                else:
                    continue
        if self.verbose and modnames_to_reload:
            modnames = modnames_to_reload
            print('\x1b[4;33m%s\x1b[24m%s\x1b[0m' % ('Reloaded modules', ': ' + ', '.join(modnames)))
        return modnames_to_reload