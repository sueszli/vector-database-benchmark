"""
A context manager for managing things injected into :mod:`builtins`.
"""
import builtins as builtin_mod
from traitlets.config.configurable import Configurable
from traitlets import Instance

class __BuiltinUndefined(object):
    pass
BuiltinUndefined = __BuiltinUndefined()

class __HideBuiltin(object):
    pass
HideBuiltin = __HideBuiltin()

class BuiltinTrap(Configurable):
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None):
        if False:
            return 10
        super(BuiltinTrap, self).__init__(shell=shell, config=None)
        self._orig_builtins = {}
        self._nested_level = 0
        self.shell = shell
        self.auto_builtins = {'exit': HideBuiltin, 'quit': HideBuiltin, 'get_ipython': self.shell.get_ipython}

    def __enter__(self):
        if False:
            while True:
                i = 10
        if self._nested_level == 0:
            self.activate()
        self._nested_level += 1
        return self

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self._nested_level == 1:
            self.deactivate()
        self._nested_level -= 1
        return False

    def add_builtin(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Add a builtin and save the original.'
        bdict = builtin_mod.__dict__
        orig = bdict.get(key, BuiltinUndefined)
        if value is HideBuiltin:
            if orig is not BuiltinUndefined:
                self._orig_builtins[key] = orig
                del bdict[key]
        else:
            self._orig_builtins[key] = orig
            bdict[key] = value

    def remove_builtin(self, key, orig):
        if False:
            while True:
                i = 10
        'Remove an added builtin and re-set the original.'
        if orig is BuiltinUndefined:
            del builtin_mod.__dict__[key]
        else:
            builtin_mod.__dict__[key] = orig

    def activate(self):
        if False:
            print('Hello World!')
        'Store ipython references in the __builtin__ namespace.'
        add_builtin = self.add_builtin
        for (name, func) in self.auto_builtins.items():
            add_builtin(name, func)

    def deactivate(self):
        if False:
            print('Hello World!')
        'Remove any builtins which might have been added by add_builtins, or\n        restore overwritten ones to their previous values.'
        remove_builtin = self.remove_builtin
        for (key, val) in self._orig_builtins.items():
            remove_builtin(key, val)
        self._orig_builtins.clear()
        self._builtins_added = False