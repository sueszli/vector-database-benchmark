class _LazyModule:

    def __init__(self, name):
        if False:
            return 10
        self.__name = name

    def _do_import(self):
        if False:
            print('Hello World!')
        import Orange
        from importlib import import_module
        mod = import_module('Orange.' + self.__name, package='Orange')
        setattr(Orange, self.__name, mod)
        return mod

    def __getattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._do_import(), key)

    def __dir__(self):
        if False:
            print('Hello World!')
        return list(self._do_import().__dict__)