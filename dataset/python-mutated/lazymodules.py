"""
Lazy modules.

They are useful to not import big modules until it's really necessary.
"""
from spyder_kernels.utils.misc import is_module_installed

class FakeObject:
    """Fake class used in replacement of missing objects"""
    pass

class LazyModule:
    """Lazy module loader class."""

    def __init__(self, modname, second_level_attrs=None):
        if False:
            return 10
        "\n        Lazy module loader class.\n\n        Parameters\n        ----------\n        modname: str\n            Module name to lazy load.\n        second_level_attrs: list (optional)\n            List of second level attributes to add to the FakeObject\n            that stands for the module in case it's not found.\n        "
        self.__spy_modname__ = modname
        self.__spy_mod__ = FakeObject
        if second_level_attrs is not None:
            for attr in second_level_attrs:
                setattr(self.__spy_mod__, attr, FakeObject)

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if is_module_installed(self.__spy_modname__):
            self.__spy_mod__ = __import__(self.__spy_modname__)
        else:
            return self.__spy_mod__
        return getattr(self.__spy_mod__, name)
numpy = LazyModule('numpy', ['MaskedArray'])
pandas = LazyModule('pandas')
PIL = LazyModule('PIL.Image', ['Image'])
bs4 = LazyModule('bs4', ['NavigableString'])
scipy = LazyModule('scipy.io')