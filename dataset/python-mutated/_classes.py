import types
import torch._C

class _ClassNamespace(types.ModuleType):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        super().__init__('torch.classes' + name)
        self.name = name

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
        if proxy is None:
            raise RuntimeError(f'Class {self.name}.{attr} not registered!')
        return proxy

class _Classes(types.ModuleType):
    __file__ = '_classes.py'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('torch.classes')

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        namespace = _ClassNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def loaded_libraries(self):
        if False:
            return 10
        return torch.ops.loaded_libraries

    def load_library(self, path):
        if False:
            return 10
        "\n        Loads a shared library from the given path into the current process.\n\n        The library being loaded may run global initialization code to register\n        custom classes with the PyTorch JIT runtime. This allows dynamically\n        loading custom classes. For this, you should compile your class\n        and the static registration code into a shared library object, and then\n        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the\n        shared object.\n\n        After the library is loaded, it is added to the\n        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected\n        for the paths of all libraries loaded using this function.\n\n        Args:\n            path (str): A path to a shared library to load.\n        "
        torch.ops.load_library(path)
classes = _Classes()