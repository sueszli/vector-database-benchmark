import sys
import importlib.util

def lazy_import(name):
    if False:
        i = 10
        return i + 15
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
lazy_module = lazy_import(sys.argv[1])
print(dir(lazy_module))