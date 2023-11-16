import importlib.util
import sys

def check_for_package(package):
    if False:
        while True:
            i = 10
    if package in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package)) is not None:
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
            return True
        except ImportError:
            return False
    else:
        return False