import importlib.machinery
import importlib.util
import os
import sys

def __import_pywin32_system_module__(modname, globs):
    if False:
        while True:
            i = 10
    suffix = '_d' if '_d.pyd' in importlib.machinery.EXTENSION_SUFFIXES else ''
    filename = '%s%d%d%s.dll' % (modname, sys.version_info[0], sys.version_info[1], suffix)
    if hasattr(sys, 'frozen'):
        for look in sys.path:
            if os.path.isfile(look):
                look = os.path.dirname(look)
            found = os.path.join(look, filename)
            if os.path.isfile(found):
                break
        else:
            raise ImportError(f"Module '{modname}' isn't in frozen sys.path {sys.path}")
    else:
        import _win32sysloader
        found = _win32sysloader.GetModuleFilename(filename)
        if found is None:
            found = _win32sysloader.LoadModule(filename)
        if found is None:
            if os.path.isfile(os.path.join(sys.prefix, filename)):
                found = os.path.join(sys.prefix, filename)
        if found is None:
            if os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                found = os.path.join(os.path.dirname(__file__), filename)
        if found is None:
            import pywin32_system32
            for path in pywin32_system32.__path__:
                maybe = os.path.join(path, filename)
                if os.path.isfile(maybe):
                    found = maybe
                    break
        if found is None:
            raise ImportError(f"No system module '{modname}' ({filename})")
    old_mod = sys.modules[modname]
    loader = importlib.machinery.ExtensionFileLoader(modname, found)
    spec = importlib.machinery.ModuleSpec(name=modname, loader=loader, origin=found)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert sys.modules[modname] is mod
    sys.modules[modname] = old_mod
    globs.update(mod.__dict__)
__import_pywin32_system_module__('pywintypes', globals())