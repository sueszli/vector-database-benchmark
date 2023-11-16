from __future__ import annotations
import contextlib
import functools
import importlib
import importlib.abc
import importlib.machinery
import os
import pathlib
import sys
import threading
import warnings
from abc import abstractmethod
from importlib._bootstrap import _ImportLockContext as ImportLock
from types import ModuleType
from typing import Any, ContextManager, Dict, List, NamedTuple
from typing_extensions import Self
from .fast_slow_proxy import _FunctionProxy, _is_function_or_method, _Unusable, get_final_type_map, get_intermediate_type_map, get_registered_functions

def rename_root_module(module: str, root: str, new_root: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Rename a module to a new root.\n\n    Parameters\n    ----------\n    module\n        Module to rename\n    root\n        Original root\n    new_root\n        New root\n\n    Returns\n    -------\n    New module name (if it matches root) otherwise original name.\n    '
    if module.startswith(root):
        return new_root + module[len(root):]
    else:
        return module

class DeducedMode(NamedTuple):
    use_fast_lib: bool
    slow_lib: str
    fast_lib: str

def deduce_cudf_pandas_mode(slow_lib: str, fast_lib: str) -> DeducedMode:
    if False:
        return 10
    '\n    Determine if cudf.pandas should use the requested fast library.\n\n    Parameters\n    ----------\n    slow_lib\n        Name of the slow library\n    fast_lib\n        Name of the fast library\n\n    Returns\n    -------\n    Whether the fast library is being used, and the resulting names of\n    the "slow" and "fast" libraries.\n    '
    if 'CUDF_PANDAS_FALLBACK_MODE' not in os.environ:
        try:
            importlib.import_module(fast_lib)
            return DeducedMode(use_fast_lib=True, slow_lib=slow_lib, fast_lib=fast_lib)
        except Exception as e:
            warnings.warn(f'Exception encountered importing {fast_lib}: {e}.Falling back to only using {slow_lib}.')
    return DeducedMode(use_fast_lib=False, slow_lib=slow_lib, fast_lib=slow_lib)

class ModuleAcceleratorBase(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _instance: ModuleAcceleratorBase | None = None
    mod_name: str
    fast_lib: str
    slow_lib: str
    _wrapped_objs: dict[Any, Any]

    def __new__(cls, mod_name: str, fast_lib: str, slow_lib: str):
        if False:
            return 10
        'Build a custom module finder that will provide wrapped modules\n        on demand.\n\n        Parameters\n        ----------\n        mod_name\n             Import name to deliver modules under.\n        fast_lib\n             Name of package that provides "fast" implementation\n        slow_lib\n             Name of package that provides "slow" fallback implementation\n        '
        if ModuleAcceleratorBase._instance is not None:
            raise RuntimeError('Only one instance of ModuleAcceleratorBase allowed')
        self = object.__new__(cls)
        self.mod_name = mod_name
        self.fast_lib = fast_lib
        self.slow_lib = slow_lib
        self._wrapped_objs = {}
        self._wrapped_objs.update(get_final_type_map())
        self._wrapped_objs.update(get_intermediate_type_map())
        self._wrapped_objs.update(get_registered_functions())
        ModuleAcceleratorBase._instance = self
        return self

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(fast={self.fast_lib}, slow={self.slow_lib})'

    def find_spec(self, fullname: str, path, target=None) -> importlib.machinery.ModuleSpec | None:
        if False:
            return 10
        "Provide ourselves as a module loader.\n\n        Parameters\n        ----------\n        fullname\n            Name of module to be imported, if it starts with the name\n            that we are using to wrap, we will deliver ourselves as a\n            loader, otherwise defer to the standard Python loaders.\n\n        Returns\n        -------\n        A ModuleSpec with ourself as loader if we're interposing,\n        otherwise None to pass off to the next loader.\n        "
        if fullname == self.mod_name or fullname.startswith(f'{self.mod_name}.'):
            return importlib.machinery.ModuleSpec(name=fullname, loader=self, origin=None, loader_state=None, is_package=True)
        return None

    def create_module(self, spec) -> ModuleType | None:
        if False:
            for i in range(10):
                print('nop')
        return None

    def exec_module(self, mod: ModuleType):
        if False:
            i = 10
            return i + 15
        self._populate_module(mod)

    @abstractmethod
    def disabled(self) -> ContextManager:
        if False:
            i = 10
            return i + 15
        pass

    def _postprocess_module(self, mod: ModuleType, slow_mod: ModuleType, fast_mod: ModuleType | None) -> ModuleType:
        if False:
            return 10
        'Ensure that the wrapped module satisfies required invariants.\n\n        Parameters\n        ----------\n        mod\n            Wrapped module to postprocess\n        slow_mod\n            Slow version that we are mimicking\n        fast_mod\n            Fast module that provides accelerated implementations (may\n            be None\n\n        Returns\n        -------\n        Checked and validated module\n\n        Notes\n        -----\n        The implementation of fast-slow proxies imposes certain\n        requirements on the wrapped modules that it delivers. This\n        function encodes those requirements and raises if the module\n        does not satisfy them.\n\n        This post-processing routine should be kept up to date with any\n        requirements encoded by fast_slow_proxy.py\n        '
        mod.__dict__['_fsproxy_slow'] = slow_mod
        if fast_mod is not None:
            mod.__dict__['_fsproxy_fast'] = fast_mod
        return mod

    @abstractmethod
    def _populate_module(self, mod: ModuleType) -> ModuleType:
        if False:
            return 10
        "Populate given module with appropriate attributes.\n\n        This traverses the attributes of the slow module corresponding\n        to mod and mirrors those in the provided module in a wrapped\n        mode that attempts to execute them using the fast module first.\n\n        Parameters\n        ----------\n        mod\n            Module to populate\n\n        Returns\n        -------\n        ModuleType\n            Populated module\n\n        Notes\n        -----\n        In addition to the attributes of the slow module,\n        the returned module must have the following attributes:\n\n        - '_fsproxy_slow': the corresponding slow module\n        - '_fsproxy_fast': the corresponding fast module\n\n        This is necessary for correct rewriting of UDFs when calling\n        to the respective fast/slow libraries.\n\n        The necessary invariants are checked and applied in\n        :meth:`_postprocess_module`.\n        "
        pass

    def _wrap_attribute(self, slow_attr: Any, fast_attr: Any | _Unusable, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the wrapped version of an attribute.\n\n        Parameters\n        ----------\n        slow_attr : Any\n            The attribute from the slow module\n        fast_mod : Any (or None)\n            The same attribute from the fast module, if it exists\n        name\n            Name of attribute\n\n        Returns\n        -------\n        Wrapped attribute\n        '
        wrapped_attr: Any
        if name in {'__all__', '__dir__', '__file__', '__doc__'}:
            wrapped_attr = slow_attr
        elif self.fast_lib == self.slow_lib:
            wrapped_attr = slow_attr
        if any([slow_attr in get_registered_functions(), slow_attr in get_final_type_map(), slow_attr in get_intermediate_type_map()]):
            return self._wrapped_objs[slow_attr]
        if isinstance(slow_attr, ModuleType) and slow_attr.__name__.startswith(self.slow_lib):
            return importlib.import_module(rename_root_module(slow_attr.__name__, self.slow_lib, self.mod_name))
        if slow_attr in self._wrapped_objs:
            if type(fast_attr) is _Unusable:
                return self._wrapped_objs[slow_attr]
        if _is_function_or_method(slow_attr):
            wrapped_attr = _FunctionProxy(fast_attr, slow_attr)
        else:
            wrapped_attr = slow_attr
        return wrapped_attr

    @classmethod
    @abstractmethod
    def install(cls, destination_module: str, fast_lib: str, slow_lib: str) -> Self | None:
        if False:
            i = 10
            return i + 15
        '\n        Install the loader in sys.meta_path.\n\n        Parameters\n        ----------\n        destination_module\n            Name under which the importer will kick in\n        fast_lib\n            Name of fast module\n        slow_lib\n            Name of slow module we are trying to mimic\n\n        Returns\n        -------\n        Instance of the class (or None if the loader was not installed)\n\n        Notes\n        -----\n        This function is idempotent. If called with the same arguments\n        a second time, it does not create a new loader, but instead\n        returns the existing loader from ``sys.meta_path``.\n\n        '
        pass

class ModuleAccelerator(ModuleAcceleratorBase):
    """
    A finder and loader that produces "accelerated" modules.

    When someone attempts to import the specified slow library with
    this finder enabled, we intercept the import and deliver an
    equivalent, accelerated, version of the module. This provides
    attributes and modules that check if they are being used from
    "within" the slow (or fast) library themselves. If this is the
    case, the implementation is forwarded to the actual slow library
    implementation, otherwise a proxy implementation is used (which
    attempts to call the fast version first).
    """
    _denylist: List[str]
    _use_fast_lib: bool
    _use_fast_lib_lock: threading.RLock
    _module_cache_prefix: str = '_slow_lib_'

    def __new__(cls, fast_lib, slow_lib):
        if False:
            i = 10
            return i + 15
        self = super().__new__(cls, slow_lib, fast_lib, slow_lib)
        slow_module = importlib.import_module(slow_lib)
        fast_module = importlib.import_module(fast_lib)
        for mod in sys.modules.copy():
            if mod.startswith(self.slow_lib):
                sys.modules[self._module_cache_prefix + mod] = sys.modules[mod]
                del sys.modules[mod]
        self._denylist = [*slow_module.__path__, *fast_module.__path__]
        self._use_fast_lib_lock = threading.RLock()
        self._use_fast_lib = True
        return self

    def _populate_module(self, mod: ModuleType):
        if False:
            for i in range(10):
                print('nop')
        mod_name = mod.__name__
        slow_mod = importlib.import_module(rename_root_module(mod_name, self.slow_lib, self._module_cache_prefix + self.slow_lib))
        try:
            fast_mod = importlib.import_module(rename_root_module(mod_name, self.slow_lib, self.fast_lib))
        except Exception:
            fast_mod = None
        real_attributes = {}
        for key in slow_mod.__dir__():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                slow_attr = getattr(slow_mod, key)
            fast_attr = getattr(fast_mod, key, _Unusable())
            real_attributes[key] = slow_attr
            try:
                wrapped_attr = self._wrap_attribute(slow_attr, fast_attr, key)
                self._wrapped_objs[slow_attr] = wrapped_attr
            except TypeError:
                pass
        setattr(mod, '__getattr__', functools.partial(self.getattr_real_or_wrapped, real=real_attributes, wrapped_objs=self._wrapped_objs, loader=self))
        setattr(mod, '__dir__', slow_mod.__dir__)
        if getattr(slow_mod, '__path__', False):
            assert mod.__spec__
            mod.__path__ = slow_mod.__path__
            mod.__spec__.submodule_search_locations = [*slow_mod.__path__]
        return self._postprocess_module(mod, slow_mod, fast_mod)

    @contextlib.contextmanager
    def disabled(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a context manager for disabling the module accelerator.\n\n        Within the block, any wrapped objects will instead deliver\n        attributes from their real counterparts (as if the current\n        nested block were in the denylist).\n\n        Returns\n        -------\n        Context manager for disabling things\n        '
        try:
            self._use_fast_lib_lock.acquire()
            saved = self._use_fast_lib
            self._use_fast_lib = False
            yield
        finally:
            self._use_fast_lib = saved
            self._use_fast_lib_lock.release()

    @staticmethod
    def getattr_real_or_wrapped(name: str, *, real: Dict[str, Any], wrapped_objs, loader: ModuleAccelerator) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Obtain an attribute from a module from either the real or\n        wrapped namespace.\n\n        Parameters\n        ----------\n        name\n            Attribute to return\n        real\n            Unwrapped "original" attributes\n        wrapped\n            Wrapped attributes\n        loader\n            Loader object that manages denylist and other skipping\n\n        Returns\n        -------\n        The requested attribute (either real or wrapped)\n        '
        with loader._use_fast_lib_lock:
            use_real = not loader._use_fast_lib
        if not use_real:
            frame = sys._getframe()
            assert frame.f_back
            calling_module = pathlib.PurePath(frame.f_back.f_code.co_filename)
            use_real = any((calling_module.is_relative_to(path) for path in loader._denylist))
        try:
            if use_real:
                return real[name]
            else:
                return wrapped_objs[real[name]]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")
        except TypeError:
            return real[name]

    @classmethod
    def install(cls, destination_module: str, fast_lib: str, slow_lib: str) -> Self | None:
        if False:
            for i in range(10):
                print('nop')
        with ImportLock():
            if destination_module != slow_lib:
                raise RuntimeError(f"Destination module '{destination_module}' must match'{slow_lib}' for this to work.")
            mode = deduce_cudf_pandas_mode(slow_lib, fast_lib)
            if mode.use_fast_lib:
                importlib.import_module(f'.._wrappers.{mode.slow_lib}', __name__)
            try:
                (self,) = (p for p in sys.meta_path if isinstance(p, cls) and p.slow_lib == mode.slow_lib and (p.fast_lib == mode.fast_lib))
            except ValueError:
                self = cls(mode.fast_lib, mode.slow_lib)
                sys.meta_path.insert(0, self)
            return self

def disable_module_accelerator() -> contextlib.ExitStack:
    if False:
        print('Hello World!')
    '\n    Temporarily disable any module acceleration.\n    '
    with contextlib.ExitStack() as stack:
        for finder in sys.meta_path:
            if isinstance(finder, ModuleAcceleratorBase):
                stack.enter_context(finder.disabled())
        return stack.pop_all()
    assert False