"""Core implementation of import.

This module is NOT meant to be directly imported! It has been designed such
that it can be bootstrapped into Python as the implementation of import. As
such it requires the injection of specific modules and attributes in order to
work. One should use importlib as the public-facing version of this module.

"""

def _object_name(obj):
    if False:
        print('Hello World!')
    try:
        return obj.__qualname__
    except AttributeError:
        return type(obj).__qualname__
_thread = None
_warnings = None
_weakref = None
_bootstrap_external = None

def _wrap(new, old):
    if False:
        print('Hello World!')
    'Simple substitute for functools.update_wrapper.'
    for replace in ['__module__', '__name__', '__qualname__', '__doc__']:
        if hasattr(old, replace):
            setattr(new, replace, getattr(old, replace))
    new.__dict__.update(old.__dict__)

def _new_module(name):
    if False:
        return 10
    return type(sys)(name)
_module_locks = {}
_blocking_on = {}

class _DeadlockError(RuntimeError):
    pass

class _ModuleLock:
    """A recursive lock implementation which is able to detect deadlocks
    (e.g. thread 1 trying to take locks A then B, and thread 2 trying to
    take locks B then A).
    """

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.lock = _thread.allocate_lock()
        self.wakeup = _thread.allocate_lock()
        self.name = name
        self.owner = None
        self.count = 0
        self.waiters = 0

    def has_deadlock(self):
        if False:
            i = 10
            return i + 15
        me = _thread.get_ident()
        tid = self.owner
        seen = set()
        while True:
            lock = _blocking_on.get(tid)
            if lock is None:
                return False
            tid = lock.owner
            if tid == me:
                return True
            if tid in seen:
                return False
            seen.add(tid)

    def acquire(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Acquire the module lock.  If a potential deadlock is detected,\n        a _DeadlockError is raised.\n        Otherwise, the lock is always acquired and True is returned.\n        '
        tid = _thread.get_ident()
        _blocking_on[tid] = self
        try:
            while True:
                with self.lock:
                    if self.count == 0 or self.owner == tid:
                        self.owner = tid
                        self.count += 1
                        return True
                    if self.has_deadlock():
                        raise _DeadlockError('deadlock detected by %r' % self)
                    if self.wakeup.acquire(False):
                        self.waiters += 1
                self.wakeup.acquire()
                self.wakeup.release()
        finally:
            del _blocking_on[tid]

    def release(self):
        if False:
            print('Hello World!')
        tid = _thread.get_ident()
        with self.lock:
            if self.owner != tid:
                raise RuntimeError('cannot release un-acquired lock')
            assert self.count > 0
            self.count -= 1
            if self.count == 0:
                self.owner = None
                if self.waiters:
                    self.waiters -= 1
                    self.wakeup.release()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '_ModuleLock({!r}) at {}'.format(self.name, id(self))

class _DummyModuleLock:
    """A simple _ModuleLock equivalent for Python builds without
    multi-threading support."""

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.count = 0

    def acquire(self):
        if False:
            while True:
                i = 10
        self.count += 1
        return True

    def release(self):
        if False:
            return 10
        if self.count == 0:
            raise RuntimeError('cannot release un-acquired lock')
        self.count -= 1

    def __repr__(self):
        if False:
            return 10
        return '_DummyModuleLock({!r}) at {}'.format(self.name, id(self))

class _ModuleLockManager:

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._name = name
        self._lock = None

    def __enter__(self):
        if False:
            print('Hello World!')
        self._lock = _get_module_lock(self._name)
        self._lock.acquire()

    def __exit__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._lock.release()

def _get_module_lock(name):
    if False:
        i = 10
        return i + 15
    'Get or create the module lock for a given module name.\n\n    Acquire/release internally the global import lock to protect\n    _module_locks.'
    _imp.acquire_lock()
    try:
        try:
            lock = _module_locks[name]()
        except KeyError:
            lock = None
        if lock is None:
            if _thread is None:
                lock = _DummyModuleLock(name)
            else:
                lock = _ModuleLock(name)

            def cb(ref, name=name):
                if False:
                    return 10
                _imp.acquire_lock()
                try:
                    if _module_locks.get(name) is ref:
                        del _module_locks[name]
                finally:
                    _imp.release_lock()
            _module_locks[name] = _weakref.ref(lock, cb)
    finally:
        _imp.release_lock()
    return lock

def _lock_unlock_module(name):
    if False:
        print('Hello World!')
    'Acquires then releases the module lock for a given module name.\n\n    This is used to ensure a module is completely initialized, in the\n    event it is being imported by another thread.\n    '
    lock = _get_module_lock(name)
    try:
        lock.acquire()
    except _DeadlockError:
        pass
    else:
        lock.release()

def _call_with_frames_removed(f, *args, **kwds):
    if False:
        print('Hello World!')
    'remove_importlib_frames in import.c will always remove sequences\n    of importlib frames that end with a call to this function\n\n    Use it instead of a normal call in places where including the importlib\n    frames introduces unwanted noise into the traceback (e.g. when executing\n    module code)\n    '
    return f(*args, **kwds)

def _verbose_message(message, *args, verbosity=1):
    if False:
        for i in range(10):
            print('nop')
    'Print the message to stderr if -v/PYTHONVERBOSE is turned on.'
    if sys.flags.verbose >= verbosity:
        if not message.startswith(('#', 'import ')):
            message = '# ' + message
        print(message.format(*args), file=sys.stderr)

def _requires_builtin(fxn):
    if False:
        for i in range(10):
            print('nop')
    'Decorator to verify the named module is built-in.'

    def _requires_builtin_wrapper(self, fullname):
        if False:
            return 10
        if fullname not in sys.builtin_module_names:
            raise ImportError('{!r} is not a built-in module'.format(fullname), name=fullname)
        return fxn(self, fullname)
    _wrap(_requires_builtin_wrapper, fxn)
    return _requires_builtin_wrapper

def _requires_frozen(fxn):
    if False:
        i = 10
        return i + 15
    'Decorator to verify the named module is frozen.'

    def _requires_frozen_wrapper(self, fullname):
        if False:
            while True:
                i = 10
        if not _imp.is_frozen(fullname):
            raise ImportError('{!r} is not a frozen module'.format(fullname), name=fullname)
        return fxn(self, fullname)
    _wrap(_requires_frozen_wrapper, fxn)
    return _requires_frozen_wrapper

def _load_module_shim(self, fullname):
    if False:
        print('Hello World!')
    'Load the specified module into sys.modules and return it.\n\n    This method is deprecated.  Use loader.exec_module() instead.\n\n    '
    msg = 'the load_module() method is deprecated and slated for removal in Python 3.12; use exec_module() instead'
    _warnings.warn(msg, DeprecationWarning)
    spec = spec_from_loader(fullname, self)
    if fullname in sys.modules:
        module = sys.modules[fullname]
        _exec(spec, module)
        return sys.modules[fullname]
    else:
        return _load(spec)

def _module_repr(module):
    if False:
        return 10
    'The implementation of ModuleType.__repr__().'
    loader = getattr(module, '__loader__', None)
    if (spec := getattr(module, '__spec__', None)):
        return _module_repr_from_spec(spec)
    elif hasattr(loader, 'module_repr'):
        try:
            return loader.module_repr(module)
        except Exception:
            pass
    try:
        name = module.__name__
    except AttributeError:
        name = '?'
    try:
        filename = module.__file__
    except AttributeError:
        if loader is None:
            return '<module {!r}>'.format(name)
        else:
            return '<module {!r} ({!r})>'.format(name, loader)
    else:
        return '<module {!r} from {!r}>'.format(name, filename)

class ModuleSpec:
    """The specification for a module, used for loading.

    A module's spec is the source for information about the module.  For
    data associated with the module, including source, use the spec's
    loader.

    `name` is the absolute name of the module.  `loader` is the loader
    to use when loading the module.  `parent` is the name of the
    package the module is in.  The parent is derived from the name.

    `is_package` determines if the module is considered a package or
    not.  On modules this is reflected by the `__path__` attribute.

    `origin` is the specific location used by the loader from which to
    load the module, if that information is available.  When filename is
    set, origin will match.

    `has_location` indicates that a spec's "origin" reflects a location.
    When this is True, `__file__` attribute of the module is set.

    `cached` is the location of the cached bytecode file, if any.  It
    corresponds to the `__cached__` attribute.

    `submodule_search_locations` is the sequence of path entries to
    search when importing submodules.  If set, is_package should be
    True--and False otherwise.

    Packages are simply modules that (may) have submodules.  If a spec
    has a non-None value in `submodule_search_locations`, the import
    system will consider modules loaded from the spec as packages.

    Only finders (see importlib.abc.MetaPathFinder and
    importlib.abc.PathEntryFinder) should modify ModuleSpec instances.

    """

    def __init__(self, name, loader, *, origin=None, loader_state=None, is_package=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.loader = loader
        self.origin = origin
        self.loader_state = loader_state
        self.submodule_search_locations = [] if is_package else None
        self._set_fileattr = False
        self._cached = None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        args = ['name={!r}'.format(self.name), 'loader={!r}'.format(self.loader)]
        if self.origin is not None:
            args.append('origin={!r}'.format(self.origin))
        if self.submodule_search_locations is not None:
            args.append('submodule_search_locations={}'.format(self.submodule_search_locations))
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        smsl = self.submodule_search_locations
        try:
            return self.name == other.name and self.loader == other.loader and (self.origin == other.origin) and (smsl == other.submodule_search_locations) and (self.cached == other.cached) and (self.has_location == other.has_location)
        except AttributeError:
            return NotImplemented

    @property
    def cached(self):
        if False:
            i = 10
            return i + 15
        if self._cached is None:
            if self.origin is not None and self._set_fileattr:
                if _bootstrap_external is None:
                    raise NotImplementedError
                self._cached = _bootstrap_external._get_cached(self.origin)
        return self._cached

    @cached.setter
    def cached(self, cached):
        if False:
            print('Hello World!')
        self._cached = cached

    @property
    def parent(self):
        if False:
            while True:
                i = 10
        "The name of the module's parent."
        if self.submodule_search_locations is None:
            return self.name.rpartition('.')[0]
        else:
            return self.name

    @property
    def has_location(self):
        if False:
            while True:
                i = 10
        return self._set_fileattr

    @has_location.setter
    def has_location(self, value):
        if False:
            while True:
                i = 10
        self._set_fileattr = bool(value)

def spec_from_loader(name, loader, *, origin=None, is_package=None):
    if False:
        for i in range(10):
            print('nop')
    'Return a module spec based on various loader methods.'
    if hasattr(loader, 'get_filename'):
        if _bootstrap_external is None:
            raise NotImplementedError
        spec_from_file_location = _bootstrap_external.spec_from_file_location
        if is_package is None:
            return spec_from_file_location(name, loader=loader)
        search = [] if is_package else None
        return spec_from_file_location(name, loader=loader, submodule_search_locations=search)
    if is_package is None:
        if hasattr(loader, 'is_package'):
            try:
                is_package = loader.is_package(name)
            except ImportError:
                is_package = None
        else:
            is_package = False
    return ModuleSpec(name, loader, origin=origin, is_package=is_package)

def _spec_from_module(module, loader=None, origin=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        spec = module.__spec__
    except AttributeError:
        pass
    else:
        if spec is not None:
            return spec
    name = module.__name__
    if loader is None:
        try:
            loader = module.__loader__
        except AttributeError:
            pass
    try:
        location = module.__file__
    except AttributeError:
        location = None
    if origin is None:
        if location is None:
            try:
                origin = loader._ORIGIN
            except AttributeError:
                origin = None
        else:
            origin = location
    try:
        cached = module.__cached__
    except AttributeError:
        cached = None
    try:
        submodule_search_locations = list(module.__path__)
    except AttributeError:
        submodule_search_locations = None
    spec = ModuleSpec(name, loader, origin=origin)
    spec._set_fileattr = False if location is None else True
    spec.cached = cached
    spec.submodule_search_locations = submodule_search_locations
    return spec

def _init_module_attrs(spec, module, *, override=False):
    if False:
        for i in range(10):
            print('nop')
    if override or getattr(module, '__name__', None) is None:
        try:
            module.__name__ = spec.name
        except AttributeError:
            pass
    if override or getattr(module, '__loader__', None) is None:
        loader = spec.loader
        if loader is None:
            if spec.submodule_search_locations is not None:
                if _bootstrap_external is None:
                    raise NotImplementedError
                _NamespaceLoader = _bootstrap_external._NamespaceLoader
                loader = _NamespaceLoader.__new__(_NamespaceLoader)
                loader._path = spec.submodule_search_locations
                spec.loader = loader
                module.__file__ = None
        try:
            module.__loader__ = loader
        except AttributeError:
            pass
    if override or getattr(module, '__package__', None) is None:
        try:
            module.__package__ = spec.parent
        except AttributeError:
            pass
    try:
        module.__spec__ = spec
    except AttributeError:
        pass
    if override or getattr(module, '__path__', None) is None:
        if spec.submodule_search_locations is not None:
            try:
                module.__path__ = spec.submodule_search_locations
            except AttributeError:
                pass
    if spec.has_location:
        if override or getattr(module, '__file__', None) is None:
            try:
                module.__file__ = spec.origin
            except AttributeError:
                pass
        if override or getattr(module, '__cached__', None) is None:
            if spec.cached is not None:
                try:
                    module.__cached__ = spec.cached
                except AttributeError:
                    pass
    return module

def module_from_spec(spec):
    if False:
        return 10
    'Create a module based on the provided spec.'
    module = None
    if hasattr(spec.loader, 'create_module'):
        module = spec.loader.create_module(spec)
    elif hasattr(spec.loader, 'exec_module'):
        raise ImportError('loaders that define exec_module() must also define create_module()')
    if module is None:
        module = _new_module(spec.name)
    _init_module_attrs(spec, module)
    return module

def _module_repr_from_spec(spec):
    if False:
        i = 10
        return i + 15
    'Return the repr to use for the module.'
    name = '?' if spec.name is None else spec.name
    if spec.origin is None:
        if spec.loader is None:
            return '<module {!r}>'.format(name)
        else:
            return '<module {!r} ({!r})>'.format(name, spec.loader)
    elif spec.has_location:
        return '<module {!r} from {!r}>'.format(name, spec.origin)
    else:
        return '<module {!r} ({})>'.format(spec.name, spec.origin)

def _exec(spec, module):
    if False:
        return 10
    "Execute the spec's specified module in an existing module's namespace."
    name = spec.name
    with _ModuleLockManager(name):
        if sys.modules.get(name) is not module:
            msg = 'module {!r} not in sys.modules'.format(name)
            raise ImportError(msg, name=name)
        try:
            if spec.loader is None:
                if spec.submodule_search_locations is None:
                    raise ImportError('missing loader', name=spec.name)
                _init_module_attrs(spec, module, override=True)
            else:
                _init_module_attrs(spec, module, override=True)
                if not hasattr(spec.loader, 'exec_module'):
                    msg = f'{_object_name(spec.loader)}.exec_module() not found; falling back to load_module()'
                    _warnings.warn(msg, ImportWarning)
                    spec.loader.load_module(name)
                else:
                    spec.loader.exec_module(module)
        finally:
            module = sys.modules.pop(spec.name)
            sys.modules[spec.name] = module
    return module

def _load_backward_compatible(spec):
    if False:
        while True:
            i = 10
    try:
        spec.loader.load_module(spec.name)
    except:
        if spec.name in sys.modules:
            module = sys.modules.pop(spec.name)
            sys.modules[spec.name] = module
        raise
    module = sys.modules.pop(spec.name)
    sys.modules[spec.name] = module
    if getattr(module, '__loader__', None) is None:
        try:
            module.__loader__ = spec.loader
        except AttributeError:
            pass
    if getattr(module, '__package__', None) is None:
        try:
            module.__package__ = module.__name__
            if not hasattr(module, '__path__'):
                module.__package__ = spec.name.rpartition('.')[0]
        except AttributeError:
            pass
    if getattr(module, '__spec__', None) is None:
        try:
            module.__spec__ = spec
        except AttributeError:
            pass
    return module

def _load_unlocked(spec):
    if False:
        for i in range(10):
            print('nop')
    if spec.loader is not None:
        if not hasattr(spec.loader, 'exec_module'):
            msg = f'{_object_name(spec.loader)}.exec_module() not found; falling back to load_module()'
            _warnings.warn(msg, ImportWarning)
            return _load_backward_compatible(spec)
    module = module_from_spec(spec)
    spec._initializing = True
    try:
        sys.modules[spec.name] = module
        try:
            if spec.loader is None:
                if spec.submodule_search_locations is None:
                    raise ImportError('missing loader', name=spec.name)
            else:
                spec.loader.exec_module(module)
        except:
            try:
                del sys.modules[spec.name]
            except KeyError:
                pass
            raise
        module = sys.modules.pop(spec.name)
        sys.modules[spec.name] = module
        _verbose_message('import {!r} # {!r}', spec.name, spec.loader)
    finally:
        spec._initializing = False
    return module

def _load(spec):
    if False:
        return 10
    "Return a new module object, loaded by the spec's loader.\n\n    The module is not added to its parent.\n\n    If a module is already in sys.modules, that existing module gets\n    clobbered.\n\n    "
    with _ModuleLockManager(spec.name):
        return _load_unlocked(spec)

class BuiltinImporter:
    """Meta path import for built-in modules.

    All methods are either class or static methods to avoid the need to
    instantiate the class.

    """
    _ORIGIN = 'built-in'

    @staticmethod
    def module_repr(module):
        if False:
            print('Hello World!')
        'Return repr for the module.\n\n        The method is deprecated.  The import machinery does the job itself.\n\n        '
        _warnings.warn('BuiltinImporter.module_repr() is deprecated and slated for removal in Python 3.12', DeprecationWarning)
        return f'<module {module.__name__!r} ({BuiltinImporter._ORIGIN})>'

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if False:
            while True:
                i = 10
        if path is not None:
            return None
        if _imp.is_builtin(fullname):
            return spec_from_loader(fullname, cls, origin=cls._ORIGIN)
        else:
            return None

    @classmethod
    def find_module(cls, fullname, path=None):
        if False:
            print('Hello World!')
        "Find the built-in module.\n\n        If 'path' is ever specified then the search is considered a failure.\n\n        This method is deprecated.  Use find_spec() instead.\n\n        "
        _warnings.warn('BuiltinImporter.find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        spec = cls.find_spec(fullname, path)
        return spec.loader if spec is not None else None

    @staticmethod
    def create_module(spec):
        if False:
            for i in range(10):
                print('nop')
        'Create a built-in module'
        if spec.name not in sys.builtin_module_names:
            raise ImportError('{!r} is not a built-in module'.format(spec.name), name=spec.name)
        return _call_with_frames_removed(_imp.create_builtin, spec)

    @staticmethod
    def exec_module(module):
        if False:
            while True:
                i = 10
        'Exec a built-in module'
        _call_with_frames_removed(_imp.exec_builtin, module)

    @classmethod
    @_requires_builtin
    def get_code(cls, fullname):
        if False:
            return 10
        'Return None as built-in modules do not have code objects.'
        return None

    @classmethod
    @_requires_builtin
    def get_source(cls, fullname):
        if False:
            print('Hello World!')
        'Return None as built-in modules do not have source code.'
        return None

    @classmethod
    @_requires_builtin
    def is_package(cls, fullname):
        if False:
            while True:
                i = 10
        'Return False as built-in modules are never packages.'
        return False
    load_module = classmethod(_load_module_shim)

class FrozenImporter:
    """Meta path import for frozen modules.

    All methods are either class or static methods to avoid the need to
    instantiate the class.

    """
    _ORIGIN = 'frozen'

    @staticmethod
    def module_repr(m):
        if False:
            while True:
                i = 10
        'Return repr for the module.\n\n        The method is deprecated.  The import machinery does the job itself.\n\n        '
        _warnings.warn('FrozenImporter.module_repr() is deprecated and slated for removal in Python 3.12', DeprecationWarning)
        return '<module {!r} ({})>'.format(m.__name__, FrozenImporter._ORIGIN)

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if False:
            while True:
                i = 10
        if _imp.is_frozen(fullname):
            return spec_from_loader(fullname, cls, origin=cls._ORIGIN)
        else:
            return None

    @classmethod
    def find_module(cls, fullname, path=None):
        if False:
            while True:
                i = 10
        'Find a frozen module.\n\n        This method is deprecated.  Use find_spec() instead.\n\n        '
        _warnings.warn('FrozenImporter.find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
        return cls if _imp.is_frozen(fullname) else None

    @staticmethod
    def create_module(spec):
        if False:
            i = 10
            return i + 15
        'Use default semantics for module creation.'

    @staticmethod
    def exec_module(module):
        if False:
            while True:
                i = 10
        name = module.__spec__.name
        if not _imp.is_frozen(name):
            raise ImportError('{!r} is not a frozen module'.format(name), name=name)
        code = _call_with_frames_removed(_imp.get_frozen_object, name)
        exec(code, module.__dict__)

    @classmethod
    def load_module(cls, fullname):
        if False:
            for i in range(10):
                print('nop')
        'Load a frozen module.\n\n        This method is deprecated.  Use exec_module() instead.\n\n        '
        return _load_module_shim(cls, fullname)

    @classmethod
    @_requires_frozen
    def get_code(cls, fullname):
        if False:
            i = 10
            return i + 15
        'Return the code object for the frozen module.'
        return _imp.get_frozen_object(fullname)

    @classmethod
    @_requires_frozen
    def get_source(cls, fullname):
        if False:
            return 10
        'Return None as frozen modules do not have source code.'
        return None

    @classmethod
    @_requires_frozen
    def is_package(cls, fullname):
        if False:
            i = 10
            return i + 15
        'Return True if the frozen module is a package.'
        return _imp.is_frozen_package(fullname)

class _ImportLockContext:
    """Context manager for the import lock."""

    def __enter__(self):
        if False:
            while True:
                i = 10
        'Acquire the import lock.'
        _imp.acquire_lock()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if False:
            return 10
        'Release the import lock regardless of any raised exceptions.'
        _imp.release_lock()

def _resolve_name(name, package, level):
    if False:
        print('Hello World!')
    'Resolve a relative module name to an absolute one.'
    bits = package.rsplit('.', level - 1)
    if len(bits) < level:
        raise ImportError('attempted relative import beyond top-level package')
    base = bits[0]
    return '{}.{}'.format(base, name) if name else base

def _find_spec_legacy(finder, name, path):
    if False:
        for i in range(10):
            print('nop')
    msg = f'{_object_name(finder)}.find_spec() not found; falling back to find_module()'
    _warnings.warn(msg, ImportWarning)
    loader = finder.find_module(name, path)
    if loader is None:
        return None
    return spec_from_loader(name, loader)

def _find_spec(name, path, target=None):
    if False:
        print('Hello World!')
    "Find a module's spec."
    meta_path = sys.meta_path
    if meta_path is None:
        raise ImportError('sys.meta_path is None, Python is likely shutting down')
    if not meta_path:
        _warnings.warn('sys.meta_path is empty', ImportWarning)
    is_reload = name in sys.modules
    for finder in meta_path:
        with _ImportLockContext():
            try:
                find_spec = finder.find_spec
            except AttributeError:
                spec = _find_spec_legacy(finder, name, path)
                if spec is None:
                    continue
            else:
                spec = find_spec(name, path, target)
        if spec is not None:
            if not is_reload and name in sys.modules:
                module = sys.modules[name]
                try:
                    __spec__ = module.__spec__
                except AttributeError:
                    return spec
                else:
                    if __spec__ is None:
                        return spec
                    else:
                        return __spec__
            else:
                return spec
    else:
        return None

def _sanity_check(name, package, level):
    if False:
        i = 10
        return i + 15
    'Verify arguments are "sane".'
    if not isinstance(name, str):
        raise TypeError('module name must be str, not {}'.format(type(name)))
    if level < 0:
        raise ValueError('level must be >= 0')
    if level > 0:
        if not isinstance(package, str):
            raise TypeError('__package__ not set to a string')
        elif not package:
            raise ImportError('attempted relative import with no known parent package')
    if not name and level == 0:
        raise ValueError('Empty module name')
_ERR_MSG_PREFIX = 'No module named '
_ERR_MSG = _ERR_MSG_PREFIX + '{!r}'

def _find_and_load_unlocked(name, import_):
    if False:
        while True:
            i = 10
    path = None
    (parent, _, child) = name.rpartition('.')
    if parent:
        if parent not in sys.modules:
            _call_with_frames_removed(import_, parent)
        if name in sys.modules:
            return sys.modules[name]
        parent_module = sys.modules[parent]
        try:
            path = parent_module.__path__
        except AttributeError:
            msg = (_ERR_MSG + '; {!r} is not a package').format(name, parent)
            raise ModuleNotFoundError(msg, name=name) from None
    spec = _find_spec(name, path)
    if spec is None:
        raise ModuleNotFoundError(_ERR_MSG.format(name), name=name)
    else:
        module = _load_unlocked(spec)
    if parent:
        parent_module = sys.modules[parent]
        try:
            _imp._maybe_set_parent_attribute(parent_module, child, module, name)
        except Exception as e:
            msg = f'Cannot set an attribute on {parent!r} for child module {child!r}: {e!r}'
            _warnings.warn(msg, ImportWarning)
    try:
        _imp._set_lazy_attributes(module, name)
    except Exception as e:
        msg = f'Cannot set lazy attributes on {name!r}: {e!r}'
        _warnings.warn(msg, ImportWarning)
    return module
_NEEDS_LOADING = object()

def _find_and_load(name, import_):
    if False:
        for i in range(10):
            print('nop')
    'Find and load the module.'
    module = sys.modules.get(name, _NEEDS_LOADING)
    if module is _NEEDS_LOADING or getattr(getattr(module, '__spec__', None), '_initializing', False):
        with _ModuleLockManager(name):
            module = sys.modules.get(name, _NEEDS_LOADING)
            if module is _NEEDS_LOADING:
                return _find_and_load_unlocked(name, import_)
        _lock_unlock_module(name)
    if module is None:
        message = 'import of {} halted; None in sys.modules'.format(name)
        raise ModuleNotFoundError(message, name=name)
    return module

def _gcd_import(name, package=None, level=0):
    if False:
        i = 10
        return i + 15
    'Import and return the module based on its name, the package the call is\n    being made from, and the level adjustment.\n\n    This function represents the greatest common denominator of functionality\n    between import_module and __import__. This includes setting __package__ if\n    the loader did not.\n\n    '
    _sanity_check(name, package, level)
    if level > 0:
        name = _resolve_name(name, package, level)
    return _find_and_load(name, _gcd_import)

def _handle_fromlist(module, fromlist, import_, *, recursive=False):
    if False:
        for i in range(10):
            print('nop')
    "Figure out what __import__ should return.\n\n    The import_ parameter is a callable which takes the name of module to\n    import. It is required to decouple the function from assuming importlib's\n    import implementation is desired.\n\n    "
    for x in fromlist:
        if not isinstance(x, str):
            if recursive:
                where = module.__name__ + '.__all__'
            else:
                where = "``from list''"
            raise TypeError(f'Item in {where} must be str, not {type(x).__name__}')
        elif x == '*':
            if not recursive and hasattr(module, '__all__'):
                _handle_fromlist(module, module.__all__, import_, recursive=True)
        elif not hasattr(module, x):
            from_name = '{}.{}'.format(module.__name__, x)
            try:
                _call_with_frames_removed(import_, from_name)
            except ModuleNotFoundError as exc:
                if exc.name == from_name and sys.modules.get(from_name, _NEEDS_LOADING) is not None:
                    continue
                raise
    return module

def _calc___package__(globals):
    if False:
        for i in range(10):
            print('nop')
    'Calculate what __package__ should be.\n\n    __package__ is not guaranteed to be defined or could be set to None\n    to represent that its proper value is unknown.\n\n    '
    package = globals.get('__package__')
    spec = globals.get('__spec__')
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn(f'__package__ != __spec__.parent ({package!r} != {spec.parent!r})', ImportWarning, stacklevel=3)
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn("can't resolve package from __spec__ or __package__, falling back on __name__ and __path__", ImportWarning, stacklevel=3)
        package = globals['__name__']
        if '__path__' not in globals:
            package = package.rpartition('.')[0]
    return package

def __import__(name, globals=None, locals=None, fromlist=(), level=0):
    if False:
        print('Hello World!')
    "Import a module.\n\n    The 'globals' argument is used to infer where the import is occurring from\n    to handle relative imports. The 'locals' argument is ignored. The\n    'fromlist' argument specifies what should exist as attributes on the module\n    being imported (e.g. ``from module import <fromlist>``).  The 'level'\n    argument represents the package location to import from in a relative\n    import (e.g. ``from ..pkg import mod`` would have a 'level' of 2).\n\n    "
    if level == 0:
        module = _gcd_import(name)
    else:
        globals_ = globals if globals is not None else {}
        package = _calc___package__(globals_)
        module = _gcd_import(name, package, level)
    if not fromlist:
        if level == 0:
            return _gcd_import(name.partition('.')[0])
        elif not name:
            return module
        else:
            cut_off = len(name) - len(name.partition('.')[0])
            return sys.modules[module.__name__[:len(module.__name__) - cut_off]]
    elif hasattr(module, '__path__'):
        return _handle_fromlist(module, fromlist, _gcd_import)
    else:
        return module

def _builtin_from_name(name):
    if False:
        for i in range(10):
            print('nop')
    spec = BuiltinImporter.find_spec(name)
    if spec is None:
        raise ImportError('no built-in module named ' + name)
    return _load_unlocked(spec)

def _setup(sys_module, _imp_module):
    if False:
        return 10
    'Setup importlib by importing needed built-in modules and injecting them\n    into the global namespace.\n\n    As sys is needed for sys.modules access and _imp is needed to load built-in\n    modules, those two modules must be explicitly passed in.\n\n    '
    global _imp, sys
    _imp = _imp_module
    sys = sys_module
    module_type = type(sys)
    for (name, module) in sys.modules.items():
        if isinstance(module, module_type):
            if name in sys.builtin_module_names:
                loader = BuiltinImporter
            elif _imp.is_frozen(name):
                loader = FrozenImporter
            else:
                continue
            spec = _spec_from_module(module, loader)
            _init_module_attrs(spec, module)
    self_module = sys.modules[__name__]
    for builtin_name in ('_thread', '_warnings', '_weakref'):
        if builtin_name not in sys.modules:
            builtin_module = _builtin_from_name(builtin_name)
        else:
            builtin_module = sys.modules[builtin_name]
        setattr(self_module, builtin_name, builtin_module)

def _install(sys_module, _imp_module):
    if False:
        print('Hello World!')
    'Install importers for builtin and frozen modules'
    _setup(sys_module, _imp_module)
    sys.meta_path.append(BuiltinImporter)
    sys.meta_path.append(FrozenImporter)

def _install_external_importers():
    if False:
        i = 10
        return i + 15
    'Install importers that require external filesystem access'
    global _bootstrap_external
    import _frozen_importlib_external
    _bootstrap_external = _frozen_importlib_external
    _frozen_importlib_external._install(sys.modules[__name__])