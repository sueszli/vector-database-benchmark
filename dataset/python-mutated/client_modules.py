import atexit
import importlib
import itertools
import pickle
import re
import sys
from .consts import OP_CALLFUNC, OP_GETVAL, OP_SETVAL
from .client import Client
from .override_decorators import LocalException

def _clean_client(client):
    if False:
        return 10
    client.cleanup()

class _WrappedModule(object):

    def __init__(self, loader, prefix, exports, exception_classes, client):
        if False:
            i = 10
            return i + 15
        self._loader = loader
        self._prefix = prefix
        self._client = client
        is_match = re.compile('^%s\\.([a-zA-Z_][a-zA-Z0-9_]*)$' % prefix.replace('.', '\\.'))
        self._exports = {}
        for k in ('classes', 'functions', 'values'):
            result = []
            for item in exports[k]:
                m = is_match.match(item)
                if m:
                    result.append(m.group(1))
            self._exports[k] = result
        self._exception_classes = {}
        for (k, v) in exception_classes.items():
            m = is_match.match(k)
            if m:
                self._exception_classes[m.group(1)] = v

    def __getattr__(self, name):
        if False:
            return 10
        if name == '__loader__':
            return self._loader
        if name in ('__name__', '__package__'):
            return self._prefix
        if name in ('__file__', '__path__'):
            return self._client.name
        if name in self._exports['classes']:
            return self._client.get_local_class('%s.%s' % (self._prefix, name))
        elif name in self._exports['functions']:

            def func(*args, **kwargs):
                if False:
                    return 10
                return self._client.stub_request(None, OP_CALLFUNC, '%s.%s' % (self._prefix, name), *args, **kwargs)
            func.__name__ = name
            func.__doc__ = 'Unknown (TODO)'
            return func
        elif name in self._exports['values']:
            return self._client.stub_request(None, OP_GETVAL, '%s.%s' % (self._prefix, name))
        elif name in self._exception_classes:
            return self._exception_classes[name]
        else:
            m = None
            try:
                m = self._loader.load_module('.'.join([self._prefix, name]))
            except ImportError:
                pass
            if m is None:
                raise AttributeError("module '%s' has no attribute '%s' -- contact the author of the configuration if this is something you expect to work (support may be added if it exists in the original library)" % (self._prefix, name))
            return m

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name in ('package', '__spec__', '_loader', '_prefix', '_client', '_exports', '_exception_classes'):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, _WrappedModule):
            object.__setattr__(self, name, value)
            return
        if name in self._exports['values']:
            self._client.stub_request(None, OP_SETVAL, '%s.%s' % (self._prefix, name), value)
        elif name in self._exports['classes'] or name in self._exports['functions']:
            raise ValueError
        else:
            raise AttributeError(name)

class ModuleImporter(object):

    def __init__(self, python_executable, pythonpath, max_pickle_version, config_dir, module_prefixes):
        if False:
            print('Hello World!')
        self._module_prefixes = module_prefixes
        self._python_executable = python_executable
        self._pythonpath = pythonpath
        self._config_dir = config_dir
        self._client = None
        self._max_pickle_version = max_pickle_version
        self._handled_modules = None
        self._aliases = {}

    def find_module(self, fullname, path=None):
        if False:
            return 10
        if self._handled_modules is not None:
            if fullname in self._handled_modules:
                return self
            return None
        if any([fullname.startswith(prefix) for prefix in self._module_prefixes]):
            return self
        return None

    def load_module(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        if fullname in sys.modules:
            return sys.modules[fullname]
        if self._client is None:
            if sys.version_info[0] < 3:
                raise NotImplementedError('Environment escape imports are not supported in Python 2')
            max_pickle_version = min(self._max_pickle_version, pickle.HIGHEST_PROTOCOL)
            self._client = Client(self._module_prefixes, self._python_executable, self._pythonpath, max_pickle_version, self._config_dir)
            atexit.register(_clean_client, self._client)
            exports = self._client.get_exports()
            ex_overrides = self._client.get_local_exception_overrides()
            prefixes = set()
            export_classes = exports.get('classes', [])
            export_functions = exports.get('functions', [])
            export_values = exports.get('values', [])
            export_exceptions = exports.get('exceptions', [])
            self._aliases = exports.get('aliases', {})
            for name in itertools.chain(export_classes, export_functions, export_values):
                splits = name.rsplit('.', 1)
                prefixes.add(splits[0])
            formed_exception_classes = {}
            for (ex_name, ex_parents) in export_exceptions:
                ex_class_dict = ex_overrides.get(ex_name, None)
                if ex_class_dict is None:
                    ex_class_dict = {}
                else:
                    ex_class_dict = dict(ex_class_dict.__dict__)
                parents = []
                for fake_base in ex_parents:
                    if fake_base.startswith('builtins.'):
                        parents.append(eval(fake_base[9:]))
                    else:
                        parents.append(formed_exception_classes[fake_base])
                splits = ex_name.rsplit('.', 1)
                ex_class_dict['__user_defined__'] = set(ex_class_dict.keys())
                new_class = type(splits[1], tuple(parents), ex_class_dict)
                new_class.__module__ = splits[0]
                new_class.__name__ = splits[1]
                formed_exception_classes[ex_name] = new_class
            for name in formed_exception_classes:
                splits = name.rsplit('.', 1)
                prefixes.add(splits[0])
            all_prefixes = list(prefixes)
            for prefix in all_prefixes:
                parts = prefix.split('.')
                cur = parts[0]
                for i in range(1, len(parts)):
                    prefixes.add(cur)
                    cur = '.'.join([cur, parts[i]])
            self._handled_modules = {}
            for prefix in prefixes:
                self._handled_modules[prefix] = _WrappedModule(self, prefix, exports, formed_exception_classes, self._client)
        fullname = self._get_canonical_name(fullname)
        module = self._handled_modules.get(fullname)
        if module is None:
            raise ImportError
        sys.modules[fullname] = module
        return module

    def _get_canonical_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        base_name = self._aliases.get(name)
        if base_name is not None:
            return base_name
        for idx in reversed([pos for (pos, char) in enumerate(name) if char == '.']):
            base_name = self._aliases.get(name[:idx])
            if base_name is not None:
                return '.'.join([base_name, name[idx + 1:]])
        return name

def create_modules(python_executable, pythonpath, max_pickle_version, path, prefixes):
    if False:
        for i in range(10):
            print('nop')
    for prefix in prefixes:
        try:
            importlib.import_module(prefix)
        except ImportError:
            pass
        else:
            raise RuntimeError('Trying to override %s when module exists in system' % prefix)
    sys.meta_path.append(ModuleImporter(python_executable, pythonpath, max_pickle_version, path, prefixes))