import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import tokenize
import unittest
import warnings
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Union
from unittest import mock
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)

def install_config_module(module):
    if False:
        return 10
    '\n    Converts a module-level config into a `ConfigModule()`.\n\n    See config_typing.pyi for instructions on how to get the converted module to typecheck.\n    '

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set({'_is_dirty', '_hash_digest'})

    def visit(source, dest, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Walk the module structure and move everything to module._config'
        for (key, value) in list(source.__dict__.items()):
            if key.startswith('__') or isinstance(value, (ModuleType, FunctionType)) or (hasattr(value, '__module__') and value.__module__ == 'typing'):
                continue
            name = f'{prefix}{key}'
            if isinstance(value, CONFIG_TYPES):
                config[name] = value
                default[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                proxy = SubConfigProxy(module, f'{name}.')
                visit(value, proxy, f'{name}.')
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f'Unhandled config {key}={value} ({type(value)})')
    config: Dict[str, Any] = dict()
    default: Dict[str, Any] = dict()
    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)
    visit(module, module, '')
    module._config = config
    module._default = default
    module._allowed_keys = set(config.keys())
    module._compile_ignored_keys = compile_ignored_keys
    module.__class__ = ConfigModuleInstance
    module._is_dirty = True
    module._hash_digest = None
COMPILE_IGNORED_MARKER = '@compile_ignored'

def get_assignments_with_compile_ignored_comments(module):
    if False:
        return 10
    source_code = inspect.getsource(module)
    assignments = set()
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
    current_comment = ('', -1)
    prev_name = ''
    prev_assigned = ('', -1)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            maybe_current = token.string.strip()
            if COMPILE_IGNORED_MARKER in maybe_current:
                assert current_comment == ('', -1), f'unconsumed {COMPILE_IGNORED_MARKER}'
                current_comment = (maybe_current, token.start[0])
                if token.start[0] == prev_assigned[1]:
                    assignments.add(prev_assigned[0])
                    current_comment = ('', -1)
        elif token.type == tokenize.NAME:
            prev_name = token.string
        elif token.type == tokenize.OP and token.string == '=':
            prev_assigned = (prev_name, token.start[0])
            if COMPILE_IGNORED_MARKER in current_comment[0] and current_comment[1] == token.start[0] - 1:
                assignments.add(prev_name)
                current_comment = ('', -1)
    assert current_comment == ('', -1), f'unconsumed {COMPILE_IGNORED_MARKER}'
    return assignments

class ConfigModule(ModuleType):
    _default: Dict[str, Any]
    _config: Dict[str, Any]
    _allowed_keys: Set[str]
    _bypass_keys: Set[str]
    _compile_ignored_keys: Set[str]
    _is_dirty: bool
    _hash_digest: Optional[bytes]

    def __init__(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'use {__name__}.install_config_module(sys.modules[__name__])')

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._allowed_keys:
            raise AttributeError(f'{self.__name__}.{name} does not exist')
        else:
            self._config[name] = value

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        try:
            return self._config[name]
        except KeyError as e:
            raise AttributeError(f'{self.__name__}.{name} does not exist') from e

    def __delattr__(self, name):
        if False:
            print('Hello World!')
        del self._config[name]

    def save_config(self) -> bytes:
        if False:
            i = 10
            return i + 15
        'Convert config to a pickled blob'
        config = dict(self._config)
        for key in config.get('_save_config_ignore', ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def codegen_config(self) -> str:
        if False:
            print('Hello World!')
        'Convert config to Python statements that replicate current config.\n        This does NOT include config settings that are at default values.\n        '
        lines = []
        mod = self.__name__
        for (k, v) in self._config.items():
            if k in self._config.get('_save_config_ignore', ()):
                continue
            if v == self._default[k]:
                continue
            lines.append(f'{mod}.{k} = {v!r}')
        return '\n'.join(lines)

    def get_hash(self) -> bytes:
        if False:
            i = 10
            return i + 15
        'Hashes the configs that are not compile_ignored'
        if self._is_dirty or self._hash_digest is None:
            dict_to_hash = {k: v for (k, v) in self._config.items() if k not in self._compile_ignored_keys}
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_digest = hashlib.md5(string_to_hash.encode('utf-8')).digest()
            self._is_dirty = False
        return self._hash_digest

    def to_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        warnings.warn('config.to_dict() has been deprecated. It may no longer change the underlying config. use config.shallow_copy_dict() or config.get_config_copy() instead', DeprecationWarning)
        return self.shallow_copy_dict()

    def shallow_copy_dict(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {**self._config}

    def load_config(self, maybe_pickled_config: Union[bytes, Dict[str, Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Restore from a prior call to save_config() or shallow_copy_dict()'
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        self._config.update(config)

    def get_config_copy(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return copy.deepcopy(self._config)

    def patch(self, arg1: Optional[Union[str, Dict[str, Any]]]=None, arg2: Any=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Decorator and/or context manager to make temporary changes to a config.\n\n        As a decorator:\n\n            @config.patch("name", val)\n            @config.patch(name1=val1, name2=val2)\n            @config.patch({"name1": val1, "name2", val2})\n            def foo(...):\n                ...\n\n        As a context manager:\n\n            with config.patch("name", val):\n                ...\n        '
        changes: Dict[str, Any]
        if arg1 is not None:
            if arg2 is not None:
                assert isinstance(arg1, str)
                changes = {arg1: arg2}
            else:
                assert isinstance(arg1, dict)
                changes = arg1
            assert not kwargs
        else:
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f'expected `dict` got {type(changes)}'
        prior: Dict[str, Any] = {}
        config = self
        dirty = False

        class ConfigPatch(ContextDecorator):

            def __enter__(self):
                if False:
                    return 10
                assert not prior
                nonlocal dirty
                for key in changes.keys():
                    prior[key] = config._config[key]
                    dirty = key not in config._compile_ignored_keys
                config._config.update(changes)
                config._is_dirty = dirty

            def __exit__(self, exc_type, exc_val, exc_tb):
                if False:
                    return 10
                nonlocal dirty
                config._config.update(prior)
                config._is_dirty = dirty
                prior.clear()
        return ConfigPatch()

class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __enter__(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('NYI')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('NYI')

    def __call__(self, func):
        if False:
            while True:
                i = 10
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):

                @classmethod
                def setUpClass(cls):
                    if False:
                        i = 10
                        return i + 15
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    if False:
                        while True:
                            i = 10
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)
            _TestCase.__name__ = func.__name__
            _TestCase.__qualname__ = func.__qualname__
            _TestCase.__module__ = func.__module__
            return _TestCase
        return super().__call__(func)

class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    def __init__(self, config, prefix):
        if False:
            i = 10
            return i + 15
        super().__setattr__('_config', config)
        super().__setattr__('_prefix', prefix)

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name):
        if False:
            return 10
        return self._config.__delattr__(self._prefix + name)

def patch_object(obj, name, value):
    if False:
        print('Hello World!')
    '\n    Workaround `mock.patch.object` issue with ConfigModule\n    '
    if isinstance(obj, ConfigModule):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)