from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from dynaconf.base import RESERVED_ATTRS
from dynaconf.base import Settings
from dynaconf.loaders.base import SourceMetadata
__all__ = ['hookable', 'EMPTY_VALUE', 'Hook', 'EagerValue', 'HookValue', 'MethodValue', 'Action', 'HookableSettings']

class Empty:
    ...
EMPTY_VALUE = Empty()

def hookable(function=None, name=None):
    if False:
        return 10
    'Adds before and after hooks to any method.\n\n    :param function: function to be decorated\n    :param name: name of the method to be decorated (default to method name)\n    :return: decorated function\n\n    Usage:\n\n        class MyHookableClass(Settings):\n            @hookable\n            def execute_loaders(....):\n                # do whatever you want here\n                return super().execute_loaders(....)\n\n        settings = Dynaconf(_wrapper_class=MyHookableClass)\n\n        def hook_function(temp_settings, value, ...):\n            # do whatever you want here\n            return value\n\n        settings.add_hook("after_execute_loaders", Hook(function))\n\n        settings.FOO\n        # will trigger execute_loaders\n        # -> will trigger the hookable method\n        # -> will execute registered hooks\n\n    see tests/test_hooking.py for more examples.\n    '
    if function and (not callable(function)):
        raise TypeError('hookable must be applied with named arguments only')

    def dispatch(fun, self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'calls the decorated function and its hooks'
        if not (_registered_hooks := get_hooks(self)):
            return fun(self, *args, **kwargs)
        function_name = name or fun.__name__
        if not set(_registered_hooks).intersection((f'before_{function_name}', f'after_{function_name}')):
            return fun(self, *args, **kwargs)
        temp_settings = Settings(dynaconf_skip_loaders=True, dynaconf_skip_validators=True)
        allowed_keys = self.__dict__.keys() - set(RESERVED_ATTRS)
        temp_data = {k: v for (k, v) in self.__dict__.items() if k in allowed_keys}
        temp_settings._store.update(temp_data)

        def _hook(action: str, value: HookValue) -> HookValue:
            if False:
                return 10
            'executes the hooks for the given action'
            hooks = _registered_hooks.get(f'{action}_{function_name}', [])
            for hook in hooks:
                value = hook.function(temp_settings, value, *args, **kwargs)
                value = HookValue.new(value)
            return value
        value = _hook('before', HookValue(EMPTY_VALUE))
        original_value = EMPTY_VALUE
        if not isinstance(value, EagerValue):
            value = MethodValue(fun(self, *args, **kwargs))
            original_value = value.value
        value = _hook('after', value)
        if value.value != original_value and function_name == 'get':
            hook_names = '_'.join([hook.function.__name__ for list_of_hooks in _registered_hooks.values() for hook in list_of_hooks])
            metadata = SourceMetadata(loader='hooking', identifier=f'{function_name}_hook_({hook_names})', merged=True)
            history = self._loaded_by_loaders.setdefault(metadata, {})
            key = args[0] if args else kwargs.get('key')
            history[key] = value.value
        return value.value
    if function:

        @wraps(function)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return dispatch(function, *args, **kwargs)
        wrapper.original_function = function
        return wrapper

    def decorator(function):
        if False:
            while True:
                i = 10

        @wraps(function)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return dispatch(function, *args, **kwargs)
        wrapper.original_function = function
        return wrapper
    return decorator

def get_hooks(obj):
    if False:
        print('Hello World!')
    'get registered hooks from object\n    must try different casing and accessors because of\n    tests and casing mode set on dynaconf.\n    '
    attr = '_registered_hooks'
    for key in [attr, attr.upper()]:
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict) and key in obj:
            return obj[key]
        elif hasattr(obj, '_store') and key in obj._store:
            return obj._store[key]
    return {}

@dataclass
class Hook:
    """Hook to wrap a callable on _registered_hooks list.

    :param callable: The callable to be wrapped

    The callable must accept the following arguments:

    - temp_settings: Settings or a Dict
    - value: The value to be processed wrapper in a HookValue
      (accumulated from previous hooks, last hook will receive the final value)
    - *args: The args passed to the original method
    - **kwargs: The kwargs passed to the original method

    The callable must return the value:

    - value: The processed value to be passed to the next hook
    """
    function: Callable

@dataclass
class HookValue:
    """Base class for hook values.
    Hooks must return a HookValue instance.
    """
    value: Any

    @classmethod
    def new(cls, value: Any) -> HookValue:
        if False:
            print('Hello World!')
        'Return a new HookValue instance with the given value.'
        if isinstance(value, HookValue):
            return value
        return cls(value)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return str(self.value)

    def __eq__(self, other) -> bool:
        if False:
            print('Hello World!')
        return self.value == other

    def __ne__(self, other) -> bool:
        if False:
            return 10
        return self.value != other

    def __bool__(self) -> bool:
        if False:
            print('Hello World!')
        return bool(self.value)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.value)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.value)

    def __getitem__(self, item):
        if False:
            return 10
        return self.value[item]

    def __setitem__(self, key, value):
        if False:
            return 10
        self.value[key] = value

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self.value[key]

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return item in self.value

    def __getattr__(self, item):
        if False:
            return 10
        return getattr(self.value, item)

    def __setattr__(self, key, value):
        if False:
            return 10
        if key == 'value':
            super().__setattr__(key, value)
        else:
            setattr(self.value, key, value)

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value + other

    def __sub__(self, other):
        if False:
            return 10
        return self.value - other

    def __mul__(self, other):
        if False:
            return 10
        return self.value * other

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        return self.value / other

    def __floordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value // other

    def __mod__(self, other):
        if False:
            while True:
                i = 10
        return self.value % other

    def __divmod__(self, other):
        if False:
            print('Hello World!')
        return divmod(self.value, other)

    def __pow__(self, power, modulo=None):
        if False:
            i = 10
            return i + 15
        return pow(self.value, power, modulo)

    def __delattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        delattr(self.value, item)

    def __repr__(self) -> str:
        if False:
            return 10
        return repr(self.value)

class MethodValue(HookValue):
    """A value returned by a method
    The main decorated method have its value wrapped in this class
    """

class EagerValue(HookValue):
    """Use this wrapper to return earlier from a hook.
    Main function is bypassed and value is passed to after hooks."""

class Action(str, Enum):
    """All the hookable functions"""
    AFTER_GET = 'after_get'
    BEFORE_GET = 'before_get'

class HookableSettings(Settings):
    """Wrapper for dynaconf.base.Settings that adds hooks to get method."""
    _REGISTERED_HOOKS: dict[Action, list[Hook]] = {}

    @hookable
    def get(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return Settings.get(self, *args, **kwargs)