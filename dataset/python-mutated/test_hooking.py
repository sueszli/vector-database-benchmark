from __future__ import annotations
import pytest
from dynaconf import Dynaconf
from dynaconf.hooking import Action
from dynaconf.hooking import EagerValue
from dynaconf.hooking import get_hooks
from dynaconf.hooking import Hook
from dynaconf.hooking import hookable
from dynaconf.hooking import HookableSettings
from dynaconf.hooking import HookValue
from dynaconf.utils.boxing import DynaBox

class BaseHookedSettings:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self._store = DynaBox(kwargs)
        self._loaded_by_loaders = {}

    @property
    def __dict__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._store

    @hookable
    def get(self, key):
        if False:
            print('Hello World!')
        return self._store.get(key)

def test_hook_dynaconf_class_get_templated():
    if False:
        return 10
    settings = Dynaconf(INTERNAL_VALUE=42, TEMPLATED='@format {this[INTERNAL_VALUE]}abc', TEMPLATED1='@int @format {this[INTERNAL_VALUE]}', TEMPLATED2='@jinja {{this.INTERNAL_VALUE}}abcd', TEMPLATED3='@int @jinja {{this.INTERNAL_VALUE}}', PERSON={'NAME': 'Joe'}, _wrapper_class=HookableSettings)

    def just_assert_values_after_get(s, v, key, *_, **__):
        if False:
            print('Hello World!')
        if key == 'INTERNAL_VALUE':
            assert v == 99
        elif key == 'PERSON':
            assert s['PERSON'] == {'NAME': 'Joe', 'city': 'Valhala', 'AGE': 18}
        else:
            assert s['PERSON'] == {'NAME': 'Joe'}
            assert s['INTERNAL_VALUE'] == 42
        return v

    def set_internal_value_to_99_before_get(s, v, key, *_, **__):
        if False:
            return 10
        if key == 'INTERNAL_VALUE':
            return EagerValue(99)
        return v

    def contribute_to_person_after_get(s, v, key, *_, **__):
        if False:
            return 10
        if key == 'PERSON':
            data = {'PERSON__AGE': 18, 'PERSON': '@merge city=Valhala'}
            s.update(data)
            return s.get(key)
        return v

    def assert_unchanged_person_value_before_get(s, v, key, *_, **__):
        if False:
            i = 10
            return i + 15
        assert s['PERSON'] == {'NAME': 'Joe'}
        return v
    settings['_registered_hooks'] = {Action.BEFORE_GET: [Hook(set_internal_value_to_99_before_get), Hook(assert_unchanged_person_value_before_get)], Action.AFTER_GET: [Hook(contribute_to_person_after_get), Hook(just_assert_values_after_get)]}
    assert settings.TEMPLATED == '99abc'
    settings.set('FOOVALUE', 100)
    assert settings.FOOVALUE == 100
    assert settings['FOOVALUE'] == 100
    assert settings.TEMPLATED1 == 99
    assert settings.TEMPLATED2 == '99abcd'
    assert settings.TEMPLATED3 == 99
    assert settings.INTERNAL_VALUE == 99
    assert settings.get('INTERNAL_VALUE') == 99
    assert settings.PERSON.NAME == 'Joe'
    assert settings.PERSON.AGE == 18
    assert settings.PERSON.CITY == 'Valhala'

def test_hook_dynaconf_class_after_get():
    if False:
        i = 10
        return i + 15
    settings = Dynaconf(INTERNAL_VALUE=42, _wrapper_class=HookableSettings)
    assert settings.internal_value == 42

    def adds_one(s, v, *_, **__):
        if False:
            return 10
        return v + 1
    settings['_registered_hooks'] = {Action.AFTER_GET: [Hook(adds_one)]}
    hooks = get_hooks(settings)
    assert len(hooks) == 1
    assert settings.INTERNAL_VALUE == 43
    assert settings.get('INTERNAL_VALUE') == 43

def test_hooked_dict():
    if False:
        print('Hello World!')

    class HookedDict(BaseHookedSettings, dict):
        _store = {}

        @hookable
        def get(self, key, default=None):
            if False:
                return 10
            return 'to'
    d = HookedDict()
    d['_registered_hooks'] = {Action.AFTER_GET: [Hook(lambda s, v, *_, **__: f'{v}fu')]}
    assert d.get('key') == 'tofu'

def test_hooked_dict_store():
    if False:
        i = 10
        return i + 15

    class HookedDict(BaseHookedSettings, dict):
        ...
    d = HookedDict(key='to', _registered_hooks={Action.AFTER_GET: [Hook(lambda s, v, *_, **__: f'{v}fu')]})
    assert d.get('key') == 'tofu'

def test_hook_before_and_after_bypass_method():
    if False:
        print('Hello World!')
    'Method is never executed, before and after hooks are called'

    class HookedSettings(BaseHookedSettings):
        _store = {}
        _registered_hooks = {Action.BEFORE_GET: [Hook(lambda s, v, *_, **__: 'ba'), Hook(lambda s, v, *_, **__: EagerValue(f'{v.value}na'))], Action.AFTER_GET: [Hook(lambda s, v, *_, **__: f'{v.value}na')]}

        @hookable(name='get')
        def get(self, key):
            if False:
                print('Hello World!')
            return 'value'
    settings = HookedSettings()
    assert settings.get('key') == 'banana'

def test_hook_runs_after_method():
    if False:
        while True:
            i = 10
    'After method the after hooks transforms value.'
    DATABASE = {'feature_enabled': True}

    def try_to_get_from_database(d, value, key, *_, **__):
        if False:
            i = 10
            return i + 15
        assert d.get('feature_enabled') is False
        return DATABASE.get(key, value.value)

    class HookedSettings(BaseHookedSettings):
        ...
    settings = HookedSettings(feature_enabled=False, something_not_in_database='default value', _registered_hooks={Action.AFTER_GET: [Hook(try_to_get_from_database)]})
    assert settings.get('feature_enabled') is True
    assert settings.get('something_not_in_database') == 'default value'

def test_hook_fail_with_wrong_parameters():
    if False:
        for i in range(10):
            print('nop')
    'Hookable decorator fails when called with wrong parameters.'
    with pytest.raises(TypeError):

        @hookable('not a function')
        def foo():
            if False:
                print('Hello World!')
            pass

def test_hook_values():
    if False:
        while True:
            i = 10
    value = HookValue(1)
    assert value == 1
    assert value != 2
    assert value == HookValue(1)
    assert value != HookValue(2)
    assert bool(value) is True
    assert str(value) == '1'
    assert repr(value) == repr(value.value)
    assert value + 1 == 2
    assert value - 1 == 0
    assert value * 2 == 2
    assert value / 2 == 0.5
    assert value // 2 == 0
    assert value % 2 == 1
    assert value ** 2 == 1
    assert divmod(value, 2) == (0, 1)
    value = HookValue([1, 2, 3])
    assert value == [1, 2, 3]
    assert value != [1, 2, 4]
    assert value == HookValue([1, 2, 3])
    assert value != HookValue([1, 2, 4])
    assert bool(value) is True
    assert str(value) == '[1, 2, 3]'
    assert repr(value) == repr(value.value)
    assert value[0] == 1
    assert value[1] == 2
    assert value[2] == 3
    assert len(value) == 3
    assert value[0:2] == [1, 2]
    assert 2 in value
    assert [x for x in value] == [1, 2, 3]

    class Dummy:
        pass
    _value = Dummy()
    value = HookValue(_value)
    assert value == value.value
    assert value != object()
    assert value == HookValue(_value)
    assert value != HookValue(object())
    assert bool(value) is True
    value.name = 'dummy value'
    assert value.name == 'dummy value'
    delattr(value, 'name')
    assert not hasattr(value, 'name')
    value = HookValue({})
    assert value == {}
    assert value != {'a': 1}
    assert value == HookValue({})
    value['a'] = 1
    assert value == {'a': 1}
    assert value['a'] == 1
    del value['a']
    assert value == {}