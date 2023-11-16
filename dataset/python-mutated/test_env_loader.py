from __future__ import annotations
import os
import sys
from collections import OrderedDict
from os import environ
import pytest
from dynaconf import settings
from dynaconf.loaders.env_loader import load
from dynaconf.loaders.env_loader import load_from_env
from dynaconf.loaders.env_loader import write
environ['DYNACONF_HOSTNAME'] = 'host.com'
environ['DYNACONF_PORT'] = '@int 5000'
environ['DYNACONF_ALIST'] = '@json ["item1", "item2", "item3", 123]'
environ['DYNACONF_ADICT'] = '@json {"key": "value", "int": 42}'
environ['DYNACONF_DEBUG'] = '@bool true'
environ['DYNACONF_MUSTBEFRESH'] = 'first'
environ['DYNACONF_MUSTBEALWAYSFRESH'] = 'first'
environ['DYNACONF_SHOULDBEFRESHINCONTEXT'] = 'first'
environ['DYNACONF_VALUE'] = '@float 42.1'
settings.configure(FRESH_VARS_FOR_DYNACONF=['MUSTBEALWAYSFRESH'], ROOT_PATH_FOR_DYNACONF=os.path.dirname(os.path.abspath(__file__)))
SETTINGS_DATA = OrderedDict()
SETTINGS_DATA['DYNACONF_INTEGER'] = 42
SETTINGS_DATA['DYNACONF_FLOAT'] = 3.14
SETTINGS_DATA['DYNACONF_STRING2'] = 'Hello'
SETTINGS_DATA['DYNACONF_STRING2_LONG'] = 'Hello World!'
SETTINGS_DATA['DYNACONF_BOOL'] = True
SETTINGS_DATA['DYNACONF_BOOL2'] = False
SETTINGS_DATA['DYNACONF_STRING21'] = '"42"'
SETTINGS_DATA['DYNACONF_STRING22'] = "'true'"
SETTINGS_DATA['DYNACONF_ARRAY'] = [1, 2, 3]
SETTINGS_DATA['DYNACONF_ARRAY2'] = [1.1, 2.2, 3.3]
SETTINGS_DATA['DYNACONF_ARRAY3'] = ['a', 'b', 'c']
SETTINGS_DATA['DYNACONF_DICT'] = {'val': 123}
SETTINGS_DATA_GROUND_TRUTH = 'DYNACONF_TESTING=true\nDYNACONF_INTEGER=42\nDYNACONF_FLOAT=3.14\nDYNACONF_STRING2=Hello\nDYNACONF_STRING2_LONG="Hello World!"\nDYNACONF_BOOL=True\nDYNACONF_BOOL2=False\nDYNACONF_STRING21="42"\nDYNACONF_STRING22="true"\nDYNACONF_ARRAY="[1, 2, 3]"\nDYNACONF_ARRAY2="[1.1, 2.2, 3.3]"\nDYNACONF_ARRAY3="[\'a\', \'b\', \'c\']"\nDYNACONF_DICT="{\'val\': 123}"\n'

def test_write(tmpdir):
    if False:
        print('Hello World!')
    settings_path = tmpdir.join('.env')
    write(settings_path, SETTINGS_DATA)
    ground_truth = SETTINGS_DATA_GROUND_TRUTH.split('\n')
    with open(str(settings_path)) as fp:
        lines = fp.readlines()
        for (idx, line) in enumerate(lines):
            line = line.strip()
            if line.split('=')[0] == 'DYNACONF_TESTING':
                continue
            assert line == ground_truth[idx].strip()

def test_env_loader():
    if False:
        print('Hello World!')
    assert settings.HOSTNAME == 'host.com'
    assert settings.PORT == 5000
    assert settings.ALIST == ['item1', 'item2', 'item3', 123]
    assert settings.ADICT == {'key': 'value', 'int': 42}

def test_single_key():
    if False:
        i = 10
        return i + 15
    environ['DYNACONF_HOSTNAME'] = 'changedhost.com'
    load(settings, key='HOSTNAME')
    assert settings.HOSTNAME == 'changedhost.com'

def test_dotenv_loader():
    if False:
        return 10
    assert settings.DOTENV_INT == 1
    assert settings.DOTENV_STR == 'hello'
    assert settings.DOTENV_FLOAT == 4.2
    assert settings.DOTENV_BOOL is False
    assert settings.DOTENV_JSON == ['1', '2']
    assert settings.DOTENV_NOTE is None

def test_get_fresh():
    if False:
        i = 10
        return i + 15
    assert settings.MUSTBEFRESH == 'first'
    environ['DYNACONF_MUSTBEFRESH'] = 'second'
    with pytest.raises(AssertionError):
        assert settings.exists('MUSTBEFRESH')
        assert settings.get_fresh('MUSTBEFRESH') == 'first'
    assert settings.get_fresh('MUSTBEFRESH') == 'second'
    environ['DYNACONF_THISMUSTEXIST'] = '@int 1'
    assert settings.exists('THISMUSTEXIST') is False
    assert settings.exists('THISMUSTEXIST', fresh=True) is True
    assert settings.get('THISMUSTEXIST') == 1
    environ['DYNACONF_THISMUSTEXIST'] = '@int 23'
    del environ['DYNACONF_THISMUSTEXIST']
    assert settings.get_fresh('THISMUSTEXIST') is None
    with pytest.raises(AttributeError):
        settings.THISMUSTEXIST
    with pytest.raises(KeyError):
        settings['THISMUSTEXIST']
    environ['DYNACONF_THISMUSTEXIST'] = '@int 23'
    load(settings)
    assert settings.get('THISMUSTEXIST') == 23

def test_always_fresh():
    if False:
        for i in range(10):
            print('nop')
    assert settings.FRESH_VARS_FOR_DYNACONF == ['MUSTBEALWAYSFRESH']
    assert settings.MUSTBEALWAYSFRESH == 'first'
    environ['DYNACONF_MUSTBEALWAYSFRESH'] = 'second'
    assert settings.MUSTBEALWAYSFRESH == 'second'
    environ['DYNACONF_MUSTBEALWAYSFRESH'] = 'third'
    assert settings.MUSTBEALWAYSFRESH == 'third'

def test_fresh_context():
    if False:
        return 10
    assert settings.SHOULDBEFRESHINCONTEXT == 'first'
    environ['DYNACONF_SHOULDBEFRESHINCONTEXT'] = 'second'
    assert settings.SHOULDBEFRESHINCONTEXT == 'first'
    with settings.fresh():
        assert settings.get('DOTENV_INT') == 1
        assert settings.SHOULDBEFRESHINCONTEXT == 'second'

def test_cleaner():
    if False:
        print('Hello World!')
    settings.clean()
    with pytest.raises(AttributeError):
        assert settings.HOSTNAME == 'host.com'

def test_empty_string_prefix():
    if False:
        return 10
    environ['_VALUE'] = 'underscored'
    load_from_env(identifier='env_global', key=None, prefix='', obj=settings, silent=True)
    assert settings.VALUE == 'underscored'

def test_no_prefix():
    if False:
        for i in range(10):
            print('nop')
    environ['VALUE'] = 'no_prefix'
    load_from_env(identifier='env_global', key=None, prefix=False, obj=settings, silent=True)
    assert settings.VALUE == 'no_prefix'

def test_none_as_string_prefix():
    if False:
        return 10
    environ['NONE_VALUE'] = 'none as prefix'
    load_from_env(identifier='env_global', key=None, prefix='none', obj=settings, silent=True)
    assert settings.VALUE == 'none as prefix'

def test_backwards_compat_using_env_argument():
    if False:
        print('Hello World!')
    environ['BLARG_VALUE'] = 'BLARG as prefix'
    load_from_env(identifier='env_global', key=None, env='BLARG', obj=settings, silent=True)
    assert settings.VALUE == 'BLARG as prefix'

def test_load_signed_integer():
    if False:
        return 10
    environ['799_SIGNED_NEG_INT'] = '-1'
    environ['799_SIGNED_POS_INT'] = '+1'
    load_from_env(identifier='env_global', key=None, prefix='799', obj=settings, silent=True)
    assert settings.SIGNED_NEG_INT == -1
    assert settings.SIGNED_POS_INT == 1

def test_env_is_not_str_raises():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        load_from_env(settings, prefix=int)
    with pytest.raises(TypeError):
        load_from_env(settings, prefix=True)

def test_can_load_in_to_dict():
    if False:
        for i in range(10):
            print('nop')
    os.environ['LOADTODICT'] = 'true'
    sets = {}
    load_from_env(sets, prefix=False, key='LOADTODICT')
    assert sets['LOADTODICT'] is True

def clean_environ(prefix):
    if False:
        while True:
            i = 10
    keys = [k for k in environ if k.startswith(prefix)]
    for key in keys:
        environ.pop(key)

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Windows env vars are case insensitive')
def test_load_dunder(clean_env):
    if False:
        i = 10
        return i + 15
    'Test load and merge with dunder settings'
    clean_environ('DYNACONF_DATABASES')
    settings.set('DATABASES', {'default': {'NAME': 'db', 'ENGINE': 'module.foo.engine', 'ARGS': {'timeout': 30}, 'PORTS': [123, 456]}})
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ENGINE'] = 'other.module'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default.ENGINE == 'other.module'
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ARGS__timeout'] = '99'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default.ARGS.timeout == 99
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ARGS'] = '@merge {retries=10}'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default.ARGS.retries == 10
    assert settings.DATABASES.default.ARGS.timeout == 99
    assert settings.DATABASES == {'default': {'NAME': 'db', 'ENGINE': 'other.module', 'ARGS': {'timeout': 99, 'retries': 10}, 'PORTS': [123, 456]}}
    assert 'default' in settings['DATABASES'].keys()
    assert 'DEFAULT' not in settings['DATABASES'].keys()
    assert 'NAME' in settings['DATABASES']['default'].keys()
    assert 'name' not in settings['DATABASES']['default'].keys()
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ARGS'] = '{timeout=8}'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default.ARGS == {'timeout': 8}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ARGS'] = '{}'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default.ARGS == {}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__ARGS'] = '@del'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert 'ARGS' not in settings.DATABASES.default.keys()
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__PORTS'] = '@merge [789, 101112]'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert 'ARGS' not in settings.DATABASES.default.keys()
    assert settings.DATABASES.default.PORTS == [123, 456, 789, 101112]
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__PORTS'] = '[789, 101112]'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert 'ARGS' not in settings.DATABASES.default.keys()
    assert settings.DATABASES.default.PORTS == [789, 101112]
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default__PORTS'] = '@del'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert 'ARGS' not in settings.DATABASES.default.keys()
    assert 'PORTS' not in settings.DATABASES.default.keys()
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default'] = '{}'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES.default == {}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__default'] = '@del'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES == {}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES__foo'] = 'bar'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES == {'foo': 'bar'}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES'] = "{hello='world'}"
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES == {'hello': 'world'}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES'] = "{yes='no'}"
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert settings.DATABASES == {'yes': 'no'}
    clean_environ('DYNACONF_DATABASES')
    environ['DYNACONF_DATABASES'] = '@del'
    load_from_env(identifier='env_global', key=None, prefix='dynaconf', obj=settings, silent=True)
    assert 'DATABASES' not in settings

def test_filtering_unknown_variables():
    if False:
        while True:
            i = 10
    settings.MYCONFIG = 'bar'
    settings.IGNORE_UNKNOWN_ENVVARS_FOR_DYNACONF = True
    environ['IGNOREME'] = 'foo'
    load_from_env(obj=settings, prefix=False, key=None, silent=True, identifier='env_global', env=False)
    assert not settings.get('IGNOREME')
    assert settings.get('MYCONFIG') == 'bar'

def test_filtering_unknown_variables_with_prefix():
    if False:
        while True:
            i = 10
    settings.MYCONFIG = 'bar'
    settings.IGNORE_UNKNOWN_ENVVARS_FOR_DYNACONF = True
    environ['APP_IGNOREME'] = 'foo'
    environ['APP_MYCONFIG'] = 'ham'
    load_from_env(obj=settings, prefix='APP', key=None, silent=True, identifier='env_global', env=False)
    assert not settings.get('IGNOREME')
    assert settings.get('MYCONFIG') == 'ham'

def test_boolean_fix():
    if False:
        for i in range(10):
            print('nop')
    environ['BOOLFIX_CAPITALTRUE'] = 'True'
    environ['BOOLFIX_CAPITALFALSE'] = 'False'
    settings.IGNORE_UNKNOWN_ENVVARS_FOR_DYNACONF = False
    load_from_env(obj=settings, prefix='BOOLFIX', key=None, silent=True, identifier='env_global', env=False)
    assert settings.CAPITALTRUE is True
    assert settings.CAPITALFALSE is False