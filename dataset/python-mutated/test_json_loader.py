from __future__ import annotations
import json
import pytest
from dynaconf import LazySettings
from dynaconf.loaders.json_loader import DynaconfEncoder
from dynaconf.loaders.json_loader import load
from dynaconf.strategies.filtering import PrefixFilter
settings = LazySettings(environments=True, ENV_FOR_DYNACONF='PRODUCTION')
JSON = '\n{\n    "a": "a,b",\n    "default": {\n        "password": "@int 99999",\n        "host": "server.com",\n        "port": "@int 8080",\n        "alist": ["item1", "item2", 23],\n        "service": {\n          "url": "service.com",\n          "port": 80,\n          "auth": {\n            "password": "qwerty",\n            "test": 1234\n          }\n        }\n    },\n    "development": {\n        "password": "@int 88888",\n        "host": "devserver.com"\n    },\n    "production": {\n        "password": "@int 11111",\n        "host": "prodserver.com"\n    },\n    "global": {\n        "global_value": "global"\n    }\n}\n'
JSON2 = '\n{\n  "global": {\n    "secret": "@float 42",\n    "password": 123456,\n    "host": "otherjson.com"\n  }\n}\n'
JSONS = [JSON, JSON2]

def test_load_from_json():
    if False:
        while True:
            i = 10
    'Assert loads from JSON string'
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'
    assert settings.PORT == 8080
    assert settings.ALIST == ['item1', 'item2', 23]
    assert settings.SERVICE['url'] == 'service.com'
    assert settings.SERVICE.url == 'service.com'
    assert settings.SERVICE.port == 80
    assert settings.SERVICE.auth.password == 'qwerty'
    assert settings.SERVICE.auth.test == 1234
    load(settings, filename=JSON, env='DEVELOPMENT')
    assert settings.HOST == 'devserver.com'
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'

def test_load_from_multiple_json():
    if False:
        i = 10
        return i + 15
    'Assert loads from JSON string'
    load(settings, filename=JSONS)
    assert settings.HOST == 'otherjson.com'
    assert settings.PASSWORD == 123456
    assert settings.SECRET == 42.0
    assert settings.PORT == 8080
    assert settings.SERVICE['url'] == 'service.com'
    assert settings.SERVICE.url == 'service.com'
    assert settings.SERVICE.port == 80
    assert settings.SERVICE.auth.password == 'qwerty'
    assert settings.SERVICE.auth.test == 1234
    load(settings, filename=JSONS, env='DEVELOPMENT')
    assert settings.PORT == 8080
    assert settings.HOST == 'otherjson.com'
    load(settings, filename=JSONS)
    assert settings.HOST == 'otherjson.com'
    assert settings.PASSWORD == 123456
    load(settings, filename=JSON, env='DEVELOPMENT')
    assert settings.PORT == 8080
    assert settings.HOST == 'devserver.com'
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'
    assert settings.PASSWORD == 11111

def test_no_filename_is_none():
    if False:
        while True:
            i = 10
    'Assert if passed no filename return is None'
    assert load(settings) is None

def test_key_error_on_invalid_env():
    if False:
        i = 10
        return i + 15
    'Assert error raised if env is not found in JSON'
    with pytest.raises(KeyError):
        load(settings, filename=JSON, env='FOOBAR', silent=False)

def test_no_key_error_on_invalid_env():
    if False:
        print('Hello World!')
    'Assert error raised if env is not found in JSON'
    load(settings, filename=JSON, env='FOOBAR', silent=True)

def test_load_single_key():
    if False:
        i = 10
        return i + 15
    'Test loading a single key'
    _JSON = '\n    {\n      "foo": {\n        "bar": "blaz",\n        "zaz": "naz"\n      }\n    }\n    '
    load(settings, filename=_JSON, env='FOO', key='bar')
    assert settings.BAR == 'blaz'
    assert settings.exists('BAR') is True
    assert settings.exists('ZAZ') is False

def test_empty_value():
    if False:
        return 10
    load(settings, filename='')

def test_multiple_filenames():
    if False:
        print('Hello World!')
    load(settings, filename='a.json,b.json,c.json,d.json')

def test_cleaner():
    if False:
        while True:
            i = 10
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'
    assert settings.PORT == 8080
    assert settings.ALIST == ['item1', 'item2', 23]
    assert settings.SERVICE['url'] == 'service.com'
    assert settings.SERVICE.url == 'service.com'
    assert settings.SERVICE.port == 80
    assert settings.SERVICE.auth.password == 'qwerty'
    assert settings.SERVICE.auth.test == 1234
    load(settings, filename=JSON, env='DEVELOPMENT')
    assert settings.HOST == 'devserver.com'
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'
    settings.clean()
    with pytest.raises(AttributeError):
        assert settings.HOST == 'prodserver.com'

def test_using_env(tmpdir):
    if False:
        return 10
    load(settings, filename=JSON)
    assert settings.HOST == 'prodserver.com'
    tmpfile = tmpdir.mkdir('sub').join('test_using_env.json')
    tmpfile.write(JSON)
    with settings.using_env('DEVELOPMENT', filename=str(tmpfile)):
        assert settings.HOST == 'devserver.com'
    assert settings.HOST == 'prodserver.com'

def test_load_dunder():
    if False:
        while True:
            i = 10
    'Test loading with dunder settings'
    _JSON = '\n    {\n      "foo": {\n        "colors__yellow__code": "#FFCC00",\n        "COLORS__yellow__name": "Yellow"\n      }\n    }\n    '
    load(settings, filename=_JSON, env='FOO')
    assert settings.COLORS.yellow.code == '#FFCC00'
    assert settings.COLORS.yellow.name == 'Yellow'

def test_dynaconf_encoder():
    if False:
        for i in range(10):
            print('nop')

    class Dummy:

        def _dynaconf_encode(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'Dummy'

    class DummyNotSerializable:
        _dynaconf_encode = 42
    data = {'dummy': Dummy()}
    data_error = {'dummy': DummyNotSerializable()}
    assert json.dumps(data, cls=DynaconfEncoder) == '{"dummy": "Dummy"}'
    with pytest.raises(TypeError):
        json.dumps(data_error, cls=DynaconfEncoder)

def test_envless():
    if False:
        for i in range(10):
            print('nop')
    settings = LazySettings()
    _json = '\n    {\n        "colors__yellow__code": "#FFCC00",\n        "COLORS__yellow__name": "Yellow"\n    }\n    '
    load(settings, filename=_json)
    assert settings.COLORS.yellow.code == '#FFCC00'
    assert settings.COLORS.yellow.name == 'Yellow'

def test_prefix():
    if False:
        for i in range(10):
            print('nop')
    settings = LazySettings(filter_strategy=PrefixFilter('prefix'))
    _json = '\n    {\n        "prefix_colors__yellow__code": "#FFCC00",\n        "COLORS__yellow__name": "Yellow"\n    }\n    '
    load(settings, filename=_json)
    assert settings.COLORS.yellow.code == '#FFCC00'
    with pytest.raises(AttributeError):
        settings.COLORS.yellow.name