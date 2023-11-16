from __future__ import annotations
import pytest
from dynaconf.base import LazySettings
TOML = '\n[default]\ndynaconf_include = ["plugin1.toml", "plugin2.toml", "plugin2.toml"]\nDEBUG = false\nSERVER = "base.example.com"\nPORT = 6666\n\n[development]\nDEBUG = false\nSERVER = "dev.example.com"\n\n[production]\nDEBUG = false\nSERVER = "prod.example.com"\n'
MIXED = '\n[default]\ndynaconf_include = ["plugin1.toml", "plugin2.{0}"]\nDEBUG = false\nSERVER = "base.example.com"\nPORT = 6666\n\n[development]\nDEBUG = false\nSERVER = "dev.example.com"\n\n[production]\nDEBUG = false\nSERVER = "prod.example.com"\n'
MIXED_MERGE = '\n[default]\ndynaconf_include = [\n    "plugin1.toml",\n    "plugin2.json",\n    "plugin2.yaml",\n    "plugin2.ini",\n    "plugin2.py"\n]\nDEBUG = false\nSERVER = "base.example.com"\nPORT = 6666\n\n[development]\nDEBUG = false\nSERVER = "dev.example.com"\n\n[production]\nDEBUG = false\nSERVER = "prod.example.com"\n\n\n[custom.nested_1]\nbase = 1\n\n[custom.nested_1.nested_2]\nbase = 2\n\n[custom.nested_1.nested_2.nested_3]\nbase = 3\n\n[custom.nested_1.nested_2.nested_3.nested_4]\nbase = 4\n'
TOML_PLUGIN = '\n[default]\nSERVER = "toml.example.com"\nPLUGIN_NAME = "testing"\n\n[development]\nSERVER = "toml.example.com"\nPLUGIN = "extra development var"\n\n[production]\nSERVER = "toml.example.com"\nPLUGIN = "extra production var"\n\n[custom.nested_1.nested_2.nested_3.nested_4]\ntoml = 5\n'
TOML_PLUGIN_2 = '\n[default]\nSERVER = "plugin2.example.com"\nPLUGIN_2_SPECIAL = true\nPORT = 4040\n\n[custom.nested_1.nested_2.nested_3.nested_4]\ntoml = 5\n'
TOML_PLUGIN_TEXT = '\n[default]\ndatabase_uri = "toml.example.com"\nport = 8080\n\n[custom.nested_1.nested_2.nested_3.nested_4]\ntoml = 5\n'
JSON_PLUGIN_TEXT = '\n{\n    "default": {\n        "database_uri": "json.example.com",\n        "port": 8080\n    },\n    "custom": {\n        "nested_1": {\n            "nested_2": {\n                "nested_3": {\n                    "nested_4": {\n                        "json": 5\n                    }\n                }\n            }\n        }\n    }\n}\n'
YAML_PLUGIN_TEXT = '\ndefault:\n  database_uri: "yaml.example.com"\n  port: 8080\n\ncustom:\n  nested_1:\n    nested_2:\n      nested_3:\n        nested_4:\n          yaml: 5\n'
INI_PLUGIN_TEXT = '\n[default]\ndatabase_uri="ini.example.com"\nport="@int 8080"\n\n[custom]\n  [[nested_1]]\n    [[[nested_2]]]\n      [[[[nested_3]]]]\n          [[[[[nested_4]]]]]\n            ini="@int 5"\n'
PY_PLUGIN_TEXT = '\nDATABASE_URI = "py.example.com"\nPORT = 8080\nNESTED_1 = {\n    "nested_2": {\n        "nested_3": {\n            "nested_4": {\n                "py": 5\n            }\n        }\n    }\n}\n'
PLUGIN_TEXT = {'toml': TOML_PLUGIN_TEXT, 'yaml': YAML_PLUGIN_TEXT, 'json': JSON_PLUGIN_TEXT, 'ini': INI_PLUGIN_TEXT, 'py': PY_PLUGIN_TEXT}

def test_invalid_include_path(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Ensure non existing paths are not loaded.'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write(TOML)
    settings = LazySettings(environments=True, ENV_FOR_DYNACONF='DEFAULT', silent=False, LOADERS_FOR_DYNACONF=False, SETTINGS_FILE_FOR_DYNACONF=str(settings_file))
    assert settings.SERVER == 'base.example.com'
    assert settings.DEBUG is False

def test_load_nested_toml(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Load a TOML file that includes other TOML files.'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write(TOML)
    toml_plugin_file = tmpdir.join('plugin1.toml')
    toml_plugin_file.write(TOML_PLUGIN)
    toml_plugin_file = tmpdir.join('plugin2.toml')
    toml_plugin_file.write(TOML_PLUGIN_2)
    settings = LazySettings(environments=True, ENV_FOR_DYNACONF='DEFAULT', silent=False, LOADERS_FOR_DYNACONF=False, ROOT_PATH_FOR_DYNACONF=str(tmpdir), SETTINGS_FILE_FOR_DYNACONF=str(settings_file))
    assert settings.SERVER == 'plugin2.example.com'
    assert settings.DEBUG is False
    assert settings.PLUGIN_NAME == 'testing'
    assert settings.PORT == 4040
    assert settings.PLUGIN_2_SPECIAL is True

@pytest.mark.parametrize('ext', ['toml', 'json', 'yaml', 'ini', 'py'])
def test_load_nested_different_types(ext, tmpdir):
    if False:
        while True:
            i = 10
    'Load a TOML file that includes other various settings file types.'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write(MIXED.format(ext))
    toml_plugin_file = tmpdir.join('plugin1.toml')
    toml_plugin_file.write(TOML_PLUGIN)
    json_plugin_file = tmpdir.join(f'plugin2.{ext}')
    json_plugin_file.write(PLUGIN_TEXT[ext])
    settings = LazySettings(environments=True, ENV_FOR_DYNACONF='DEFAULT', silent=False, LOADERS_FOR_DYNACONF=False, ROOT_PATH_FOR_DYNACONF=str(tmpdir), SETTINGS_FILE_FOR_DYNACONF=str(settings_file))
    assert settings.DEBUG is False
    assert settings.DATABASE_URI == f'{ext}.example.com'
    assert settings.PORT == 8080
    assert settings.SERVER == 'toml.example.com'
    assert settings.PLUGIN_NAME == 'testing'

def test_load_nested_different_types_with_merge(tmpdir):
    if False:
        i = 10
        return i + 15
    'Check merge works for includes.'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write(MIXED_MERGE)
    toml_plugin_file = tmpdir.join('plugin1.toml')
    toml_plugin_file.write(TOML_PLUGIN)
    for ext in ['toml', 'json', 'yaml', 'ini', 'py']:
        json_plugin_file = tmpdir.join(f'plugin2.{ext}')
        json_plugin_file.write(PLUGIN_TEXT[ext])
    settings = LazySettings(environments=True, ENV_FOR_DYNACONF='custom', silent=False, LOADERS_FOR_DYNACONF=False, ROOT_PATH_FOR_DYNACONF=str(tmpdir), SETTINGS_FILE_FOR_DYNACONF=str(settings_file), MERGE_ENABLED_FOR_DYNACONF=True)
    assert settings.DEBUG is False
    assert settings.DATABASE_URI == f'{ext}.example.com'
    assert settings.PORT == 8080
    assert settings.SERVER == 'toml.example.com'
    assert settings.PLUGIN_NAME == 'testing'
    assert settings.NESTED_1.base == 1
    assert settings.NESTED_1.nested_2.base == 2
    assert settings.NESTED_1.nested_2.nested_3.base == 3
    assert settings.NESTED_1.nested_2.nested_3.nested_4.base == 4
    for ext in ['toml', 'json', 'yaml', 'ini', 'py']:
        assert settings.NESTED_1.nested_2.nested_3.nested_4[ext] == 5

def test_programmatically_file_load(tmpdir):
    if False:
        i = 10
        return i + 15
    'Check file can be included programmatically'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write("\n       [default]\n       default_var = 'default'\n    ")
    settings = LazySettings(environments=True, SETTINGS_FILE_FOR_DYNACONF=str(settings_file))
    assert settings.DEFAULT_VAR == 'default'
    toml_plugin_file = tmpdir.join('plugin1.toml')
    toml_plugin_file.write("\n        [development]\n        plugin_value = 'plugin'\n    ")
    settings.load_file(path=str(toml_plugin_file))
    assert settings.PLUGIN_VALUE == 'plugin'
    settings.setenv('production')
    assert settings.DEFAULT_VAR == 'default'
    with pytest.raises(AttributeError):
        assert settings.PLUGIN_VALUE == 'plugin'

def test_include_via_python_module_name(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Check if an include can be a Python module name'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write("\n       [default]\n       default_var = 'default'\n    ")
    dummy_folder = tmpdir.mkdir('dummy')
    dummy_folder.join('dummy_module.py').write('FOO = "164110"')
    dummy_folder.join('__init__.py').write('print("initing dummy...")')
    settings = LazySettings(environments=True, SETTINGS_FILE_FOR_DYNACONF=str(settings_file), INCLUDES_FOR_DYNACONF=['dummy.dummy_module'])
    assert settings.DEFAULT_VAR == 'default'
    assert settings.FOO == '164110'

def test_include_via_python_module_name_and_others(tmpdir):
    if False:
        i = 10
        return i + 15
    'Check if an include can be a Python module name plus others'
    settings_file = tmpdir.join('settings.toml')
    settings_file.write("\n       [default]\n       default_var = 'default'\n    ")
    dummy_folder = tmpdir.mkdir('dummy')
    dummy_folder.join('dummy_module.py').write('FOO = "164110"')
    dummy_folder.join('__init__.py').write('print("initing dummy...")')
    yaml_file = tmpdir.join('otherfile.yaml')
    yaml_file.write('\n       default:\n         yaml_value: 748632\n    ')
    settings = LazySettings(environments=True, SETTINGS_FILE_FOR_DYNACONF=str(settings_file), INCLUDES_FOR_DYNACONF=['dummy.dummy_module', 'otherfile.yaml'])
    assert settings.DEFAULT_VAR == 'default'
    assert settings.FOO == '164110'
    assert settings.YAML_VALUE == 748632