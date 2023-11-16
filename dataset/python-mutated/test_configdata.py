"""Tests for qutebrowser.config.configdata."""
import textwrap
import yaml
import pytest
from qutebrowser import app
from qutebrowser.config import configdata, configtypes
from qutebrowser.utils import usertypes

def test_init(config_stub):
    if False:
        for i in range(10):
            print('nop')
    'Test reading the default yaml file.'
    config_stub.val.aliases = {}
    assert isinstance(configdata.DATA, dict)
    assert 'search.ignore_case' in configdata.DATA

def test_data(config_stub):
    if False:
        while True:
            i = 10
    'Test various properties of the default values.'
    for option in configdata.DATA.values():
        option.typ.to_py(option.default)
        option.typ.to_str(option.default)
        if isinstance(option.typ, (configtypes.Dict, configtypes.List)):
            assert option.default is not None, option
        if isinstance(option.typ, configtypes.ListOrValue):
            assert isinstance(option.default, list), option
        if isinstance(option.typ, configtypes.Float):
            for value in [option.default, option.typ.minval, option.typ.maxval]:
                assert value is None or isinstance(value, float), option
        assert '.  ' not in option.description, option

def test_init_benchmark(benchmark):
    if False:
        for i in range(10):
            print('nop')
    benchmark(configdata.init)

def test_is_valid_prefix(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(configdata, 'DATA', ['foo.bar'])
    assert configdata.is_valid_prefix('foo')
    assert not configdata.is_valid_prefix('foo.bar')
    assert not configdata.is_valid_prefix('foa')

class TestReadYaml:

    def test_valid(self):
        if False:
            print('Hello World!')
        yaml_data = textwrap.dedent('\n            test1:\n                type: Bool\n                default: true\n                desc: Hello World\n\n            test2:\n                type: String\n                default: foo\n                backend: QtWebKit\n                desc: Hello World 2\n        ')
        (data, _migrations) = configdata._read_yaml(yaml_data)
        assert data.keys() == {'test1', 'test2'}
        assert data['test1'].description == 'Hello World'
        assert data['test2'].default == 'foo'
        assert data['test2'].backends == [usertypes.Backend.QtWebKit]
        assert isinstance(data['test1'].typ, configtypes.Bool)

    def test_invalid_keys(self):
        if False:
            for i in range(10):
                print('nop')
        'Test reading with unknown keys.'
        data = textwrap.dedent('\n            test:\n                type: Bool\n                default: true\n                desc: Hello World\n                hello: world\n        ')
        with pytest.raises(ValueError, match='Invalid keys'):
            configdata._read_yaml(data)

    @pytest.mark.parametrize('first, second, shadowing', [('foo', 'foo.bar', True), ('foo.bar', 'foo', True), ('foo.bar', 'foo.bar.baz', True), ('foo.bar', 'foo.baz', False)])
    def test_shadowing(self, first, second, shadowing):
        if False:
            return 10
        "Make sure a setting can't shadow another."
        data = textwrap.dedent('\n            {first}:\n                type: Bool\n                default: true\n                desc: Hello World\n\n            {second}:\n                type: Bool\n                default: true\n                desc: Hello World\n        '.format(first=first, second=second))
        if shadowing:
            with pytest.raises(ValueError, match='Shadowing keys'):
                configdata._read_yaml(data)
        else:
            configdata._read_yaml(data)

    def test_rename(self):
        if False:
            print('Hello World!')
        yaml_data = textwrap.dedent('\n            test:\n                renamed: test_new\n\n            test_new:\n                type: Bool\n                default: true\n                desc: Hello World\n        ')
        (data, migrations) = configdata._read_yaml(yaml_data)
        assert data.keys() == {'test_new'}
        assert migrations.renamed == {'test': 'test_new'}

    def test_rename_unknown_target(self):
        if False:
            for i in range(10):
                print('nop')
        yaml_data = textwrap.dedent('\n            test:\n                renamed: test2\n        ')
        with pytest.raises(ValueError, match='Renaming test to unknown test2'):
            configdata._read_yaml(yaml_data)

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        yaml_data = textwrap.dedent('\n            test:\n                deleted: true\n        ')
        (data, migrations) = configdata._read_yaml(yaml_data)
        assert not data.keys()
        assert migrations.deleted == ['test']

    def test_delete_invalid_value(self):
        if False:
            i = 10
            return i + 15
        yaml_data = textwrap.dedent('\n            test:\n                deleted: false\n        ')
        with pytest.raises(ValueError, match='Invalid deleted value: False'):
            configdata._read_yaml(yaml_data)

class TestParseYamlType:

    def _yaml(self, s):
        if False:
            print('Hello World!')
        'Get the type from parsed YAML data.'
        return yaml.safe_load(textwrap.dedent(s))['type']

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        'Test type which is only a name.'
        data = self._yaml('type: Bool')
        typ = configdata._parse_yaml_type('test', data)
        assert isinstance(typ, configtypes.Bool)
        assert not typ.none_ok

    def test_complex(self):
        if False:
            print('Hello World!')
        'Test type parsing with arguments.'
        data = self._yaml('\n            type:\n              name: String\n              minlen: 2\n        ')
        typ = configdata._parse_yaml_type('test', data)
        assert isinstance(typ, configtypes.String)
        assert not typ.none_ok
        assert typ.minlen == 2

    def test_list(self):
        if False:
            return 10
        'Test type parsing with a list and subtypes.'
        data = self._yaml('\n            type:\n              name: List\n              valtype: String\n        ')
        typ = configdata._parse_yaml_type('test', data)
        assert isinstance(typ, configtypes.List)
        assert isinstance(typ.valtype, configtypes.String)
        assert not typ.none_ok
        assert not typ.valtype.none_ok

    def test_dict(self):
        if False:
            i = 10
            return i + 15
        'Test type parsing with a dict and subtypes.'
        data = self._yaml('\n            type:\n              name: Dict\n              keytype: String\n              valtype:\n                name: Int\n                minval: 10\n        ')
        typ = configdata._parse_yaml_type('test', data)
        assert isinstance(typ, configtypes.Dict)
        assert isinstance(typ.keytype, configtypes.String)
        assert isinstance(typ.valtype, configtypes.Int)
        assert not typ.none_ok
        assert typ.valtype.minval == 10

    def test_invalid_node(self):
        if False:
            for i in range(10):
                print('nop')
        'Test type parsing with invalid node type.'
        data = self._yaml('type: 42')
        with pytest.raises(ValueError, match='Invalid node for test while reading type: 42'):
            configdata._parse_yaml_type('test', data)

    def test_unknown_type(self):
        if False:
            while True:
                i = 10
        "Test type parsing with type which doesn't exist."
        data = self._yaml('type: Foobar')
        with pytest.raises(AttributeError, match='Did not find type Foobar for test'):
            configdata._parse_yaml_type('test', data)

    def test_unknown_dict(self):
        if False:
            print('Hello World!')
        'Test type parsing with a dict without keytype.'
        data = self._yaml('type: Dict')
        with pytest.raises(ValueError, match="Invalid node for test while reading 'keytype': 'Dict'"):
            configdata._parse_yaml_type('test', data)

    def test_unknown_args(self):
        if False:
            i = 10
            return i + 15
        'Test type parsing with unknown type arguments.'
        data = self._yaml('\n            type:\n              name: Int\n              answer: 42\n        ')
        with pytest.raises(TypeError, match='Error while creating Int'):
            configdata._parse_yaml_type('test', data)

class TestParseYamlBackend:

    def _yaml(self, s):
        if False:
            print('Hello World!')
        'Get the type from parsed YAML data.'
        return yaml.safe_load(textwrap.dedent(s))['backend']

    @pytest.mark.parametrize('backend, expected', [('QtWebKit', [usertypes.Backend.QtWebKit]), ('QtWebEngine', [usertypes.Backend.QtWebEngine]), ('null', [usertypes.Backend.QtWebKit, usertypes.Backend.QtWebEngine])])
    def test_simple(self, backend, expected):
        if False:
            while True:
                i = 10
        'Check a simple "backend: QtWebKit".'
        data = self._yaml('backend: {}'.format(backend))
        backends = configdata._parse_yaml_backends('test', data)
        assert backends == expected

    @pytest.mark.parametrize('webkit, has_new_version, expected', [(True, True, [usertypes.Backend.QtWebEngine, usertypes.Backend.QtWebKit]), (False, True, [usertypes.Backend.QtWebEngine]), (True, False, [usertypes.Backend.QtWebKit])])
    def test_dict(self, monkeypatch, webkit, has_new_version, expected):
        if False:
            return 10
        data = self._yaml('\n            backend:\n              QtWebKit: {}\n              QtWebEngine: Qt 5.15\n        '.format('true' if webkit else 'false'))
        monkeypatch.setattr(configdata.qtutils, 'version_check', lambda v: has_new_version)
        backends = configdata._parse_yaml_backends('test', data)
        assert backends == expected

    @pytest.mark.parametrize('yaml_data', ['backend: 42', '\n        backend:\n          QtWebKit: true\n          QtWebEngine: true\n          foo: bar\n        ', '\n        backend:\n          QtWebKit: true\n        '])
    def test_invalid_backend(self, yaml_data):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError, match='Invalid node for test while reading backends:'):
            configdata._parse_yaml_backends('test', self._yaml(yaml_data))