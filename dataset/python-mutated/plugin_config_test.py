from os import path
import pytest
from errbot.botplugin import ValidationException, recurse_check_structure
extra_plugin_dir = path.join(path.dirname(path.realpath(__file__)), 'config_plugin')

def test_recurse_check_structure_valid():
    if False:
        for i in range(10):
            print('nop')
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list=['Foo', 'Bar', 'Bas'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    recurse_check_structure(sample, to_check)

def test_recurse_check_structure_missingitem():
    if False:
        return 10
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_recurse_check_structure_extrasubitem():
    if False:
        while True:
            i = 10
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list=['Foo', 'Bar', 'Bas'], dict={'foo': 'Bar', 'Bar': 'Foo'}, none=None, true=True, false=False)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_recurse_check_structure_missingsubitem():
    if False:
        return 10
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list=['Foo', 'Bar', 'Bas'], dict={}, none=None, true=True, false=False)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_recurse_check_structure_wrongtype_1():
    if False:
        i = 10
        return i + 15
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string=None, list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_recurse_check_structure_wrongtype_2():
    if False:
        for i in range(10):
            print('nop')
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list={'foo': 'Bar'}, dict={'foo': 'Bar'}, none=None, true=True, false=False)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_recurse_check_structure_wrongtype_3():
    if False:
        print('Hello World!')
    sample = dict(string='Foobar', list=['Foo', 'Bar'], dict={'foo': 'Bar'}, none=None, true=True, false=False)
    to_check = dict(string='Foobar', list=['Foo', 'Bar'], dict=['Foo', 'Bar'], none=None, true=True, false=False)
    with pytest.raises(ValidationException):
        recurse_check_structure(sample, to_check)

def test_failed_config(testbot):
    if False:
        return 10
    assert 'Plugin configuration done.' in testbot.exec_command('!plugin config Config {"One": "two"}')