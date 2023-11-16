"""Configuration.from_ini() tests."""
from dependency_injector import errors
from pytest import mark, raises

def test(config, ini_config_file_1):
    if False:
        for i in range(10):
            print('nop')
    config.from_ini(ini_config_file_1)
    assert config() == {'section1': {'value1': '1'}, 'section2': {'value2': '2'}}
    assert config.section1() == {'value1': '1'}
    assert config.section1.value1() == '1'
    assert config.section2() == {'value2': '2'}
    assert config.section2.value2() == '2'

def test_option(config, ini_config_file_1):
    if False:
        return 10
    config.option.from_ini(ini_config_file_1)
    assert config() == {'option': {'section1': {'value1': '1'}, 'section2': {'value2': '2'}}}
    assert config.option() == {'section1': {'value1': '1'}, 'section2': {'value2': '2'}}
    assert config.option.section1() == {'value1': '1'}
    assert config.option.section1.value1() == '1'
    assert config.option.section2() == {'value2': '2'}
    assert config.option.section2.value2() == '2'

def test_merge(config, ini_config_file_1, ini_config_file_2):
    if False:
        i = 10
        return i + 15
    config.from_ini(ini_config_file_1)
    config.from_ini(ini_config_file_2)
    assert config() == {'section1': {'value1': '11', 'value11': '11'}, 'section2': {'value2': '2'}, 'section3': {'value3': '3'}}
    assert config.section1() == {'value1': '11', 'value11': '11'}
    assert config.section1.value1() == '11'
    assert config.section1.value11() == '11'
    assert config.section2() == {'value2': '2'}
    assert config.section2.value2() == '2'
    assert config.section3() == {'value3': '3'}
    assert config.section3.value3() == '3'

def test_file_does_not_exist(config):
    if False:
        while True:
            i = 10
    config.from_ini('./does_not_exist.ini')
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_file_does_not_exist_strict_mode(config):
    if False:
        while True:
            i = 10
    with raises(IOError):
        config.from_ini('./does_not_exist.ini')

def test_option_file_does_not_exist(config):
    if False:
        i = 10
        return i + 15
    config.option.from_ini('does_not_exist.ini')
    assert config.option.undefined() is None

@mark.parametrize('config_type', ['strict'])
def test_option_file_does_not_exist_strict_mode(config):
    if False:
        while True:
            i = 10
    with raises(IOError):
        config.option.from_ini('./does_not_exist.ini')

def test_required_file_does_not_exist(config):
    if False:
        return 10
    with raises(IOError):
        config.from_ini('./does_not_exist.ini', required=True)

def test_required_option_file_does_not_exist(config):
    if False:
        for i in range(10):
            print('nop')
    with raises(IOError):
        config.option.from_ini('./does_not_exist.ini', required=True)

@mark.parametrize('config_type', ['strict'])
def test_not_required_file_does_not_exist_strict_mode(config):
    if False:
        print('Hello World!')
    config.from_ini('./does_not_exist.ini', required=False)
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_not_required_option_file_does_not_exist_strict_mode(config):
    if False:
        return 10
    config.option.from_ini('./does_not_exist.ini', required=False)
    with raises(errors.Error):
        config.option()