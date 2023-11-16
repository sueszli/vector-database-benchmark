"""Configuration.from_json() tests."""
from dependency_injector import errors
from pytest import mark, raises

def test(config, json_config_file_1):
    if False:
        i = 10
        return i + 15
    config.from_json(json_config_file_1)
    assert config() == {'section1': {'value1': 1}, 'section2': {'value2': 2}}
    assert config.section1() == {'value1': 1}
    assert config.section1.value1() == 1
    assert config.section2() == {'value2': 2}
    assert config.section2.value2() == 2

def test_merge(config, json_config_file_1, json_config_file_2):
    if False:
        return 10
    config.from_json(json_config_file_1)
    config.from_json(json_config_file_2)
    assert config() == {'section1': {'value1': 11, 'value11': 11}, 'section2': {'value2': 2}, 'section3': {'value3': 3}}
    assert config.section1() == {'value1': 11, 'value11': 11}
    assert config.section1.value1() == 11
    assert config.section1.value11() == 11
    assert config.section2() == {'value2': 2}
    assert config.section2.value2() == 2
    assert config.section3() == {'value3': 3}
    assert config.section3.value3() == 3

def test_file_does_not_exist(config):
    if False:
        i = 10
        return i + 15
    config.from_json('./does_not_exist.json')
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_file_does_not_exist_strict_mode(config):
    if False:
        print('Hello World!')
    with raises(IOError):
        config.from_json('./does_not_exist.json')

def test_option_file_does_not_exist(config):
    if False:
        i = 10
        return i + 15
    config.option.from_json('./does_not_exist.json')
    assert config.option() is None

@mark.parametrize('config_type', ['strict'])
def test_option_file_does_not_exist_strict_mode(config):
    if False:
        while True:
            i = 10
    with raises(IOError):
        config.option.from_json('./does_not_exist.json')

def test_required_file_does_not_exist(config):
    if False:
        print('Hello World!')
    with raises(IOError):
        config.from_json('./does_not_exist.json', required=True)

def test_required_option_file_does_not_exist(config):
    if False:
        i = 10
        return i + 15
    with raises(IOError):
        config.option.from_json('./does_not_exist.json', required=True)

@mark.parametrize('config_type', ['strict'])
def test_not_required_file_does_not_exist_strict_mode(config):
    if False:
        while True:
            i = 10
    config.from_json('./does_not_exist.json', required=False)
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_not_required_option_file_does_not_exist_strict_mode(config):
    if False:
        return 10
    config.option.from_json('./does_not_exist.json', required=False)
    with raises(errors.Error):
        config.option()