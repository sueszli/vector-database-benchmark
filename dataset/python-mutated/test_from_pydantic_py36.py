"""Configuration.from_pydantic() tests."""
import pydantic
from dependency_injector import providers, errors
from pytest import fixture, mark, raises

class Section11(pydantic.BaseModel):
    value1 = 1

class Section12(pydantic.BaseModel):
    value2 = 2

class Settings1(pydantic.BaseSettings):
    section1 = Section11()
    section2 = Section12()

class Section21(pydantic.BaseModel):
    value1 = 11
    value11 = 11

class Section3(pydantic.BaseModel):
    value3 = 3

class Settings2(pydantic.BaseSettings):
    section1 = Section21()
    section3 = Section3()

@fixture
def no_pydantic_module_installed():
    if False:
        i = 10
        return i + 15
    providers.pydantic = None
    yield
    providers.pydantic = pydantic

def test(config):
    if False:
        while True:
            i = 10
    config.from_pydantic(Settings1())
    assert config() == {'section1': {'value1': 1}, 'section2': {'value2': 2}}
    assert config.section1() == {'value1': 1}
    assert config.section1.value1() == 1
    assert config.section2() == {'value2': 2}
    assert config.section2.value2() == 2

def test_kwarg(config):
    if False:
        print('Hello World!')
    config.from_pydantic(Settings1(), exclude={'section2'})
    assert config() == {'section1': {'value1': 1}}
    assert config.section1() == {'value1': 1}
    assert config.section1.value1() == 1

def test_merge(config):
    if False:
        i = 10
        return i + 15
    config.from_pydantic(Settings1())
    config.from_pydantic(Settings2())
    assert config() == {'section1': {'value1': 11, 'value11': 11}, 'section2': {'value2': 2}, 'section3': {'value3': 3}}
    assert config.section1() == {'value1': 11, 'value11': 11}
    assert config.section1.value1() == 11
    assert config.section1.value11() == 11
    assert config.section2() == {'value2': 2}
    assert config.section2.value2() == 2
    assert config.section3() == {'value3': 3}
    assert config.section3.value3() == 3

def test_empty_settings(config):
    if False:
        return 10
    config.from_pydantic(pydantic.BaseSettings())
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_empty_settings_strict_mode(config):
    if False:
        print('Hello World!')
    with raises(ValueError):
        config.from_pydantic(pydantic.BaseSettings())

def test_option_empty_settings(config):
    if False:
        for i in range(10):
            print('nop')
    config.option.from_pydantic(pydantic.BaseSettings())
    assert config.option() == {}

@mark.parametrize('config_type', ['strict'])
def test_option_empty_settings_strict_mode(config):
    if False:
        return 10
    with raises(ValueError):
        config.option.from_pydantic(pydantic.BaseSettings())

def test_required_empty_settings(config):
    if False:
        i = 10
        return i + 15
    with raises(ValueError):
        config.from_pydantic(pydantic.BaseSettings(), required=True)

def test_required_option_empty_settings(config):
    if False:
        for i in range(10):
            print('nop')
    with raises(ValueError):
        config.option.from_pydantic(pydantic.BaseSettings(), required=True)

@mark.parametrize('config_type', ['strict'])
def test_not_required_empty_settings_strict_mode(config):
    if False:
        while True:
            i = 10
    config.from_pydantic(pydantic.BaseSettings(), required=False)
    assert config() == {}

@mark.parametrize('config_type', ['strict'])
def test_not_required_option_empty_settings_strict_mode(config):
    if False:
        i = 10
        return i + 15
    config.option.from_pydantic(pydantic.BaseSettings(), required=False)
    assert config.option() == {}
    assert config() == {'option': {}}

def test_not_instance_of_settings(config):
    if False:
        return 10
    with raises(errors.Error) as error:
        config.from_pydantic({})
    assert error.value.args[0] == 'Unable to recognize settings instance, expect "pydantic.BaseSettings", got {0} instead'.format({})

def test_option_not_instance_of_settings(config):
    if False:
        while True:
            i = 10
    with raises(errors.Error) as error:
        config.option.from_pydantic({})
    assert error.value.args[0] == 'Unable to recognize settings instance, expect "pydantic.BaseSettings", got {0} instead'.format({})

def test_subclass_instead_of_instance(config):
    if False:
        while True:
            i = 10
    with raises(errors.Error) as error:
        config.from_pydantic(Settings1)
    assert error.value.args[0] == 'Got settings class, but expect instance: instead "Settings1" use "Settings1()"'

def test_option_subclass_instead_of_instance(config):
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error) as error:
        config.option.from_pydantic(Settings1)
    assert error.value.args[0] == 'Got settings class, but expect instance: instead "Settings1" use "Settings1()"'

@mark.usefixtures('no_pydantic_module_installed')
def test_no_pydantic_installed(config):
    if False:
        i = 10
        return i + 15
    with raises(errors.Error) as error:
        config.from_pydantic(Settings1())
    assert error.value.args[0] == 'Unable to load pydantic configuration - pydantic is not installed. Install pydantic or install Dependency Injector with pydantic extras: "pip install dependency-injector[pydantic]"'

@mark.usefixtures('no_pydantic_module_installed')
def test_option_no_pydantic_installed(config):
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error) as error:
        config.option.from_pydantic(Settings1())
    assert error.value.args[0] == 'Unable to load pydantic configuration - pydantic is not installed. Install pydantic or install Dependency Injector with pydantic extras: "pip install dependency-injector[pydantic]"'