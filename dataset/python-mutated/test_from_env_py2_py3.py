"""Configuration.from_env() tests."""
from pytest import mark, raises

def test(config):
    if False:
        while True:
            i = 10
    config.from_env('CONFIG_TEST_ENV')
    assert config() == 'test-value'

def test_with_children(config):
    if False:
        print('Hello World!')
    config.section1.value1.from_env('CONFIG_TEST_ENV')
    assert config() == {'section1': {'value1': 'test-value'}}
    assert config.section1() == {'value1': 'test-value'}
    assert config.section1.value1() == 'test-value'

def test_default(config):
    if False:
        for i in range(10):
            print('nop')
    config.from_env('UNDEFINED_ENV', 'default-value')
    assert config() == 'default-value'

def test_default_none(config):
    if False:
        for i in range(10):
            print('nop')
    config.from_env('UNDEFINED_ENV')
    assert config() is None

def test_option_default_none(config):
    if False:
        for i in range(10):
            print('nop')
    config.option.from_env('UNDEFINED_ENV')
    assert config.option() is None

def test_as_(config):
    if False:
        print('Hello World!')
    config.from_env('CONFIG_INT', as_=int)
    assert config() == 42
    assert isinstance(config(), int)

def test_as__default(config):
    if False:
        print('Hello World!')
    config.from_env('UNDEFINED', as_=int, default='33')
    assert config() == 33
    assert isinstance(config(), int)

def test_as__undefined_required(config):
    if False:
        return 10
    with raises(ValueError):
        config.from_env('UNDEFINED', as_=int, required=True)
    assert config() == {}

def test_as__defined_empty(config):
    if False:
        while True:
            i = 10
    with raises(ValueError):
        config.from_env('EMPTY', as_=int)
    assert config() == {}

def test_option_as_(config):
    if False:
        i = 10
        return i + 15
    config.option.from_env('CONFIG_INT', as_=int)
    assert config.option() == 42
    assert isinstance(config.option(), int)

def test_option_as__default(config):
    if False:
        print('Hello World!')
    config.option.from_env('UNDEFINED', as_=int, default='33')
    assert config.option() == 33
    assert isinstance(config.option(), int)

def test_option_as__undefined_required(config):
    if False:
        while True:
            i = 10
    with raises(ValueError):
        config.option.from_env('UNDEFINED', as_=int, required=True)
    assert config.option() is None

def test_option_as__defined_empty(config):
    if False:
        i = 10
        return i + 15
    with raises(ValueError):
        config.option.from_env('EMPTY', as_=int)
    assert config.option() is None

@mark.parametrize('config_type', ['strict'])
def test_undefined_in_strict_mode(config):
    if False:
        print('Hello World!')
    with raises(ValueError):
        config.from_env('UNDEFINED_ENV')

@mark.parametrize('config_type', ['strict'])
def test_option_undefined_in_strict_mode(config):
    if False:
        return 10
    with raises(ValueError):
        config.option.from_env('UNDEFINED_ENV')

def test_undefined_in_strict_mode_with_default(config):
    if False:
        while True:
            i = 10
    config.from_env('UNDEFINED_ENV', 'default-value')
    assert config() == 'default-value'

@mark.parametrize('config_type', ['strict'])
def test_option_undefined_in_strict_mode_with_default(config):
    if False:
        for i in range(10):
            print('nop')
    config.option.from_env('UNDEFINED_ENV', 'default-value')
    assert config.option() == 'default-value'

def test_required_undefined(config):
    if False:
        i = 10
        return i + 15
    with raises(ValueError):
        config.from_env('UNDEFINED_ENV', required=True)

def test_required_undefined_with_default(config):
    if False:
        for i in range(10):
            print('nop')
    config.from_env('UNDEFINED_ENV', default='default-value', required=True)
    assert config() == 'default-value'

def test_option_required_undefined(config):
    if False:
        for i in range(10):
            print('nop')
    with raises(ValueError):
        config.option.from_env('UNDEFINED_ENV', required=True)

def test_option_required_undefined_with_default(config):
    if False:
        return 10
    config.option.from_env('UNDEFINED_ENV', default='default-value', required=True)
    assert config.option() == 'default-value'

@mark.parametrize('config_type', ['strict'])
def test_not_required_undefined_in_strict_mode(config):
    if False:
        i = 10
        return i + 15
    config.from_env('UNDEFINED_ENV', required=False)
    assert config() is None

@mark.parametrize('config_type', ['strict'])
def test_option_not_required_undefined_in_strict_mode(config):
    if False:
        return 10
    config.option.from_env('UNDEFINED_ENV', required=False)
    assert config.option() is None

@mark.parametrize('config_type', ['strict'])
def test_not_required_undefined_with_default_in_strict_mode(config):
    if False:
        print('Hello World!')
    config.from_env('UNDEFINED_ENV', default='default-value', required=False)
    assert config() == 'default-value'

@mark.parametrize('config_type', ['strict'])
def test_option_not_required_undefined_with_default_in_strict_mode(config):
    if False:
        i = 10
        return i + 15
    config.option.from_env('UNDEFINED_ENV', default='default-value', required=False)
    assert config.option() == 'default-value'