import os
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import EnvironmentVariable, ExactStr, _check_vars

@pytest.fixture
def make_unknown_env():
    if False:
        return 10
    varname = 'MODIN_UNKNOWN'
    os.environ[varname] = 'foo'
    yield varname
    del os.environ[varname]

@pytest.fixture(params=[str, ExactStr])
def make_custom_envvar(request):
    if False:
        return 10

    class CustomVar(EnvironmentVariable, type=request.param):
        """custom var"""
        default = 10
        varname = 'MODIN_CUSTOM'
        choices = (1, 5, 10)
    return CustomVar

@pytest.fixture
def set_custom_envvar(make_custom_envvar):
    if False:
        return 10
    os.environ[make_custom_envvar.varname] = '  custom  '
    yield ('Custom' if make_custom_envvar.type is str else '  custom  ')
    del os.environ[make_custom_envvar.varname]

def test_unknown(make_unknown_env):
    if False:
        print('Hello World!')
    with pytest.warns(UserWarning, match=f'Found unknown .*{make_unknown_env}.*'):
        _check_vars()

def test_custom_default(make_custom_envvar):
    if False:
        print('Hello World!')
    assert make_custom_envvar.get() == 10

def test_custom_set(make_custom_envvar, set_custom_envvar):
    if False:
        print('Hello World!')
    assert make_custom_envvar.get() == set_custom_envvar

def test_custom_help(make_custom_envvar):
    if False:
        for i in range(10):
            print('nop')
    assert 'MODIN_CUSTOM' in make_custom_envvar.get_help()
    assert 'custom var' in make_custom_envvar.get_help()

def test_hdk_envvar():
    if False:
        return 10
    try:
        import pyhdk
        defaults = cfg.HdkLaunchParameters.get()
        assert defaults['enable_union'] == 1
        if version.parse(pyhdk.__version__) >= version.parse('0.6.1'):
            assert defaults['log_dir'] == 'pyhdk_log'
        del cfg.HdkLaunchParameters._value
    except ImportError:
        pass
    os.environ[cfg.HdkLaunchParameters.varname] = 'enable_union=2,enable_thrift_logs=3'
    params = cfg.HdkLaunchParameters.get()
    assert params['enable_union'] == 2
    assert params['enable_thrift_logs'] == 3
    os.environ[cfg.HdkLaunchParameters.varname] = 'unsupported=X'
    del cfg.HdkLaunchParameters._value
    params = cfg.HdkLaunchParameters.get()
    assert params['unsupported'] == 'X'
    try:
        import pyhdk
        pyhdk.buildConfig(**cfg.HdkLaunchParameters.get())
    except RuntimeError as e:
        assert str(e) == "unrecognised option '--unsupported'"
    except ImportError:
        pass
    os.environ[cfg.HdkLaunchParameters.varname] = 'enable_union=4,enable_thrift_logs=5,enable_lazy_dict_materialization=6'
    del cfg.HdkLaunchParameters._value
    params = cfg.HdkLaunchParameters.get()
    assert params['enable_union'] == 4
    assert params['enable_thrift_logs'] == 5
    assert params['enable_lazy_dict_materialization'] == 6