import os
import pytest

@pytest.fixture
def nameset():
    if False:
        print('Hello World!')
    name = 'hey_i_am_an_env_var'
    os.environ[name] = 'i am a value'
    yield name
    del os.environ[name]

def test_envvar_catcher(nameset):
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        os.environ.get('Modin_FOO', 'bar')
    with pytest.raises(AssertionError):
        'modin_qux' not in os.environ
    assert 'yay_random_name' not in os.environ
    assert os.environ[nameset]