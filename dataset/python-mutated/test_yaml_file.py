import random
from unittest import mock
from conda_env import env
from conda_env.specs.yaml_file import YamlFileSpec

def test_no_environment_file():
    if False:
        while True:
            i = 10
    spec = YamlFileSpec(name=None, filename='not-a-file')
    assert not spec.can_handle()

def test_environment_file_exist():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(env, 'from_file', return_value={}):
        spec = YamlFileSpec(name=None, filename='environment.yaml')
        assert spec.can_handle()

def test_get_environment():
    if False:
        return 10
    r = random.randint(100, 200)
    with mock.patch.object(env, 'from_file', return_value=r):
        spec = YamlFileSpec(name=None, filename='environment.yaml')
        assert spec.environment == r

def test_filename():
    if False:
        while True:
            i = 10
    filename = f'filename_{random.randint(100, 200)}'
    with mock.patch.object(env, 'from_file') as from_file:
        spec = YamlFileSpec(filename=filename)
        spec.environment
    from_file.assert_called_with(filename)