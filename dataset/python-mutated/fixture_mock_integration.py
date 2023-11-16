"""Reproduces issue #3774"""
from unittest import mock
import pytest
config = {'mykey': 'ORIGINAL'}

@pytest.fixture(scope='function')
@mock.patch.dict(config, {'mykey': 'MOCKED'})
def my_fixture():
    if False:
        for i in range(10):
            print('nop')
    return config['mykey']

def test_foobar(my_fixture):
    if False:
        for i in range(10):
            print('nop')
    assert my_fixture == 'MOCKED'