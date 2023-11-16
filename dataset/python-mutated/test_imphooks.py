"""Testing xonsh import hooks"""
import os
from importlib import import_module
import pytest
from xonsh import imphooks

@pytest.fixture(autouse=True)
def imp_env(xession):
    if False:
        for i in range(10):
            print('nop')
    xession.env.update({'PATH': [], 'PATHEXT': []})
    imphooks.install_import_hooks(xession.execer)
    yield

def test_import():
    if False:
        print('Hello World!')
    import sample
    assert 'hello mom jawaka\n' == sample.x

def test_import_empty():
    if False:
        print('Hello World!')
    from xpack import empty_xsh
    assert empty_xsh

def test_absolute_import():
    if False:
        print('Hello World!')
    from xpack import sample
    assert 'hello mom jawaka\n' == sample.x

def test_relative_import():
    if False:
        return 10
    from xpack import relimp
    assert 'hello mom jawaka\n' == relimp.sample.x
    assert 'hello mom jawaka\ndark chest of wonders' == relimp.y

def test_sub_import():
    if False:
        while True:
            i = 10
    from xpack.sub import sample
    assert 'hello mom jawaka\n' == sample.x
TEST_DIR = os.path.dirname(__file__)

def test_module_dunder_file_attribute():
    if False:
        print('Hello World!')
    import sample
    exp = os.path.join(TEST_DIR, 'sample.xsh')
    assert os.path.abspath(sample.__file__) == exp

def test_module_dunder_file_attribute_sub():
    if False:
        for i in range(10):
            print('nop')
    from xpack.sub import sample
    exp = os.path.join(TEST_DIR, 'xpack', 'sub', 'sample.xsh')
    assert os.path.abspath(sample.__file__) == exp

def test_get_source():
    if False:
        print('Hello World!')
    mod = import_module('sample')
    loader = mod.__loader__
    source = loader.get_source('sample')
    with open(os.path.join(TEST_DIR, 'sample.xsh')) as srcfile:
        assert source == srcfile.read()