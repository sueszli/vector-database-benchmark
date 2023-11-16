from __future__ import annotations
import os.path
import pytest
from pre_commit.prefix import Prefix

def norm_slash(*args):
    if False:
        for i in range(10):
            print('nop')
    return tuple((x.replace('/', os.sep) for x in args))

@pytest.mark.parametrize(('prefix', 'path_end', 'expected_output'), (norm_slash('foo', '', 'foo'), norm_slash('foo', 'bar', 'foo/bar'), norm_slash('foo/bar', '../baz', 'foo/baz'), norm_slash('./', 'bar', 'bar'), norm_slash('./', '', '.'), norm_slash('/tmp/foo', '/tmp/bar', '/tmp/bar')))
def test_path(prefix, path_end, expected_output):
    if False:
        print('Hello World!')
    instance = Prefix(prefix)
    ret = instance.path(path_end)
    assert ret == expected_output

def test_path_multiple_args():
    if False:
        return 10
    instance = Prefix('foo')
    ret = instance.path('bar', 'baz')
    assert ret == os.path.join('foo', 'bar', 'baz')

def test_exists(tmpdir):
    if False:
        while True:
            i = 10
    assert not Prefix(str(tmpdir)).exists('foo')
    tmpdir.ensure('foo')
    assert Prefix(str(tmpdir)).exists('foo')

def test_star(tmpdir):
    if False:
        return 10
    for f in ('a.txt', 'b.txt', 'c.py'):
        tmpdir.join(f).ensure()
    assert set(Prefix(str(tmpdir)).star('.txt')) == {'a.txt', 'b.txt'}