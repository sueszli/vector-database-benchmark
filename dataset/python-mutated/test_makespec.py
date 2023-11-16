import os
import argparse
import pytest
from PyInstaller.building import makespec

def test_make_variable_path():
    if False:
        while True:
            i = 10
    p = os.path.join(makespec.HOMEPATH, 'aaa', 'bbb', 'ccc')
    assert makespec.make_variable_path(p) == ('HOMEPATH', os.path.join('aaa', 'bbb', 'ccc'))

def test_make_variable_path_regression():
    if False:
        for i in range(10):
            print('nop')
    p = os.path.join(makespec.HOMEPATH + 'aaa', 'bbb', 'ccc')
    assert makespec.make_variable_path(p) == (None, p)

def test_Path_constructor():
    if False:
        print('Hello World!')
    p = makespec.Path('aaa', 'bbb', 'ccc')
    assert p.path == os.path.join('aaa', 'bbb', 'ccc')

def test_Path_repr():
    if False:
        print('Hello World!')
    p = makespec.Path(makespec.HOMEPATH, 'aaa', 'bbb', 'ccc')
    assert p.path == os.path.join(makespec.HOMEPATH, 'aaa', 'bbb', 'ccc')
    assert repr(p) == 'os.path.join(HOMEPATH,%r)' % os.path.join('aaa', 'bbb', 'ccc')

def test_Path_repr_relative():
    if False:
        i = 10
        return i + 15
    p = makespec.Path('aaa', 'bbb', 'ccc.py')
    assert p.path == os.path.join('aaa', 'bbb', 'ccc.py')
    assert repr(p) == '%r' % os.path.join('aaa', 'bbb', 'ccc.py')

def test_Path_regression():
    if False:
        print('Hello World!')
    p = makespec.Path(makespec.HOMEPATH + '-aaa', 'bbb', 'ccc')
    assert p.path == os.path.join(makespec.HOMEPATH + '-aaa', 'bbb', 'ccc')
    assert repr(p) == repr(os.path.join(makespec.HOMEPATH + '-aaa', 'bbb', 'ccc'))

def test_add_data(capsys):
    if False:
        i = 10
        return i + 15
    '\n    Test CLI parsing of --add-data and --add-binary.\n    '
    parser = argparse.ArgumentParser()
    makespec.__add_options(parser)
    assert parser.parse_args([]).datas == []
    assert parser.parse_args(['--add-data', '/foo/bar:.']).datas == [('/foo/bar', '.')]
    assert parser.parse_args(['--add-data=C:\\foo\\bar:baz']).datas == [('C:\\foo\\bar', 'baz')]
    assert parser.parse_args(['--add-data=c:/foo/bar:baz']).datas == [('c:/foo/bar', 'baz')]
    assert parser.parse_args(['--add-data=/foo/:bar']).datas == [('/foo/', 'bar')]
    for args in [['--add-data', 'foo/bar'], ['--add-data', 'C:/foo/bar']]:
        with pytest.raises(SystemExit):
            parser.parse_args(args)
        assert '--add-data: Wrong syntax, should be --add-data=SOURCE:DEST' in capsys.readouterr().err
    if os.pathsep == ';':
        assert parser.parse_args(['--add-data', 'foo;.']).datas == [('foo', '.')]
    else:
        assert parser.parse_args(['--add-data', 'foo;bar:.']).datas == [('foo;bar', '.')]
    with pytest.raises(SystemExit):
        parser.parse_args(['--add-data', 'foo:'])
    assert '--add-data: You have to specify both SOURCE and DEST' in capsys.readouterr().err
    options = parser.parse_args(['--add-data=a:b', '--add-data=c:d', '--add-binary=e:f'])
    assert options.datas == [('a', 'b'), ('c', 'd')]
    assert options.binaries == [('e', 'f')]