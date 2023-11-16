import pytest
import os
import pathlib
from importlib.machinery import EXTENSION_SUFFIXES
from PyInstaller.building import utils

def test_format_binaries_and_datas_not_found_raises_error(tmpdir):
    if False:
        print('Hello World!')
    datas = [('non-existing.txt', '.')]
    tmpdir.join('existing.txt').ensure()
    with pytest.raises(SystemExit):
        utils.format_binaries_and_datas(datas, str(tmpdir))

def test_format_binaries_and_datas_empty_src(tmpdir):
    if False:
        print('Hello World!')
    datas = [('', '.')]
    with pytest.raises(SystemExit, match='Empty SRC is not allowed'):
        utils.format_binaries_and_datas(datas, str(tmpdir))

def test_format_binaries_and_datas_1(tmpdir):
    if False:
        return 10

    def _(path):
        if False:
            while True:
                i = 10
        return os.path.join(*path.split('/'))
    datas = [(_('existing.txt'), '.'), (_('other.txt'), 'foo'), (_('*.log'), 'logs'), (_('a/*.log'), 'lll'), (_('a/here.tex'), '.'), (_('b/[abc].tex'), 'tex')]
    expected = set()
    for (dest, src) in (('existing.txt', 'existing.txt'), ('foo/other.txt', 'other.txt'), ('logs/aaa.log', 'aaa.log'), ('logs/bbb.log', 'bbb.log'), ('lll/xxx.log', 'a/xxx.log'), ('lll/yyy.log', 'a/yyy.log'), ('here.tex', 'a/here.tex'), ('tex/a.tex', 'b/a.tex'), ('tex/b.tex', 'b/b.tex')):
        src = tmpdir.join(_(src)).ensure()
        expected.add((_(dest), str(src)))
    tmpdir.join(_('not.txt')).ensure()
    tmpdir.join(_('a/not.txt')).ensure()
    tmpdir.join(_('b/not.txt')).ensure()
    res = utils.format_binaries_and_datas(datas, str(tmpdir))
    assert res == expected

def test_format_binaries_and_datas_with_bracket(tmpdir):
    if False:
        i = 10
        return i + 15

    def _(path):
        if False:
            while True:
                i = 10
        return os.path.join(*path.split('/'))
    datas = [(_('b/[abc].tex'), 'tex')]
    expected = set()
    for (dest, src) in (('tex/[abc].tex', 'b/[abc].tex'),):
        src = tmpdir.join(_(src)).ensure()
        expected.add((_(dest), str(src)))
    tmpdir.join(_('tex/not.txt')).ensure()
    res = utils.format_binaries_and_datas(datas, str(tmpdir))
    assert res == expected

def test_add_suffix_to_extension():
    if False:
        while True:
            i = 10
    SUFFIX = EXTENSION_SUFFIXES[0]
    CASES = [('mypkg', 'mypkg' + SUFFIX, 'lib38/site-packages/mypkg' + SUFFIX, 'EXTENSION'), ('pkg.subpkg._extension', 'pkg/subpkg/_extension' + SUFFIX, 'lib38/site-packages/pkg/subpkg/_extension' + SUFFIX, 'EXTENSION'), ('lib-dynload/_extension', 'lib-dynload/_extension' + SUFFIX, 'lib38/lib-dynload/_extension' + SUFFIX, 'EXTENSION')]
    for case in CASES:
        dest_name1 = str(pathlib.PurePath(case[0]))
        dest_name2 = str(pathlib.PurePath(case[1]))
        src_name = str(pathlib.PurePath(case[2]))
        typecode = case[3]
        toc = (dest_name1, src_name, typecode)
        toc_expected = (dest_name2, src_name, typecode)
        toc2 = utils.add_suffix_to_extension(*toc)
        assert toc2 == toc_expected
        toc3 = utils.add_suffix_to_extension(*toc2)
        assert toc3 == toc2

def test_should_include_system_binary():
    if False:
        i = 10
        return i + 15
    CASES = [('lib-dynload/any', '/usr/lib64/any', [], True), ('libany', '/lib64/libpython.so', [], True), ('any', '/lib/python/site-packages/any', [], True), ('libany', '/etc/libany', [], True), ('libany', '/usr/lib/libany', ['*any*'], True), ('libany2', '/lib/libany2', ['libnone*', 'libany*'], True), ('libnomatch', '/lib/libnomatch', ['libnone*', 'libany*'], False)]
    for case in CASES:
        tuple = (case[0], case[1])
        excepts = case[2]
        expected = case[3]
        assert utils._should_include_system_binary(tuple, excepts) == expected