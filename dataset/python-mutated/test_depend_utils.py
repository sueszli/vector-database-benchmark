import os
import pytest
import textwrap
from PyInstaller.depend import utils
from PyInstaller import compat
CTYPES_CLASSNAMES = ('CDLL', 'ctypes.CDLL', 'WinDLL', 'ctypes.WinDLL', 'OleDLL', 'ctypes.OleDLL', 'PyDLL', 'ctypes.PyDLL')

def __scan_code_for_ctypes(code, monkeypatch, extended_args):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(utils, '_resolveCtypesImports', lambda cbinaries: cbinaries)
    code = textwrap.dedent(code)
    if extended_args:
        from test_bytecode import many_constants, many_globals
        code = many_constants() + many_globals() + code
    co = compile(code, 'dummy', 'exec')
    return utils.scan_code_for_ctypes(co)

@pytest.mark.parametrize('classname', CTYPES_CLASSNAMES)
@pytest.mark.parametrize('extended_args', [False, True])
def test_ctypes_CDLL_call(monkeypatch, classname, extended_args):
    if False:
        return 10
    code = "%s('somelib.xxx')" % classname
    res = __scan_code_for_ctypes(code, monkeypatch, extended_args)
    assert res == set(['somelib.xxx'])

@pytest.mark.parametrize('classname', CTYPES_CLASSNAMES)
@pytest.mark.parametrize('extended_args', [False, True])
def test_ctypes_LibraryLoader(monkeypatch, classname, extended_args):
    if False:
        i = 10
        return i + 15
    code = '%s.somelib' % classname.lower()
    res = __scan_code_for_ctypes(code, monkeypatch, extended_args)
    assert res == set(['somelib.dll'])

@pytest.mark.parametrize('classname', CTYPES_CLASSNAMES)
@pytest.mark.parametrize('extended_args', [False, True])
def test_ctypes_LibraryLoader_LoadLibrary(monkeypatch, classname, extended_args):
    if False:
        print('Hello World!')
    code = "%s.LoadLibrary('somelib.xxx')" % classname.lower()
    res = __scan_code_for_ctypes(code, monkeypatch, extended_args)
    assert res == set(['somelib.xxx'])

@pytest.mark.parametrize('extended_args', [False, True])
@pytest.mark.skipif(compat.is_musl, reason="find_library() doesn't work on musl")
@pytest.mark.skipif(compat.is_macos_11 and (not (compat.is_macos_11_native and compat.is_py39)), reason='find_library() requires python >= 3.9 built with Big Sur support.')
def test_ctypes_util_find_library(monkeypatch, extended_args):
    if False:
        for i in range(10):
            print('nop')
    if compat.is_win:
        libname = 'KERNEL32'
    else:
        libname = 'c'
    code = "ctypes.util.find_library('%s')" % libname
    res = __scan_code_for_ctypes(code, monkeypatch, extended_args)
    assert res

def test_ctypes_util_find_library_as_default_argument():
    if False:
        while True:
            i = 10
    code = '\n    def locate_library(loader=ctypes.util.find_library):\n        pass\n    '
    code = textwrap.dedent(code)
    co = compile(code, '<ctypes_util_find_library_as_default_argument>', 'exec')
    utils.scan_code_for_ctypes(co)

@pytest.mark.linux
def test_ldconfig_cache():
    if False:
        for i in range(10):
            print('nop')
    utils.load_ldconfig_cache()
    if compat.is_musl:
        assert not utils.LDCONFIG_CACHE
        return
    libpath = None
    for soname in utils.LDCONFIG_CACHE:
        if soname.startswith('libc.so.'):
            libpath = utils.LDCONFIG_CACHE[soname]
            break
    assert libpath, 'libc.so not found'
    assert os.path.isfile(libpath)