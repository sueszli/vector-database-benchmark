"""Tests for the compilerop module.
"""
import linecache
import sys
from IPython.core import compilerop

def test_code_name():
    if False:
        i = 10
        return i + 15
    code = 'x=1'
    name = compilerop.code_name(code)
    assert name.startswith('<ipython-input-0')

def test_code_name2():
    if False:
        while True:
            i = 10
    code = 'x=1'
    name = compilerop.code_name(code, 9)
    assert name.startswith('<ipython-input-9')

def test_cache():
    if False:
        i = 10
        return i + 15
    'Test the compiler correctly compiles and caches inputs\n    '
    cp = compilerop.CachingCompiler()
    ncache = len(linecache.cache)
    cp.cache('x=1')
    assert len(linecache.cache) > ncache

def test_proper_default_encoding():
    if False:
        return 10
    assert sys.getdefaultencoding() == 'utf-8'

def test_cache_unicode():
    if False:
        return 10
    cp = compilerop.CachingCompiler()
    ncache = len(linecache.cache)
    cp.cache(u"t = 'žćčšđ'")
    assert len(linecache.cache) > ncache

def test_compiler_check_cache():
    if False:
        while True:
            i = 10
    'Test the compiler properly manages the cache.\n    '
    cp = compilerop.CachingCompiler()
    cp.cache('x=1', 99)
    linecache.checkcache()
    assert any((k.startswith('<ipython-input-99') for k in linecache.cache)), 'Entry for input-99 missing from linecache'