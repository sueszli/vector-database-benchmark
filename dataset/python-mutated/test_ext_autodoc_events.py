"""Test the autodoc extension.  This tests mainly for autodoc events"""
import pytest
from sphinx.ext.autodoc import between, cut_lines
from .test_ext_autodoc import do_autodoc

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_process_docstring(app):
    if False:
        return 10

    def on_process_docstring(app, what, name, obj, options, lines):
        if False:
            for i in range(10):
                print('nop')
        lines.clear()
        lines.append('my docstring')
    app.connect('autodoc-process-docstring', on_process_docstring)
    actual = do_autodoc(app, 'function', 'target.process_docstring.func')
    assert list(actual) == ['', '.. py:function:: func()', '   :module: target.process_docstring', '', '   my docstring', '']

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_process_docstring_for_nondatadescriptor(app):
    if False:
        return 10

    def on_process_docstring(app, what, name, obj, options, lines):
        if False:
            i = 10
            return i + 15
        raise
    app.connect('autodoc-process-docstring', on_process_docstring)
    actual = do_autodoc(app, 'attribute', 'target.AttCls.a1')
    assert list(actual) == ['', '.. py:attribute:: AttCls.a1', '   :module: target', '   :value: hello world', '']

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_cut_lines(app):
    if False:
        print('Hello World!')
    app.connect('autodoc-process-docstring', cut_lines(2, 2, ['function']))
    actual = do_autodoc(app, 'function', 'target.process_docstring.func')
    assert list(actual) == ['', '.. py:function:: func()', '   :module: target.process_docstring', '', '   second line', '']

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_between(app):
    if False:
        return 10
    app.connect('autodoc-process-docstring', between('---', ['function']))
    actual = do_autodoc(app, 'function', 'target.process_docstring.func')
    assert list(actual) == ['', '.. py:function:: func()', '   :module: target.process_docstring', '', '   second line', '']

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_between_exclude(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('autodoc-process-docstring', between('---', ['function'], exclude=True))
    actual = do_autodoc(app, 'function', 'target.process_docstring.func')
    assert list(actual) == ['', '.. py:function:: func()', '   :module: target.process_docstring', '', '   first line', '   third line', '']

@pytest.mark.sphinx('html', testroot='ext-autodoc')
def test_skip_module_member(app):
    if False:
        i = 10
        return i + 15

    def autodoc_skip_member(app, what, name, obj, skip, options):
        if False:
            for i in range(10):
                print('nop')
        if name == 'Class':
            return True
        elif name == 'raises':
            return False
        return None
    app.connect('autodoc-skip-member', autodoc_skip_member)
    options = {'members': None}
    actual = do_autodoc(app, 'module', 'target', options)
    assert list(actual) == ['', '.. py:module:: target', '', '', '.. py:function:: raises(exc, func, *args, **kwds)', '   :module: target', '', '   Raise AssertionError if ``func(*args, **kwds)`` does not raise *exc*.', '']