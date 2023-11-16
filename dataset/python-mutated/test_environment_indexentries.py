"""Test the sphinx.environment.adapters.indexentries."""
import pytest
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.testing import restructuredtext

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_single_index(app):
    if False:
        i = 10
        return i + 15
    text = '.. index:: docutils\n.. index:: Python\n.. index:: pip; install\n.. index:: pip; upgrade\n.. index:: Sphinx\n.. index:: Ель\n.. index:: ёлка\n.. index:: \u200fעברית\u200e\n.. index:: 9-symbol\n.. index:: &-symbol\n.. index:: £100\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 6
    assert index[0] == ('Symbols', [('&-symbol', [[('', '#index-9')], [], None]), ('9-symbol', [[('', '#index-8')], [], None]), ('£100', [[('', '#index-10')], [], None])])
    assert index[1] == ('D', [('docutils', [[('', '#index-0')], [], None])])
    assert index[2] == ('P', [('pip', [[], [('install', [('', '#index-2')]), ('upgrade', [('', '#index-3')])], None]), ('Python', [[('', '#index-1')], [], None])])
    assert index[3] == ('S', [('Sphinx', [[('', '#index-4')], [], None])])
    assert index[4] == ('Е', [('ёлка', [[('', '#index-6')], [], None]), ('Ель', [[('', '#index-5')], [], None])])
    assert index[5] == ('ע', [('\u200fעברית\u200e', [[('', '#index-7')], [], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_pair_index(app):
    if False:
        while True:
            i = 10
    text = '.. index:: pair: docutils; reStructuredText\n.. index:: pair: Python; interpreter\n.. index:: pair: Sphinx; documentation tool\n.. index:: pair: Sphinx; :+1:\n.. index:: pair: Sphinx; Ель\n.. index:: pair: Sphinx; ёлка\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 7
    assert index[0] == ('Symbols', [(':+1:', [[], [('Sphinx', [('', '#index-3')])], None])])
    assert index[1] == ('D', [('documentation tool', [[], [('Sphinx', [('', '#index-2')])], None]), ('docutils', [[], [('reStructuredText', [('', '#index-0')])], None])])
    assert index[2] == ('I', [('interpreter', [[], [('Python', [('', '#index-1')])], None])])
    assert index[3] == ('P', [('Python', [[], [('interpreter', [('', '#index-1')])], None])])
    assert index[4] == ('R', [('reStructuredText', [[], [('docutils', [('', '#index-0')])], None])])
    assert index[5] == ('S', [('Sphinx', [[], [(':+1:', [('', '#index-3')]), ('documentation tool', [('', '#index-2')]), ('ёлка', [('', '#index-5')]), ('Ель', [('', '#index-4')])], None])])
    assert index[6] == ('Е', [('ёлка', [[], [('Sphinx', [('', '#index-5')])], None]), ('Ель', [[], [('Sphinx', [('', '#index-4')])], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_triple_index(app):
    if False:
        while True:
            i = 10
    text = '.. index:: triple: foo; bar; baz\n.. index:: triple: Python; Sphinx; reST\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 5
    assert index[0] == ('B', [('bar', [[], [('baz, foo', [('', '#index-0')])], None]), ('baz', [[], [('foo bar', [('', '#index-0')])], None])])
    assert index[1] == ('F', [('foo', [[], [('bar baz', [('', '#index-0')])], None])])
    assert index[2] == ('P', [('Python', [[], [('Sphinx reST', [('', '#index-1')])], None])])
    assert index[3] == ('R', [('reST', [[], [('Python Sphinx', [('', '#index-1')])], None])])
    assert index[4] == ('S', [('Sphinx', [[], [('reST, Python', [('', '#index-1')])], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_see_index(app):
    if False:
        while True:
            i = 10
    text = '.. index:: see: docutils; reStructuredText\n.. index:: see: Python; interpreter\n.. index:: see: Sphinx; documentation tool\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 3
    assert index[0] == ('D', [('docutils', [[], [('see reStructuredText', [])], None])])
    assert index[1] == ('P', [('Python', [[], [('see interpreter', [])], None])])
    assert index[2] == ('S', [('Sphinx', [[], [('see documentation tool', [])], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_seealso_index(app):
    if False:
        i = 10
        return i + 15
    text = '.. index:: seealso: docutils; reStructuredText\n.. index:: seealso: Python; interpreter\n.. index:: seealso: Sphinx; documentation tool\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 3
    assert index[0] == ('D', [('docutils', [[], [('see also reStructuredText', [])], None])])
    assert index[1] == ('P', [('Python', [[], [('see also interpreter', [])], None])])
    assert index[2] == ('S', [('Sphinx', [[], [('see also documentation tool', [])], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_main_index(app):
    if False:
        return 10
    text = '.. index:: !docutils\n.. index:: docutils\n.. index:: pip; install\n.. index:: !pip; install\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 2
    assert index[0] == ('D', [('docutils', [[('main', '#index-0'), ('', '#index-1')], [], None])])
    assert index[1] == ('P', [('pip', [[], [('install', [('main', '#index-3'), ('', '#index-2')])], None])])

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_index_with_name(app):
    if False:
        for i in range(10):
            print('nop')
    text = '.. index:: single: docutils\n   :name: ref1\n.. index:: single: Python\n   :name: ref2\n.. index:: Sphinx\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 3
    assert index[0] == ('D', [('docutils', [[('', '#ref1')], [], None])])
    assert index[1] == ('P', [('Python', [[('', '#ref2')], [], None])])
    assert index[2] == ('S', [('Sphinx', [[('', '#index-0')], [], None])])
    std = app.env.get_domain('std')
    assert std.anonlabels['ref1'] == ('index', 'ref1')
    assert std.anonlabels['ref2'] == ('index', 'ref2')

@pytest.mark.sphinx('dummy', freshenv=True)
def test_create_index_by_key(app):
    if False:
        for i in range(10):
            print('nop')
    text = '.. glossary::\n\n   docutils\n   Python\n   スフィンクス : ス\n'
    restructuredtext.parse(app, text)
    index = IndexEntries(app.env).create_index(app.builder)
    assert len(index) == 3
    assert index[0] == ('D', [('docutils', [[('main', '#term-docutils')], [], None])])
    assert index[1] == ('P', [('Python', [[('main', '#term-Python')], [], None])])
    assert index[2] == ('ス', [('スフィンクス', [[('main', '#term-0')], [], 'ス'])])