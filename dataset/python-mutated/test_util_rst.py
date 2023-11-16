"""Tests sphinx.util.rst functions."""
from docutils.statemachine import StringList
from jinja2 import Environment
from sphinx.util.rst import append_epilog, escape, heading, prepend_prolog, textwidth

def test_escape():
    if False:
        while True:
            i = 10
    assert escape(':ref:`id`') == '\\:ref\\:\\`id\\`'
    assert escape('footnote [#]_') == 'footnote \\[\\#\\]\\_'
    assert escape('sphinx.application') == 'sphinx.application'
    assert escape('.. toctree::') == '\\.. toctree\\:\\:'

def test_append_epilog(app):
    if False:
        i = 10
        return i + 15
    epilog = 'this is rst_epilog\ngood-bye reST!'
    content = StringList(['hello Sphinx world', 'Sphinx is a document generator'], 'dummy.rst')
    append_epilog(content, epilog)
    assert list(content.xitems()) == [('dummy.rst', 0, 'hello Sphinx world'), ('dummy.rst', 1, 'Sphinx is a document generator'), ('dummy.rst', 2, ''), ('<rst_epilog>', 0, 'this is rst_epilog'), ('<rst_epilog>', 1, 'good-bye reST!')]

def test_prepend_prolog(app):
    if False:
        for i in range(10):
            print('nop')
    prolog = 'this is rst_prolog\nhello reST!'
    content = StringList([':title: test of SphinxFileInput', ':author: Sphinx team', '', 'hello Sphinx world', 'Sphinx is a document generator'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('dummy.rst', 0, ':title: test of SphinxFileInput'), ('dummy.rst', 1, ':author: Sphinx team'), ('<generated>', 0, ''), ('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 2, ''), ('dummy.rst', 3, 'hello Sphinx world'), ('dummy.rst', 4, 'Sphinx is a document generator')]

def test_prepend_prolog_with_CR(app):
    if False:
        while True:
            i = 10
    prolog = 'this is rst_prolog\nhello reST!\n'
    content = StringList(['hello Sphinx world', 'Sphinx is a document generator'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 0, 'hello Sphinx world'), ('dummy.rst', 1, 'Sphinx is a document generator')]

def test_prepend_prolog_without_CR(app):
    if False:
        print('Hello World!')
    prolog = 'this is rst_prolog\nhello reST!'
    content = StringList(['hello Sphinx world', 'Sphinx is a document generator'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 0, 'hello Sphinx world'), ('dummy.rst', 1, 'Sphinx is a document generator')]

def test_prepend_prolog_with_roles_in_sections(app):
    if False:
        i = 10
        return i + 15
    prolog = 'this is rst_prolog\nhello reST!'
    content = StringList([':title: test of SphinxFileInput', ':author: Sphinx team', '', ':mod:`foo`', '----------', '', 'hello'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('dummy.rst', 0, ':title: test of SphinxFileInput'), ('dummy.rst', 1, ':author: Sphinx team'), ('<generated>', 0, ''), ('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 2, ''), ('dummy.rst', 3, ':mod:`foo`'), ('dummy.rst', 4, '----------'), ('dummy.rst', 5, ''), ('dummy.rst', 6, 'hello')]

def test_prepend_prolog_with_roles_in_sections_with_newline(app):
    if False:
        return 10
    prolog = 'this is rst_prolog\nhello reST!\n'
    content = StringList([':mod:`foo`', '-' * 10, '', 'hello'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 0, ':mod:`foo`'), ('dummy.rst', 1, '----------'), ('dummy.rst', 2, ''), ('dummy.rst', 3, 'hello')]

def test_prepend_prolog_with_roles_in_sections_without_newline(app):
    if False:
        return 10
    prolog = 'this is rst_prolog\nhello reST!'
    content = StringList([':mod:`foo`', '-' * 10, '', 'hello'], 'dummy.rst')
    prepend_prolog(content, prolog)
    assert list(content.xitems()) == [('<rst_prolog>', 0, 'this is rst_prolog'), ('<rst_prolog>', 1, 'hello reST!'), ('<generated>', 0, ''), ('dummy.rst', 0, ':mod:`foo`'), ('dummy.rst', 1, '----------'), ('dummy.rst', 2, ''), ('dummy.rst', 3, 'hello')]

def test_textwidth():
    if False:
        while True:
            i = 10
    assert textwidth('Hello') == 5
    assert textwidth('русский язык') == 12
    assert textwidth('русский язык', 'WFA') == 23

def test_heading():
    if False:
        for i in range(10):
            print('nop')
    env = Environment()
    env.extend(language=None)
    assert heading(env, 'Hello') == 'Hello\n====='
    assert heading(env, 'Hello', 1) == 'Hello\n====='
    assert heading(env, 'Hello', 2) == 'Hello\n-----'
    assert heading(env, 'Hello', 3) == 'Hello\n~~~~~'
    assert heading(env, 'русский язык', 1) == 'русский язык\n============'
    env.language = 'ja'
    assert heading(env, 'русский язык', 1) == 'русский язык\n======================='