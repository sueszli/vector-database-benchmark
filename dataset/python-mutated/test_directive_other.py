"""Test the other directives."""
from pathlib import Path
import pytest
from docutils import nodes
from sphinx import addnodes
from sphinx.testing import restructuredtext
from sphinx.testing.util import assert_node

@pytest.mark.sphinx(testroot='toctree-glob')
def test_toctree(app):
    if False:
        while True:
            i = 10
    text = '.. toctree::\n\n   foo\n   bar/index\n   baz\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'foo'), (None, 'bar/index'), (None, 'baz')], includefiles=['foo', 'bar/index', 'baz'])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_relative_toctree(app):
    if False:
        print('Hello World!')
    text = '.. toctree::\n\n   bar_1\n   bar_2\n   bar_3\n   ../quux\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'bar/index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'bar/bar_1'), (None, 'bar/bar_2'), (None, 'bar/bar_3'), (None, 'quux')], includefiles=['bar/bar_1', 'bar/bar_2', 'bar/bar_3', 'quux'])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_toctree_urls_and_titles(app):
    if False:
        for i in range(10):
            print('nop')
    text = '.. toctree::\n\n   Sphinx <https://www.sphinx-doc.org/>\n   https://readthedocs.org/\n   The BAR <bar/index>\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[('Sphinx', 'https://www.sphinx-doc.org/'), (None, 'https://readthedocs.org/'), ('The BAR', 'bar/index')], includefiles=['bar/index'])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_toctree_glob(app):
    if False:
        while True:
            i = 10
    text = '.. toctree::\n   :glob:\n\n   *\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'baz'), (None, 'foo'), (None, 'quux')], includefiles=['baz', 'foo', 'quux'])
    text = '.. toctree::\n   :glob:\n\n   foo\n   *\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'foo'), (None, 'baz'), (None, 'quux')], includefiles=['foo', 'baz', 'quux'])
    text = '.. toctree::\n   :glob:\n\n   *\n   foo\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'baz'), (None, 'foo'), (None, 'quux'), (None, 'foo')], includefiles=['baz', 'foo', 'quux', 'foo'])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_toctree_glob_and_url(app):
    if False:
        return 10
    text = '.. toctree::\n   :glob:\n\n   https://example.com/?q=sphinx\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'https://example.com/?q=sphinx')], includefiles=[])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_reversed_toctree(app):
    if False:
        return 10
    text = '.. toctree::\n   :reversed:\n\n   foo\n   bar/index\n   baz\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'baz'), (None, 'bar/index'), (None, 'foo')], includefiles=['baz', 'bar/index', 'foo'])

@pytest.mark.sphinx(testroot='toctree-glob')
def test_toctree_twice(app):
    if False:
        return 10
    text = '.. toctree::\n\n   foo\n   foo\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, [nodes.document, nodes.compound, addnodes.toctree])
    assert_node(doctree[0][0], entries=[(None, 'foo'), (None, 'foo')], includefiles=['foo', 'foo'])

@pytest.mark.sphinx(testroot='directive-include')
def test_include_include_read_event(app):
    if False:
        for i in range(10):
            print('nop')
    sources_reported = []

    def source_read_handler(_app, relative_path, parent_docname, source):
        if False:
            return 10
        sources_reported.append((relative_path, parent_docname, source[0]))
    app.connect('include-read', source_read_handler)
    text = '.. include:: baz/baz.rst\n   :start-line: 4\n.. include:: text.txt\n   :literal:\n.. include:: bar.txt\n'
    app.env.find_files(app.config, app.builder)
    restructuredtext.parse(app, text, 'index')
    included_files = {filename.as_posix() for (filename, p, s) in sources_reported}
    assert 'index.rst' not in included_files
    assert 'baz/baz.rst' in included_files
    assert 'text.txt' not in included_files
    assert 'bar.txt' in included_files
    assert (Path('baz/baz.rst'), 'index', '\nBaz was here.') in sources_reported

@pytest.mark.sphinx(testroot='directive-include')
def test_include_include_read_event_nested_includes(app):
    if False:
        while True:
            i = 10

    def source_read_handler(_app, _relative_path, _parent_docname, source):
        if False:
            for i in range(10):
                print('nop')
        text = source[0].replace('#magical', 'amazing')
        source[0] = text
    app.connect('include-read', source_read_handler)
    text = '.. include:: baz/baz.rst\n'
    app.env.find_files(app.config, app.builder)
    doctree = restructuredtext.parse(app, text, 'index')
    assert_node(doctree, addnodes.document)
    assert len(doctree.children) == 3
    assert_node(doctree.children[1], nodes.paragraph)
    assert doctree.children[1].rawsource == 'The amazing foo.'