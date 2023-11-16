"""Tests uti.nodes functions."""
from __future__ import annotations
import warnings
from textwrap import dedent
from typing import Any
import pytest
from docutils import frontend, nodes
from docutils.parsers import rst
from docutils.utils import new_document
from sphinx.transforms import ApplySourceWorkaround
from sphinx.util.nodes import NodeMatcher, apply_source_workaround, clean_astext, extract_messages, make_id, split_explicit_title

def _transform(doctree):
    if False:
        for i in range(10):
            print('nop')
    ApplySourceWorkaround(doctree).apply()

def create_new_document():
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        settings = frontend.OptionParser(components=(rst.Parser,)).get_default_values()
    settings.id_prefix = 'id'
    document = new_document('dummy.txt', settings)
    return document

def _get_doctree(text):
    if False:
        return 10
    document = create_new_document()
    rst.Parser().parse(text, document)
    _transform(document)
    return document

def assert_node_count(messages, node_type, expect_count):
    if False:
        return 10
    count = 0
    node_list = [node for (node, msg) in messages]
    for node in node_list:
        if isinstance(node, node_type):
            count += 1
    assert count == expect_count, 'Count of %r in the %r is %d instead of %d' % (node_type, node_list, count, expect_count)

def test_NodeMatcher():
    if False:
        print('Hello World!')
    doctree = nodes.document(None, None)
    doctree += nodes.paragraph('', 'Hello')
    doctree += nodes.paragraph('', 'Sphinx', block=1)
    doctree += nodes.paragraph('', 'World', block=2)
    doctree += nodes.literal_block('', 'blah blah blah', block=3)
    matcher = NodeMatcher(nodes.paragraph)
    assert len(list(doctree.findall(matcher))) == 3
    matcher = NodeMatcher(nodes.paragraph, nodes.literal_block)
    assert len(list(doctree.findall(matcher))) == 4
    matcher = NodeMatcher(block=1)
    assert len(list(doctree.findall(matcher))) == 1
    matcher = NodeMatcher(block=Any)
    assert len(list(doctree.findall(matcher))) == 3
    matcher = NodeMatcher(nodes.paragraph, block=Any)
    assert len(list(doctree.findall(matcher))) == 2
    matcher = NodeMatcher(nodes.title)
    assert len(list(doctree.findall(matcher))) == 0
    matcher = NodeMatcher(blah=Any)
    assert len(list(doctree.findall(matcher))) == 0

@pytest.mark.parametrize(('rst', 'node_cls', 'count'), [('\n           .. admonition:: admonition title\n\n              admonition body\n           ', nodes.title, 1), ('\n           .. figure:: foo.jpg\n\n              this is title\n           ', nodes.caption, 1), ('\n           .. rubric:: spam\n           ', nodes.rubric, 1), ('\n           | spam\n           | egg\n           ', nodes.line, 2), ('\n           section\n           =======\n\n           +----------------+\n           | | **Title 1**  |\n           | | Message 1    |\n           +----------------+\n           ', nodes.line, 2), ('\n           * | **Title 1**\n             | Message 1\n           ', nodes.line, 2)])
def test_extract_messages(rst, node_cls, count):
    if False:
        print('Hello World!')
    msg = extract_messages(_get_doctree(dedent(rst)))
    assert_node_count(msg, node_cls, count)

def test_extract_messages_without_rawsource():
    if False:
        print('Hello World!')
    "\n    Check node.rawsource is fall-backed by using node.astext() value.\n\n    `extract_message` which is used from Sphinx i18n feature drop ``not node.rawsource``\n    nodes. So, all nodes which want to translate must have ``rawsource`` value.\n    However, sometimes node.rawsource is not set.\n\n    For example: recommonmark-0.2.0 doesn't set rawsource to `paragraph` node.\n\n    refs #1994: Fall back to node's astext() during i18n message extraction.\n    "
    p = nodes.paragraph()
    p.append(nodes.Text('test'))
    p.append(nodes.Text('sentence'))
    assert not p.rawsource
    document = create_new_document()
    document.append(p)
    _transform(document)
    assert_node_count(extract_messages(document), nodes.TextElement, 1)
    assert [m for (n, m) in extract_messages(document)][0], 'text sentence'

def test_clean_astext():
    if False:
        for i in range(10):
            print('nop')
    node = nodes.paragraph(text='hello world')
    assert clean_astext(node) == 'hello world'
    node = nodes.image(alt='hello world')
    assert clean_astext(node) == ''
    node = nodes.paragraph(text='hello world')
    node += nodes.raw('', 'raw text', format='html')
    assert clean_astext(node) == 'hello world'

@pytest.mark.parametrize(('prefix', 'term', 'expected'), [('', '', 'id0'), ('term', '', 'term-0'), ('term', 'Sphinx', 'term-Sphinx'), ('', 'io.StringIO', 'io.StringIO'), ('', 'sphinx.setup_command', 'sphinx.setup_command'), ('', '_io.StringIO', 'io.StringIO'), ('', 'ｓｐｈｉｎｘ', 'sphinx'), ('', '悠好', 'id0'), ('', 'Hello=悠好=こんにちは', 'Hello'), ('', 'fünf', 'funf'), ('', '0sphinx', 'sphinx'), ('', 'sphinx-', 'sphinx')])
def test_make_id(app, prefix, term, expected):
    if False:
        for i in range(10):
            print('nop')
    document = create_new_document()
    assert make_id(app.env, document, prefix, term) == expected

def test_make_id_already_registered(app):
    if False:
        print('Hello World!')
    document = create_new_document()
    document.ids['term-Sphinx'] = True
    assert make_id(app.env, document, 'term', 'Sphinx') == 'term-0'

def test_make_id_sequential(app):
    if False:
        while True:
            i = 10
    document = create_new_document()
    document.ids['term-0'] = True
    assert make_id(app.env, document, 'term') == 'term-1'

@pytest.mark.parametrize(('title', 'expected'), [('hello', (False, 'hello', 'hello')), ('hello <world>', (True, 'hello', 'world')), ('hello <world> <sphinx>', (True, 'hello <world>', 'sphinx'))])
def test_split_explicit_target(title, expected):
    if False:
        return 10
    assert expected == split_explicit_title(title)

def test_apply_source_workaround_literal_block_no_source():
    if False:
        for i in range(10):
            print('nop')
    "Regression test for #11091.\n\n     Test that apply_source_workaround doesn't raise.\n     "
    literal_block = nodes.literal_block('', '')
    list_item = nodes.list_item('', literal_block)
    bullet_list = nodes.bullet_list('', list_item)
    assert literal_block.source is None
    assert list_item.source is None
    assert bullet_list.source is None
    apply_source_workaround(literal_block)
    assert literal_block.source is None
    assert list_item.source is None
    assert bullet_list.source is None