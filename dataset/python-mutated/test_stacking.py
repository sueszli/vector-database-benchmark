"""Test CSS stacking contexts."""
import pytest
from weasyprint.stacking import StackingContext
from .testing_utils import assert_no_logs, render_pages, serialize
z_index_source = '\n  <style>\n    @page { size: 10px }\n    div, div * { width: 10px; height: 10px; position: absolute }\n    article { background: red; z-index: %s }\n    section { background: blue; z-index: %s }\n    nav { background: lime; z-index: %s }\n  </style>\n  <div>\n    <article></article>\n    <section></section>\n    <nav></nav>\n  </div>'

def serialize_stacking(context):
    if False:
        print('Hello World!')
    return (context.box.element_tag, [b.element_tag for b in context.blocks_and_cells], [serialize_stacking(c) for c in context.zero_z_contexts])

@assert_no_logs
@pytest.mark.parametrize('source, contexts', (('\n      <p id=lorem></p>\n      <div style="position: relative">\n        <p id=lipsum></p>\n      </div>', ('html', ['body', 'p'], [('div', ['p'], [])])), ('\n      <div style="position: relative">\n        <p style="position: relative"></p>\n      </div>', ('html', ['body'], [('div', [], []), ('p', [], [])]))))
def test_nested(source, contexts):
    if False:
        i = 10
        return i + 15
    (page,) = render_pages(source)
    (html,) = page.children
    assert serialize_stacking(StackingContext.from_box(html, page)) == contexts

@assert_no_logs
def test_image_contexts():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <body>Some text: <img style="position: relative" src=pattern.png>')
    (html,) = page.children
    context = StackingContext.from_box(html, page)
    assert serialize([context.box]) == [('html', 'Block', [('body', 'Block', [('body', 'Line', [('body', 'Text', 'Some text: ')])])])]
    assert serialize((c.box for c in context.zero_z_contexts)) == [('img', 'InlineReplaced', '<replaced>')]

@assert_no_logs
@pytest.mark.parametrize('z_indexes, color', (((3, 2, 1), 'R'), ((1, 2, 3), 'G'), ((1, 2, -3), 'B'), ((1, 2, 'auto'), 'B'), ((-1, 'auto', -2), 'B')))
def test_z_index(assert_pixels, z_indexes, color):
    if False:
        return 10
    assert_pixels('\n'.join([color * 10] * 10), z_index_source % z_indexes)