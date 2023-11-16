"""Test that the "before layout" box tree is correctly constructed."""
import pytest
from weasyprint.css import PageType, get_all_computed_styles
from weasyprint.formatting_structure import boxes, build
from weasyprint.layout.page import set_page_type_computed_styles
from .testing_utils import FakeHTML, assert_no_logs, assert_tree, capture_logs, parse, parse_all, render_pages

def _get_grid(html):
    if False:
        i = 10
        return i + 15
    html = parse_all(html)
    (body,) = html.children
    (table_wrapper,) = body.children
    (table,) = table_wrapper.children
    return tuple(([[(style, width, color) if width else None for (_score, (style, width, color)) in column] for column in grid] for grid in table.collapsed_border_grid))

@assert_no_logs
def test_box_tree():
    if False:
        i = 10
        return i + 15
    assert_tree(parse('<p>'), [('p', 'Block', [])])
    assert_tree(parse('\n      <style>\n        span { display: inline-block }\n      </style>\n      <p>Hello <em>World <img src="pattern.png"><span>L</span></em>!</p>'), [('p', 'Block', [('p', 'Text', 'Hello '), ('em', 'Inline', [('em', 'Text', 'World '), ('img', 'InlineReplaced', '<replaced>'), ('span', 'InlineBlock', [('span', 'Text', 'L')])]), ('p', 'Text', '!')])])

@assert_no_logs
def test_html_entities():
    if False:
        for i in range(10):
            print('nop')
    for quote in ['"', '&quot;', '&#x22;', '&#34;']:
        assert_tree(parse('<p>{0}abc{1}'.format(quote, quote)), [('p', 'Block', [('p', 'Text', '"abc"')])])

@assert_no_logs
def test_inline_in_block_1():
    if False:
        i = 10
        return i + 15
    source = '<div>Hello, <em>World</em>!\n<p>Lipsum.</p></div>'
    expected = [('div', 'Block', [('div', 'Block', [('div', 'Line', [('div', 'Text', 'Hello, '), ('em', 'Inline', [('em', 'Text', 'World')]), ('div', 'Text', '! ')])]), ('p', 'Block', [('p', 'Line', [('p', 'Text', 'Lipsum.')])])])]
    box = parse(source)
    box = build.inline_in_block(box)
    assert_tree(box, expected)

@assert_no_logs
def test_inline_in_block_2():
    if False:
        while True:
            i = 10
    source = '<div><p>Lipsum.</p>Hello, <em>World</em>!\n</div>'
    expected = [('div', 'Block', [('p', 'Block', [('p', 'Line', [('p', 'Text', 'Lipsum.')])]), ('div', 'Block', [('div', 'Line', [('div', 'Text', 'Hello, '), ('em', 'Inline', [('em', 'Text', 'World')]), ('div', 'Text', '! ')])])])]
    box = parse(source)
    box = build.inline_in_block(box)
    assert_tree(box, expected)

@assert_no_logs
def test_inline_in_block_3():
    if False:
        print('Hello World!')
    source = '<p>Hello <em style="position:absolute;\n                                    display: block">World</em>!</p>'
    expected = [('p', 'Block', [('p', 'Line', [('p', 'Text', 'Hello '), ('em', 'Block', [('em', 'Line', [('em', 'Text', 'World')])]), ('p', 'Text', '!')])])]
    box = parse(source)
    box = build.inline_in_block(box)
    assert_tree(box, expected)
    box = build.block_in_inline(box)
    assert_tree(box, expected)

@assert_no_logs
def test_inline_in_block_4():
    if False:
        while True:
            i = 10
    source = '<p>Hello <em style="float: left">World</em>!</p>'
    box = parse(source)
    box = build.inline_in_block(box)
    box = build.block_in_inline(box)
    assert_tree(box, [('p', 'Block', [('p', 'Line', [('p', 'Text', 'Hello '), ('em', 'Block', [('em', 'Line', [('em', 'Text', 'World')])]), ('p', 'Text', '!')])])])

@assert_no_logs
def test_block_in_inline():
    if False:
        while True:
            i = 10
    box = parse('\n      <style>\n        p { display: inline-block; }\n        span, i { display: block; }\n      </style>\n      <p>Lorem <em>ipsum <strong>dolor <span>sit</span>\n      <span>amet,</span></strong><span><em>conse<i>')
    box = build.inline_in_block(box)
    assert_tree(box, [('body', 'Line', [('p', 'InlineBlock', [('p', 'Line', [('p', 'Text', 'Lorem '), ('em', 'Inline', [('em', 'Text', 'ipsum '), ('strong', 'Inline', [('strong', 'Text', 'dolor '), ('span', 'Block', [('span', 'Line', [('span', 'Text', 'sit')])]), ('strong', 'Text', ' '), ('span', 'Block', [('span', 'Line', [('span', 'Text', 'amet,')])])]), ('span', 'Block', [('span', 'Line', [('em', 'Inline', [('em', 'Text', 'conse'), ('i', 'Block', [])])])])])])])])])
    box = build.block_in_inline(box)
    assert_tree(box, [('body', 'Line', [('p', 'InlineBlock', [('p', 'Block', [('p', 'Line', [('p', 'Text', 'Lorem '), ('em', 'Inline', [('em', 'Text', 'ipsum '), ('strong', 'Inline', [('strong', 'Text', 'dolor ')])])])]), ('span', 'Block', [('span', 'Line', [('span', 'Text', 'sit')])]), ('p', 'Block', [('p', 'Line', [('em', 'Inline', [('strong', 'Inline', [('strong', 'Text', ' ')])])])]), ('span', 'Block', [('span', 'Line', [('span', 'Text', 'amet,')])]), ('p', 'Block', [('p', 'Line', [('em', 'Inline', [('strong', 'Inline', [])])])]), ('span', 'Block', [('span', 'Block', [('span', 'Line', [('em', 'Inline', [('em', 'Text', 'conse')])])]), ('i', 'Block', []), ('span', 'Block', [('span', 'Line', [('em', 'Inline', [])])])]), ('p', 'Block', [('p', 'Line', [('em', 'Inline', [])])])])])])

@assert_no_logs
def test_styles():
    if False:
        while True:
            i = 10
    box = parse('\n      <style>\n        span { display: block; }\n        * { margin: 42px }\n        html { color: blue }\n      </style>\n      <p>Lorem <em>ipsum <strong>dolor <span>sit</span>\n        <span>amet,</span></strong><span>consectetur</span></em></p>')
    box = build.inline_in_block(box)
    box = build.block_in_inline(box)
    descendants = list(box.descendants())
    assert len(descendants) == 31
    assert descendants[0] == box
    for child in descendants:
        assert child.style['color'] == (0, 0, 1, 1)
        assert child.style['margin_top'] in ((0, 'px'), (42, 'px'))

@assert_no_logs
def test_whitespace():
    if False:
        for i in range(10):
            print('nop')
    assert_tree(parse_all('\n      <p>Lorem \t\r\n  ipsum\t<strong>  dolor\n        <img src=pattern.png> sit\n        <span style="position: absolute"></span> <em> amet </em>\n        consectetur</strong>.</p>\n      <pre>\t  foo\n</pre>\n      <pre style="white-space: pre-wrap">\t  foo\n</pre>\n      <pre style="white-space: pre-line">\t  foo\n</pre>\n    '), [('p', 'Block', [('p', 'Line', [('p', 'Text', 'Lorem ipsum '), ('strong', 'Inline', [('strong', 'Text', 'dolor '), ('img', 'InlineReplaced', '<replaced>'), ('strong', 'Text', ' sit '), ('span', 'Block', []), ('em', 'Inline', [('em', 'Text', 'amet ')]), ('strong', 'Text', 'consectetur')]), ('p', 'Text', '.')])]), ('pre', 'Block', [('pre', 'Line', [('pre', 'Text', '\t  foo\n')])]), ('pre', 'Block', [('pre', 'Line', [('pre', 'Text', '\t  foo\n')])]), ('pre', 'Block', [('pre', 'Line', [('pre', 'Text', ' foo\n')])])])

@assert_no_logs
@pytest.mark.parametrize('page_type, top, right, bottom, left', ((PageType(side='left', first=True, index=0, blank=None, name=None), 20, 3, 3, 10), (PageType(side='right', first=True, index=0, blank=None, name=None), 20, 10, 3, 3), (PageType(side='left', first=None, index=1, blank=None, name=None), 10, 3, 3, 10), (PageType(side='right', first=None, index=1, blank=None, name=None), 10, 10, 3, 3), (PageType(side='right', first=None, index=1, blank=None, name='name'), 5, 10, 3, 15), (PageType(side='right', first=None, index=2, blank=None, name='name'), 5, 10, 1, 15), (PageType(side='right', first=None, index=8, blank=None, name='name'), 5, 10, 2, 15)))
def test_page_style(page_type, top, right, bottom, left):
    if False:
        print('Hello World!')
    document = FakeHTML(string='\n      <style>\n        @page { margin: 3px }\n        @page name { margin-left: 15px; margin-top: 5px }\n        @page :nth(3) { margin-bottom: 1px }\n        @page :nth(5n+4) { margin-bottom: 2px }\n        @page :first { margin-top: 20px }\n        @page :right { margin-right: 10px; margin-top: 10px }\n        @page :left { margin-left: 10px; margin-top: 10px }\n      </style>\n    ')
    style_for = get_all_computed_styles(document)
    set_page_type_computed_styles(page_type, document, style_for)
    style = style_for(page_type)
    assert style['margin_top'] == (top, 'px')
    assert style['margin_right'] == (right, 'px')
    assert style['margin_bottom'] == (bottom, 'px')
    assert style['margin_left'] == (left, 'px')

@assert_no_logs
def test_images_1():
    if False:
        i = 10
        return i + 15
    with capture_logs() as logs:
        result = parse_all('\n          <p><img src=pattern.png\n            /><img alt="No src"\n            /><img src=inexistent.jpg alt="Inexistent src" /></p>\n        ')
    assert len(logs) == 1
    assert 'ERROR: Failed to load image' in logs[0]
    assert 'inexistent.jpg' in logs[0]
    assert_tree(result, [('p', 'Block', [('p', 'Line', [('img', 'InlineReplaced', '<replaced>'), ('img', 'Inline', [('img', 'Text', 'No src')]), ('img', 'Inline', [('img', 'Text', 'Inexistent src')])])])])

@assert_no_logs
def test_images_2():
    if False:
        print('Hello World!')
    with capture_logs() as logs:
        result = parse_all('<p><img src=pattern.png alt="No base_url">', base_url=None)
    assert len(logs) == 1
    assert 'ERROR: Relative URI reference without a base URI' in logs[0]
    assert_tree(result, [('p', 'Block', [('p', 'Line', [('img', 'Inline', [('img', 'Text', 'No base_url')])])])])

@assert_no_logs
def test_tables_1():
    if False:
        i = 10
        return i + 15
    assert_tree(parse_all('\n      <x-table>\n        <x-tr>\n          <x-th>foo</x-th>\n          <x-th>bar</x-th>\n        </x-tr>\n        <x-tfoot></x-tfoot>\n        <x-thead><x-th></x-th></x-thead>\n        <x-caption style="caption-side: bottom"></x-caption>\n        <x-thead></x-thead>\n        <x-col></x-col>\n        <x-caption>top caption</x-caption>\n        <x-tr>\n          <x-td>baz</x-td>\n        </x-tr>\n      </x-table>\n    '), [('x-table', 'Block', [('x-caption', 'TableCaption', [('x-caption', 'Line', [('x-caption', 'Text', 'top caption')])]), ('x-table', 'Table', [('x-table', 'TableColumnGroup', [('x-col', 'TableColumn', [])]), ('x-thead', 'TableRowGroup', [('x-thead', 'TableRow', [('x-th', 'TableCell', [])])]), ('x-table', 'TableRowGroup', [('x-tr', 'TableRow', [('x-th', 'TableCell', [('x-th', 'Line', [('x-th', 'Text', 'foo')])]), ('x-th', 'TableCell', [('x-th', 'Line', [('x-th', 'Text', 'bar')])])])]), ('x-thead', 'TableRowGroup', []), ('x-table', 'TableRowGroup', [('x-tr', 'TableRow', [('x-td', 'TableCell', [('x-td', 'Line', [('x-td', 'Text', 'baz')])])])]), ('x-tfoot', 'TableRowGroup', [])]), ('x-caption', 'TableCaption', [])])])

@assert_no_logs
def test_tables_2():
    if False:
        i = 10
        return i + 15
    assert_tree(parse_all('\n      <span style="display: table-cell">foo</span>\n      <span style="display: table-cell">bar</span>\n    '), [('body', 'Block', [('body', 'Table', [('body', 'TableRowGroup', [('body', 'TableRow', [('span', 'TableCell', [('span', 'Line', [('span', 'Text', 'foo')])]), ('span', 'TableCell', [('span', 'Line', [('span', 'Text', 'bar')])])])])])])])

@assert_no_logs
def test_tables_3():
    if False:
        return 10
    assert_tree(parse_all('\n      <span style="display: table-column-group">\n        1\n        <em style="display: table-column">\n          2\n          <strong>3</strong>\n        </em>\n        <strong>4</strong>\n      </span>\n      <ins style="display: table-column-group"></ins>\n    '), [('body', 'Block', [('body', 'Table', [('span', 'TableColumnGroup', [('em', 'TableColumn', [])]), ('ins', 'TableColumnGroup', [('ins', 'TableColumn', [])])])])])

@assert_no_logs
def test_tables_4():
    if False:
        for i in range(10):
            print('nop')
    assert_tree(parse_all('<x-table>foo <div></div></x-table>'), [('x-table', 'Block', [('x-table', 'Table', [('x-table', 'TableRowGroup', [('x-table', 'TableRow', [('x-table', 'TableCell', [('x-table', 'Block', [('x-table', 'Line', [('x-table', 'Text', 'foo ')])]), ('div', 'Block', [])])])])])])])

@assert_no_logs
def test_tables_5():
    if False:
        for i in range(10):
            print('nop')
    assert_tree(parse_all('<x-thead style="display: table-header-group"><div></div><x-td></x-td></x-thead>'), [('body', 'Block', [('body', 'Table', [('x-thead', 'TableRowGroup', [('x-thead', 'TableRow', [('x-thead', 'TableCell', [('div', 'Block', [])]), ('x-td', 'TableCell', [])])])])])])

@assert_no_logs
def test_tables_6():
    if False:
        i = 10
        return i + 15
    assert_tree(parse_all('<span><x-tr></x-tr></span>'), [('body', 'Line', [('span', 'Inline', [('span', 'InlineBlock', [('span', 'InlineTable', [('span', 'TableRowGroup', [('x-tr', 'TableRow', [])])])])])])])

@assert_no_logs
def test_tables_7():
    if False:
        print('Hello World!')
    assert_tree(parse_all('\n      <span>\n        <em style="display: table-cell"></em>\n        <em style="display: table-cell"></em>\n      </span>\n    '), [('body', 'Line', [('span', 'Inline', [('span', 'Text', ' '), ('span', 'InlineBlock', [('span', 'InlineTable', [('span', 'TableRowGroup', [('span', 'TableRow', [('em', 'TableCell', []), ('em', 'TableCell', [])])])])]), ('span', 'Text', ' ')])])])

@assert_no_logs
def test_tables_8():
    if False:
        return 10
    assert_tree(parse_all('<x-tr></x-tr>\t<x-tr></x-tr>'), [('body', 'Block', [('body', 'Table', [('body', 'TableRowGroup', [('x-tr', 'TableRow', []), ('x-tr', 'TableRow', [])])])])])

@assert_no_logs
def test_tables_9():
    if False:
        return 10
    assert_tree(parse_all('<x-col></x-col>\n<x-colgroup></x-colgroup>'), [('body', 'Block', [('body', 'Table', [('body', 'TableColumnGroup', [('x-col', 'TableColumn', [])]), ('x-colgroup', 'TableColumnGroup', [('x-colgroup', 'TableColumn', [])])])])])

@assert_no_logs
def test_table_style():
    if False:
        while True:
            i = 10
    html = parse_all('<table style="margin: 1px; padding: 2px"></table>')
    (body,) = html.children
    (wrapper,) = body.children
    (table,) = wrapper.children
    assert isinstance(wrapper, boxes.BlockBox)
    assert isinstance(table, boxes.TableBox)
    assert wrapper.style['margin_top'] == (1, 'px')
    assert wrapper.style['padding_top'] == (0, 'px')
    assert table.style['margin_top'] == (0, 'px')
    assert table.style['padding_top'] == (2, 'px')

@assert_no_logs
def test_column_style():
    if False:
        for i in range(10):
            print('nop')
    html = parse_all('\n      <table>\n        <col span=3 style="width: 10px"></col>\n        <col span=2></col>\n      </table>\n    ')
    (body,) = html.children
    (wrapper,) = body.children
    (table,) = wrapper.children
    (colgroup,) = table.column_groups
    widths = [col.style['width'] for col in colgroup.children]
    assert widths == [(10, 'px'), (10, 'px'), (10, 'px'), 'auto', 'auto']
    assert [col.grid_x for col in colgroup.children] == [0, 1, 2, 3, 4]
    assert colgroup.children[0] is not colgroup.children[1]

@assert_no_logs
def test_nested_grid_x():
    if False:
        for i in range(10):
            print('nop')
    html = parse_all('\n      <table>\n        <col span=2></col>\n        <colgroup span=2></colgroup>\n        <colgroup>\n          <col></col>\n          <col span=2></col>\n        </colgroup>\n        <col></col>\n      </table>\n    ')
    (body,) = html.children
    (wrapper,) = body.children
    (table,) = wrapper.children
    grid = [(colgroup.grid_x, [col.grid_x for col in colgroup.children]) for colgroup in table.column_groups]
    assert grid == [(0, [0, 1]), (2, [2, 3]), (4, [4, 5, 6]), (7, [7])]

@assert_no_logs
def test_colspan_rowspan_1():
    if False:
        for i in range(10):
            print('nop')
    html = parse_all('\n      <table>\n        <tr>\n          <td>A <td>B <td>C\n        </tr>\n        <tr>\n          <td>D <td colspan=2 rowspan=2>E\n        </tr>\n        <tr>\n          <td colspan=2>F <td rowspan=0>G\n        </tr>\n        <tr>\n          <td>H\n        </tr>\n        <tr>\n          <td>I <td>J\n        </tr>\n      </table>\n    ')
    (body,) = html.children
    (wrapper,) = body.children
    (table,) = wrapper.children
    (group,) = table.children
    assert [[c.grid_x for c in row.children] for row in group.children] == [[0, 1, 2], [0, 1], [0, 3], [0], [0, 1]]
    assert [[c.colspan for c in row.children] for row in group.children] == [[1, 1, 1], [1, 2], [2, 1], [1], [1, 1]]
    assert [[c.rowspan for c in row.children] for row in group.children] == [[1, 1, 1], [1, 2], [1, 3], [1], [1, 1]]

@assert_no_logs
def test_colspan_rowspan_2():
    if False:
        while True:
            i = 10
    html = parse_all('\n        <table>\n            <tr>\n                <td rowspan=5></td>\n                <td></td>\n            </tr>\n            <tr>\n                <td></td>\n            </tr>\n        </table>\n    ')
    (body,) = html.children
    (wrapper,) = body.children
    (table,) = wrapper.children
    (group,) = table.children
    assert [[c.grid_x for c in row.children] for row in group.children] == [[0, 1], [1]]
    assert [[c.colspan for c in row.children] for row in group.children] == [[1, 1], [1]]
    assert [[c.rowspan for c in row.children] for row in group.children] == [[2, 1], [1]]

@assert_no_logs
def test_before_after_1():
    if False:
        return 10
    assert_tree(parse_all('\n      <style>\n        p:before { content: normal }\n        div:before { content: none }\n        section::before { color: black }\n      </style>\n      <p></p>\n      <div></div>\n      <section></section>\n    '), [('p', 'Block', []), ('div', 'Block', []), ('section', 'Block', [])])

@assert_no_logs
def test_before_after_2():
    if False:
        for i in range(10):
            print('nop')
    assert_tree(parse_all("\n      <style>\n        p:before { content: 'a' 'b' }\n        p::after { content: 'd' 'e' }\n      </style>\n      <p> c </p>\n    "), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', 'ab')]), ('p', 'Text', ' c '), ('p::after', 'Inline', [('p::after', 'Text', 'de')])])])])

@assert_no_logs
def test_before_after_3():
    if False:
        i = 10
        return i + 15
    assert_tree(parse_all('\n      <style>\n        a[href]:before { content: \'[\' attr(href) \'] \' }\n      </style>\n      <p><a href="some url">some text</a></p>\n    '), [('p', 'Block', [('p', 'Line', [('a', 'Inline', [('a::before', 'Inline', [('a::before', 'Text', '[some url] ')]), ('a', 'Text', 'some text')])])])])

@assert_no_logs
def test_before_after_4():
    if False:
        print('Hello World!')
    assert_tree(parse_all("\n      <style>\n        body { quotes: '«' '»' '“' '”' }\n        q:before { content: open-quote '\xa0'}\n        q:after { content: '\xa0' close-quote }\n      </style>\n      <p><q>Lorem ipsum <q>dolor</q> sit amet</q></p>\n    "), [('p', 'Block', [('p', 'Line', [('q', 'Inline', [('q::before', 'Inline', [('q::before', 'Text', '«\xa0')]), ('q', 'Text', 'Lorem ipsum '), ('q', 'Inline', [('q::before', 'Inline', [('q::before', 'Text', '“\xa0')]), ('q', 'Text', 'dolor'), ('q::after', 'Inline', [('q::after', 'Text', '\xa0”')])]), ('q', 'Text', ' sit amet'), ('q::after', 'Inline', [('q::after', 'Text', '\xa0»')])])])])])

@assert_no_logs
def test_before_after_5():
    if False:
        while True:
            i = 10
    with capture_logs() as logs:
        assert_tree(parse_all("\n          <style>\n            p:before {\n              content: 'a' url(pattern.png) 'b';\n\n              /* Invalid, ignored in favor of the one above.\n                 Regression test: this used to crash: */\n              content: some-function(nested-function(something));\n            }\n          </style>\n          <p>c</p>\n        "), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', 'a'), ('p::before', 'InlineReplaced', '<replaced>'), ('p::before', 'Text', 'b')]), ('p', 'Text', 'c')])])])
    assert len(logs) == 1
    assert 'nested-function(' in logs[0]
    assert 'invalid value' in logs[0]

@assert_no_logs
def test_margin_boxes():
    if False:
        while True:
            i = 10
    (page_1, page_2) = render_pages('\n      <style>\n        @page {\n          /* Make the page content area only 10px high and wide,\n             so every word in <p> end up on a page of its own. */\n          size: 30px;\n          margin: 10px;\n          @top-center { content: "Title" }\n        }\n        @page :first {\n          @bottom-left { content: "foo" }\n          @bottom-left-corner { content: "baz" }\n        }\n      </style>\n      <p>lorem ipsum\n    ')
    assert page_1.children[0].element_tag == 'html'
    assert page_2.children[0].element_tag == 'html'
    margin_boxes_1 = [box.at_keyword for box in page_1.children[1:]]
    margin_boxes_2 = [box.at_keyword for box in page_2.children[1:]]
    assert margin_boxes_1 == ['@top-center', '@bottom-left', '@bottom-left-corner']
    assert margin_boxes_2 == ['@top-center']
    (html, top_center) = page_2.children
    (line_box,) = top_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'Title'

@assert_no_logs
def test_margin_box_string_set_1():
    if False:
        for i in range(10):
            print('nop')
    (page_1, page_2) = render_pages('\n      <style>\n        @page {\n          @bottom-center { content: string(text_header) }\n        }\n        p {\n          string-set: text_header content();\n        }\n        .page {\n          page-break-before: always;\n        }\n      </style>\n      <p>first assignment</p>\n      <div class="page"></div>\n    ')
    (html, bottom_center) = page_2.children
    (line_box,) = bottom_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'first assignment'
    (html, bottom_center) = page_1.children
    (line_box,) = bottom_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'first assignment'

@assert_no_logs
def test_margin_box_string_set_2():
    if False:
        while True:
            i = 10

    def simple_string_set_test(content_val, extra_style=''):
        if False:
            print('Hello World!')
        (page_1,) = render_pages('\n          <style>\n            @page {\n              @top-center { content: string(text_header) }\n            }\n            p {\n              string-set: text_header content(%(content_val)s);\n            }\n            %(extra_style)s\n          </style>\n          <p>first assignment</p>\n        ' % dict(content_val=content_val, extra_style=extra_style))
        (html, top_center) = page_1.children
        (line_box,) = top_center.children
        (text_box,) = line_box.children
        if content_val in ('before', 'after'):
            assert text_box.text == 'pseudo'
        else:
            assert text_box.text == 'first assignment'
    for value in ('', 'text', 'before', 'after'):
        if value in ('before', 'after'):
            extra_style = 'p:%s{content: "pseudo"}' % value
            simple_string_set_test(value, extra_style)
        else:
            simple_string_set_test(value)

@assert_no_logs
def test_margin_box_string_set_3():
    if False:
        while True:
            i = 10
    (page_1,) = render_pages('\n      <style>\n        @page {\n          @top-center { content: string(text_header, first) }\n        }\n        p {\n          string-set: text_header content();\n        }\n      </style>\n      <p>first assignment</p>\n      <p>Second assignment</p>\n    ')
    (html, top_center) = page_1.children
    (line_box,) = top_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'first assignment'

@assert_no_logs
def test_margin_box_string_set_4():
    if False:
        i = 10
        return i + 15
    (page_1, page_2) = render_pages('\n      <style>\n        @page {\n          @top-center { content: string(header_nofirst, first-except) }\n        }\n        p{\n          string-set: header_nofirst content();\n        }\n        .page{\n          page-break-before: always;\n        }\n      </style>\n      <p>first_excepted</p>\n      <div class="page"></div>\n    ')
    (html, top_center) = page_1.children
    assert len(top_center.children) == 0
    (html, top_center) = page_2.children
    (line_box,) = top_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'first_excepted'

@assert_no_logs
def test_margin_box_string_set_5():
    if False:
        for i in range(10):
            print('nop')
    (page_1,) = render_pages('\n      <style>\n        @page {\n          @top-center { content: string(header_last, last) }\n        }\n        p {\n          string-set: header_last content();\n        }\n      </style>\n      <p>String set</p>\n      <p>Second assignment</p>\n    ')
    (html, top_center) = page_1.children[:2]
    (line_box,) = top_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'Second assignment'

@assert_no_logs
def test_margin_box_string_set_6():
    if False:
        i = 10
        return i + 15
    (page_1,) = render_pages('\n      <style>\n        @page {\n          @top-center { content: string(text_header, first) }\n          @bottom-center { content: string(text_footer, last) }\n        }\n        html { counter-reset: a }\n        body { counter-increment: a }\n        ul { counter-reset: b }\n        li {\n          counter-increment: b;\n          string-set:\n            text_header content(before) "-" content() "-" content(after)\n                        counter(a, upper-roman) \'.\' counters(b, \'|\'),\n            text_footer content(before) \'-\' attr(class)\n                        counters(b, \'|\') "/" counter(a, upper-roman);\n        }\n        li:before { content: \'before!\' }\n        li:after { content: \'after!\' }\n        li:last-child:before { content: \'before!last\' }\n        li:last-child:after { content: \'after!last\' }\n      </style>\n      <ul>\n        <li class="firstclass">first\n        <li>\n          <ul>\n            <li class="secondclass">second\n    ')
    (html, top_center, bottom_center) = page_1.children
    (top_line_box,) = top_center.children
    (top_text_box,) = top_line_box.children
    assert top_text_box.text == 'before!-first-after!I.1'
    (bottom_line_box,) = bottom_center.children
    (bottom_text_box,) = bottom_line_box.children
    assert bottom_text_box.text == 'before!last-secondclass2|1/I'

def test_margin_box_string_set_7():
    if False:
        while True:
            i = 10
    (page_1,) = render_pages('\n      <style>\n        img { string-set: left attr(alt) }\n        img + img { string-set: right attr(alt) }\n        @page { @top-left  { content: \'[\' string(left)  \']\' }\n                @top-right { content: \'{\' string(right) \'}\' } }\n      </style>\n      <img src=pattern.png alt="Chocolate">\n      <img src=no_such_file.png alt="Cake">\n    ')
    (html, top_left, top_right) = page_1.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[Chocolate]'
    (right_line_box,) = top_right.children
    (right_text_box,) = right_line_box.children
    assert right_text_box.text == '{Cake}'

@assert_no_logs
def test_margin_box_string_set_8():
    if False:
        print('Hello World!')
    (page_1, page_2, page_3) = render_pages('\n      <style>\n        @page { @top-left  { content: \'[\' string(left) \']\' } }\n        p { page-break-before: always }\n        .initial { string-set: left \'initial\' }\n        .empty   { string-set: left \'\'        }\n        .space   { string-set: left \' \'       }\n      </style>\n\n      <p class="initial">Initial</p>\n      <p class="empty">Empty</p>\n      <p class="space">Space</p>\n    ')
    (html, top_left) = page_1.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[initial]'
    (html, top_left) = page_2.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[]'
    (html, top_left) = page_3.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[ ]'

@assert_no_logs
def test_margin_box_string_set_9():
    if False:
        while True:
            i = 10
    (page_1,) = render_pages("\n      <style>\n        @page {\n          @top-center {\n            content: string(text_header, first)\n                     ' ' string(TEXT_header, first)\n          }\n        }\n        p { string-set: text_header content() }\n        div { string-set: TEXT_header content() }\n      </style>\n      <p>first assignment</p>\n      <div>second assignment</div>\n    ")
    (html, top_center) = page_1.children
    (line_box,) = top_center.children
    (text_box,) = line_box.children
    assert text_box.text == 'first assignment second assignment'

@assert_no_logs
def test_margin_box_string_set_10():
    if False:
        i = 10
        return i + 15
    (page_1, page_2, page_3, page_4) = render_pages("\n      <style>\n        @page { @top-left  { content: '[' string(p, start) ']' } }\n        p { string-set: p content(); page-break-after: always }\n      </style>\n      <article></article>\n      <p>1</p>\n      <article></article>\n      <p>2</p>\n      <p>3</p>\n      <article></article>\n    ")
    (html, top_left) = page_1.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[]'
    (html, top_left) = page_2.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[1]'
    (html, top_left) = page_3.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[3]'
    (html, top_left) = page_4.children
    (left_line_box,) = top_left.children
    (left_text_box,) = left_line_box.children
    assert left_text_box.text == '[3]'

@assert_no_logs
def test_page_counters():
    if False:
        for i in range(10):
            print('nop')
    'Test page-based counters.'
    pages = render_pages('\n      <style>\n        @page {\n          /* Make the page content area only 10px high and wide,\n             so every word in <p> end up on a page of its own. */\n          size: 30px;\n          margin: 10px;\n          @bottom-center {\n            content: "Page\xa0" counter(page) "\xa0of\xa0" counter(pages) ".";\n          }\n        }\n      </style>\n      <p>lorem ipsum dolor\n    ')
    for (page_number, page) in enumerate(pages, 1):
        (html, bottom_center) = page.children
        (line_box,) = bottom_center.children
        (text_box,) = line_box.children
        assert text_box.text == 'Page\xa0{0}\xa0of\xa03.'.format(page_number)
black = (0, 0, 0, 1)
red = (1, 0, 0, 1)
green = (0, 1, 0, 1)
blue = (0, 0, 1, 1)
yellow = (1, 1, 0, 1)
black_3 = ('solid', 3, black)
red_1 = ('solid', 1, red)
yellow_5 = ('solid', 5, yellow)
green_5 = ('solid', 5, green)
dashed_blue_5 = ('dashed', 5, blue)

@assert_no_logs
def test_border_collapse_1():
    if False:
        return 10
    html = parse_all('<table></table>')
    (body,) = html.children
    (table_wrapper,) = body.children
    (table,) = table_wrapper.children
    assert isinstance(table, boxes.TableBox)
    assert not hasattr(table, 'collapsed_border_grid')
    grid = _get_grid('<table style="border-collapse: collapse"></table>')
    assert grid == ([], [])

@assert_no_logs
def test_border_collapse_2():
    if False:
        return 10
    (vertical_borders, horizontal_borders) = _get_grid('\n      <style>td { border: 1px solid red }</style>\n      <table style="border-collapse: collapse; border: 3px solid black">\n        <tr> <td>A</td> <td>B</td> </tr>\n        <tr> <td>C</td> <td>D</td> </tr>\n      </table>\n    ')
    assert vertical_borders == [[black_3, red_1, black_3], [black_3, red_1, black_3]]
    assert horizontal_borders == [[black_3, black_3], [red_1, red_1], [black_3, black_3]]

@assert_no_logs
def test_border_collapse_3():
    if False:
        return 10
    (vertical_borders, horizontal_borders) = _get_grid('\n      <style>table, td { border: 3px solid }</style>\n      <table style="border-collapse: collapse">\n        <tr> <td>A</td> <td style="border-style: hidden">B</td> </tr>\n        <tr> <td>C</td> <td style="border-style: none">D</td> </tr>\n      </table>\n    ')
    assert vertical_borders == [[black_3, None, None], [black_3, black_3, black_3]]
    assert horizontal_borders == [[black_3, None], [black_3, None], [black_3, black_3]]

@assert_no_logs
def test_border_collapse_4():
    if False:
        return 10
    (vertical_borders, horizontal_borders) = _get_grid('\n      <style>td { border: 1px solid red }</style>\n      <table style="border-collapse: collapse; border: 5px solid yellow">\n        <col style="border: 3px solid black" />\n        <tr> <td></td> <td></td> <td></td> </tr>\n        <tr> <td></td> <td style="border: 5px dashed blue"></td>\n          <td style="border: 5px solid lime"></td> </tr>\n        <tr> <td></td> <td></td> <td></td> </tr>\n        <tr> <td></td> <td></td> <td></td> </tr>\n      </table>\n    ')
    assert vertical_borders == [[yellow_5, black_3, red_1, yellow_5], [yellow_5, dashed_blue_5, green_5, green_5], [yellow_5, black_3, red_1, yellow_5], [yellow_5, black_3, red_1, yellow_5]]
    assert horizontal_borders == [[yellow_5, yellow_5, yellow_5], [red_1, dashed_blue_5, green_5], [red_1, dashed_blue_5, green_5], [red_1, red_1, red_1], [yellow_5, yellow_5, yellow_5]]

@assert_no_logs
def test_border_collapse_5():
    if False:
        print('Hello World!')
    (vertical_borders, horizontal_borders) = _get_grid('\n        <style>col, tr { border: 3px solid }</style>\n        <table style="border-collapse: collapse">\n            <col /><col /><col />\n            <tr> <td rowspan=2></td> <td></td> <td></td> </tr>\n            <tr>                     <td colspan=2></td> </tr>\n        </table>\n    ')
    assert vertical_borders == [[black_3, black_3, black_3, black_3], [black_3, black_3, None, black_3]]
    assert horizontal_borders == [[black_3, black_3, black_3], [None, black_3, black_3], [black_3, black_3, black_3]]

@assert_no_logs
@pytest.mark.parametrize('html', ('<html style="display: none">', '<html style="display: none">abc', '<html style="display: none"><p>abc', '<body style="display: none"><p>abc'))
def test_display_none_root(html):
    if False:
        return 10
    box = parse_all(html)
    assert box.style['display'] == ('block', 'flow')
    assert not box.children