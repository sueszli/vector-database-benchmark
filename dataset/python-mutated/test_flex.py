"""Tests for flex layout."""
import pytest
from ..testing_utils import assert_no_logs, render_pages

@assert_no_logs
def test_flex_direction_row():
    if False:
        return 10
    (page,) = render_pages('\n      <article style="display: flex">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y
    assert div_1.position_x == article.position_x
    assert div_1.position_x < div_2.position_x < div_3.position_x

@assert_no_logs
def test_flex_direction_row_rtl():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <article style="display: flex; direction: rtl">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y
    assert div_1.position_x + div_1.width == article.position_x + article.width
    assert div_1.position_x > div_2.position_x > div_3.position_x

@assert_no_logs
def test_flex_direction_row_reverse():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: row-reverse">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'A'
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y
    assert div_3.position_x + div_3.width == article.position_x + article.width
    assert div_1.position_x < div_2.position_x < div_3.position_x

@assert_no_logs
def test_flex_direction_row_reverse_rtl():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: row-reverse;\n      direction: rtl">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'A'
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y
    assert div_3.position_x == article.position_x
    assert div_1.position_x > div_2.position_x > div_3.position_x

@assert_no_logs
def test_flex_direction_column():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_1.position_y == article.position_y
    assert div_1.position_y < div_2.position_y < div_3.position_y

@assert_no_logs
def test_flex_direction_column_rtl():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column;\n      direction: rtl">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_1.position_y == article.position_y
    assert div_1.position_y < div_2.position_y < div_3.position_y

@assert_no_logs
def test_flex_direction_column_reverse():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column-reverse">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'A'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_3.position_y + div_3.height == article.position_y + article.height
    assert div_1.position_y < div_2.position_y < div_3.position_y

@assert_no_logs
def test_flex_direction_column_reverse_rtl():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column-reverse;\n      direction: rtl">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'A'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_3.position_y + div_3.height == article.position_y + article.height
    assert div_1.position_y < div_2.position_y < div_3.position_y

@assert_no_logs
def test_flex_row_wrap():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <article style="display: flex; flex-flow: wrap; width: 50px">\n        <div style="width: 20px">A</div>\n        <div style="width: 20px">B</div>\n        <div style="width: 20px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_y == div_2.position_y == article.position_y
    assert div_3.position_y == article.position_y + div_2.height
    assert div_1.position_x == div_3.position_x == article.position_x
    assert div_1.position_x < div_2.position_x

@assert_no_logs
def test_flex_column_wrap():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <article style="display: flex; flex-flow: column wrap; height: 50px">\n        <div style="height: 20px">A</div>\n        <div style="height: 20px">B</div>\n        <div style="height: 20px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == div_2.position_x == article.position_x
    assert div_3.position_x == article.position_x + div_2.width
    assert div_1.position_y == div_3.position_y == article.position_y
    assert div_1.position_y < div_2.position_y

@assert_no_logs
def test_flex_row_wrap_reverse():
    if False:
        return 10
    (page,) = render_pages('\n      <article style="display: flex; flex-flow: wrap-reverse; width: 50px">\n        <div style="width: 20px">A</div>\n        <div style="width: 20px">B</div>\n        <div style="width: 20px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'A'
    assert div_3.children[0].children[0].text == 'B'
    assert div_1.position_y == article.position_y
    assert div_2.position_y == div_3.position_y == article.position_y + div_1.height
    assert div_1.position_x == div_2.position_x == article.position_x
    assert div_2.position_x < div_3.position_x

@assert_no_logs
def test_flex_column_wrap_reverse():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <article style="display: flex; flex-flow: column wrap-reverse;\n                      height: 50px">\n        <div style="height: 20px">A</div>\n        <div style="height: 20px">B</div>\n        <div style="height: 20px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'C'
    assert div_2.children[0].children[0].text == 'A'
    assert div_3.children[0].children[0].text == 'B'
    assert div_1.position_x == article.position_x
    assert div_2.position_x == div_3.position_x == article.position_x + div_1.width
    assert div_1.position_y == div_2.position_y == article.position_y
    assert div_2.position_y < div_3.position_y

@assert_no_logs
def test_flex_direction_column_fixed_height_container():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <section style="height: 10px">\n        <article style="display: flex; flex-direction: column">\n          <div>A</div>\n          <div>B</div>\n          <div>C</div>\n        </article>\n      </section>\n    ')
    (html,) = page.children
    (body,) = html.children
    (section,) = body.children
    (article,) = section.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_1.position_y == article.position_y
    assert div_1.position_y < div_2.position_y < div_3.position_y
    assert section.height == 10
    assert article.height > 10

@pytest.mark.xfail
@assert_no_logs
def test_flex_direction_column_fixed_height():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column; height: 10px">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == div_2.position_x == div_3.position_x == article.position_x
    assert div_1.position_y == article.position_y
    assert div_1.position_y < div_2.position_y < div_3.position_y
    assert article.height == 10
    assert div_3.position_y > 10

@assert_no_logs
def test_flex_direction_column_fixed_height_wrap():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex; flex-direction: column; height: 10px;\n                      flex-wrap: wrap">\n        <div>A</div>\n        <div>B</div>\n        <div>C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x != div_2.position_x != div_3.position_x
    assert div_1.position_y == article.position_y
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y
    assert article.height == 10

@assert_no_logs
def test_flex_item_min_width():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <article style="display: flex">\n        <div style="min-width: 30px">A</div>\n        <div style="min-width: 50px">B</div>\n        <div style="min-width: 5px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.position_x == 0
    assert div_1.width == 30
    assert div_2.position_x == 30
    assert div_2.width == 50
    assert div_3.position_x == 80
    assert div_3.width > 5
    assert div_1.position_y == div_2.position_y == div_3.position_y == article.position_y

@assert_no_logs
def test_flex_item_min_height():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <article style="display: flex">\n        <div style="min-height: 30px">A</div>\n        <div style="min-height: 50px">B</div>\n        <div style="min-height: 5px">C</div>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (div_1, div_2, div_3) = article.children
    assert div_1.children[0].children[0].text == 'A'
    assert div_2.children[0].children[0].text == 'B'
    assert div_3.children[0].children[0].text == 'C'
    assert div_1.height == div_2.height == div_3.height == article.height == 50

@assert_no_logs
def test_flex_auto_margin():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('<div style="display: flex; margin: auto">')
    (page,) = render_pages('<div style="display: flex; flex-direction: column; margin: auto">')

@assert_no_logs
def test_flex_no_baseline():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <div class="references" style="display: flex; align-items: baseline;">\n        <div></div>\n      </div>')

@assert_no_logs
@pytest.mark.parametrize('align, height, y1, y2', (('flex-start', 50, 0, 10), ('flex-end', 50, 30, 40), ('space-around', 60, 10, 40), ('space-between', 50, 0, 40), ('space-evenly', 50, 10, 30)))
def test_flex_align_content(align, height, y1, y2):
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        article {\n          align-content: %s;\n          display: flex;\n          flex-wrap: wrap;\n          font-family: weasyprint;\n          font-size: 10px;\n          height: %dpx;\n          line-height: 1;\n        }\n        section {\n          width: 100%%;\n        }\n      </style>\n      <article>\n        <section><span>Lorem</span></section>\n        <section><span>Lorem</span></section>\n      </article>\n    ' % (align, height))
    (html,) = page.children
    (body,) = html.children
    (article,) = body.children
    (section1, section2) = article.children
    (line1,) = section1.children
    (line2,) = section2.children
    (span1,) = line1.children
    (span2,) = line2.children
    assert section1.position_x == span1.position_x == 0
    assert section1.position_y == span1.position_y == y1
    assert section2.position_x == span2.position_x == 0
    assert section2.position_y == span2.position_y == y2

@assert_no_logs
def test_flex_item_percentage():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <div style="display: flex; font-size: 15px; line-height: 1">\n        <div style="height: 100%">a</div>\n      </div>')
    (html,) = page.children
    (body,) = html.children
    (flex,) = body.children
    (flex_item,) = flex.children
    assert flex_item.height == 15

@assert_no_logs
def test_flex_undefined_percentage_height_multiple_lines():
    if False:
        return 10
    (page,) = render_pages('\n      <div style="display: flex; flex-wrap: wrap; height: 100%">\n        <div style="width: 100%">a</div>\n        <div style="width: 100%">b</div>\n      </div>')

@assert_no_logs
def test_flex_absolute():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <div style="display: flex; position: absolute">\n        <div>a</div>\n      </div>')