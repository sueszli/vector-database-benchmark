"""Tests for inline blocks layout."""
from ..testing_utils import assert_no_logs, render_pages

@assert_no_logs
def test_inline_block_sizes():
    if False:
        return 10
    (page,) = render_pages('\n      <style>\n        @page { margin: 0; size: 200px 2000px }\n        body { margin: 0 }\n        div { display: inline-block; }\n      </style>\n      <div> </div>\n      <div>a</div>\n      <div style="margin: 10px; height: 100px"></div>\n      <div style="margin-left: 10px; margin-top: -50px;\n                  padding-right: 20px;"></div>\n      <div>\n        Ipsum dolor sit amet,\n        consectetur adipiscing elit.\n        Sed sollicitudin nibh\n        et turpis molestie tristique.\n      </div>\n      <div style="width: 100px; height: 100px;\n                  padding-left: 10px; margin-right: 10px;\n                  margin-top: -10px; margin-bottom: 50px"></div>\n      <div style="font-size: 0">\n        <div style="min-width: 10px; height: 10px"></div>\n        <div style="width: 10%">\n          <div style="width: 10px; height: 10px"></div>\n        </div>\n      </div>\n      <div style="min-width: 150px">foo</div>\n      <div style="max-width: 10px\n        ">Supercalifragilisticexpialidocious</div>')
    (html,) = page.children
    assert html.element_tag == 'html'
    (body,) = html.children
    assert body.element_tag == 'body'
    assert body.width == 200
    (line_1, line_2, line_3, line_4) = body.children
    (div_1, _, div_2, _, div_3, _, div_4, _) = line_1.children
    assert div_1.element_tag == 'div'
    assert div_1.width == 0
    assert div_2.element_tag == 'div'
    assert 0 < div_2.width < 20
    assert div_3.element_tag == 'div'
    assert div_3.width == 0
    assert div_3.margin_width() == 20
    assert div_3.height == 100
    assert div_4.element_tag == 'div'
    assert div_4.width == 0
    assert div_4.margin_width() == 30
    (div_5, _) = line_2.children
    assert div_5.element_tag == 'div'
    assert len(div_5.children) > 1
    assert div_5.width == 200
    (div_6, _, div_7, _) = line_3.children
    assert div_6.element_tag == 'div'
    assert div_6.width == 100
    assert div_6.margin_width() == 120
    assert div_6.height == 100
    assert div_6.margin_height() == 140
    assert div_7.element_tag == 'div'
    assert div_7.width == 20
    (child_line,) = div_7.children
    (child_div_1, child_div_2) = child_line.children
    assert child_div_1.element_tag == 'div'
    assert child_div_1.width == 10
    assert child_div_2.element_tag == 'div'
    assert child_div_2.width == 2
    (grandchild,) = child_div_2.children
    assert grandchild.element_tag == 'div'
    assert grandchild.width == 10
    (div_8, _, div_9) = line_4.children
    assert div_8.width == 150
    assert div_9.width == 10

@assert_no_logs
def test_inline_block_with_margin():
    if False:
        print('Hello World!')
    (page_1,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        @page { size: 100px }\n        span { font-family: weasyprint; display: inline-block; margin: 0 30px }\n      </style>\n      <span>a b c d e f g h i j k l</span>')
    (html,) = page_1.children
    (body,) = html.children
    (line_1,) = body.children
    (span,) = line_1.children
    assert span.width == 40