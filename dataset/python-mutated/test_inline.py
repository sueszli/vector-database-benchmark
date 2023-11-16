"""Tests for inlines layout."""
import pytest
from weasyprint.formatting_structure import boxes
from ..testing_utils import SANS_FONTS, assert_no_logs, render_pages

@assert_no_logs
def test_empty_linebox():
    if False:
        return 10
    (page,) = render_pages('<p> </p>')
    (html,) = page.children
    (body,) = html.children
    (paragraph,) = body.children
    assert len(paragraph.children) == 0
    assert paragraph.height == 0

@pytest.mark.xfail
@assert_no_logs
def test_empty_linebox_removed_space():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        p { width: 1px }\n      </style>\n      <p><br>  </p>\n    ')
    (page,) = render_pages('<p> </p>')
    (html,) = page.children
    (body,) = html.children
    (paragraph,) = body.children
    assert len(paragraph.children) == 1

@assert_no_logs
def test_breaking_linebox():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style>\n      p { font-size: 13px;\n          width: 300px;\n          font-family: %(fonts)s;\n          background-color: #393939;\n          color: #FFFFFF;\n          line-height: 1;\n          text-decoration: underline overline line-through;}\n      </style>\n      <p><em>Lorem<strong> Ipsum <span>is very</span>simply</strong><em>\n      dummy</em>text of the printing and. naaaa </em> naaaa naaaa naaaa\n      naaaa naaaa naaaa naaaa naaaa</p>\n    ' % {'fonts': SANS_FONTS})
    (html,) = page.children
    (body,) = html.children
    (paragraph,) = body.children
    assert len(list(paragraph.children)) == 3
    lines = paragraph.children
    for line in lines:
        assert line.style['font_size'] == 13
        assert line.element_tag == 'p'
        for child in line.children:
            assert child.element_tag in ('em', 'p')
            assert child.style['font_size'] == 13
            if isinstance(child, boxes.ParentBox):
                for child_child in child.children:
                    assert child.element_tag in ('em', 'strong', 'span')
                    assert child.style['font_size'] == 13

@assert_no_logs
def test_position_x_ltr():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        span {\n          padding: 0 10px 0 15px;\n          margin: 0 2px 0 3px;\n          border: 1px solid;\n         }\n      </style>\n      <body><span>a<br>b<br>c</span>')
    (html,) = page.children
    (body,) = html.children
    (line1, line2, line3) = body.children
    (span1,) = line1.children
    assert span1.position_x == 0
    (text1, br1) = span1.children
    assert text1.position_x == 15 + 3 + 1
    (span2,) = line2.children
    assert span2.position_x == 0
    (text2, br2) = span2.children
    assert text2.position_x == 0
    (span3,) = line3.children
    assert span3.position_x == 0
    (text3,) = span3.children
    assert text3.position_x == 0

@assert_no_logs
def test_position_x_rtl():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        body {\n          direction: rtl;\n          width: 100px;\n        }\n        span {\n          padding: 0 10px 0 15px;\n          margin: 0 2px 0 3px;\n          border: 1px solid;\n         }\n      </style>\n      <body><span>a<br>b<br>c</span>')
    (html,) = page.children
    (body,) = html.children
    (line1, line2, line3) = body.children
    (span1,) = line1.children
    (text1, br1) = span1.children
    assert span1.position_x == 100 - text1.width - (10 + 2 + 1)
    assert text1.position_x == 100 - text1.width - (10 + 2 + 1)
    (span2,) = line2.children
    (text2, br2) = span2.children
    assert span2.position_x == 100 - text2.width
    assert text2.position_x == 100 - text2.width
    (span3,) = line3.children
    (text3,) = span3.children
    assert span3.position_x == 100 - text3.width - (15 + 3 + 1)
    assert text3.position_x == 100 - text3.width

@assert_no_logs
def test_breaking_linebox_regression_1():
    if False:
        return 10
    (page,) = render_pages('<pre>a\nb\rc\r\nd\u2029e</pre>')
    (html,) = page.children
    (body,) = html.children
    (pre,) = body.children
    lines = pre.children
    texts = []
    for line in lines:
        (text_box,) = line.children
        texts.append(text_box.text)
    assert texts == ['a', 'b', 'c', 'd', 'e']

@assert_no_logs
def test_breaking_linebox_regression_2():
    if False:
        while True:
            i = 10
    html_sample = '\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n      </style>\n      <p style="width: %d.5em; font-family: weasyprint">ab\n      <span style="padding-right: 1em; margin-right: 1em">c def</span>g\n      hi</p>'
    for i in range(16):
        (page,) = render_pages(html_sample % i)
        (html,) = page.children
        (body,) = html.children
        (p,) = body.children
        lines = p.children
        if i in (0, 1, 2, 3):
            (line_1, line_2, line_3, line_4) = lines
            (textbox_1,) = line_1.children
            assert textbox_1.text == 'ab'
            (span_1,) = line_2.children
            (textbox_1,) = span_1.children
            assert textbox_1.text == 'c'
            (span_1, textbox_2) = line_3.children
            (textbox_1,) = span_1.children
            assert textbox_1.text == 'def'
            assert textbox_2.text == 'g'
            (textbox_1,) = line_4.children
            assert textbox_1.text == 'hi'
        elif i in (4, 5, 6, 7, 8):
            (line_1, line_2, line_3) = lines
            (textbox_1, span_1) = line_1.children
            assert textbox_1.text == 'ab '
            (textbox_2,) = span_1.children
            assert textbox_2.text == 'c'
            (span_1, textbox_2) = line_2.children
            (textbox_1,) = span_1.children
            assert textbox_1.text == 'def'
            assert textbox_2.text == 'g'
            (textbox_1,) = line_3.children
            assert textbox_1.text == 'hi'
        elif i in (9, 10):
            (line_1, line_2) = lines
            (textbox_1, span_1) = line_1.children
            assert textbox_1.text == 'ab '
            (textbox_2,) = span_1.children
            assert textbox_2.text == 'c'
            (span_1, textbox_2) = line_2.children
            (textbox_1,) = span_1.children
            assert textbox_1.text == 'def'
            assert textbox_2.text == 'g hi'
        elif i in (11, 12, 13):
            (line_1, line_2) = lines
            (textbox_1, span_1, textbox_3) = line_1.children
            assert textbox_1.text == 'ab '
            (textbox_2,) = span_1.children
            assert textbox_2.text == 'c def'
            assert textbox_3.text == 'g'
            (textbox_1,) = line_2.children
            assert textbox_1.text == 'hi'
        else:
            (line_1,) = lines
            (textbox_1, span_1, textbox_3) = line_1.children
            assert textbox_1.text == 'ab '
            (textbox_2,) = span_1.children
            assert textbox_2.text == 'c def'
            assert textbox_3.text == 'g hi'

@assert_no_logs
def test_breaking_linebox_regression_3():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><div style="width: 5.5em; font-family: weasyprint">aaaa aaaa a [<span>aaa</span>]')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (line1, line2, line3, line4) = div.children
    assert line1.children[0].text == line2.children[0].text == 'aaaa'
    assert line3.children[0].text == 'a'
    (text1, span, text2) = line4.children
    assert text1.text == '['
    assert text2.text == ']'
    assert span.children[0].text == 'aaa'

@assert_no_logs
def test_breaking_linebox_regression_4():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><div style="width: 5.5em; font-family: weasyprint">aaaa a <span>b c</span>d')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (line1, line2, line3) = div.children
    assert line1.children[0].text == 'aaaa'
    assert line2.children[0].text == 'a '
    assert line2.children[1].children[0].text == 'b'
    assert line3.children[0].children[0].text == 'c'
    assert line3.children[1].text == 'd'

@assert_no_logs
def test_breaking_linebox_regression_5():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><div style="width: 5.5em; font-family: weasyprint"><span>aaaa aaaa a a a</span><span>bc</span>')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (line1, line2, line3, line4) = div.children
    assert line1.children[0].children[0].text == 'aaaa'
    assert line2.children[0].children[0].text == 'aaaa'
    assert line3.children[0].children[0].text == 'a a'
    assert line4.children[0].children[0].text == 'a'
    assert line4.children[1].children[0].text == 'bc'

@assert_no_logs
def test_breaking_linebox_regression_6():
    if False:
        print('Hello World!')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><div style="width: 5.5em; font-family: weasyprint">a a <span style="white-space: nowrap">/ccc</span>')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (line1, line2) = div.children
    assert line1.children[0].text == 'a a'
    assert line2.children[0].children[0].text == '/ccc'

@assert_no_logs
def test_breaking_linebox_regression_7():
    if False:
        return 10
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><div style="width: 3.5em; font-family: weasyprint"><span><span>abc d e</span></span><span>f')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (line1, line2, line3) = div.children
    assert line1.children[0].children[0].children[0].text == 'abc'
    assert line2.children[0].children[0].children[0].text == 'd'
    assert line3.children[0].children[0].children[0].text == 'e'
    assert line3.children[1].children[0].text == 'f'

@assert_no_logs
def test_breaking_linebox_regression_8():
    if False:
        return 10
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="font-family: weasyprint"><span>\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nbbbbbbbbbbb\n<b>cccc</b></span>ddd</p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2) = p.children
    assert line1.children[0].children[0].text == 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbb'
    assert line2.children[0].children[0].children[0].text == 'cccc'
    assert line2.children[1].text == 'ddd'

@pytest.mark.xfail
@assert_no_logs
def test_breaking_linebox_regression_9():
    if False:
        print('Hello World!')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="font-family: weasyprint"><span>\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbb\n<b>cccc</b></span>ddd</p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2) = p.children
    assert line1.children[0].children[0].text == 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbb'
    assert line2.children[0].children[0].children[0].text == 'cccc'
    assert line2.children[1].text == 'ddd'

@assert_no_logs
def test_breaking_linebox_regression_10():
    if False:
        print('Hello World!')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="width:195px; font-family: weasyprint">  <span>    <span>xxxxxx YYY yyyyyy yyy</span>    ZZZZZZ zzzzz  </span> )x </p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2, line3, line4) = p.children
    assert line1.children[0].children[0].children[0].text == 'xxxxxx YYY'
    assert line2.children[0].children[0].children[0].text == 'yyyyyy yyy'
    assert line3.children[0].children[0].text == 'ZZZZZZ zzzzz'
    assert line4.children[0].text == ')x'

@assert_no_logs
def test_breaking_linebox_regression_11():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="width:10em; font-family: weasyprint">  line 1<br><span>123 567 90</span>x</p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2, line3) = p.children
    assert line1.children[0].text == 'line 1'
    assert line2.children[0].children[0].text == '123 567'
    assert line3.children[0].children[0].text == '90'
    assert line3.children[1].text == 'x'

@assert_no_logs
def test_breaking_linebox_regression_12():
    if False:
        while True:
            i = 10
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="width:10em; font-family: weasyprint">  <br><span>123 567 90</span>x</p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2, line3) = p.children
    assert line2.children[0].children[0].text == '123 567'
    assert line3.children[0].children[0].text == '90'
    assert line3.children[1].text == 'x'

@assert_no_logs
def test_breaking_linebox_regression_13():
    if False:
        while True:
            i = 10
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><p style="width:10em; font-family: weasyprint">  123 567 90 <span>123 567 90</span>x</p>')
    (html,) = page.children
    (body,) = html.children
    (p,) = body.children
    (line1, line2, line3) = p.children
    assert line1.children[0].text == '123 567 90'
    assert line2.children[0].children[0].text == '123 567'
    assert line3.children[0].children[0].text == '90'
    assert line3.children[1].text == 'x'

@assert_no_logs
def test_breaking_linebox_regression_14():
    if False:
        print('Hello World!')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}  body {font-family: weasyprint; width: 3em}</style><span> <span>a</span> b</span><span>c</span>')
    (html,) = page.children
    (body,) = html.children
    (line1, line2) = body.children
    assert line1.children[0].children[0].children[0].text == 'a'
    assert line2.children[0].children[0].text == 'b'
    assert line2.children[1].children[0].text == 'c'

@assert_no_logs
def test_breaking_linebox_regression_15():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}  body {font-family: weasyprint; font-size: 4px}  pre {float: left}</style><pre>ab©\ndéf\nghïj\nklm</pre>')
    (html,) = page.children
    (body,) = html.children
    (pre,) = body.children
    (line1, line2, line3, line4) = pre.children
    assert line1.children[0].text == 'ab©'
    assert line2.children[0].text == 'déf'
    assert line3.children[0].text == 'ghïj'
    assert line4.children[0].text == 'klm'
    assert line1.children[0].width == 4 * 3
    assert line2.children[0].width == 4 * 3
    assert line3.children[0].width == 4 * 4
    assert line4.children[0].width == 4 * 3
    assert pre.width == 4 * 4

@assert_no_logs
def test_breaking_linebox_regression_16():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}  body {font-family: weasyprint; font-size: 4px}  p {float: left}</style><p>tést</p><pre>ab©\ndéf\nghïj\nklm</pre>')
    (html,) = page.children
    (body,) = html.children
    (p, pre) = body.children
    (line1,) = p.children
    assert line1.children[0].text == 'tést'
    assert p.width == 4 * 4
    (line1, line2, line3, line4) = pre.children
    assert line1.children[0].text == 'ab©'
    assert line2.children[0].text == 'déf'
    assert line3.children[0].text == 'ghïj'
    assert line4.children[0].text == 'klm'
    assert line1.children[0].width == 4 * 3
    assert line2.children[0].width == 4 * 3
    assert line3.children[0].width == 4 * 4
    assert line4.children[0].width == 4 * 3

@assert_no_logs
def test_linebox_text():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <style>\n        p { width: 165px; font-family:%(fonts)s;}\n      </style>\n      <p><em>Lorem Ipsum</em>is very <strong>coool</strong></p>\n    ' % {'fonts': SANS_FONTS})
    (html,) = page.children
    (body,) = html.children
    (paragraph,) = body.children
    lines = list(paragraph.children)
    assert len(lines) == 2
    text = ' '.join((''.join((box.text for box in line.descendants() if isinstance(box, boxes.TextBox))) for line in lines))
    assert text == 'Lorem Ipsumis very coool'

@assert_no_logs
def test_linebox_positions():
    if False:
        for i in range(10):
            print('nop')
    for (width, expected_lines) in [(165, 2), (1, 5), (0, 5)]:
        page = '\n          <style>\n            p { width:%(width)spx; font-family:%(fonts)s;\n                line-height: 20px }\n          </style>\n          <p>this is test for <strong>Weasyprint</strong></p>'
        (page,) = render_pages(page % {'fonts': SANS_FONTS, 'width': width})
        (html,) = page.children
        (body,) = html.children
        (paragraph,) = body.children
        lines = list(paragraph.children)
        assert len(lines) == expected_lines
        ref_position_y = lines[0].position_y
        ref_position_x = lines[0].position_x
        for line in lines:
            assert ref_position_y == line.position_y
            assert ref_position_x == line.position_x
            for box in line.children:
                assert ref_position_x == box.position_x
                ref_position_x += box.width
                assert ref_position_y == box.position_y
            assert ref_position_x - line.position_x <= line.width
            ref_position_x = line.position_x
            ref_position_y += line.height

@assert_no_logs
def test_forced_line_breaks_pre():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style> pre { line-height: 42px }</style>\n      <pre>Lorem ipsum dolor sit amet,\n          consectetur adipiscing elit.\n\n\n          Sed sollicitudin nibh\n\n          et turpis molestie tristique.</pre>\n    ')
    (html,) = page.children
    (body,) = html.children
    (pre,) = body.children
    assert pre.element_tag == 'pre'
    lines = pre.children
    assert all((isinstance(line, boxes.LineBox) for line in lines))
    assert len(lines) == 7
    assert [line.height for line in lines] == [42] * 7

@assert_no_logs
def test_forced_line_breaks_paragraph():
    if False:
        return 10
    (page,) = render_pages('\n      <style> p { line-height: 42px }</style>\n      <p>Lorem ipsum dolor sit amet,<br>\n        consectetur adipiscing elit.<br><br><br>\n        Sed sollicitudin nibh<br>\n        <br>\n\n        et turpis molestie tristique.</p>\n    ')
    (html,) = page.children
    (body,) = html.children
    (paragraph,) = body.children
    assert paragraph.element_tag == 'p'
    lines = paragraph.children
    assert all((isinstance(line, boxes.LineBox) for line in lines))
    assert len(lines) == 7
    assert [line.height for line in lines] == [42] * 7

@assert_no_logs
def test_inlinebox_splitting():
    if False:
        print('Hello World!')
    for width in [10000, 100, 10, 0]:
        (page,) = render_pages('\n          <style>p { font-family:%(fonts)s; width: %(width)spx; }</style>\n          <p><strong>WeasyPrint is a frée softwäre ./ visual rendèring enginè\n                     for HTML !!! and CSS.</strong></p>\n        ' % {'fonts': SANS_FONTS, 'width': width})
        (html,) = page.children
        (body,) = html.children
        (paragraph,) = body.children
        lines = paragraph.children
        if width == 10000:
            assert len(lines) == 1
        else:
            assert len(lines) > 1
        text_parts = []
        for line in lines:
            (strong,) = line.children
            (text,) = strong.children
            text_parts.append(text.text)
        assert ' '.join(text_parts) == 'WeasyPrint is a frée softwäre ./ visual rendèring enginè for HTML !!! and CSS.'

@assert_no_logs
def test_whitespace_processing():
    if False:
        for i in range(10):
            print('nop')
    for source in ['a', '  a  ', ' \n  \ta', ' a\t ']:
        (page,) = render_pages('<p><em>%s</em></p>' % source)
        (html,) = page.children
        (body,) = html.children
        (p,) = body.children
        (line,) = p.children
        (em,) = line.children
        (text,) = em.children
        assert text.text == 'a', 'source was %r' % (source,)
        (page,) = render_pages('<p style="white-space: pre-line">\n\n<em>%s</em></pre>' % source.replace('\n', ' '))
        (html,) = page.children
        (body,) = html.children
        (p,) = body.children
        (_line1, _line2, line3) = p.children
        (em,) = line3.children
        (text,) = em.children
        assert text.text == 'a', 'source was %r' % (source,)

@assert_no_logs
def test_inline_replaced_auto_margins():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <style>\n        @page { size: 200px }\n        img { display: inline; margin: auto; width: 50px }\n      </style>\n      <body><img src="pattern.png" />')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (img,) = line.children
    assert img.margin_top == 0
    assert img.margin_right == 0
    assert img.margin_bottom == 0
    assert img.margin_left == 0

@assert_no_logs
def test_empty_inline_auto_margins():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        @page { size: 200px }\n        span { margin: auto }\n      </style>\n      <body><span></span>')
    (html,) = page.children
    (body,) = html.children
    (block,) = body.children
    (span,) = block.children
    assert span.margin_top != 0
    assert span.margin_right == 0
    assert span.margin_bottom != 0
    assert span.margin_left == 0

@assert_no_logs
def test_font_stretch():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style>\n        p { float: left; font-family: %s }\n      </style>\n      <p>Hello, world!</p>\n      <p style="font-stretch: condensed">Hello, world!</p>\n    ' % SANS_FONTS)
    (html,) = page.children
    (body,) = html.children
    (p_1, p_2) = body.children
    normal = p_1.width
    condensed = p_2.width
    assert condensed < normal

@assert_no_logs
@pytest.mark.parametrize('source, lines_count', (('<body>hyphénation', 1), ('<body lang=fr>hyphénation', 1), ('<body style="hyphens: auto">hyphénation', 1), ('<body style="hyphens: auto" lang=fr>hyphénation', 4), ('<body>hyp&shy;hénation', 2), ('<body style="hyphens: none">hyp&shy;hénation', 1)))
def test_line_count(source, lines_count):
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('<html style="width: 5em; font-family: weasyprint"><style>@font-face {  src:url(weasyprint.otf); font-family :weasyprint}</style>' + source)
    (html,) = page.children
    (body,) = html.children
    lines = body.children
    assert len(lines) == lines_count

@assert_no_logs
def test_vertical_align_1():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <span>\n        <img src="pattern.png" style="width: 40px"\n        ><img src="pattern.png" style="width: 60px"\n      ></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_1.position_y == 20
    assert img_2.position_y == 0
    assert 60 < line.height < 70
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_2():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <span>\n        <img src="pattern.png" style="width: 40px; vertical-align: -15px"\n        ><img src="pattern.png" style="width: 60px"></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_1.position_y == 35
    assert img_2.position_y == 0
    assert line.height == 75
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_3():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <span style="line-height: 10px">\n        <img src="pattern.png" style="width: 40px; vertical-align: -150%"\n        ><img src="pattern.png" style="width: 60px"></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_1.position_y == 35
    assert img_2.position_y == 0
    assert line.height == 75
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_4():
    if False:
        return 10
    (page,) = render_pages('\n      <span style="line-height: 10px">\n        <span style="line-height: 10px; vertical-align: -15px">\n          <img src="pattern.png" style="width: 40px"></span>\n        <img src="pattern.png" style="width: 60px"></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span_1,) = line.children
    (span_2, _whitespace, img_2) = span_1.children
    (img_1,) = span_2.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_1.position_y == 35
    assert img_2.position_y == 0
    assert line.height == 75
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_5():
    if False:
        print('Hello World!')
    (page,) = render_pages('<style>  @font-face {src: url(weasyprint.otf); font-family: weasyprint}</style><span style="line-height: 12px; font-size: 12px;             font-family: weasyprint"><img src="pattern.png" style="width: 40px; vertical-align: middle"><img src="pattern.png" style="width: 60px"></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_2.position_y == 0
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_6():
    if False:
        return 10
    (page,) = render_pages('\n      <span style="line-height: 10px">\n        <img src="pattern.png" style="width: 60px"\n        ><img src="pattern.png" style="width: 40px; vertical-align: super"\n        ><img src="pattern.png" style="width: 40px; vertical-align: sub"\n      ></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2, img_3) = span.children
    assert img_1.height == 60
    assert img_2.height == 40
    assert img_3.height == 40
    assert img_1.position_y == 0
    assert img_2.position_y == 12
    assert img_3.position_y == 28
    assert line.height == 68
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_7():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <body style="line-height: 10px">\n        <span>\n          <img src="pattern.png" style="vertical-align: text-top"\n          ><img src="pattern.png" style="vertical-align: text-bottom"\n        ></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.height == 4
    assert img_2.height == 4
    assert img_1.position_y == 0
    assert img_2.position_y == 12
    assert line.height == 16
    assert body.height == line.height

@assert_no_logs
def test_vertical_align_8():
    if False:
        print('Hello World!')
    (page,) = render_pages('<span style="line-height: 1.5">\n      <span style="padding: 1px"></span></span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span_1,) = line.children
    (span_2,) = span_1.children
    assert span_1.height == 16
    assert span_2.height == 16
    assert span_1.margin_height() == 24
    assert span_2.margin_height() == 24
    assert line.height == 24

@assert_no_logs
def test_vertical_align_9():
    if False:
        return 10
    (page,) = render_pages('\n      <span>\n        <img src="pattern.png" style="width: 40px; vertical-align: -15px"\n        ><img src="pattern.png" style="width: 60px"\n      ></span><div style="display: inline-block; vertical-align: 3px">\n        <div>\n          <div style="height: 100px">foo</div>\n          <div>\n            <img src="pattern.png" style="\n                 width: 40px; vertical-align: -15px"\n            ><img src="pattern.png" style="width: 60px"\n          ></div>\n        </div>\n      </div>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span, div_1) = line.children
    assert line.height == 178
    assert body.height == line.height
    (img_1, img_2) = span.children
    assert img_1.height == 40
    assert img_2.height == 60
    assert img_1.position_y == 138
    assert img_2.position_y == 103
    (div_2,) = div_1.children
    (div_3, div_4) = div_2.children
    (div_line,) = div_4.children
    (div_img_1, div_img_2) = div_line.children
    assert div_1.position_y == 0
    assert div_1.height == 175
    assert div_3.height == 100
    assert div_line.height == 75
    assert div_img_1.height == 40
    assert div_img_2.height == 60
    assert div_img_1.position_y == 135
    assert div_img_2.position_y == 100

@assert_no_logs
def test_vertical_align_10():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <span style="font-size: 0">\n        <img src="pattern.png" style="vertical-align: 26px">\n        <img src="pattern.png" style="vertical-align: -10px">\n        <span style="vertical-align: top">\n          <img src="pattern.png" style="vertical-align: -10px">\n          <span style="vertical-align: -10px">\n            <img src="pattern.png" style="vertical-align: bottom">\n          </span>\n        </span>\n        <span style="vertical-align: bottom">\n          <img src="pattern.png" style="vertical-align: 6px">\n        </span>\n      </span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span_1,) = line.children
    (img_1, img_2, span_2, span_4) = span_1.children
    (img_3, span_3) = span_2.children
    (img_4,) = span_3.children
    (img_5,) = span_4.children
    assert body.height == line.height
    assert line.height == 40
    assert img_1.position_y == 0
    assert img_2.position_y == 36
    assert img_3.position_y == 6
    assert img_4.position_y == 36
    assert img_5.position_y == 30

@assert_no_logs
def test_vertical_align_11():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <span style="font-size: 0">\n        <img src="pattern.png" style="vertical-align: bottom">\n        <img src="pattern.png" style="vertical-align: top; height: 100px">\n      </span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1, img_2) = span.children
    assert img_1.position_y == 96
    assert img_2.position_y == 0

@assert_no_logs
def test_vertical_align_12():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <span style="font-size: 0; vertical-align: top">\n        <img src="pattern.png">\n      </span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span,) = line.children
    (img_1,) = span.children
    assert img_1.position_y == 0

@assert_no_logs
def test_vertical_align_13():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <span style="font-size: 0; vertical-align: top; display: inline-block">\n        <img src="pattern.png">\n      </span>')
    (html,) = page.children
    (body,) = html.children
    (line_1,) = body.children
    (span,) = line_1.children
    (line_2,) = span.children
    (img_1,) = line_2.children
    assert img_1.element_tag == 'img'
    assert img_1.position_y == 0

@assert_no_logs
def test_box_decoration_break_inline_slice():
    if False:
        return 10
    (page_1,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        @page { size: 100px }\n        span { font-family: weasyprint; box-decoration-break: slice;\n               padding: 5px; border: 1px solid black }\n      </style>\n      <span>a<br/>b<br/>c</span>')
    (html,) = page_1.children
    (body,) = html.children
    (line_1, line_2, line_3) = body.children
    (span,) = line_1.children
    assert span.width == 16
    assert span.margin_width() == 16 + 5 + 1
    (text, br) = span.children
    assert text.position_x == 5 + 1
    (span,) = line_2.children
    assert span.width == 16
    assert span.margin_width() == 16
    (text, br) = span.children
    assert text.position_x == 0
    (span,) = line_3.children
    assert span.width == 16
    assert span.margin_width() == 16 + 5 + 1
    (text,) = span.children
    assert text.position_x == 0

@assert_no_logs
def test_box_decoration_break_inline_clone():
    if False:
        while True:
            i = 10
    (page_1,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        @page { size: 100px }\n        span { font-size: 12pt; font-family: weasyprint;\n               box-decoration-break: clone;\n               padding: 5px; border: 1px solid black }\n      </style>\n      <span>a<br/>b<br/>c</span>')
    (html,) = page_1.children
    (body,) = html.children
    (line_1, line_2, line_3) = body.children
    (span,) = line_1.children
    assert span.width == 16
    assert span.margin_width() == 16 + 2 * (5 + 1)
    (text, br) = span.children
    assert text.position_x == 5 + 1
    (span,) = line_2.children
    assert span.width == 16
    assert span.margin_width() == 16 + 2 * (5 + 1)
    (text, br) = span.children
    assert text.position_x == 5 + 1
    (span,) = line_3.children
    assert span.width == 16
    assert span.margin_width() == 16 + 2 * (5 + 1)
    (text,) = span.children
    assert text.position_x == 5 + 1