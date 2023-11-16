"""Test CSS counters."""
import pytest
from weasyprint import HTML
from .testing_utils import assert_no_logs, assert_tree, parse_all, render_pages

@assert_no_logs
def test_counters_1():
    if False:
        while True:
            i = 10
    assert_tree(parse_all('\n      <style>\n        p { counter-increment: p 2 }\n        p:before { content: counter(p); }\n        p:nth-child(1) { counter-increment: none; }\n        p:nth-child(2) { counter-increment: p; }\n      </style>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p 117 p"></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p -13"></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p 42"></p>\n      <p></p>\n      <p></p>'), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', counter)])])]) for counter in '0 1 3  2 4 6  -11 -9 -7  44 46 48'.split()])

@assert_no_logs
def test_counters_2():
    if False:
        while True:
            i = 10
    assert_tree(parse_all('\n      <ol style="list-style-position: inside">\n        <li></li>\n        <li></li>\n        <li></li>\n        <li><ol>\n          <li></li>\n          <li style="counter-increment: none"></li>\n          <li></li>\n        </ol></li>\n        <li></li>\n      </ol>'), [('ol', 'Block', [('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '1. ')])])]), ('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '2. ')])])]), ('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '3. ')])])]), ('li', 'Block', [('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '4. ')])])]), ('ol', 'Block', [('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '1. ')])])]), ('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '1. ')])])]), ('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '2. ')])])])])]), ('li', 'Block', [('li', 'Line', [('li::marker', 'Inline', [('li::marker', 'Text', '5. ')])])])])])

@assert_no_logs
def test_counters_3():
    if False:
        print('Hello World!')
    assert_tree(parse_all('\n      <style>\n        p { display: list-item; list-style: inside decimal }\n      </style>\n      <div>\n        <p></p>\n        <p></p>\n        <p style="counter-reset: list-item 7 list-item -56"></p>\n      </div>\n      <p></p>'), [('div', 'Block', [('p', 'Block', [('p', 'Line', [('p::marker', 'Inline', [('p::marker', 'Text', '1. ')])])]), ('p', 'Block', [('p', 'Line', [('p::marker', 'Inline', [('p::marker', 'Text', '2. ')])])]), ('p', 'Block', [('p', 'Line', [('p::marker', 'Inline', [('p::marker', 'Text', '-55. ')])])])]), ('p', 'Block', [('p', 'Line', [('p::marker', 'Inline', [('p::marker', 'Text', '1. ')])])])])

@assert_no_logs
def test_counters_4():
    if False:
        return 10
    assert_tree(parse_all("\n      <style>\n        section:before { counter-reset: h; content: '' }\n        h1:before { counter-increment: h; content: counters(h, '.') }\n      </style>\n      <body>\n        <section><h1></h1>\n          <h1></h1>\n          <section><h1></h1>\n            <h1></h1>\n          </section>\n          <h1></h1>\n        </section>\n      </body>"), [('section', 'Block', [('section', 'Block', [('section', 'Line', [('section::before', 'Inline', [])])]), ('h1', 'Block', [('h1', 'Line', [('h1::before', 'Inline', [('h1::before', 'Text', '1')])])]), ('h1', 'Block', [('h1', 'Line', [('h1::before', 'Inline', [('h1::before', 'Text', '2')])])]), ('section', 'Block', [('section', 'Block', [('section', 'Line', [('section::before', 'Inline', [])])]), ('h1', 'Block', [('h1', 'Line', [('h1::before', 'Inline', [('h1::before', 'Text', '2.1')])])]), ('h1', 'Block', [('h1', 'Line', [('h1::before', 'Inline', [('h1::before', 'Text', '2.2')])])])]), ('h1', 'Block', [('h1', 'Line', [('h1::before', 'Inline', [('h1::before', 'Text', '3')])])])])])

@assert_no_logs
def test_counters_5():
    if False:
        for i in range(10):
            print('nop')
    assert_tree(parse_all('\n      <style>\n        p:before { content: counter(c) }\n      </style>\n      <div>\n        <span style="counter-reset: c">\n          Scope created now, deleted after the div\n        </span>\n      </div>\n      <p></p>'), [('div', 'Block', [('div', 'Line', [('span', 'Inline', [('span', 'Text', 'Scope created now, deleted after the div ')])])]), ('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', '0')])])])])

@assert_no_logs
def test_counters_6():
    if False:
        return 10
    assert_tree(parse_all('\n      <p style="counter-increment: c;\n                display: list-item; list-style: inside decimal">'), [('p', 'Block', [('p', 'Line', [('p::marker', 'Inline', [('p::marker', 'Text', '0. ')])])])])

@assert_no_logs
def test_counters_7():
    if False:
        return 10
    assert_tree(parse_all('\n      <style>\n        p { counter-increment: p 2 }\n        p:before { content: counter(p) \'.\' counter(P); }\n      </style>\n      <p></p>\n      <p style="counter-increment: P 3"></p>\n      <p></p>'), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', counter)])])]) for counter in '2.0 2.3 4.3'.split()])

@assert_no_logs
def test_counters_8():
    if False:
        print('Hello World!')
    assert_tree(parse_all("\n      <style>\n        p:before { content: 'a'; display: list-item }\n      </style>\n      <p></p>\n      <p></p>"), 2 * [('p', 'Block', [('p::before', 'Block', [('p::marker', 'Block', [('p::marker', 'Line', [('p::marker', 'Text', '• ')])]), ('p::before', 'Block', [('p::before', 'Line', [('p::before', 'Text', 'a')])])])])])

@pytest.mark.xfail
@assert_no_logs
def test_counters_9():
    if False:
        while True:
            i = 10
    document = HTML(string='\n      <ol>\n        <li></li>\n        <li>\n          <ol style="counter-reset: a">\n            <li></li>\n            <li></li>\n          </ol>\n        </li>\n        <li></li>\n      </ol>\n    ').render()
    (page,) = document.pages
    (html,) = page._page_box.children
    (body,) = html.children
    (ol1,) = body.children
    (oli1, oli2, oli3) = ol1.children
    (marker, ol2) = oli2.children
    (oli21, oli22) = ol2.children
    assert oli1.children[0].children[0].children[0].text == '1. '
    assert oli2.children[0].children[0].children[0].text == '2. '
    assert oli21.children[0].children[0].children[0].text == '1. '
    assert oli22.children[0].children[0].children[0].text == '2. '
    assert oli3.children[0].children[0].children[0].text == '3. '

@assert_no_logs
def test_counter_styles_1():
    if False:
        i = 10
        return i + 15
    assert_tree(parse_all("\n      <style>\n        body { --var: 'Counter'; counter-reset: p -12 }\n        p { counter-increment: p }\n        p:nth-child(1):before { content: '-' counter(p, none) '-'; }\n        p:nth-child(2):before { content: counter(p, disc); }\n        p:nth-child(3):before { content: counter(p, circle); }\n        p:nth-child(4):before { content: counter(p, square); }\n        p:nth-child(5):before { content: counter(p); }\n        p:nth-child(6):before { content: var(--var) ':' counter(p); }\n        p:nth-child(7):before { content: counter(p) ':' var(--var); }\n      </style>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p></p>\n    "), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', counter)])])]) for counter in '--  •  ◦  ▪  -7 Counter:-6 -5:Counter'.split()])

@assert_no_logs
def test_counter_styles_2():
    if False:
        return 10
    assert_tree(parse_all('\n      <style>\n        p { counter-increment: p }\n        p::before { content: counter(p, decimal-leading-zero); }\n      </style>\n      <p style="counter-reset: p -1987"></p>\n      <p></p>\n      <p style="counter-reset: p -12"></p>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p -2"></p>\n      <p></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p 8"></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p 98"></p>\n      <p></p>\n      <p></p>\n      <p style="counter-reset: p 4134"></p>\n      <p></p>\n    '), [('p', 'Block', [('p', 'Line', [('p::before', 'Inline', [('p::before', 'Text', counter)])])]) for counter in '-1986 -1985  -11 -10 -9 -8  -1 00 01 02  09 10 11\n                            99 100 101  4135 4136'.split()])

@assert_no_logs
def test_counter_styles_3():
    if False:
        return 10
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'decimal-leading-zero') for value in [-1986, -1985, -11, -10, -9, -8, -1, 0, 1, 2, 9, 10, 11, 99, 100, 101, 4135, 4136]] == '\n        -1986 -1985  -11 -10 -9 -8  -1 00 01 02  09 10 11\n        99 100 101  4135 4136\n    '.split()

@assert_no_logs
def test_counter_styles_4():
    if False:
        while True:
            i = 10
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'lower-roman') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 49, 50, 389, 390, 3489, 3490, 3491, 4999, 5000, 5001]] == '\n        -1986 -1985  -1 0 i ii iii iv v vi vii viii ix x xi xii\n        xlix l  ccclxxxix cccxc  mmmcdlxxxix mmmcdxc mmmcdxci\n        4999 5000 5001\n    '.split()

@assert_no_logs
def test_counter_styles_5():
    if False:
        i = 10
        return i + 15
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'upper-roman') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 49, 50, 389, 390, 3489, 3490, 3491, 4999, 5000, 5001]] == '\n        -1986 -1985  -1 0 I II III IV V VI VII VIII IX X XI XII\n        XLIX L  CCCLXXXIX CCCXC  MMMCDLXXXIX MMMCDXC MMMCDXCI\n        4999 5000 5001\n    '.split()

@assert_no_logs
def test_counter_styles_6():
    if False:
        i = 10
        return i + 15
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'lower-alpha') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 2002, 2003]] == '\n        -1986 -1985  -1 0 a b c d  y z aa ab ac bxz bya\n    '.split()

@assert_no_logs
def test_counter_styles_7():
    if False:
        for i in range(10):
            print('nop')
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'upper-alpha') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 2002, 2003]] == '\n        -1986 -1985  -1 0 A B C D  Y Z AA AB AC BXZ BYA\n    '.split()

@assert_no_logs
def test_counter_styles_8():
    if False:
        return 10
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'lower-latin') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 2002, 2003]] == '\n        -1986 -1985  -1 0 a b c d  y z aa ab ac bxz bya\n    '.split()

@assert_no_logs
def test_counter_styles_9():
    if False:
        for i in range(10):
            print('nop')
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'upper-latin') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 2002, 2003]] == '\n        -1986 -1985  -1 0 A B C D  Y Z AA AB AC BXZ BYA\n    '.split()

@assert_no_logs
def test_counter_styles_10():
    if False:
        for i in range(10):
            print('nop')
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'georgian') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 19999, 20000, 20001]] == '\n        -1986 -1985  -1 0 ა\n        ბ გ დ ე ვ ზ ჱ თ ი ია იბ\n        კ ლ მ ნ ჲ ო პ ჟ რ\n        ს ტ ჳ ფ ქ ღ ყ შ ჩ\n        ც ძ წ ჭ ხ ჴ ჯ ჰ ჵ\n        ჵჰშჟთ 20000 20001\n    '.split()

@assert_no_logs
def test_counter_styles_11():
    if False:
        print('Hello World!')
    render = HTML(string='')._ua_counter_style()[0].render_value
    assert [render(value, 'armenian') for value in [-1986, -1985, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999, 10000, 10001]] == '\n        -1986 -1985  -1 0 Ա\n        Բ Գ Դ Ե Զ Է Ը Թ Ժ ԺԱ ԺԲ\n        Ի Լ Խ Ծ Կ Հ Ձ Ղ Ճ\n        Մ Յ Ն Շ Ո Չ Պ Ջ Ռ\n        Ս Վ Տ Ր Ց Ւ Փ Ք\n        ՔՋՂԹ 10000 10001\n    '.split()

@assert_no_logs
@pytest.mark.parametrize('arguments, values', (('cyclic "a" "b" "c"', ('a ', 'b ', 'c ', 'a ')), ('symbolic "a" "b"', ('a ', 'b ', 'aa ', 'bb ')), ('"a" "b"', ('a ', 'b ', 'aa ', 'bb ')), ('alphabetic "a" "b"', ('a ', 'b ', 'aa ', 'ab ')), ('fixed "a" "b"', ('a ', 'b ', '3 ', '4 ')), ('numeric "0" "1" "2"', ('1 ', '2 ', '10 ', '11 '))))
def test_counter_symbols(arguments, values):
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style>\n        ol { list-style-type: symbols(%s) }\n      </style>\n      <ol>\n        <li>abc</li>\n        <li>abc</li>\n        <li>abc</li>\n        <li>abc</li>\n      </ol>\n    ' % arguments)
    (html,) = page.children
    (body,) = html.children
    (ol,) = body.children
    (li_1, li_2, li_3, li_4) = ol.children
    assert li_1.children[0].children[0].children[0].text == values[0]
    assert li_2.children[0].children[0].children[0].text == values[1]
    assert li_3.children[0].children[0].children[0].text == values[2]
    assert li_4.children[0].children[0].children[0].text == values[3]

@assert_no_logs
@pytest.mark.parametrize('style_type, values', (('decimal', ('1. ', '2. ', '3. ', '4. ')), ('"/"', ('/', '/', '/', '/'))))
def test_list_style_types(style_type, values):
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        ol { list-style-type: %s }\n      </style>\n      <ol>\n        <li>abc</li>\n        <li>abc</li>\n        <li>abc</li>\n        <li>abc</li>\n      </ol>\n    ' % style_type)
    (html,) = page.children
    (body,) = html.children
    (ol,) = body.children
    (li_1, li_2, li_3, li_4) = ol.children
    assert li_1.children[0].children[0].children[0].text == values[0]
    assert li_2.children[0].children[0].children[0].text == values[1]
    assert li_3.children[0].children[0].children[0].text == values[2]
    assert li_4.children[0].children[0].children[0].text == values[3]

def test_counter_set():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style>\n        body { counter-reset: h2 0 h3 4; font-size: 1px }\n        article { counter-reset: h2 2 }\n        h1 { counter-increment: h1 }\n        h1::before { content: counter(h1) }\n        h2 { counter-increment: h2; counter-set: h3 3 }\n        h2::before { content: counter(h2) }\n        h3 { counter-increment: h3 }\n        h3::before { content: counter(h3) }\n      </style>\n      <article>\n        <h1></h1>\n      </article>\n      <article>\n        <h2></h2>\n        <h3></h3>\n      </article>\n      <article>\n        <h3></h3>\n      </article>\n      <article>\n        <h2></h2>\n      </article>\n      <article>\n        <h3></h3>\n        <h3></h3>\n      </article>\n      <article>\n        <h1></h1>\n        <h2></h2>\n        <h3></h3>\n      </article>\n    ')
    (html,) = page.children
    (body,) = html.children
    (art_1, art_2, art_3, art_4, art_5, art_6) = body.children
    (h1,) = art_1.children
    assert h1.children[0].children[0].children[0].text == '1'
    (h2, h3) = art_2.children
    assert h2.children[0].children[0].children[0].text == '3'
    assert h3.children[0].children[0].children[0].text == '4'
    (h3,) = art_3.children
    assert h3.children[0].children[0].children[0].text == '5'
    (h2,) = art_4.children
    assert h2.children[0].children[0].children[0].text == '3'
    (h3_1, h3_2) = art_5.children
    assert h3_1.children[0].children[0].children[0].text == '4'
    assert h3_2.children[0].children[0].children[0].text == '5'
    (h1, h2, h3) = art_6.children
    assert h1.children[0].children[0].children[0].text == '1'
    assert h2.children[0].children[0].children[0].text == '3'
    assert h3.children[0].children[0].children[0].text == '4'

def test_counter_multiple_extends():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <style>\n        @counter-style a {\n          system: extends b;\n          prefix: a;\n        }\n        @counter-style b {\n          system: extends c;\n          suffix: b;\n        }\n        @counter-style c {\n          system: extends b;\n          pad: 2 c;\n        }\n        @counter-style d {\n          system: extends d;\n          prefix: d;\n        }\n        @counter-style e {\n          system: extends unknown;\n          prefix: e;\n        }\n        @counter-style f {\n          system: extends decimal;\n          symbols: a;\n        }\n        @counter-style g {\n          system: extends decimal;\n          additive-symbols: 1 a;\n        }\n      </style>\n      <ol>\n        <li style="list-style-type: a"></li>\n        <li style="list-style-type: b"></li>\n        <li style="list-style-type: c"></li>\n        <li style="list-style-type: d"></li>\n        <li style="list-style-type: e"></li>\n        <li style="list-style-type: f"></li>\n        <li style="list-style-type: g"></li>\n        <li style="list-style-type: h"></li>\n      </ol>\n    ')
    (html,) = page.children
    (body,) = html.children
    (ol,) = body.children
    (li_1, li_2, li_3, li_4, li_5, li_6, li_7, li_8) = ol.children
    assert li_1.children[0].children[0].children[0].text == 'a1b'
    assert li_2.children[0].children[0].children[0].text == '2b'
    assert li_3.children[0].children[0].children[0].text == 'c3. '
    assert li_4.children[0].children[0].children[0].text == 'd4. '
    assert li_5.children[0].children[0].children[0].text == 'e5. '
    assert li_6.children[0].children[0].children[0].text == '6. '
    assert li_7.children[0].children[0].children[0].text == '7. '
    assert li_8.children[0].children[0].children[0].text == '8. '