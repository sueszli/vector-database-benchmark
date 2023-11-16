"""Tests for position property."""
from ..testing_utils import assert_no_logs, render_pages

@assert_no_logs
def test_relative_positioning_1():
    if False:
        return 10
    (page,) = render_pages('\n      <style>\n        p { height: 20px }\n      </style>\n      <p>1</p>\n      <div style="position: relative; top: 10px">\n        <p>2</p>\n        <p style="position: relative; top: -5px; left: 5px">3</p>\n        <p>4</p>\n        <p style="position: relative; bottom: 5px; right: 5px">5</p>\n        <p style="position: relative">6</p>\n        <p>7</p>\n      </div>\n      <p>8</p>\n    ')
    (html,) = page.children
    (body,) = html.children
    (p1, div, p8) = body.children
    (p2, p3, p4, p5, p6, p7) = div.children
    assert (p1.position_x, p1.position_y) == (0, 0)
    assert (div.position_x, div.position_y) == (0, 30)
    assert (p2.position_x, p2.position_y) == (0, 30)
    assert (p3.position_x, p3.position_y) == (5, 45)
    assert (p4.position_x, p4.position_y) == (0, 70)
    assert (p5.position_x, p5.position_y) == (-5, 85)
    assert (p6.position_x, p6.position_y) == (0, 110)
    assert (p7.position_x, p7.position_y) == (0, 130)
    assert (p8.position_x, p8.position_y) == (0, 140)
    assert div.height == 120

@assert_no_logs
def test_relative_positioning_2():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <style>\n        img { width: 20px }\n        body { font-size: 0 } /* Remove spaces */\n      </style>\n      <body>\n      <span><img src=pattern.png></span>\n      <span style="position: relative; left: 10px">\n        <img src=pattern.png>\n        <img src=pattern.png\n             style="position: relative; left: -5px; top: 5px">\n        <img src=pattern.png>\n        <img src=pattern.png\n             style="position: relative; right: 5px; bottom: 5px">\n        <img src=pattern.png style="position: relative">\n        <img src=pattern.png>\n      </span>\n      <span><img src=pattern.png></span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2, span3) = line.children
    (img1,) = span1.children
    (img2, img3, img4, img5, img6, img7) = span2.children
    (img8,) = span3.children
    assert (img1.position_x, img1.position_y) == (0, 0)
    assert span2.position_x == 30
    assert (img2.position_x, img2.position_y) == (30, 0)
    assert (img3.position_x, img3.position_y) == (45, 5)
    assert (img4.position_x, img4.position_y) == (70, 0)
    assert (img5.position_x, img5.position_y) == (85, -5)
    assert (img6.position_x, img6.position_y) == (110, 0)
    assert (img7.position_x, img7.position_y) == (130, 0)
    assert (img8.position_x, img8.position_y) == (140, 0)
    assert span2.width == 120

@assert_no_logs
def test_relative_positioning_3():
    if False:
        return 10
    (page,) = render_pages('\n      <style>\n        img { width: 20px }\n        body { font-size: 0 } /* Remove spaces */\n      </style>\n      <body>\n      <span><img src=pattern.png></span>\n      <span style="position: relative; left: 10px; right: 5px\n        "><img src=pattern.png></span>\n      <span><img src=pattern.png></span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2, span3) = line.children
    assert span2.position_x == 20 + 10

@assert_no_logs
def test_relative_positioning_4():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <style>\n        img { width: 20px }\n        body { direction: rtl; width: 100px;\n               font-size: 0 } /* Remove spaces */\n      </style>\n      <body>\n      <span><img src=pattern.png></span>\n      <span style="position: relative; left: 10px; right: 5px\n        "><img src=pattern.png></span>\n      <span><img src=pattern.png></span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2, span3) = line.children
    assert span2.position_x == 100 - 20 - 5 - 20

@assert_no_logs
def test_absolute_positioning_1():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <div style="margin: 3px">\n        <div style="height: 20px; width: 20px; position: absolute"></div>\n        <div style="height: 20px; width: 20px; position: absolute;\n                    left: 0"></div>\n        <div style="height: 20px; width: 20px; position: absolute;\n                    top: 0"></div>\n      </div>\n    ')
    (html,) = page.children
    (body,) = html.children
    (div1,) = body.children
    (div2, div3, div4) = div1.children
    assert div1.height == 0
    assert (div1.position_x, div1.position_y) == (0, 0)
    assert (div2.width, div2.height) == (20, 20)
    assert (div2.position_x, div2.position_y) == (3, 3)
    assert (div3.width, div3.height) == (20, 20)
    assert (div3.position_x, div3.position_y) == (0, 3)
    assert (div4.width, div4.height) == (20, 20)
    assert (div4.position_x, div4.position_y) == (3, 0)

@assert_no_logs
def test_absolute_positioning_2():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <div style="position: relative; width: 20px">\n        <div style="height: 20px; width: 20px; position: absolute"></div>\n        <div style="height: 20px; width: 20px"></div>\n      </div>\n    ')
    (html,) = page.children
    (body,) = html.children
    (div1,) = body.children
    (div2, div3) = div1.children
    for div in (div1, div2, div3):
        assert (div.position_x, div.position_y) == (0, 0)
        assert (div.width, div.height) == (20, 20)

@assert_no_logs
def test_absolute_positioning_3():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <body style="font-size: 0">\n        <img src=pattern.png>\n        <span style="position: relative">\n          <span style="position: absolute">2</span>\n          <span style="position: absolute">3</span>\n          <span>4</span>\n        </span>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (img, span1) = line.children
    (span2, span3, span4) = span1.children
    assert span1.position_x == 4
    assert (span2.position_x, span2.position_y) == (4, 0)
    assert (span3.position_x, span3.position_y) == (4, 0)
    assert span4.position_x == 4

@assert_no_logs
def test_absolute_positioning_4():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style> img { width: 5px; height: 20px} </style>\n      <body style="font-size: 0">\n        <img src=pattern.png>\n        <span style="position: absolute">2</span>\n        <img src=pattern.png>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (img1, span, img2) = line.children
    assert (img1.position_x, img1.position_y) == (0, 0)
    assert (span.position_x, span.position_y) == (5, 0)
    assert (img2.position_x, img2.position_y) == (5, 0)

@assert_no_logs
def test_absolute_positioning_5():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <style> img { width: 5px; height: 20px} </style>\n      <body style="font-size: 0">\n        <img src=pattern.png>\n        <span style="position: absolute; display: block">2</span>\n        <img src=pattern.png>\n    ')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (img1, span, img2) = line.children
    assert (img1.position_x, img1.position_y) == (0, 0)
    assert (span.position_x, span.position_y) == (0, 20)
    assert (img2.position_x, img2.position_y) == (5, 0)

@assert_no_logs
def test_absolute_positioning_6():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <div style="position: relative; width: 20px; height: 60px;\n                  border: 10px solid; padding-top: 6px; top: 5px; left: 1px">\n        <div style="height: 20px; width: 20px; position: absolute;\n                    bottom: 50%"></div>\n        <div style="height: 20px; width: 20px; position: absolute;\n                    top: 13px"></div>\n      </div>\n    ')
    (html,) = page.children
    (body,) = html.children
    (div1,) = body.children
    (div2, div3) = div1.children
    assert (div1.position_x, div1.position_y) == (1, 5)
    assert (div1.width, div1.height) == (20, 60)
    assert (div1.border_width(), div1.border_height()) == (40, 86)
    assert (div2.position_x, div2.position_y) == (11, 28)
    assert (div2.width, div2.height) == (20, 20)
    assert (div3.position_x, div3.position_y) == (11, 28)
    assert (div3.width, div3.height) == (20, 20)

@assert_no_logs
def test_absolute_positioning_7():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <style>\n        @page { size: 1000px 2000px }\n        html { font-size: 0 }\n        p { height: 20px }\n      </style>\n      <p>1</p>\n      <div style="width: 100px">\n        <p>2</p>\n        <p style="position: absolute; top: -5px; left: 5px">3</p>\n        <p style="margin: 3px">4</p>\n        <p style="position: absolute; bottom: 5px; right: 15px;\n                  width: 50px; height: 10%;\n                  padding: 3px; margin: 7px">5\n          <span>\n            <img src="pattern.png">\n            <span style="position: absolute"></span>\n            <span style="position: absolute; top: -10px; right: 5px;\n                         width: 20px; height: 15px"></span>\n          </span>\n        </p>\n        <p style="margin-top: 8px">6</p>\n      </div>\n      <p>7</p>\n    ')
    (html,) = page.children
    (body,) = html.children
    (p1, div, p7) = body.children
    (p2, p3, p4, p5, p6) = div.children
    (line,) = p5.children
    (span1,) = line.children
    (img, span2, span3) = span1.children
    assert (p1.position_x, p1.position_y) == (0, 0)
    assert (div.position_x, div.position_y) == (0, 20)
    assert (p2.position_x, p2.position_y) == (0, 20)
    assert (p3.position_x, p3.position_y) == (5, -5)
    assert (p4.position_x, p4.position_y) == (0, 40)
    assert (p5.position_x, p5.position_y) == (915, 1775)
    assert (img.position_x, img.position_y) == (925, 1785)
    assert (span2.position_x, span2.position_y) == (929, 1785)
    assert (span3.position_x, span3.position_y) == (953, 1772)
    assert (p6.position_x, p6.position_y) == (0, 63)
    assert div.height == 71
    assert (p7.position_x, p7.position_y) == (0, 91)

@assert_no_logs
def test_absolute_positioning_8():
    if False:
        return 10
    (page,) = render_pages('\n      <style>@page{ width: 50px; height: 50px }</style>\n      <body style="font-size: 0">\n        <div style="position: absolute; margin: auto;\n                    left: 0; right: 10px;\n                    top: 0; bottom: 10px;\n                    width: 10px; height: 20px">\n    ')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    assert (div.content_box_x(), div.content_box_y()) == (15, 10)
    assert (div.width, div.height) == (10, 20)

@assert_no_logs
def test_absolute_images():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n      <style>\n        @page { size: 50px; }\n        img { display: block; position: absolute }\n      </style>\n      <div style="margin: 10px">\n        <img src=pattern.png />\n        <img src=pattern.png style="left: 15px" />\n        <img src=pattern.png style="top: 15px" />\n        <img src=pattern.png style="bottom: 25px" />\n      </div>\n    ')
    (html,) = page.children
    (body,) = html.children
    (div,) = body.children
    (img1, img2, img3, img4) = div.children
    assert div.height == 0
    assert (div.position_x, div.position_y) == (0, 0)
    assert (img1.position_x, img1.position_y) == (10, 10)
    assert (img1.width, img1.height) == (4, 4)
    assert (img2.position_x, img2.position_y) == (15, 10)
    assert (img2.width, img2.height) == (4, 4)
    assert (img3.position_x, img3.position_y) == (10, 15)
    assert (img3.width, img3.height) == (4, 4)
    assert (img4.position_x, img4.position_y) == (10, 21)
    assert (img4.width, img4.height) == (4, 4)

@assert_no_logs
def test_fixed_positioning():
    if False:
        for i in range(10):
            print('nop')
    (page_1, page_2, page_3) = render_pages('\n      a\n      <div style="page-break-before: always; page-break-after: always">\n        <p style="position: fixed">b</p>\n      </div>\n      c\n    ')
    (html,) = page_1.children
    assert [c.element_tag for c in html.children] == ['body', 'p']
    (html,) = page_2.children
    (body,) = html.children
    (div,) = body.children
    assert [c.element_tag for c in div.children] == ['p']
    (html,) = page_3.children
    assert [c.element_tag for c in html.children] == ['p', 'body']

@assert_no_logs
def test_fixed_positioning_regression_1():
    if False:
        for i in range(10):
            print('nop')
    (page_1, page_2) = render_pages('\n      <style>\n        @page:first { size: 100px 200px }\n        @page { size: 200px 100px; margin: 0 }\n        article { break-after: page }\n        .fixed { position: fixed; top: 10px; width: 20px }\n      </style>\n      <ul class="fixed" style="right: 0"><li>a</li></ul>\n      <img class="fixed" style="right: 20px" src="pattern.png" />\n      <div class="fixed" style="right: 40px">b</div>\n      <article>page1</article>\n      <article>page2</article>\n    ')
    (html,) = page_1.children
    (body,) = html.children
    (ul, img, div, article) = body.children
    marker = ul.children[0]
    assert (ul.position_x, ul.position_y) == (80, 10)
    assert (img.position_x, img.position_y) == (60, 10)
    assert (div.position_x, div.position_y) == (40, 10)
    assert (article.position_x, article.position_y) == (0, 0)
    assert marker.position_x == ul.position_x
    (html,) = page_2.children
    (ul, img, div, body) = html.children
    marker = ul.children[0]
    assert (ul.position_x, ul.position_y) == (180, 10)
    assert (img.position_x, img.position_y) == (160, 10)
    assert (div.position_x, div.position_y) == (140, 10)
    assert (article.position_x, article.position_y) == (0, 0)
    assert marker.position_x == ul.position_x

@assert_no_logs
def test_fixed_positioning_regression_2():
    if False:
        while True:
            i = 10
    (page_1, page_2) = render_pages('\n      <style>\n        @page { size: 100px 100px }\n        section { break-after: page }\n        .fixed { position: fixed; top: 10px; left: 15px; width: 20px }\n      </style>\n      <div class="fixed">\n        <article class="fixed" style="top: 20px">\n          <header class="fixed" style="left: 5px"></header>\n        </article>\n      </div>\n      <section></section>\n      <pre></pre>\n    ')
    (html,) = page_1.children
    (body,) = html.children
    (div, section) = body.children
    assert (div.position_x, div.position_y) == (15, 10)
    (article,) = div.children
    assert (article.position_x, article.position_y) == (15, 20)
    (header,) = article.children
    assert (header.position_x, header.position_y) == (5, 10)
    (html,) = page_2.children
    (div, body) = html.children
    assert (div.position_x, div.position_y) == (15, 10)
    (article,) = div.children
    assert (article.position_x, article.position_y) == (15, 20)
    (header,) = article.children
    assert (header.position_x, header.position_y) == (5, 10)