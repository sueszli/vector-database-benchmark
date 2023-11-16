"""Test the fonts features."""
from .testing_utils import assert_no_logs, render_pages

@assert_no_logs
def test_font_face():
    if False:
        print('Hello World!')
    (page,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        body { font-family: weasyprint }\n      </style>\n      <span>abc</span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    assert line.width == 3 * 16

@assert_no_logs
def test_kerning_default():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        body { font-family: weasyprint }\n      </style>\n      <span>kk</span><span>liga</span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2) = line.children
    assert span1.width == 1.5 * 16
    assert span2.width == 1.5 * 16

@assert_no_logs
def test_ligatures_word_space():
    if False:
        for i in range(10):
            print('nop')
    (page,) = render_pages('\n      <style>\n        @font-face { src: url(weasyprint.otf); font-family: weasyprint }\n        body { font-family: weasyprint; word-spacing: 1em; width: 10em }\n      </style>\n      aa liga aa')
    (html,) = page.children
    (body,) = html.children
    assert len(body.children) == 1

@assert_no_logs
def test_kerning_deactivate():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages("\n      <style>\n        @font-face {\n          src: url(weasyprint.otf);\n          font-family: no-kern;\n          font-feature-settings: 'kern' off;\n        }\n        @font-face {\n          src: url(weasyprint.otf);\n          font-family: kern;\n        }\n        span:nth-child(1) { font-family: kern }\n        span:nth-child(2) { font-family: no-kern }\n      </style>\n      <span>kk</span><span>kk</span>")
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2) = line.children
    assert span1.width == 1.5 * 16
    assert span2.width == 2 * 16

@assert_no_logs
def test_kerning_ligature_deactivate():
    if False:
        while True:
            i = 10
    (page,) = render_pages("\n      <style>\n        @font-face {\n          src: url(weasyprint.otf);\n          font-family: no-kern-liga;\n          font-feature-settings: 'kern' off;\n          font-variant: no-common-ligatures;\n        }\n        @font-face {\n          src: url(weasyprint.otf);\n          font-family: kern-liga;\n        }\n        span:nth-child(1) { font-family: kern-liga }\n        span:nth-child(2) { font-family: no-kern-liga }\n      </style>\n      <span>kk liga</span><span>kk liga</span>")
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2) = line.children
    assert span1.width == (1.5 + 1 + 1.5) * 16
    assert span2.width == (2 + 1 + 4) * 16

@assert_no_logs
def test_font_face_descriptors():
    if False:
        i = 10
        return i + 15
    (page,) = render_pages('\n        <style>\n          @font-face {\n            src: url(weasyprint.otf);\n            font-family: weasyprint;\n            font-variant: sub\n                          discretionary-ligatures\n                          oldstyle-nums\n                          slashed-zero;\n          }\n          span { font-family: weasyprint }\n        </style><span>kk</span><span>subs</span><span>dlig</span><span>onum</span><span>zero</span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (kern, subs, dlig, onum, zero) = line.children
    assert kern.width == 1.5 * 16
    assert subs.width == 1.5 * 16
    assert dlig.width == 1.5 * 16
    assert onum.width == 1.5 * 16
    assert zero.width == 1.5 * 16

@assert_no_logs
def test_woff_simple():
    if False:
        while True:
            i = 10
    (page,) = render_pages('\n      <style>\n        @font-face {\n          src: url(weasyprint.otf);\n          font-family: weasyprint-otf;\n        }\n        @font-face {\n          src: url(weasyprint.woff);\n          font-family: weasyprint-woff;\n        }\n        @font-face {\n          src: url(weasyprint.woff);\n          font-family: weasyprint-woff-cached;\n        }\n        span:nth-child(1) { font-family: weasyprint-otf }\n        span:nth-child(2) { font-family: weasyprint-woff }\n        span:nth-child(3) { font-family: weasyprint-woff-cached }\n        span:nth-child(4) { font-family: sans }\n      </style><span>woff font</span><span>woff font</span><span>woff font</span><span>woff font</span>')
    (html,) = page.children
    (body,) = html.children
    (line,) = body.children
    (span1, span2, span3, span4) = line.children
    assert span1.width == span2.width
    assert span1.width == span3.width
    assert span1.width != span4.width