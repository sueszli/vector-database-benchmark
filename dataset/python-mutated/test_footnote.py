"""Test how footnotes are drawn."""
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_inline_footnote(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 7px;\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        span {\n            float: footnote;\n        }\n    </style>\n    <div>abc<span>de</span></div>')

@assert_no_logs
def test_block_footnote(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 7px;\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n        }\n    </style>\n    <div>abc<div class="footnote">de</div></div>')

@assert_no_logs
def test_long_footnote(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RR_______\n        RR_______\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 7px;\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        span {\n            float: footnote;\n        }\n    </style>\n    <div>abc<span>de f</span></div>')

@assert_no_logs
def test_footnote_margin(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        _________\n        _RRRRRR__\n        _RRRRRR__\n        _________\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 7px;\n\n            @footnote {\n                margin: 1px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        span {\n            float: footnote;\n        }\n    </style>\n    <div>abc<span>d</span></div>')

@assert_no_logs
def test_footnote_with_absolute(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        _RRRR____\n        _RRRR____\n        _________\n        _RRRR____\n        _RRRR____\n        BB_______\n        BB_______\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 7px;\n            margin: 0 1px 2px;\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        span {\n            float: footnote;\n        }\n        mark {\n            display: block;\n            position: absolute;\n            left: -1px;\n            color: blue;\n        }\n    </style>\n    <div>a<span><mark>d</mark></span></div>')

@assert_no_logs
def test_footnote_max_height_1(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_____\n        RRRR_____\n        _BBBBBB__\n        _BBBBBB__\n        _________\n        _________\n        _________\n        _________\n        _BBBBBB__\n        _BBBBBB__\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 6px;\n\n            @footnote {\n                margin-left: 1px;\n                max-height: 3px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n            color: blue;\n        }\n    </style>\n    <div>ab<div class="footnote">c</div><div class="footnote">d</div></div>\n    <div>ef</div>')

@assert_no_logs
def test_footnote_max_height_2(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        _________\n        _BBBBBB__\n        _BBBBBB__\n        _________\n        _________\n        _________\n        _________\n        _BBBBBB__\n        _BBBBBB__\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 6px;\n\n            @footnote {\n                margin-left: 1px;\n                max-height: 3px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n            color: blue;\n        }\n    </style>\n    <div>ab<div class="footnote">c</div><div class="footnote">d</div></div>')

@assert_no_logs
def test_footnote_max_height_3(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _BBBBBB__\n        _________\n        _________\n        _________\n        _________\n        _________\n        _BBBBBB__\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 6px;\n\n            @footnote {\n                margin-left: 1px;\n                max-height: 1px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n            color: blue;\n        }\n    </style>\n    <div>ab<div class="footnote">c</div><div class="footnote">d</div></div>')

@assert_no_logs
def test_footnote_max_height_4(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_____\n        RRRR_____\n        _BBBBBB__\n        _BBBBBB__\n        RRRR_____\n        RRRR_____\n        _________\n        _________\n        _BBBBBB__\n        _BBBBBB__\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 9px 6px;\n\n            @footnote {\n                margin-left: 1px;\n                max-height: 3px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n            color: blue;\n        }\n    </style>\n    <div>ab<div class="footnote">c</div><div class="footnote">d</div></div>\n    <div>ef</div>\n    <div>gh</div>')

@assert_no_logs
def test_footnote_max_height_5(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRR__RR\n        RRRRRRRR__RR\n        _BBBBBB_____\n        _BBBBBB_____\n        _BBBBBB_____\n        _BBBBBB_____\n        RRRR________\n        RRRR________\n        ____________\n        ____________\n        _BBBBBB_____\n        _BBBBBB_____\n    ', '\n    <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n            size: 12px 6px;\n\n            @footnote {\n                margin-left: 1px;\n                max-height: 4px;\n            }\n        }\n        div {\n            color: red;\n            font-family: weasyprint;\n            font-size: 2px;\n            line-height: 1;\n        }\n        div.footnote {\n            float: footnote;\n            color: blue;\n        }\n    </style>\n    <div>ab<div class="footnote">c</div><div class="footnote">d</div>\n    <div class="footnote">e</div></div>\n    <div>fg</div>')