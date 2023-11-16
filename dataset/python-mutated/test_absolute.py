"""Test how absolutes are drawn."""
import pytest
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_absolute_split_1(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        BBBBRRRRRRRR____\n        BBBBRRRRRRRR____\n        BBBBRR__________\n        BBBBRR__________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                left: 0;\n                position: absolute;\n                top: 0;\n                width: 4px;\n            }\n        </style>\n        <div class="split">aa aa</div>\n        <div>bbbbbb bbb</div>\n    ')

@assert_no_logs
def test_absolute_split_2(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRRRRRRBBBB\n        RRRRRRRRRRRRBBBB\n        RRRR________BBBB\n        RRRR________BBBB\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                top: 0;\n                right: 0;\n                width: 4px;\n            }\n        </style>\n        <div class="split">aa aa</div>\n        <div>bbbbbb bb</div>\n    ')

@assert_no_logs
def test_absolute_split_3(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        BBBBRRRRRRRR____\n        BBBBRRRRRRRR____\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                top: 0;\n                left: 0;\n                width: 4px;\n            }\n        </style>\n        <div class="split">aa</div>\n        <div>bbbbbb bbbbb</div>\n    ')

@assert_no_logs
def test_absolute_split_4(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RRRRRRRRRRRRBBBB\n        RRRRRRRRRRRRBBBB\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                top: 0;\n                right: 0;\n                width: 4px;\n            }\n        </style>\n        <div class="split">aa</div>\n        <div>bbbbbb bbbbb</div>\n    ')

@assert_no_logs
def test_absolute_split_5(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        BBBBRRRR____gggg\n        BBBBRRRR____gggg\n        BBBBRRRRRR__gggg\n        BBBBRRRRRR__gggg\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                top: 0;\n                left: 0;\n                width: 4px;\n            }\n            div.split2 {\n                color: green;\n                position: absolute;\n                top: 0;\n                right: 0;\n                width: 4px;\n        </style>\n        <div class="split">aa aa</div>\n        <div class="split2">cc cc</div>\n        <div>bbbb bbbbb</div>\n    ')

@assert_no_logs
def test_absolute_split_6(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        BBBBRRRR____gggg\n        BBBBRRRR____gggg\n        BBBBRRRRRR______\n        BBBBRRRRRR______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                width: 4px;\n            }\n            div.split2 {\n                color: green;\n                position: absolute;\n                top: 0;\n                right: 0;\n                width: 4px;\n        </style>\n        <div class="split">aa aa</div>\n        <div class="split2">cc</div>\n        <div>bbbb bbbbb</div>\n    ')

@assert_no_logs
def test_absolute_split_7(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        BBBBRRRRRRRRgggg\n        BBBBRRRRRRRRgggg\n        ____RRRR____gggg\n        ____RRRR____gggg\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 2px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                width: 4px;\n            }\n            div.split2 {\n                color: green;\n                position: absolute;\n                top: 0;\n                right: 0;\n                width: 4px;\n            }\n            div.push {\n                margin-left: 4px;\n            }\n        </style>\n        <div class="split">aa</div>\n        <div class="split2">cc cc</div>\n        <div class="push">bbbb bb</div>\n    ')

@assert_no_logs
def test_absolute_split_8(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ______\n        ______\n        ______\n        ______\n        __RR__\n        __RR__\n        ______\n        ______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                margin: 2px 0;\n                size: 6px 8px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div {\n                position: absolute;\n                left: 2px;\n                top: 2px;\n                width: 2px;\n            }\n        </style>\n        <div>a a a a</div>\n    ')

@assert_no_logs
def test_absolute_split_9(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        ______\n        ______\n        BBRRBB\n        BBRRBB\n        BBRR__\n        BBRR__\n        ______\n        ______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                margin: 2px 0;\n                size: 6px 8px;\n            }\n            body {\n                color: blue;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div {\n                color: red;\n                position: absolute;\n                left: 2px;\n                top: 0;\n                width: 2px;\n            }\n        </style>\n        aaa a<div>a a a a</div>\n    ')

@assert_no_logs
def test_absolute_split_10(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        BB____\n        BB____\n        __RR__\n        __RR__\n        __RR__\n        __RR__\n\n        BBRR__\n        BBRR__\n        __RR__\n        __RR__\n        ______\n        ______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 6px;\n            }\n            body {\n                color: blue;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div {\n                color: red;\n                position: absolute;\n                left: 2px;\n                top: 2px;\n                width: 2px;\n            }\n            div + article {\n                break-before: page;\n            }\n        </style>\n        <article>a</article>\n        <div>a a a a</div>\n        <article>a</article>\n    ')

@assert_no_logs
def test_absolute_split_11(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        BBBBBB\n        BBBBBB\n        BBRRBB\n        BBRRBB\n        __RR__\n        __RR__\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 6px;\n            }\n            body {\n                color: blue;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div {\n                bottom: 0;\n                color: red;\n                position: absolute;\n                left: 2px;\n                width: 2px;\n            }\n        </style>\n        aaa aaa<div>a a</div>\n    ')

@pytest.mark.xfail
@assert_no_logs
def test_absolute_next_page(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n        BBBBBBRRRR______\n        BBBBBBRRRR______\n        BBBBBB__________\n        ________________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 4px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div.split {\n                color: blue;\n                position: absolute;\n                font-size: 3px;\n            }\n        </style>\n        aaaaa aaaaa\n        <div class="split">bb</div>\n        aaaaa\n    ')

@assert_no_logs
def test_absolute_rtl_1(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        __________RRRRRR\n        __________RRRRRR\n        ________________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 3px;\n            }\n            body {\n                direction: rtl;\n            }\n            div {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                position: absolute;\n            }\n        </style>\n        <div>bbb</div>\n    ')

@assert_no_logs
def test_absolute_rtl_2(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        ________________\n        _________RRRRRR_\n        _________RRRRRR_\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 3px;\n            }\n            body {\n                direction: rtl;\n            }\n            div {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                padding: 1px;\n                position: absolute;\n            }\n        </style>\n        <div>bbb</div>\n    ')

@assert_no_logs
def test_absolute_rtl_3(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        ________________\n        RRRRRR__________\n        RRRRRR__________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 3px;\n            }\n            body {\n                direction: rtl;\n            }\n            div {\n                bottom: 0;\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                left: 0;\n                line-height: 1;\n                position: absolute;\n            }\n        </style>\n        <div>bbb</div>\n    ')

@assert_no_logs
def test_absolute_rtl_4(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ________________\n        _________RRRRRR_\n        _________RRRRRR_\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 3px;\n            }\n            body {\n                direction: rtl;\n            }\n            div {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                position: absolute;\n                right: 1px;\n                top: 1px;\n            }\n        </style>\n        <div>bbb</div>\n    ')

@assert_no_logs
def test_absolute_rtl_5(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RRRRRR__________\n        RRRRRR__________\n        ________________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                size: 16px 3px;\n            }\n            div {\n                color: red;\n                direction: rtl;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                position: absolute;\n            }\n        </style>\n        <div>bbb</div>\n    ')

@assert_no_logs
def test_absolute_pages_counter(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        ______\n        _RR___\n        _RR___\n        _RR___\n        _RR___\n        _____B\n        ______\n        _RR___\n        _RR___\n        _BB___\n        _BB___\n        _____B\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                font-family: weasyprint;\n                margin: 1px;\n                size: 6px 6px;\n                @bottom-right-corner {\n                    color: blue;\n                    content: counter(pages);\n                    font-size: 1px;\n                }\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n            }\n            div {\n                color: blue;\n                position: absolute;\n            }\n        </style>\n        a a a <div>a a</div>\n    ')

@assert_no_logs
def test_absolute_pages_counter_orphans(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        ______\n        _RR___\n        _RR___\n        _RR___\n        _RR___\n        ______\n        ______\n        ______\n        _____B\n        ______\n        _RR___\n        _RR___\n        _BB___\n        _BB___\n        _GG___\n        _GG___\n        ______\n        _____B\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                font-family: weasyprint;\n                margin: 1px;\n                size: 6px 9px;\n                @bottom-right-corner {\n                    color: blue;\n                    content: counter(pages);\n                    font-size: 1px;\n                }\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                orphans: 2;\n                widows: 2;\n            }\n            div {\n                color: blue;\n                position: absolute;\n            }\n            div ~ div {\n                color: lime;\n            }\n        </style>\n        a a a <div>a a a</div> a <div>a a a</div>\n    ')

@assert_no_logs
def test_absolute_in_inline(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ______\n        _GG___\n        _GG___\n        _GG___\n        _GG___\n        ______\n        ______\n        ______\n        ______\n\n        ______\n        _RR___\n        _RR___\n        _RR___\n        _RR___\n        _BB___\n        _BB___\n        ______\n        ______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                margin: 1px;\n                size: 6px 9px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                orphans: 2;\n                widows: 2;\n            }\n            p {\n                color: lime;\n            }\n            div {\n                color: blue;\n                position: absolute;\n            }\n        </style>\n        <p>a a</p> a a <div>a</div>\n    ')

@assert_no_logs
def test_fixed_in_inline(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        ______\n        _GG___\n        _GG___\n        _GG___\n        _GG___\n        _BB___\n        _BB___\n        ______\n        ______\n\n        ______\n        _RR___\n        _RR___\n        _RR___\n        _RR___\n        _BB___\n        _BB___\n        ______\n        ______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n                margin: 1px;\n                size: 6px 9px;\n            }\n            body {\n                color: red;\n                font-family: weasyprint;\n                font-size: 2px;\n                line-height: 1;\n                orphans: 2;\n                widows: 2;\n            }\n            p {\n                color: lime;\n            }\n            div {\n                color: blue;\n                position: fixed;\n            }\n        </style>\n        <p>a a</p> a a <div>a</div>\n    ')

@assert_no_logs
def test_absolute_image_background(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ____\n        _RBB\n        _BBB\n        _BBB\n    ', '\n        <style>\n          @page {\n            size: 4px;\n          }\n          img {\n            background: blue;\n            position: absolute;\n            top: 1px;\n            left: 1px;\n          }\n        </style>\n        <img src="pattern-transparent.svg" />\n    ')