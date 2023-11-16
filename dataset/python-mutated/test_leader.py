"""Test how leaders are drawn."""
import pytest
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_leader_simple(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RR__BBBBBBBB__BB\n        RR__BBBBBBBB__BB\n        RRRR__BBBB__BBBB\n        RRRR__BBBB__BBBB\n        RR__BBBB__BBBBBB\n        RR__BBBB__BBBBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 6px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n      </style>\n      <div>a</div>\n      <div>bb</div>\n      <div>c</div>\n    ")

@assert_no_logs
def test_leader_too_long(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRRRRRRRR______\n        RRRRRRRRRR______\n        BBBBBBBBBBBB__BB\n        BBBBBBBBBBBB__BB\n        RR__RR__RR__RR__\n        RR__RR__RR__RR__\n        RR__RR__RR______\n        RR__RR__RR______\n        BBBBBBBBBB__BBBB\n        BBBBBBBBBB__BBBB\n        RR__RR__RR__RR__\n        RR__RR__RR__RR__\n        RR__BBBB__BBBBBB\n        RR__BBBB__BBBBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 14px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n      </style>\n      <div>aaaaa</div>\n      <div>a a a a a a a</div>\n      <div>a a a a a</div>\n    ")

@assert_no_logs
def test_leader_alone(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRBBBBBBBBBBBBBB\n        RRBBBBBBBBBBBBBB\n    ', '\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader(dotted);\n        }\n      </style>\n      <div>a</div>\n    ')

@assert_no_logs
def test_leader_content(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RR____BB______BB\n        RR____BB______BB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader(' . ') 'a';\n        }\n      </style>\n      <div>a</div>\n    ")

@pytest.mark.xfail
@assert_no_logs
def test_leader_float(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        bbGRR___BB____BB\n        bbGRR___BB____BB\n        GGGRR___BB____BB\n        ___RR___BB____BB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        article {\n          background: lime;\n          color: navy;\n          float: left;\n          height: 3px;\n          width: 3px;\n        }\n        div::after {\n          color: blue;\n          content: leader('. ') 'a';\n        }\n      </style>\n      <div>a<article>a</article></div>\n      <div>a</div>\n    ")

@pytest.mark.xfail
@assert_no_logs
def test_leader_float_small(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        bbRRBB__BB____BB\n        bbRRBB__BB____BB\n        RR__BB__BB____BB\n        RR__BB__BB____BB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        article {\n          background: lime;\n          color: navy;\n          float: left;\n        }\n        div::after {\n          color: blue;\n          content: leader('. ') 'a';\n        }\n      </style>\n      <div>a<article>a</article></div>\n      <div>a</div>\n    ")

@assert_no_logs
def test_leader_in_inline(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RR__GGBBBBBB__RR\n        RR__GGBBBBBB__RR\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        span {\n          color: lime;\n        }\n        span::after {\n          color: blue;\n          content: leader('-');\n        }\n      </style>\n      <div>a <span>a</span> a</div>\n    ")

@pytest.mark.xfail
@assert_no_logs
def test_leader_bad_alignment(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRR__________\n        RRRRRR__________\n        ______BB______RR\n        ______BB______RR\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader(' - ') 'a';\n        }\n      </style>\n      <div>aaa</div>\n    ")

@assert_no_logs
def test_leader_simple_rtl(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        BB__BBBBBBBB__RR\n        BB__BBBBBBBB__RR\n        BBBB__BBBB__RRRR\n        BBBB__BBBB__RRRR\n        BBBBBB__BBBB__RR\n        BBBBBB__BBBB__RR\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 6px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          direction: rtl;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          /* RTL Mark used in second space */\n          content: '\xa0' leader(dotted) '\u200f\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n      </style>\n      <div>a</div>\n      <div>bb</div>\n      <div>c</div>\n    ")

@assert_no_logs
def test_leader_too_long_rtl(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        ______RRRRRRRRRR\n        ______RRRRRRRRRR\n        BB__BBBBBBBBBBBB\n        BB__BBBBBBBBBBBB\n        __RR__RR__RR__RR\n        __RR__RR__RR__RR\n        ______RR__RR__RR\n        ______RR__RR__RR\n        BBBB__BBBBBBBBBB\n        BBBB__BBBBBBBBBB\n        __RR__RR__RR__RR\n        __RR__RR__RR__RR\n        BBBBBB__BBBB__RR\n        BBBBBB__BBBB__RR\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 14px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          direction: rtl;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          /* RTL Mark used in second space */\n          content: '\xa0' leader(dotted) '\u200f\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n      </style>\n      <div>aaaaa</div>\n      <div>a a a a a a a</div>\n      <div>a a a a a</div>\n    ")

@assert_no_logs
def test_leader_float_leader(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RR____________BB\n        RR____________BB\n        RRRR__________BB\n        RRRR__________BB\n        RR____________BB\n        RR____________BB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 6px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader('\u202f.\u202f') 'a';\n          float: right;\n        }\n      </style>\n      <div>a</div>\n      <div>bb</div>\n      <div>c</div>\n    ")

@assert_no_logs
def test_leader_empty_string(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRR____\n        ________\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 8px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 1px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader('');\n        }\n      </style>\n      <div>aaaa</div>\n    ")

@assert_no_logs
def test_leader_zero_width_string(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRR____\n        ________\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 8px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 1px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: leader('\u200b');  /* zero-width space */\n        }\n      </style>\n      <div>aaaa</div>\n    ")

@assert_no_logs
def test_leader_absolute(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        BBBBRRRR\n        ______GG\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 8px 2px;\n        }\n        body {\n          color: red;\n          font-family: weasyprint;\n          font-size: 1px;\n          line-height: 1;\n        }\n        div::before {\n          color: blue;\n          content: leader('z');\n        }\n        article {\n          bottom: 0;\n          color: lime;\n          position: absolute;\n          right: 0;\n        }\n      </style>\n      <div>aa<article>bb</article>aa</div>\n    ")

@assert_no_logs
def test_leader_padding(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RR__BBBBBBBB__BB\n        RR__BBBBBBBB__BB\n        __RR__BBBB__BBBB\n        __RR__BBBB__BBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n        div + div {\n          padding-left: 2px;\n        }\n      </style>\n      <div>a</div>\n      <div>b</div>\n    ")

@assert_no_logs
def test_leader_inline_padding(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RR__BBBBBBBB__BB\n        RR__BBBBBBBB__BB\n        __RR__BBBB__BBBB\n        __RR__BBBB__BBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        span::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n        div + div span {\n          padding-left: 2px;\n        }\n      </style>\n      <div><span>a</span></div>\n      <div><span>b</span></div>\n    ")

@assert_no_logs
def test_leader_margin(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RR__BBBBBBBB__BB\n        RR__BBBBBBBB__BB\n        __RR__BBBB__BBBB\n        __RR__BBBB__BBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        div::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n        div + div {\n          margin-left: 2px;\n        }\n      </style>\n      <div>a</div>\n      <div>b</div>\n    ")

@assert_no_logs
def test_leader_inline_margin(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RR__BBBBBBBB__BB\n        RR__BBBBBBBB__BB\n        __RR__BBBB__BBBB\n        __RR__BBBB__BBBB\n    ', "\n      <style>\n        @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n        @page {\n          size: 16px 4px;\n        }\n        body {\n          color: red;\n          counter-reset: count;\n          font-family: weasyprint;\n          font-size: 2px;\n          line-height: 1;\n        }\n        span::after {\n          color: blue;\n          content: '\xa0' leader(dotted) '\xa0' counter(count, lower-roman);\n          counter-increment: count;\n        }\n        div + div span {\n          margin-left: 2px;\n        }\n      </style>\n      <div><span>a</span></div>\n      <div><span>b</span></div>\n    ")