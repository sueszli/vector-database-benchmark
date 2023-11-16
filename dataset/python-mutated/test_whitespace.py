"""Test how white spaces collapse."""
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_whitespace_inline(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RRRR__RRRR____\n        RRRR__RRRR____\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n        </style>\n        <span>aa </span><span> aa</span>\n    ')

@assert_no_logs
def test_whitespace_nested_inline(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRR__RRRR____\n        RRRR__RRRR____\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n        </style>\n        <span><span>aa </span></span><span><span> aa</span></span>\n    ')

@assert_no_logs
def test_whitespace_inline_space_between(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRR__RRRR____\n        RRRR__RRRR____\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n        </style>\n        <span>aa </span> <span> aa</span>\n    ')

@assert_no_logs
def test_whitespace_float_between(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RRRR__RRRR__BB\n        RRRR__RRRR__BB\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {float: right; color: blue}\n        </style>\n        <span>aa </span><div>a</div><span> aa</span>\n    ')

@assert_no_logs
def test_whitespace_in_float(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RRRRRRRR____BB\n        RRRRRRRR____BB\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {\n              color: blue;\n              float: right;\n            }\n        </style>\n        <span>aa</span><div> a </div><span>aa</span>\n    ')

@assert_no_logs
def test_whitespace_absolute_between(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RRRR__RRRR__BB\n        RRRR__RRRR__BB\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {\n              color: blue;\n              position: absolute;\n              right: 0;\n              top: 0;\n            }\n        </style>\n        <span>aa </span><div>a</div><span> aa</span>\n    ')

@assert_no_logs
def test_whitespace_in_absolute(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRRRRRR____BB\n        RRRRRRRR____BB\n        ______________\n        ______________\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {size: 14px 4px}\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {\n              color: blue;\n              position: absolute;\n              right: 0;\n              top: 0;\n            }\n        </style>\n        <span>aa</span><div> a </div><span>aa</span>\n    ')

@assert_no_logs
def test_whitespace_running_between(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        RRRR__RRRR____\n        RRRR__RRRR____\n        ______BB______\n        ______BB______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n              size: 14px 4px;\n              margin: 0 0 2px;\n              @bottom-center {\n                content: element(test);\n              }\n            }\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {\n              background: green;\n              color: blue;\n              position: running(test);\n            }\n        </style>\n        <span>aa </span><div>a</div><span> aa</span>\n    ')

@assert_no_logs
def test_whitespace_in_running(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRRRRRR______\n        RRRRRRRR______\n        ______BB______\n        ______BB______\n    ', '\n        <style>\n            @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n            @page {\n              size: 14px 4px;\n              margin: 0 0 2px;\n              @bottom-center {\n                content: element(test);\n              }\n            }\n            body {\n              color: red;\n              font-family: weasyprint;\n              font-size: 2px;\n              line-height: 1;\n            }\n            div {\n              background: green;\n              color: blue;\n              position: running(test);\n            }\n        </style>\n        <span>aa</span><div> a </div><span>aa</span>\n    ')