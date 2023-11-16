"""Test how footnotes in columns are drawn."""
import pytest
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_footnote_column_margin_top(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_____\n        RRRR_____\n        _________\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 7px;\n        @footnote {\n          margin-top: 2px;\n        }\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>a<span>de</span> ab ab ab ab ab ab</div>')

@assert_no_logs
def test_footnote_column_fill_auto(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRR_____\n        RRRR_____\n        RRRR_____\n        RRRR_____\n        RRRR_____\n        RRRR_____\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 13px;\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-fill: auto;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>a<span>de</span> a<span>de</span> a<span>de</span></div>')

@assert_no_logs
def test_footnote_column_fill_auto_break_inside_avoid(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 13px;\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-fill: auto;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      article {\n        break-inside: avoid;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>\n      <article>a<span>de</span> a<span>de</span></article>\n      <article>ab</article>\n      <article>a<span>de</span> ab</article>\n      <article>ab</article>\n    </div>')

@assert_no_logs
def test_footnote_column_p_after(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        KK__KK___\n        KK__KK___\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        KK__KK___\n        KK__KK___\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 11px;\n      }\n      body {\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>a<span>de</span> a<span>de</span> ab ab</div>\n    <p>a a a a</p>')

@assert_no_logs
def test_footnote_column_p_before(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        KKKK_____\n        KKKK_____\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RR__\n        RRRR_RR__\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_RR__\n        RRRR_RR__\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n        _________\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 13px;\n      }\n      body {\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <p>ab</p>\n    <div>\n    a<span>de</span> a<span>de</span>\n    a<span>de</span> a ab a </div>')

@assert_no_logs
def test_footnote_column_3(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRR_RRRR_RRRR\n        RRRR_RRRR_RRRR\n        ______________\n        RRRRRRRR______\n        RRRRRRRR______\n        RRRR_RRRR_____\n        RRRR_RRRR_____\n        ______________\n        ______________\n        ______________\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 14px 5px;\n      }\n      body {\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      div {\n        color: red;\n        columns: 3;\n        column-gap: 1px;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>ab ab a<span>de</span> ab ab </div>')

@assert_no_logs
def test_footnote_column_3_p_before(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        KKKK__________\n        KKKK__________\n        RRRR_RRRR_RRRR\n        RRRR_RRRR_RRRR\n        RRRR_RRRR_RRRR\n        RRRR_RRRR_RRRR\n        ______________\n        RRRRRRRR______\n        RRRRRRRR______\n        RRRR_RRRR_____\n        RRRR_RRRR_____\n        ______________\n        ______________\n        ______________\n        ______________\n        ______________\n        RRRRRRRR______\n        RRRRRRRR______\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 14px 9px;\n      }\n      body {\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      div {\n        color: red;\n        columns: 3;\n        column-gap: 1px;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <p>ab</p>\n    <div>ab ab a<span>de</span> ab ab ab a<span>de</span> ab </div>')

@assert_no_logs
def test_footnote_column_clone_decoration(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        _________\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        _________\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        _________\n        _________\n        _________\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 7px;\n      }\n      div {\n        box-decoration-break: clone;\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n        padding: 1px 0;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>a<span>de</span> ab ab ab</div>')

@assert_no_logs
def test_footnote_column_max_height(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        _________\n        _________\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 9px;\n        @footnote {\n          max-height: 2em;\n        }\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>\n      a<span>de</span> a<span>de</span>\n      a<span>de</span> ab\n      ab ab\n    </div>')

@pytest.mark.xfail
@assert_no_logs
def test_footnote_column_reported_split(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        RRRR_RRRR\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRRRRRR_\n        RRRR_____\n        RRRR_____\n        _________\n        _________\n        _________\n        _________\n        _________\n        RRRRRRRR_\n        RRRRRRRR_\n    ', '\n    <style>\n      @font-face {src: url(weasyprint.otf); font-family: weasyprint}\n      @page {\n        size: 9px 9px;\n      }\n      div {\n        color: red;\n        columns: 2;\n        column-gap: 1px;\n        font-family: weasyprint;\n        font-size: 2px;\n        line-height: 1;\n      }\n      span {\n        float: footnote;\n      }\n    </style>\n    <div>\n      <article>a<span>de</span> a<span>de</span></article>\n      <article>a<span>de</span> ab ab</article>\n    </div>')