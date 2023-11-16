"""Test the currentColor value."""
import pytest
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_current_color_1(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('GG\nGG', '\n      <style>\n        @page { size: 2px }\n        html, body { height: 100%; margin: 0 }\n        html { color: red; background: currentColor }\n        body { color: lime; background: inherit }\n      </style>\n      <body>')

@assert_no_logs
def test_current_color_2(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('GG\nGG', '\n      <style>\n        @page { size: 2px }\n        html { color: red; border-color: currentColor }\n        body { color: lime; border: 1px solid; border-color: inherit;\n               margin: 0 }\n      </style>\n      <body>')

@assert_no_logs
def test_current_color_3(assert_pixels):
    if False:
        return 10
    assert_pixels('GG\nGG', '\n      <style>\n        @page { size: 2px }\n        html { color: red; outline-color: currentColor }\n        body { color: lime; outline: 1px solid; outline-color: inherit;\n               margin: 1px }\n      </style>\n      <body>')

@assert_no_logs
def test_current_color_4(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('GG\nGG', '\n      <style>\n        @page { size: 2px }\n        html { color: red; border-color: currentColor; }\n        body { margin: 0 }\n        table { border-collapse: collapse;\n                color: lime; border: 1px solid; border-color: inherit }\n      </style>\n      <table><td>')

@assert_no_logs
def test_current_color_svg_1(assert_pixels):
    if False:
        return 10
    assert_pixels('KK\nKK', '\n      <style>\n        @page { size: 2px }\n        svg { display: block }\n      </style>\n      <svg xmlns="http://www.w3.org/2000/svg"\n           width="2" height="2" fill="currentColor">\n        <rect width="2" height="2"></rect>\n      </svg>')

@pytest.mark.xfail
@assert_no_logs
def test_current_color_svg_2(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('GG\nGG', '\n      <style>\n        @page { size: 2px }\n        svg { display: block }\n        body { color: lime }\n      </style>\n      <svg xmlns="http://www.w3.org/2000/svg"\n           width="2" height="2">\n        <rect width="2" height="2" fill="currentColor"></rect>\n      </svg>')