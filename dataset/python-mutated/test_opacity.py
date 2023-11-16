"""Test opacity."""
from ..testing_utils import assert_no_logs
opacity_source = '\n    <style>\n        @page { size: 60px 60px }\n        div { background: #000; width: 20px; height: 20px }\n    </style>\n    %s'

@assert_no_logs
def test_opacity_1(assert_same_renderings):
    if False:
        return 10
    assert_same_renderings(opacity_source % '<div></div>', opacity_source % '<div></div><div style="opacity: 0"></div>')

@assert_no_logs
def test_opacity_2(assert_same_renderings):
    if False:
        return 10
    assert_same_renderings(opacity_source % '<div style="background: rgb(102, 102, 102)"></div>', opacity_source % '<div style="opacity: 0.6"></div>')

@assert_no_logs
def test_opacity_3(assert_same_renderings):
    if False:
        return 10
    assert_same_renderings(opacity_source % '<div style="background: rgb(102, 102, 102)"></div>', opacity_source % '<div style="opacity: 0.6"></div>', opacity_source % '\n          <div style="background: none; opacity: 0.666666">\n            <div style="opacity: 0.9"></div>\n          </div>\n        ')