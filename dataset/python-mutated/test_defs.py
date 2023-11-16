"""Test how SVG definitions are drawn."""
from base64 import b64encode
from ...testing_utils import assert_no_logs
SVG = '\n<svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n  <defs>\n    <rect id="rectangle" width="5" height="2" fill="red" />\n  </defs>\n  <use href="#rectangle" />\n  <use href="#rectangle" x="3" y="3" />\n  <use href="#rectangle" x="5" y="6" />\n</svg>\n'
RESULT = '\n  RRRRR_____\n  RRRRR_____\n  __________\n  ___RRRRR__\n  ___RRRRR__\n  __________\n  _____RRRRR\n  _____RRRRR\n  __________\n  __________\n'

@assert_no_logs
def test_use(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels(RESULT, '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n    ' + SVG)

@assert_no_logs
def test_use_base64(assert_pixels):
    if False:
        while True:
            i = 10
    base64_svg = b64encode(SVG.encode()).decode()
    assert_pixels(RESULT, '\n      <style>\n        @page { size: 10px }\n        img { display: block }\n      </style>\n      <img src="data:image/svg+xml;base64,' + base64_svg + '"/>')