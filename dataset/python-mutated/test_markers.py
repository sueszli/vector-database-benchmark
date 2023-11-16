"""Test how SVG markers are drawn."""
from ...testing_utils import assert_no_logs

@assert_no_logs
def test_markers(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n    ', '\n      <style>\n        @page { size: 11px 13px }\n        svg { display: block }\n      </style>\n      <svg width="11px" height="13px" xmlns="http://www.w3.org/2000/svg">\n        <defs>\n          <marker id="rectangle">\n            <rect x="-1" y="-1" width="3" height="3" fill="red" />\n          </marker>\n        </defs>\n        <path\n          d="M 5 2 v 4 v 4"\n          marker-start="url(#rectangle)"\n          marker-mid="url(#rectangle)"\n          marker-end="url(#rectangle)" />\n      </svg>\n    ')

@assert_no_logs
def test_markers_viewbox(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n    ', '\n      <style>\n        @page { size: 11px 13px }\n        svg { display: block }\n      </style>\n      <svg width="11px" height="13px" xmlns="http://www.w3.org/2000/svg">\n        <defs>\n          <marker id="rectangle" viewBox="-1 -1 3 3">\n            <rect x="-10" y="-10" width="20" height="20" fill="red" />\n          </marker>\n        </defs>\n        <path\n          d="M 5 2 v 4 v 4"\n          marker-start="url(#rectangle)"\n          marker-mid="url(#rectangle)"\n          marker-end="url(#rectangle)" />\n      </svg>\n    ')

@assert_no_logs
def test_markers_size(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n    ', '\n      <style>\n        @page { size: 11px 13px }\n        svg { display: block }\n      </style>\n      <svg width="11px" height="13px" xmlns="http://www.w3.org/2000/svg">\n        <defs>\n          <marker id="rectangle"\n                  refX="1" refY="1" markerWidth="3" markerHeight="3">\n            <rect width="6" height="6" fill="red" />\n          </marker>\n        </defs>\n        <path\n          d="M 5 2 v 4 v 4"\n          marker-start="url(#rectangle)"\n          marker-mid="url(#rectangle)"\n          marker-end="url(#rectangle)" />\n      </svg>\n    ')

@assert_no_logs
def test_markers_viewbox_size(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n        ____RRR____\n        ____RRR____\n        ____RRR____\n        ___________\n    ', '\n      <style>\n        @page { size: 11px 13px }\n        svg { display: block }\n      </style>\n      <svg width="11px" height="13px" xmlns="http://www.w3.org/2000/svg">\n        <defs>\n          <marker id="rectangle" viewBox="-10 -10 6 6"\n                  refX="-8" refY="-8" markerWidth="3" markerHeight="3">\n            <rect x="-10" y="-10" width="6" height="6" fill="red" />\n          </marker>\n        </defs>\n        <path\n          d="M 5 2 v 4 v 4"\n          marker-start="url(#rectangle)"\n          marker-mid="url(#rectangle)"\n          marker-end="url(#rectangle)" />\n      </svg>\n    ')