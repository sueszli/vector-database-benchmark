"""
Tests for InfiniteLineVisual
All images are of size (100,100) to keep a small file size
"""
import numpy as np
from vispy.scene import visuals
from vispy.testing import requires_application, TestingCanvas, run_tests_if_main
from vispy.testing.image_tester import assert_image_approved
from vispy.testing import assert_raises

@requires_application()
def test_set_data():
    if False:
        while True:
            i = 10
    'Test InfiniteLineVisual'
    pos = 5.0
    color = [1.0, 1.0, 0.5, 0.5]
    expected_color = np.array(color, dtype=np.float32)
    for (is_vertical, reference_image) in [(True, 'infinite_line.png'), (False, 'infinite_line_h.png')]:
        with TestingCanvas() as c:
            region = visuals.InfiniteLine(pos=pos, color=color, vertical=is_vertical, parent=c.scene)
            assert region.pos == pos
            assert np.all(region.color == expected_color)
            assert region.is_vertical == is_vertical
            region.set_data(color=tuple(color))
            assert np.all(region.color == expected_color)
            assert_image_approved(c.render(), 'visuals/%s' % reference_image)
            assert_raises(TypeError, region.set_data, pos=[[1, 2], [3, 4]])
            assert_raises(ValueError, region.set_data, color=[[1, 2], [3, 4]])
            assert_raises(ValueError, region.set_data, color=[1, 2])
run_tests_if_main()