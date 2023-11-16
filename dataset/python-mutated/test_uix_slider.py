from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.base import EventLoop

class _TestSliderHandle(Slider):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(_TestSliderHandle, self).__init__(**kwargs)
        self.sensitivity = 'handle'

class _TestSliderAll(Slider):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(_TestSliderAll, self).__init__(**kwargs)
        self.sensitivity = 'all'

class SliderMoveTestCase(GraphicUnitTest):
    framecount = 0

    def setUp(self):
        if False:
            i = 10
            return i + 15
        import kivy.lang.builder as builder
        if not hasattr(self, '_trace'):
            self._trace = builder.trace
        self.builder = builder
        builder.trace = lambda *_, **__: None
        super(SliderMoveTestCase, self).setUp()

    def tearDown(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        import kivy.lang.builder as builder
        builder.trace = self._trace
        super(SliderMoveTestCase, self).tearDown(*args, **kwargs)

    def test_slider_move(self):
        if False:
            while True:
                i = 10
        EventLoop.ensure_window()
        win = EventLoop.window
        layout = BoxLayout(orientation='vertical')
        s_handle = _TestSliderHandle()
        s_all = _TestSliderAll()
        layout.add_widget(s_handle)
        layout.add_widget(s_all)
        win.add_widget(layout)
        EventLoop.idle()
        cur1 = s_handle.children[0]
        cur2 = s_all.children[0]
        h1 = cur1.to_window(*cur1.center)[1]
        h2 = h1 - s_handle.cursor_height
        h3 = cur2.to_window(*cur2.center)[1]
        h4 = h3 - s_all.cursor_height
        w1 = cur1.to_window(*cur1.center)[0]
        w2 = cur2.to_window(*cur2.center)[0]
        wh = win.width / 2.0
        dt = 2
        points = [[w1, h1, wh, h1, 'handle'], [w1, h2, wh, h2, 'handle'], [w2, h3, wh, h3, 'all'], [w2, h4, wh, h4, 'all']]
        for point in points:
            (x, y, nx, ny, id) = point
            touch = UTMotionEvent('unittest', 1, {'x': x / float(win.width), 'y': y / float(win.height)})
            EventLoop.post_dispatch_input('begin', touch)
            if id == 'handle':
                if x == w1 and y == h1:
                    self.assertAlmostEqual(s_handle.value, 0.0, delta=dt)
                elif x == w1 and y == h2:
                    self.assertAlmostEqual(s_handle.value, 50.0, delta=dt)
            elif id == 'all':
                if x == w1 and y == h3:
                    self.assertAlmostEqual(s_all.value, 0.0, delta=dt)
                elif x == w1 and y == h4:
                    self.assertAlmostEqual(s_all.value, 0.0, delta=dt)
            touch.move({'x': nx / float(win.width), 'y': ny / float(win.height)})
            EventLoop.post_dispatch_input('update', touch)
            if id == 'handle':
                if nx == wh and ny == h1:
                    self.assertAlmostEqual(s_handle.value, 50.0, delta=dt)
                elif nx == wh and ny == h2:
                    self.assertAlmostEqual(s_handle.value, 50.0, delta=dt)
            elif id == 'all':
                if nx == wh and ny == h3:
                    self.assertAlmostEqual(s_all.value, 50.0, delta=dt)
                elif nx == wh and ny == h4:
                    self.assertAlmostEqual(s_all.value, 50.0, delta=dt)
            EventLoop.post_dispatch_input('end', touch)
        self.render(layout)
if __name__ == '__main__':
    import unittest
    unittest.main()