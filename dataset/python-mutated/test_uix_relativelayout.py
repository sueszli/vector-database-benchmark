"""
uix.relativelayout tests
========================
"""
import unittest
from kivy.base import EventLoop
from kivy.tests import UTMotionEvent
from kivy.uix.relativelayout import RelativeLayout

class UixRelativeLayoutTest(unittest.TestCase):

    def test_relativelayout_on_touch_move(self):
        if False:
            while True:
                i = 10
        EventLoop.ensure_window()
        rl = RelativeLayout()
        EventLoop.window.add_widget(rl)
        touch = UTMotionEvent('unittest', 1, {'x': 0.5, 'y': 0.5})
        EventLoop.post_dispatch_input('begin', touch)
        touch.move({'x': 0.6, 'y': 0.4})
        EventLoop.post_dispatch_input('update', touch)
        EventLoop.post_dispatch_input('end', touch)

    def test_relativelayout_coordinates(self):
        if False:
            i = 10
            return i + 15
        EventLoop.ensure_window()
        rl = RelativeLayout(pos=(100, 100))
        EventLoop.window.add_widget(rl)
        self.assertEqual(rl.to_parent(50, 50), (150, 150))
        self.assertEqual(rl.to_local(50, 50), (-50, -50))