from kivy.tests.common import GraphicUnitTest

class MultitouchSimulatorTestCase(GraphicUnitTest):
    framecount = 3

    def render(self, root, framecount=1):
        if False:
            while True:
                i = 10
        pass

    def correct_y(self, win, y):
        if False:
            i = 10
            return i + 15
        return win.height - 1.0 - y

    def mouse_init(self, on_demand=False, disabled=False, scatter=False):
        if False:
            print('Hello World!')
        from kivy.base import EventLoop
        from kivy.uix.button import Button
        from kivy.uix.scatter import Scatter
        eventloop = EventLoop
        win = eventloop.window
        eventloop.idle()
        wid = Scatter() if scatter else Button()
        if on_demand:
            mode = 'multitouch_on_demand'
        elif disabled:
            mode = 'disable_multitouch'
        else:
            mode = ''
        from kivy.input.providers.mouse import MouseMotionEventProvider
        mouse = MouseMotionEventProvider('unittest', mode)
        mouse.is_touch = True
        mouse.scale_for_screen = lambda *_, **__: None
        mouse.grab_exclusive_class = None
        mouse.grab_list = []
        if on_demand:
            self.assertTrue(mouse.multitouch_on_demand)
        return (eventloop, win, mouse, wid)

    def multitouch_dot_touch(self, button, **kwargs):
        if False:
            return 10
        (eventloop, win, mouse, wid) = self.mouse_init(**kwargs)
        mouse.start()
        eventloop.add_input_provider(mouse)
        self.assertEqual(mouse.counter, 0)
        self.assertEqual(mouse.touches, {})
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
        event_id = next(iter(mouse.touches))
        self.assertEqual(mouse.counter, 1)
        if 'on_demand' in kwargs and 'scatter' not in kwargs:
            self.render(wid)
            mouse.stop()
            eventloop.remove_input_provider(mouse)
            return
        elif 'on_demand' in kwargs and 'scatter' in kwargs:
            self.assertIn('multitouch_sim', mouse.touches[event_id].profile)
            self.assertTrue(mouse.multitouch_on_demand)
            self.advance_frames(1)
            wid.on_touch_down(mouse.touches[event_id])
            wid.on_touch_up(mouse.touches[event_id])
            self.assertTrue(mouse.touches[event_id].multitouch_sim)
        elif 'disabled' in kwargs:
            self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
        else:
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
        self.assertEqual(mouse.counter, 1)
        if 'disabled' not in kwargs:
            self.assertIn(event_id, mouse.touches)
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), button, {})
        self.assertEqual(mouse.counter, 1 + int('disabled' in kwargs))
        if 'disabled' in kwargs:
            self.assertNotIn(event_id, mouse.touches)
            mouse.stop()
            eventloop.remove_input_provider(mouse)
            return
        else:
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        dot_proxy = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), button, {})
        if button == 'left':
            with self.assertRaises(ReferenceError):
                print(dot_proxy)
            self.assertEqual(mouse.counter, 1)
            self.assertNotIn(event_id, mouse.touches)
            self.assertEqual(mouse.touches, {})
        elif button == 'right':
            self.assertEqual(mouse.counter, 1)
            self.assertIn(event_id, mouse.touches)
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        self.render(wid)
        mouse.stop()
        eventloop.remove_input_provider(mouse)

    def multitouch_dot_move(self, button, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (eventloop, win, mouse, wid) = self.mouse_init(**kwargs)
        mouse.start()
        eventloop.add_input_provider(mouse)
        self.assertEqual(mouse.counter, 0)
        self.assertEqual(mouse.touches, {})
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
        event_id = next(iter(mouse.touches))
        self.assertEqual(mouse.counter, 1)
        if 'on_demand' in kwargs and 'scatter' not in kwargs:
            self.render(wid)
            mouse.stop()
            eventloop.remove_input_provider(mouse)
            return
        elif 'on_demand' in kwargs and 'scatter' in kwargs:
            self.assertIn('multitouch_sim', mouse.touches[event_id].profile)
            self.assertTrue(mouse.multitouch_on_demand)
            self.advance_frames(1)
            wid.on_touch_down(mouse.touches[event_id])
            wid.on_touch_up(mouse.touches[event_id])
            self.assertTrue(mouse.touches[event_id].multitouch_sim)
            win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
            ellipse = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
            win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
        elif 'disabled' in kwargs:
            self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
        else:
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        if 'disabled' in kwargs:
            self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
            mouse.stop()
            eventloop.remove_input_provider(mouse)
            return
        else:
            ellipse = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
        win.dispatch('on_mouse_move', 11, self.correct_y(win, 11), {})
        self.assertEqual(ellipse.pos, (1, 1))
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
        self.assertEqual(mouse.counter, 1)
        self.assertIn(event_id, mouse.touches)
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), button, {})
        self.assertEqual(mouse.counter, 1)
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        win.dispatch('on_mouse_move', 50, self.correct_y(win, 50), {})
        self.assertEqual(ellipse.pos, (40, 40))
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), button, {})
        self.assertEqual(mouse.counter, 1)
        if button == 'left':
            self.assertNotIn(event_id, mouse.touches)
        elif button == 'right':
            self.assertIn(event_id, mouse.touches)
            self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        self.render(wid)
        mouse.stop()
        eventloop.remove_input_provider(mouse)

    def test_multitouch_dontappear(self):
        if False:
            while True:
                i = 10
        (eventloop, win, mouse, wid) = self.mouse_init()
        mouse.start()
        eventloop.add_input_provider(mouse)
        self.assertEqual(mouse.counter, 0)
        self.assertEqual(mouse.touches, {})
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'left', {})
        event_id = next(iter(mouse.touches))
        win.dispatch('on_mouse_move', 11, self.correct_y(win, 11), {})
        self.assertEqual(mouse.counter, 1)
        self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'left', {})
        self.assertEqual(mouse.counter, 1)
        self.assertNotIn(event_id, mouse.touches)
        self.advance_frames(1)
        self.render(wid)
        mouse.stop()
        eventloop.remove_input_provider(mouse)

    def test_multitouch_appear(self):
        if False:
            return 10
        (eventloop, win, mouse, wid) = self.mouse_init()
        mouse.start()
        eventloop.add_input_provider(mouse)
        self.assertEqual(mouse.counter, 0)
        self.assertEqual(mouse.touches, {})
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
        event_id = next(iter(mouse.touches))
        self.assertEqual(mouse.counter, 1)
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        ellipse = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
        self.assertAlmostEqual(ellipse.pos[0], 0, delta=0.0001)
        self.assertAlmostEqual(ellipse.pos[1], 0, delta=0.0001)
        win.dispatch('on_mouse_move', 11, self.correct_y(win, 11), {})
        self.assertEqual(ellipse.pos, (1, 1))
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
        self.assertEqual(ellipse.pos, (1, 1))
        self.assertEqual(mouse.counter, 1)
        self.assertIn(event_id, mouse.touches)
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
        self.render(wid)
        mouse.stop()
        eventloop.remove_input_provider(mouse)

    def test_multitouch_dot_lefttouch(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_touch('left')

    def test_multitouch_dot_leftmove(self):
        if False:
            i = 10
            return i + 15
        self.multitouch_dot_move('left')

    def test_multitouch_dot_righttouch(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_touch('right')

    def test_multitouch_dot_rightmove(self):
        if False:
            i = 10
            return i + 15
        self.multitouch_dot_move('right')

    def test_multitouch_on_demand_noscatter_lefttouch(self):
        if False:
            print('Hello World!')
        self.multitouch_dot_touch('left', on_demand=True)

    def test_multitouch_on_demand_noscatter_leftmove(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_move('left', on_demand=True)

    def test_multitouch_on_demand_noscatter_righttouch(self):
        if False:
            return 10
        self.multitouch_dot_touch('right', on_demand=True)

    def test_multitouch_on_demand_noscatter_rightmove(self):
        if False:
            print('Hello World!')
        self.multitouch_dot_move('right', on_demand=True)

    def test_multitouch_on_demand_scatter_lefttouch(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_touch('left', on_demand=True, scatter=True)

    def test_multitouch_on_demand_scatter_leftmove(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_move('left', on_demand=True, scatter=True)

    def test_multitouch_on_demand_scatter_righttouch(self):
        if False:
            while True:
                i = 10
        self.multitouch_dot_touch('right', on_demand=True, scatter=True)

    def test_multitouch_on_demand_scatter_rightmove(self):
        if False:
            print('Hello World!')
        self.multitouch_dot_move('right', on_demand=True, scatter=True)

    def test_multitouch_disabled_lefttouch(self):
        if False:
            return 10
        self.multitouch_dot_touch('left', disabled=True)

    def test_multitouch_disabled_leftmove(self):
        if False:
            i = 10
            return i + 15
        self.multitouch_dot_move('left', disabled=True)

    def test_multitouch_disabled_righttouch(self):
        if False:
            return 10
        self.multitouch_dot_touch('right', disabled=True)

    def test_multitouch_disabled_rightmove(self):
        if False:
            for i in range(10):
                print('nop')
        self.multitouch_dot_move('right', disabled=True)
if __name__ == '__main__':
    import unittest
    unittest.main()