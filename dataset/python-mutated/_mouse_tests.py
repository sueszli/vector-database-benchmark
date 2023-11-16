import unittest
import time
from ._mouse_event import MoveEvent, ButtonEvent, WheelEvent, LEFT, RIGHT, MIDDLE, X, X2, UP, DOWN, DOUBLE
from keyboard import mouse

class FakeOsMouse(object):

    def __init__(self):
        if False:
            return 10
        self.append = None
        self.position = (0, 0)
        self.queue = None
        self.init = lambda : None

    def listen(self, queue):
        if False:
            return 10
        self.listening = True
        self.queue = queue

    def press(self, button):
        if False:
            while True:
                i = 10
        self.append((DOWN, button))

    def release(self, button):
        if False:
            while True:
                i = 10
        self.append((UP, button))

    def get_position(self):
        if False:
            print('Hello World!')
        return self.position

    def move_to(self, x, y):
        if False:
            return 10
        self.append(('move', (x, y)))
        self.position = (x, y)

    def wheel(self, delta):
        if False:
            while True:
                i = 10
        self.append(('wheel', delta))

    def move_relative(self, x, y):
        if False:
            print('Hello World!')
        self.position = (self.position[0] + x, self.position[1] + y)

class TestMouse(unittest.TestCase):

    @staticmethod
    def setUpClass():
        if False:
            for i in range(10):
                print('nop')
        mouse._os_mouse = FakeOsMouse()
        mouse._listener.start_if_necessary()
        assert mouse._os_mouse.listening

    def setUp(self):
        if False:
            while True:
                i = 10
        self.events = []
        mouse._pressed_events.clear()
        mouse._os_mouse.append = self.events.append

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        mouse.unhook_all()
        self.wait_for_events_queue()

    def wait_for_events_queue(self):
        if False:
            print('Hello World!')
        mouse._listener.queue.join()

    def flush_events(self):
        if False:
            while True:
                i = 10
        self.wait_for_events_queue()
        events = list(self.events)
        del self.events[:]
        return events

    def press(self, button=LEFT):
        if False:
            i = 10
            return i + 15
        mouse._os_mouse.queue.put(ButtonEvent(DOWN, button, time.time()))
        self.wait_for_events_queue()

    def release(self, button=LEFT):
        if False:
            return 10
        mouse._os_mouse.queue.put(ButtonEvent(UP, button, time.time()))
        self.wait_for_events_queue()

    def double_click(self, button=LEFT):
        if False:
            while True:
                i = 10
        mouse._os_mouse.queue.put(ButtonEvent(DOUBLE, button, time.time()))
        self.wait_for_events_queue()

    def click(self, button=LEFT):
        if False:
            i = 10
            return i + 15
        self.press(button)
        self.release(button)

    def wheel(self, delta=1):
        if False:
            while True:
                i = 10
        mouse._os_mouse.queue.put(WheelEvent(delta, time.time()))
        self.wait_for_events_queue()

    def move(self, x=0, y=0):
        if False:
            print('Hello World!')
        mouse._os_mouse.queue.put(MoveEvent(x, y, time.time()))
        self.wait_for_events_queue()

    def test_hook(self):
        if False:
            return 10
        events = []
        self.press()
        mouse.hook(events.append)
        self.press()
        mouse.unhook(events.append)
        self.press()
        self.assertEqual(len(events), 1)

    def test_is_pressed(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(mouse.is_pressed())
        self.press()
        self.assertTrue(mouse.is_pressed())
        self.release()
        self.press(X2)
        self.assertFalse(mouse.is_pressed())
        self.assertTrue(mouse.is_pressed(X2))
        self.press(X2)
        self.assertTrue(mouse.is_pressed(X2))
        self.release(X2)
        self.release(X2)
        self.assertFalse(mouse.is_pressed(X2))

    def test_buttons(self):
        if False:
            print('Hello World!')
        mouse.press()
        self.assertEqual(self.flush_events(), [(DOWN, LEFT)])
        mouse.release()
        self.assertEqual(self.flush_events(), [(UP, LEFT)])
        mouse.click()
        self.assertEqual(self.flush_events(), [(DOWN, LEFT), (UP, LEFT)])
        mouse.double_click()
        self.assertEqual(self.flush_events(), [(DOWN, LEFT), (UP, LEFT), (DOWN, LEFT), (UP, LEFT)])
        mouse.right_click()
        self.assertEqual(self.flush_events(), [(DOWN, RIGHT), (UP, RIGHT)])
        mouse.click(RIGHT)
        self.assertEqual(self.flush_events(), [(DOWN, RIGHT), (UP, RIGHT)])
        mouse.press(X2)
        self.assertEqual(self.flush_events(), [(DOWN, X2)])

    def test_position(self):
        if False:
            while True:
                i = 10
        self.assertEqual(mouse.get_position(), mouse._os_mouse.get_position())

    def test_move(self):
        if False:
            print('Hello World!')
        mouse.move(0, 0)
        self.assertEqual(mouse._os_mouse.get_position(), (0, 0))
        mouse.move(100, 500)
        self.assertEqual(mouse._os_mouse.get_position(), (100, 500))
        mouse.move(1, 2, False)
        self.assertEqual(mouse._os_mouse.get_position(), (101, 502))
        mouse.move(0, 0)
        mouse.move(100, 499, True, duration=0.01)
        self.assertEqual(mouse._os_mouse.get_position(), (100, 499))
        mouse.move(100, 1, False, duration=0.01)
        self.assertEqual(mouse._os_mouse.get_position(), (200, 500))
        mouse.move(0, 0, False, duration=0.01)
        self.assertEqual(mouse._os_mouse.get_position(), (200, 500))

    def triggers(self, fn, events, **kwargs):
        if False:
            print('Hello World!')
        self.triggered = False

        def callback():
            if False:
                while True:
                    i = 10
            self.triggered = True
        handler = fn(callback, **kwargs)
        for (event_type, arg) in events:
            if event_type == DOWN:
                self.press(arg)
            elif event_type == UP:
                self.release(arg)
            elif event_type == DOUBLE:
                self.double_click(arg)
            elif event_type == 'WHEEL':
                self.wheel()
        mouse._listener.remove_handler(handler)
        return self.triggered

    def test_on_button(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, LEFT)]))
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, RIGHT)]))
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, X)]))
        self.assertFalse(self.triggers(mouse.on_button, [('WHEEL', '')]))
        self.assertFalse(self.triggers(mouse.on_button, [(DOWN, X)], buttons=MIDDLE))
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, MIDDLE)], buttons=MIDDLE))
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, MIDDLE)], buttons=MIDDLE))
        self.assertFalse(self.triggers(mouse.on_button, [(DOWN, MIDDLE)], buttons=MIDDLE, types=UP))
        self.assertTrue(self.triggers(mouse.on_button, [(UP, MIDDLE)], buttons=MIDDLE, types=UP))
        self.assertTrue(self.triggers(mouse.on_button, [(UP, MIDDLE)], buttons=[MIDDLE, LEFT], types=[UP, DOWN]))
        self.assertTrue(self.triggers(mouse.on_button, [(DOWN, LEFT)], buttons=[MIDDLE, LEFT], types=[UP, DOWN]))
        self.assertFalse(self.triggers(mouse.on_button, [(UP, X)], buttons=[MIDDLE, LEFT], types=[UP, DOWN]))

    def test_ons(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.triggers(mouse.on_click, [(UP, LEFT)]))
        self.assertFalse(self.triggers(mouse.on_click, [(UP, RIGHT)]))
        self.assertFalse(self.triggers(mouse.on_click, [(DOWN, LEFT)]))
        self.assertFalse(self.triggers(mouse.on_click, [(DOWN, RIGHT)]))
        self.assertTrue(self.triggers(mouse.on_double_click, [(DOUBLE, LEFT)]))
        self.assertFalse(self.triggers(mouse.on_double_click, [(DOUBLE, RIGHT)]))
        self.assertFalse(self.triggers(mouse.on_double_click, [(DOWN, RIGHT)]))
        self.assertTrue(self.triggers(mouse.on_right_click, [(UP, RIGHT)]))
        self.assertTrue(self.triggers(mouse.on_middle_click, [(UP, MIDDLE)]))

    def test_wait(self):
        if False:
            for i in range(10):
                print('nop')
        from threading import Thread, Lock
        lock = Lock()
        lock.acquire()

        def t():
            if False:
                while True:
                    i = 10
            mouse.wait()
            lock.release()
        Thread(target=t).start()
        self.press()
        lock.acquire()

    def test_record_play(self):
        if False:
            print('Hello World!')
        from threading import Thread, Lock
        lock = Lock()
        lock.acquire()

        def t():
            if False:
                i = 10
                return i + 15
            self.recorded = mouse.record(RIGHT)
            lock.release()
        Thread(target=t).start()
        self.click()
        self.wheel(5)
        self.move(100, 50)
        self.press(RIGHT)
        lock.acquire()
        self.assertEqual(len(self.recorded), 5)
        self.assertEqual(self.recorded[0]._replace(time=None), ButtonEvent(DOWN, LEFT, None))
        self.assertEqual(self.recorded[1]._replace(time=None), ButtonEvent(UP, LEFT, None))
        self.assertEqual(self.recorded[2]._replace(time=None), WheelEvent(5, None))
        self.assertEqual(self.recorded[3]._replace(time=None), MoveEvent(100, 50, None))
        self.assertEqual(self.recorded[4]._replace(time=None), ButtonEvent(DOWN, RIGHT, None))
        mouse.play(self.recorded, speed_factor=0)
        events = self.flush_events()
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0], (DOWN, LEFT))
        self.assertEqual(events[1], (UP, LEFT))
        self.assertEqual(events[2], ('wheel', 5))
        self.assertEqual(events[3], ('move', (100, 50)))
        self.assertEqual(events[4], (DOWN, RIGHT))
        mouse.play(self.recorded)
        events = self.flush_events()
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0], (DOWN, LEFT))
        self.assertEqual(events[1], (UP, LEFT))
        self.assertEqual(events[2], ('wheel', 5))
        self.assertEqual(events[3], ('move', (100, 50)))
        self.assertEqual(events[4], (DOWN, RIGHT))
        mouse.play(self.recorded, include_clicks=False)
        events = self.flush_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], ('wheel', 5))
        self.assertEqual(events[1], ('move', (100, 50)))
        mouse.play(self.recorded, include_moves=False)
        events = self.flush_events()
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0], (DOWN, LEFT))
        self.assertEqual(events[1], (UP, LEFT))
        self.assertEqual(events[2], ('wheel', 5))
        self.assertEqual(events[3], (DOWN, RIGHT))
        mouse.play(self.recorded, include_wheel=False)
        events = self.flush_events()
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0], (DOWN, LEFT))
        self.assertEqual(events[1], (UP, LEFT))
        self.assertEqual(events[2], ('move', (100, 50)))
        self.assertEqual(events[3], (DOWN, RIGHT))
if __name__ == '__main__':
    unittest.main()