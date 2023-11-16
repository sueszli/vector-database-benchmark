import os
import platform
import unittest
import pygame
import time
Clock = pygame.time.Clock

class ClockTypeTest(unittest.TestCase):
    __tags__ = ['timing']

    def test_construction(self):
        if False:
            i = 10
            return i + 15
        'Ensure a Clock object can be created'
        c = Clock()
        self.assertTrue(c, 'Clock cannot be constructed')

    def test_get_fps(self):
        if False:
            while True:
                i = 10
        'test_get_fps tests pygame.time.get_fps()'
        c = Clock()
        self.assertEqual(c.get_fps(), 0)
        self.assertTrue(type(c.get_fps()) == float)
        delta = 0.3
        self._fps_test(c, 100, delta)
        self._fps_test(c, 60, delta)
        self._fps_test(c, 30, delta)

    def _fps_test(self, clock, fps, delta):
        if False:
            print('Hello World!')
        'ticks fps times each second, hence get_fps() should return fps'
        delay_per_frame = 1.0 / fps
        for f in range(fps):
            clock.tick()
            time.sleep(delay_per_frame)
        self.assertAlmostEqual(clock.get_fps(), fps, delta=fps * delta)

    def test_get_rawtime(self):
        if False:
            return 10
        iterations = 10
        delay = 0.1
        delay_miliseconds = delay * 10 ** 3
        framerate_limit = 5
        delta = 50
        c = Clock()
        self.assertEqual(c.get_rawtime(), 0)
        for f in range(iterations):
            time.sleep(delay)
            c.tick(framerate_limit)
            c1 = c.get_rawtime()
            self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
        for f in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_rawtime()
            c2 = c.get_time()
            self.assertAlmostEqual(c1, c2, delta=delta)

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_get_time(self):
        if False:
            for i in range(10):
                print('nop')
        delay = 0.1
        delay_miliseconds = delay * 10 ** 3
        iterations = 10
        delta = 50
        c = Clock()
        self.assertEqual(c.get_time(), 0)
        for i in range(iterations):
            time.sleep(delay)
            c.tick()
            c1 = c.get_time()
            self.assertAlmostEqual(delay_miliseconds, c1, delta=delta)
        for i in range(iterations):
            t0 = time.time()
            time.sleep(delay)
            c.tick()
            t1 = time.time()
            c1 = c.get_time()
            d0 = (t1 - t0) * 10 ** 3
            self.assertAlmostEqual(d0, c1, delta=delta)

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_tick(self):
        if False:
            return 10
        'Tests time.Clock.tick()'
        '\n        Loops with a set delay a few times then checks what tick reports to\n        verify its accuracy. Then calls tick with a desired frame-rate and\n        verifies it is not faster than the desired frame-rate nor is it taking\n        a dramatically long time to complete\n        '
        epsilon = 5
        epsilon2 = 0.3
        epsilon3 = 20
        testing_framerate = 60
        milliseconds = 5.0
        collection = []
        c = Clock()
        c.tick()
        for i in range(100):
            time.sleep(milliseconds / 1000)
            collection.append(c.tick())
        for outlier in [min(collection), max(collection)]:
            if outlier != milliseconds:
                collection.remove(outlier)
        average_time = float(sum(collection)) / len(collection)
        self.assertAlmostEqual(average_time, milliseconds, delta=epsilon)
        c = Clock()
        collection = []
        start = time.time()
        for i in range(testing_framerate):
            collection.append(c.tick(testing_framerate))
        for outlier in [min(collection), max(collection)]:
            if outlier != round(1000 / testing_framerate):
                collection.remove(outlier)
        end = time.time()
        self.assertAlmostEqual(end - start, 1, delta=epsilon2)
        average_tick_time = float(sum(collection)) / len(collection)
        self.assertAlmostEqual(1000 / average_tick_time, testing_framerate, delta=epsilon3)

    def test_tick_busy_loop(self):
        if False:
            i = 10
            return i + 15
        'Test tick_busy_loop'
        c = Clock()
        second_length = 1000
        shortfall_tolerance = 1
        sample_fps = 40
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        pygame.time.wait(10)
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        pygame.time.wait(200)
        self.assertGreaterEqual(c.tick_busy_loop(sample_fps), second_length / sample_fps - shortfall_tolerance)
        high_fps = 500
        self.assertGreaterEqual(c.tick_busy_loop(high_fps), second_length / high_fps - shortfall_tolerance)
        low_fps = 1
        self.assertGreaterEqual(c.tick_busy_loop(low_fps), second_length / low_fps - shortfall_tolerance)
        low_non_factor_fps = 35
        frame_length_without_decimal_places = int(second_length / low_non_factor_fps)
        self.assertGreaterEqual(c.tick_busy_loop(low_non_factor_fps), frame_length_without_decimal_places - shortfall_tolerance)
        high_non_factor_fps = 750
        frame_length_without_decimal_places_2 = int(second_length / high_non_factor_fps)
        self.assertGreaterEqual(c.tick_busy_loop(high_non_factor_fps), frame_length_without_decimal_places_2 - shortfall_tolerance)
        zero_fps = 0
        self.assertEqual(c.tick_busy_loop(zero_fps), 0)
        negative_fps = -1
        self.assertEqual(c.tick_busy_loop(negative_fps), 0)
        fractional_fps = 32.75
        frame_length_without_decimal_places_3 = int(second_length / fractional_fps)
        self.assertGreaterEqual(c.tick_busy_loop(fractional_fps), frame_length_without_decimal_places_3 - shortfall_tolerance)
        bool_fps = True
        self.assertGreaterEqual(c.tick_busy_loop(bool_fps), second_length / bool_fps - shortfall_tolerance)

class TimeModuleTest(unittest.TestCase):
    __tags__ = ['timing']

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_delay(self):
        if False:
            return 10
        'Tests time.delay() function.'
        millis = 50
        iterations = 20
        delta = 150
        self._wait_delay_check(pygame.time.delay, millis, iterations, delta)
        self._type_error_checks(pygame.time.delay)

    def test_get_ticks(self):
        if False:
            print('Hello World!')
        'Tests time.get_ticks()'
        '\n         Iterates and delays for arbitrary amount of time for each iteration,\n         check get_ticks to equal correct gap time\n        '
        iterations = 20
        millis = 50
        delta = 15
        self.assertTrue(type(pygame.time.get_ticks()) == int)
        for i in range(iterations):
            curr_ticks = pygame.time.get_ticks()
            curr_time = time.time()
            pygame.time.delay(millis)
            time_diff = round((time.time() - curr_time) * 1000)
            ticks_diff = pygame.time.get_ticks() - curr_ticks
            self.assertAlmostEqual(ticks_diff, time_diff, delta=delta)

    @unittest.skipIf(platform.machine() == 's390x', 'Fails on s390x')
    @unittest.skipIf(os.environ.get('CI', None), 'CI can have variable time slices, slow.')
    def test_set_timer(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests time.set_timer()'
        '\n        Tests if a timer will post the correct amount of eventid events in\n        the specified delay. Test is posting event objects work.\n        Also tests if setting milliseconds to 0 stops the timer and if\n        the once argument and repeat arguments work.\n        '
        pygame.init()
        TIMER_EVENT_TYPE = pygame.event.custom_type()
        timer_event = pygame.event.Event(TIMER_EVENT_TYPE)
        delta = 50
        timer_delay = 100
        test_number = 8
        events = 0
        pygame.event.clear()
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
        t1 = pygame.time.get_ticks()
        max_test_time = t1 + timer_delay * test_number + delta
        while events < test_number:
            for event in pygame.event.get():
                if event == timer_event:
                    events += 1
            if pygame.time.get_ticks() > max_test_time:
                break
        pygame.time.set_timer(TIMER_EVENT_TYPE, 0)
        t2 = pygame.time.get_ticks()
        self.assertEqual(events, test_number)
        self.assertAlmostEqual(timer_delay * test_number, t2 - t1, delta=delta)
        pygame.time.delay(200)
        self.assertNotIn(timer_event, pygame.event.get())
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
        pygame.time.delay(int(timer_delay * 3.5))
        self.assertEqual(pygame.event.get().count(timer_event), 3)
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay * 10)
        pygame.time.delay(timer_delay * 5)
        self.assertNotIn(timer_event, pygame.event.get())
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay * 3)
        pygame.time.delay(timer_delay * 7)
        self.assertEqual(pygame.event.get().count(timer_event), 2)
        pygame.time.set_timer(TIMER_EVENT_TYPE, timer_delay)
        pygame.time.delay(int(timer_delay * 5.5))
        self.assertEqual(pygame.event.get().count(timer_event), 5)
        pygame.time.set_timer(TIMER_EVENT_TYPE, 10, True)
        pygame.time.delay(40)
        self.assertEqual(pygame.event.get().count(timer_event), 1)
        events_to_test = [pygame.event.Event(TIMER_EVENT_TYPE), pygame.event.Event(TIMER_EVENT_TYPE, foo='9gwz5', baz=12, lol=[124, (34, '')]), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode='a')]
        repeat = 3
        millis = 50
        for e in events_to_test:
            pygame.time.set_timer(e, millis, loops=repeat)
            pygame.time.delay(2 * millis * repeat)
            self.assertEqual(pygame.event.get().count(e), repeat)
        pygame.quit()

    def test_wait(self):
        if False:
            i = 10
            return i + 15
        'Tests time.wait() function.'
        millis = 100
        iterations = 10
        delta = 50
        self._wait_delay_check(pygame.time.wait, millis, iterations, delta)
        self._type_error_checks(pygame.time.wait)

    def _wait_delay_check(self, func_to_check, millis, iterations, delta):
        if False:
            while True:
                i = 10
        ' "\n        call func_to_check(millis) "iterations" times and check each time if\n        function "waited" for given millisecond (+- delta). At the end, take\n        average time for each call (whole_duration/iterations), which should\n        be equal to millis (+- delta - acceptable margin of error).\n        *Created to avoid code duplication during delay and wait tests\n        '
        start_time = time.time()
        for i in range(iterations):
            wait_time = func_to_check(millis)
            self.assertAlmostEqual(wait_time, millis, delta=delta)
        stop_time = time.time()
        duration = round((stop_time - start_time) * 1000)
        self.assertAlmostEqual(duration / iterations, millis, delta=delta)

    def _type_error_checks(self, func_to_check):
        if False:
            return 10
        'Checks 3 TypeError (float, tuple, string) for the func_to_check'
        'Intended for time.delay and time.wait functions'
        self.assertRaises(TypeError, func_to_check, 0.1)
        self.assertRaises(TypeError, pygame.time.delay, (0, 1))
        self.assertRaises(TypeError, pygame.time.delay, '10')
if __name__ == '__main__':
    unittest.main()