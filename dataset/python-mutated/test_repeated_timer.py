__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2015 The OctoPrint Project - Released under terms of the AGPLv3 License'
import time
import unittest
from unittest import mock
from octoprint.util import RepeatedTimer

class Countdown:

    def __init__(self, start):
        if False:
            while True:
                i = 10
        self._counter = start

    def step(self):
        if False:
            i = 10
            return i + 15
        self._counter -= 1

    @property
    def counter(self):
        if False:
            i = 10
            return i + 15
        return self._counter

class IncreasingInterval(Countdown):

    def __init__(self, start, factor):
        if False:
            while True:
                i = 10
        Countdown.__init__(self, start)
        self._start = start
        self._factor = factor

    def interval(self):
        if False:
            while True:
                i = 10
        result = (self._start - self._counter + 1) * self._factor
        return result

class RepeatedTimerTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def test_condition(self):
        if False:
            return 10
        countdown = Countdown(5)
        timer_task = mock.MagicMock()
        timer_task.side_effect = countdown.step
        timer = RepeatedTimer(0.1, timer_task, condition=lambda : countdown.counter > 0)
        timer.start()
        timer.join()
        self.assertEqual(5, timer_task.call_count)

    def test_finished_callback(self):
        if False:
            return 10
        countdown = Countdown(5)
        timer_task = mock.MagicMock()
        timer_task.side_effect = countdown.step
        on_finished = mock.MagicMock()
        timer = RepeatedTimer(0.1, timer_task, condition=lambda : countdown.counter > 0, on_finish=on_finished)
        timer.start()
        timer.join()
        self.assertEqual(1, on_finished.call_count)

    def test_condition_callback(self):
        if False:
            for i in range(10):
                print('nop')
        countdown = Countdown(5)
        timer_task = mock.MagicMock()
        timer_task.side_effect = countdown.step
        on_cancelled = mock.MagicMock()
        on_condition_false = mock.MagicMock()
        timer = RepeatedTimer(0.1, timer_task, condition=lambda : countdown.counter > 0, on_condition_false=on_condition_false, on_cancelled=on_cancelled)
        timer.start()
        timer.join()
        self.assertEqual(1, on_condition_false.call_count)
        self.assertEqual(0, on_cancelled.call_count)

    def test_cancelled_callback(self):
        if False:
            while True:
                i = 10
        countdown = Countdown(5)
        timer_task = mock.MagicMock()
        timer_task.side_effect = countdown.step
        on_cancelled = mock.MagicMock()
        on_condition_false = mock.MagicMock()
        timer = RepeatedTimer(10, timer_task, condition=lambda : countdown.counter > 0, on_condition_false=on_condition_false, on_cancelled=on_cancelled)
        timer.start()
        time.sleep(1)
        timer.cancel()
        timer.join()
        self.assertEqual(0, on_condition_false.call_count)
        self.assertEqual(1, on_cancelled.call_count)

    def test_run_first(self):
        if False:
            while True:
                i = 10
        timer_task = mock.MagicMock()
        timer = RepeatedTimer(60, timer_task, run_first=True)
        timer.start()
        time.sleep(1)
        timer.cancel()
        timer.join()
        self.assertEqual(1, timer_task.call_count)

    def test_not_run_first(self):
        if False:
            i = 10
            return i + 15
        timer_task = mock.MagicMock()
        timer = RepeatedTimer(60, timer_task)
        timer.start()
        time.sleep(1)
        timer.cancel()
        timer.join()
        self.assertEqual(0, timer_task.call_count)

    def test_adjusted_interval(self):
        if False:
            while True:
                i = 10
        increasing_interval = IncreasingInterval(3, 1)
        timer_task = mock.MagicMock()
        timer_task.side_effect = increasing_interval.step
        timer = RepeatedTimer(increasing_interval.interval, timer_task, condition=lambda : increasing_interval.counter > 0)
        start_time = time.time()
        timer.start()
        timer.join()
        duration = time.time() - start_time
        self.assertEqual(3, timer_task.call_count)
        self.assertGreaterEqual(duration, 6)
        self.assertLess(duration, 7)

    def test_condition_change_during_task(self):
        if False:
            while True:
                i = 10

        def sleep():
            if False:
                i = 10
                return i + 15
            time.sleep(2)
        timer_task = mock.MagicMock()
        timer_task.side_effect = sleep
        timer = RepeatedTimer(0.1, timer_task, run_first=True)
        timer.start()
        time.sleep(1)
        timer.condition = lambda : False
        timer.join()
        self.assertEqual(1, timer_task.call_count)