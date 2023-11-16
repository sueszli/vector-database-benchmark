from typing import Iterator
import testslide
from ..timer import Timer

def regular_interval_ticker(interval: int) -> Iterator[int]:
    if False:
        while True:
            i = 10
    tick = 0
    while True:
        yield (interval * tick)
        tick += 1

class TimerTest(testslide.TestCase):

    def test_resolution(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        base_interval = 1000000000
        ticker = regular_interval_ticker(base_interval)
        timer = Timer(get_current_time_in_nanosecond=lambda : next(ticker))
        self.assertEqual(timer.stop_in_nanosecond(), base_interval)
        self.assertEqual(timer.stop_in_microsecond(), 2 * base_interval / 1000.0)
        self.assertEqual(timer.stop_in_millisecond(), 3 * base_interval / 1000000.0)
        self.assertEqual(timer.stop_in_second(), 4 * base_interval / 1000000000.0)

    def test_reset(self) -> None:
        if False:
            while True:
                i = 10
        ticker = regular_interval_ticker(100)
        timer = Timer(get_current_time_in_nanosecond=lambda : next(ticker))
        self.assertEqual(timer.stop_in_nanosecond(), 100)
        timer.reset()
        self.assertEqual(timer.stop_in_nanosecond(), 100)