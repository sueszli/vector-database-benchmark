from __future__ import annotations
import contextlib
import functools
import math
import random
import signal
import time

class TimingResult:
    """Timing result."""

    def __init__(self):
        if False:
            return 10
        self.start_time = 0
        self.end_time = 0
        self.value = 0

@contextlib.contextmanager
def timing(repeat_count: int=1):
    if False:
        i = 10
        return i + 15
    '\n    Measures code execution time.\n\n    :param repeat_count: If passed, the result will be divided by the value.\n    '
    result = TimingResult()
    result.start_time = time.monotonic()
    try:
        yield result
    finally:
        end_time = time.monotonic()
        diff = (end_time - result.start_time) * 1000.0
        result.end_time = end_time
        if repeat_count == 1:
            result.value = diff
            print(f'Loop time: {diff:.3f} ms')
        else:
            average_time = diff / repeat_count
            result.value = average_time
            print(f'Average time: {average_time:.3f} ms')

def repeat(repeat_count=5):
    if False:
        return 10
    '\n    Function decorators that repeat function many times.\n\n    :param repeat_count: The repeat count\n    '

    def repeat_decorator(f):
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        def wrap(*args, **kwargs):
            if False:
                while True:
                    i = 10
            last_result = None
            for _ in range(repeat_count):
                last_result = f(*args, **kwargs)
            return last_result
        return wrap
    return repeat_decorator

class TimeoutException(Exception):
    """Exception when the test timeo uts"""

@contextlib.contextmanager
def timeout(seconds=1):
    if False:
        while True:
            i = 10
    '\n    Executes code only  limited seconds. If the code does not end during this time, it will be interrupted.\n\n    :param seconds: Number of seconds\n    '

    def handle_timeout(signum, frame):
        if False:
            print('Hello World!')
        raise TimeoutException('Process timed out.')
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(seconds)
    except ValueError:
        raise Exception("timeout can't be used in the current context")
    try:
        yield
    except TimeoutException:
        print('Process timed out.')
    finally:
        try:
            signal.alarm(0)
        except ValueError:
            raise Exception("timeout can't be used in the current context")
if __name__ == '__main__':

    def monte_carlo(total=10000):
        if False:
            i = 10
            return i + 15
        'Monte Carlo'
        inside = 0
        for _ in range(total):
            x_val = random.random() ** 2
            y_val = random.random() ** 2
            if math.sqrt(x_val + y_val) < 1:
                inside += 1
        return inside / total * 4
    with timeout(1):
        print('Sleep 5s with 1s timeout')
        time.sleep(4)
        print(':-/')
    print()
    REPEAT_COUNT = 5

    @timing(REPEAT_COUNT)
    @repeat(REPEAT_COUNT)
    @timing()
    def get_pi():
        if False:
            return 10
        'Returns PI value:'
        return monte_carlo()
    res = get_pi()
    print('PI: ', res)
    print()
    with timing():
        res = monte_carlo()
    print('PI: ', res)