import timeit
import unittest
import sys
import io
from textwrap import dedent
from test.support import captured_stdout
from test.support import captured_stderr
DEFAULT_NUMBER = 1000000
DEFAULT_REPEAT = 5

class FakeTimer:
    BASE_TIME = 42.0

    def __init__(self, seconds_per_increment=1.0):
        if False:
            return 10
        self.count = 0
        self.setup_calls = 0
        self.seconds_per_increment = seconds_per_increment
        timeit._fake_timer = self

    def __call__(self):
        if False:
            i = 10
            return i + 15
        return self.BASE_TIME + self.count * self.seconds_per_increment

    def inc(self):
        if False:
            print('Hello World!')
        self.count += 1

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.setup_calls += 1

    def wrap_timer(self, timer):
        if False:
            while True:
                i = 10
        "Records 'timer' and returns self as callable timer."
        self.saved_timer = timer
        return self

class TestTimeit(unittest.TestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            del timeit._fake_timer
        except AttributeError:
            pass

    def test_reindent_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(timeit.reindent('', 0), '')
        self.assertEqual(timeit.reindent('', 4), '')

    def test_reindent_single(self):
        if False:
            return 10
        self.assertEqual(timeit.reindent('pass', 0), 'pass')
        self.assertEqual(timeit.reindent('pass', 4), 'pass')

    def test_reindent_multi_empty(self):
        if False:
            print('Hello World!')
        self.assertEqual(timeit.reindent('\n\n', 0), '\n\n')
        self.assertEqual(timeit.reindent('\n\n', 4), '\n    \n    ')

    def test_reindent_multi(self):
        if False:
            return 10
        self.assertEqual(timeit.reindent('print()\npass\nbreak', 0), 'print()\npass\nbreak')
        self.assertEqual(timeit.reindent('print()\npass\nbreak', 4), 'print()\n    pass\n    break')

    def test_timer_invalid_stmt(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, timeit.Timer, stmt=None)
        self.assertRaises(SyntaxError, timeit.Timer, stmt='return')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='yield')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='yield from ()')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='break')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='continue')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='from timeit import *')
        self.assertRaises(SyntaxError, timeit.Timer, stmt='  pass')
        self.assertRaises(SyntaxError, timeit.Timer, setup='while False:\n  pass', stmt='  break')

    def test_timer_invalid_setup(self):
        if False:
            return 10
        self.assertRaises(ValueError, timeit.Timer, setup=None)
        self.assertRaises(SyntaxError, timeit.Timer, setup='return')
        self.assertRaises(SyntaxError, timeit.Timer, setup='yield')
        self.assertRaises(SyntaxError, timeit.Timer, setup='yield from ()')
        self.assertRaises(SyntaxError, timeit.Timer, setup='break')
        self.assertRaises(SyntaxError, timeit.Timer, setup='continue')
        self.assertRaises(SyntaxError, timeit.Timer, setup='from timeit import *')
        self.assertRaises(SyntaxError, timeit.Timer, setup='  pass')

    def test_timer_empty_stmt(self):
        if False:
            for i in range(10):
                print('nop')
        timeit.Timer(stmt='')
        timeit.Timer(stmt=' \n\t\x0c')
        timeit.Timer(stmt='# comment')
    fake_setup = 'import timeit\ntimeit._fake_timer.setup()'
    fake_stmt = 'import timeit\ntimeit._fake_timer.inc()'

    def fake_callable_setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake_timer.setup()

    def fake_callable_stmt(self):
        if False:
            i = 10
            return i + 15
        self.fake_timer.inc()

    def timeit(self, stmt, setup, number=None, globals=None):
        if False:
            return 10
        self.fake_timer = FakeTimer()
        t = timeit.Timer(stmt=stmt, setup=setup, timer=self.fake_timer, globals=globals)
        kwargs = {}
        if number is None:
            number = DEFAULT_NUMBER
        else:
            kwargs['number'] = number
        delta_time = t.timeit(**kwargs)
        self.assertEqual(self.fake_timer.setup_calls, 1)
        self.assertEqual(self.fake_timer.count, number)
        self.assertEqual(delta_time, number)

    def test_timeit_zero_iters(self):
        if False:
            print('Hello World!')
        self.timeit(self.fake_stmt, self.fake_setup, number=0)

    def test_timeit_few_iters(self):
        if False:
            for i in range(10):
                print('nop')
        self.timeit(self.fake_stmt, self.fake_setup, number=3)

    def test_timeit_callable_stmt(self):
        if False:
            i = 10
            return i + 15
        self.timeit(self.fake_callable_stmt, self.fake_setup, number=3)

    def test_timeit_callable_setup(self):
        if False:
            return 10
        self.timeit(self.fake_stmt, self.fake_callable_setup, number=3)

    def test_timeit_callable_stmt_and_setup(self):
        if False:
            return 10
        self.timeit(self.fake_callable_stmt, self.fake_callable_setup, number=3)

    def test_timeit_function_zero_iters(self):
        if False:
            i = 10
            return i + 15
        delta_time = timeit.timeit(self.fake_stmt, self.fake_setup, number=0, timer=FakeTimer())
        self.assertEqual(delta_time, 0)

    def test_timeit_globals_args(self):
        if False:
            for i in range(10):
                print('nop')
        global _global_timer
        _global_timer = FakeTimer()
        t = timeit.Timer(stmt='_global_timer.inc()', timer=_global_timer)
        self.assertRaises(NameError, t.timeit, number=3)
        timeit.timeit(stmt='_global_timer.inc()', timer=_global_timer, globals=globals(), number=3)
        local_timer = FakeTimer()
        timeit.timeit(stmt='local_timer.inc()', timer=local_timer, globals=locals(), number=3)

    def repeat(self, stmt, setup, repeat=None, number=None):
        if False:
            return 10
        self.fake_timer = FakeTimer()
        t = timeit.Timer(stmt=stmt, setup=setup, timer=self.fake_timer)
        kwargs = {}
        if repeat is None:
            repeat = DEFAULT_REPEAT
        else:
            kwargs['repeat'] = repeat
        if number is None:
            number = DEFAULT_NUMBER
        else:
            kwargs['number'] = number
        delta_times = t.repeat(**kwargs)
        self.assertEqual(self.fake_timer.setup_calls, repeat)
        self.assertEqual(self.fake_timer.count, repeat * number)
        self.assertEqual(delta_times, repeat * [float(number)])

    def test_repeat_zero_reps(self):
        if False:
            print('Hello World!')
        self.repeat(self.fake_stmt, self.fake_setup, repeat=0)

    def test_repeat_zero_iters(self):
        if False:
            return 10
        self.repeat(self.fake_stmt, self.fake_setup, number=0)

    def test_repeat_few_reps_and_iters(self):
        if False:
            print('Hello World!')
        self.repeat(self.fake_stmt, self.fake_setup, repeat=3, number=5)

    def test_repeat_callable_stmt(self):
        if False:
            return 10
        self.repeat(self.fake_callable_stmt, self.fake_setup, repeat=3, number=5)

    def test_repeat_callable_setup(self):
        if False:
            return 10
        self.repeat(self.fake_stmt, self.fake_callable_setup, repeat=3, number=5)

    def test_repeat_callable_stmt_and_setup(self):
        if False:
            print('Hello World!')
        self.repeat(self.fake_callable_stmt, self.fake_callable_setup, repeat=3, number=5)

    def test_repeat_function_zero_reps(self):
        if False:
            while True:
                i = 10
        delta_times = timeit.repeat(self.fake_stmt, self.fake_setup, repeat=0, timer=FakeTimer())
        self.assertEqual(delta_times, [])

    def test_repeat_function_zero_iters(self):
        if False:
            while True:
                i = 10
        delta_times = timeit.repeat(self.fake_stmt, self.fake_setup, number=0, timer=FakeTimer())
        self.assertEqual(delta_times, DEFAULT_REPEAT * [0.0])

    def assert_exc_string(self, exc_string, expected_exc_name):
        if False:
            for i in range(10):
                print('nop')
        exc_lines = exc_string.splitlines()
        self.assertGreater(len(exc_lines), 2)
        self.assertTrue(exc_lines[0].startswith('Traceback'))
        self.assertTrue(exc_lines[-1].startswith(expected_exc_name))

    def test_print_exc(self):
        if False:
            while True:
                i = 10
        s = io.StringIO()
        t = timeit.Timer('1/0')
        try:
            t.timeit()
        except:
            t.print_exc(s)
        self.assert_exc_string(s.getvalue(), 'ZeroDivisionError')
    MAIN_DEFAULT_OUTPUT = '1 loop, best of 5: 1 sec per loop\n'

    def run_main(self, seconds_per_increment=1.0, switches=None, timer=None):
        if False:
            return 10
        if timer is None:
            timer = FakeTimer(seconds_per_increment=seconds_per_increment)
        if switches is None:
            args = []
        else:
            args = switches[:]
        args.append(self.fake_stmt)
        orig_sys_path = sys.path[:]
        with captured_stdout() as s:
            timeit.main(args=args, _wrap_timer=timer.wrap_timer)
        sys.path[:] = orig_sys_path[:]
        return s.getvalue()

    def test_main_bad_switch(self):
        if False:
            return 10
        s = self.run_main(switches=['--bad-switch'])
        self.assertEqual(s, dedent('            option --bad-switch not recognized\n            use -h/--help for command line help\n            '))

    def test_main_seconds(self):
        if False:
            i = 10
            return i + 15
        s = self.run_main(seconds_per_increment=5.5)
        self.assertEqual(s, '1 loop, best of 5: 5.5 sec per loop\n')

    def test_main_milliseconds(self):
        if False:
            i = 10
            return i + 15
        s = self.run_main(seconds_per_increment=0.0055)
        self.assertEqual(s, '50 loops, best of 5: 5.5 msec per loop\n')

    def test_main_microseconds(self):
        if False:
            i = 10
            return i + 15
        s = self.run_main(seconds_per_increment=2.5e-06, switches=['-n100'])
        self.assertEqual(s, '100 loops, best of 5: 2.5 usec per loop\n')

    def test_main_fixed_iters(self):
        if False:
            while True:
                i = 10
        s = self.run_main(seconds_per_increment=2.0, switches=['-n35'])
        self.assertEqual(s, '35 loops, best of 5: 2 sec per loop\n')

    def test_main_setup(self):
        if False:
            while True:
                i = 10
        s = self.run_main(seconds_per_increment=2.0, switches=['-n35', '-s', 'print("CustomSetup")'])
        self.assertEqual(s, 'CustomSetup\n' * DEFAULT_REPEAT + '35 loops, best of 5: 2 sec per loop\n')

    def test_main_multiple_setups(self):
        if False:
            while True:
                i = 10
        s = self.run_main(seconds_per_increment=2.0, switches=['-n35', '-s', 'a = "CustomSetup"', '-s', 'print(a)'])
        self.assertEqual(s, 'CustomSetup\n' * DEFAULT_REPEAT + '35 loops, best of 5: 2 sec per loop\n')

    def test_main_fixed_reps(self):
        if False:
            while True:
                i = 10
        s = self.run_main(seconds_per_increment=60.0, switches=['-r9'])
        self.assertEqual(s, '1 loop, best of 9: 60 sec per loop\n')

    def test_main_negative_reps(self):
        if False:
            while True:
                i = 10
        s = self.run_main(seconds_per_increment=60.0, switches=['-r-5'])
        self.assertEqual(s, '1 loop, best of 1: 60 sec per loop\n')

    @unittest.skipIf(sys.flags.optimize >= 2, 'need __doc__')
    def test_main_help(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.run_main(switches=['-h'])
        self.assertEqual(s, timeit.__doc__ + ' ')

    def test_main_verbose(self):
        if False:
            print('Hello World!')
        s = self.run_main(switches=['-v'])
        self.assertEqual(s, dedent('                1 loop -> 1 secs\n\n                raw times: 1 sec, 1 sec, 1 sec, 1 sec, 1 sec\n\n                1 loop, best of 5: 1 sec per loop\n            '))

    def test_main_very_verbose(self):
        if False:
            return 10
        s = self.run_main(seconds_per_increment=3e-05, switches=['-vv'])
        self.assertEqual(s, dedent('                1 loop -> 3e-05 secs\n                2 loops -> 6e-05 secs\n                5 loops -> 0.00015 secs\n                10 loops -> 0.0003 secs\n                20 loops -> 0.0006 secs\n                50 loops -> 0.0015 secs\n                100 loops -> 0.003 secs\n                200 loops -> 0.006 secs\n                500 loops -> 0.015 secs\n                1000 loops -> 0.03 secs\n                2000 loops -> 0.06 secs\n                5000 loops -> 0.15 secs\n                10000 loops -> 0.3 secs\n\n                raw times: 300 msec, 300 msec, 300 msec, 300 msec, 300 msec\n\n                10000 loops, best of 5: 30 usec per loop\n            '))

    def test_main_with_time_unit(self):
        if False:
            return 10
        unit_sec = self.run_main(seconds_per_increment=0.003, switches=['-u', 'sec'])
        self.assertEqual(unit_sec, '100 loops, best of 5: 0.003 sec per loop\n')
        unit_msec = self.run_main(seconds_per_increment=0.003, switches=['-u', 'msec'])
        self.assertEqual(unit_msec, '100 loops, best of 5: 3 msec per loop\n')
        unit_usec = self.run_main(seconds_per_increment=0.003, switches=['-u', 'usec'])
        self.assertEqual(unit_usec, '100 loops, best of 5: 3e+03 usec per loop\n')
        with captured_stderr() as error_stringio:
            invalid = self.run_main(seconds_per_increment=0.003, switches=['-u', 'parsec'])
        self.assertEqual(error_stringio.getvalue(), 'Unrecognized unit. Please select nsec, usec, msec, or sec.\n')

    def test_main_exception(self):
        if False:
            print('Hello World!')
        with captured_stderr() as error_stringio:
            s = self.run_main(switches=['1/0'])
        self.assert_exc_string(error_stringio.getvalue(), 'ZeroDivisionError')

    def test_main_exception_fixed_reps(self):
        if False:
            return 10
        with captured_stderr() as error_stringio:
            s = self.run_main(switches=['-n1', '1/0'])
        self.assert_exc_string(error_stringio.getvalue(), 'ZeroDivisionError')

    def autorange(self, seconds_per_increment=1 / 1024, callback=None):
        if False:
            for i in range(10):
                print('nop')
        timer = FakeTimer(seconds_per_increment=seconds_per_increment)
        t = timeit.Timer(stmt=self.fake_stmt, setup=self.fake_setup, timer=timer)
        return t.autorange(callback)

    def test_autorange(self):
        if False:
            return 10
        (num_loops, time_taken) = self.autorange()
        self.assertEqual(num_loops, 500)
        self.assertEqual(time_taken, 500 / 1024)

    def test_autorange_second(self):
        if False:
            return 10
        (num_loops, time_taken) = self.autorange(seconds_per_increment=1.0)
        self.assertEqual(num_loops, 1)
        self.assertEqual(time_taken, 1.0)

    def test_autorange_with_callback(self):
        if False:
            print('Hello World!')

        def callback(a, b):
            if False:
                for i in range(10):
                    print('nop')
            print('{} {:.3f}'.format(a, b))
        with captured_stdout() as s:
            (num_loops, time_taken) = self.autorange(callback=callback)
        self.assertEqual(num_loops, 500)
        self.assertEqual(time_taken, 500 / 1024)
        expected = '1 0.001\n2 0.002\n5 0.005\n10 0.010\n20 0.020\n50 0.049\n100 0.098\n200 0.195\n500 0.488\n'
        self.assertEqual(s.getvalue(), expected)
if __name__ == '__main__':
    unittest.main()