import contextlib
import cProfile
import gc
import logging
import os
import random
import tempfile
import time
from viztracer import VizTracer
from .base_tmpl import BaseTmpl

class Timer:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.timer = 0

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.timer = time.perf_counter()
        return self

    def __exit__(self, type, value, trace):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_time(self):
        if False:
            while True:
                i = 10
        return time.perf_counter() - self.timer

class BenchmarkTimer:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.timer_baseline = None
        self.timer_experiments = {}
        self._set_up_funcs = []

    @contextlib.contextmanager
    def time(self, title, section=None, baseline=False):
        if False:
            i = 10
            return i + 15
        for (func, args, kwargs) in self._set_up_funcs:
            func(*args, **kwargs)
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            data = {'dur': end_time - start_time, 'section': section}
            if baseline:
                self.timer_baseline = data
            else:
                if title not in self.timer_experiments:
                    self.timer_experiments[title] = []
                self.timer_experiments[title].append(data)

    def print_result(self):
        if False:
            print('Hello World!')

        def time_str(baseline, experiment):
            if False:
                while True:
                    i = 10
            return f"{experiment['dur']:.9f}({experiment['dur'] / baseline['dur']:.2f})[{experiment['section']}]"
        for experiments in self.timer_experiments.values():
            logging.info(' '.join([time_str(self.timer_baseline, experiment) for experiment in experiments]))

    def add_set_up_func(self, func, *args, **kwargs):
        if False:
            return 10
        self._set_up_funcs.append((func, args, kwargs))

class TestPerformance(BaseTmpl):

    def do_one_function(self, func):
        if False:
            for i in range(10):
                print('nop')
        bm_timer = BenchmarkTimer()
        bm_timer.add_set_up_func(gc.collect)
        gc.collect()
        gc.disable()
        with bm_timer.time('baseline', 'baseline', baseline=True):
            func()
        tracer = VizTracer(verbose=0)
        tracer.start()
        with bm_timer.time('c', 'c'):
            func()
        tracer.stop()
        with bm_timer.time('c', 'parse'):
            tracer.parse()
        with tempfile.TemporaryDirectory() as tmpdir:
            ofile = os.path.join(tmpdir, 'result.json')
            with bm_timer.time('c', 'save'):
                tracer.save(output_file=ofile)
        tracer.start()
        func()
        tracer.stop()
        with tempfile.TemporaryDirectory() as tmpdir:
            ofile = os.path.join(tmpdir, 'result.json')
            with bm_timer.time('c', 'dump'):
                tracer.dump(ofile)
        tracer.clear()
        pr = cProfile.Profile()
        pr.enable()
        with bm_timer.time('cProfile', 'cProfile'):
            func()
        pr.disable()
        gc.enable()
        bm_timer.print_result()

    def test_fib(self):
        if False:
            while True:
                i = 10

        def fib():
            if False:
                return 10

            def _fib(n):
                if False:
                    return 10
                if n <= 1:
                    return 1
                return _fib(n - 1) + _fib(n - 2)
            return _fib(23)
        self.do_one_function(fib)

    def test_slow_fib(self):
        if False:
            while True:
                i = 10

        def slow_fib():
            if False:
                print('Hello World!')

            def _fib(n):
                if False:
                    return 10
                if n <= 1:
                    return 1
                time.sleep(1e-05)
                return _fib(n - 1) + _fib(n - 2)
            return _fib(15)
        self.do_one_function(slow_fib)

    def test_qsort(self):
        if False:
            for i in range(10):
                print('nop')

        def qsort():
            if False:
                for i in range(10):
                    print('nop')

            def quicksort(array):
                if False:
                    i = 10
                    return i + 15
                if len(array) < 2:
                    return array
                (low, same, high) = ([], [], [])
                pivot = array[random.randint(0, len(array) - 1)]
                for item in array:
                    if item < pivot:
                        low.append(item)
                    elif item == pivot:
                        same.append(item)
                    elif item > pivot:
                        high.append(item)
                return quicksort(low) + same + quicksort(high)
            arr = [random.randrange(100000) for _ in range(5000)]
            quicksort(arr)
        self.do_one_function(qsort)

    def test_hanoi(self):
        if False:
            while True:
                i = 10

        def hanoi():
            if False:
                while True:
                    i = 10

            def TowerOfHanoi(n, source, destination, auxiliary):
                if False:
                    while True:
                        i = 10
                if n == 1:
                    return
                TowerOfHanoi(n - 1, source, auxiliary, destination)
                TowerOfHanoi(n - 1, auxiliary, destination, source)
            TowerOfHanoi(16, 'A', 'B', 'C')
        self.do_one_function(hanoi)

    def test_list(self):
        if False:
            return 10

        def list_operation():
            if False:
                for i in range(10):
                    print('nop')

            def ListOperation(n):
                if False:
                    i = 10
                    return i + 15
                if n == 1:
                    return [1]
                ret = ListOperation(n - 1)
                for i in range(n):
                    ret.append(i)
                return ret
            ListOperation(205)
        self.do_one_function(list_operation)

    def test_float(self):
        if False:
            return 10
        from math import cos, sin, sqrt

        class Point:
            __slots__ = ('x', 'y', 'z')

            def __init__(self, i):
                if False:
                    while True:
                        i = 10
                self.x = x = sin(i)
                self.y = cos(i) * 3
                self.z = x * x / 2

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return f'<Point: x={self.x}, y={self.y}, z={self.z}>'

            def normalize(self):
                if False:
                    i = 10
                    return i + 15
                x = self.x
                y = self.y
                z = self.z
                norm = sqrt(x * x + y * y + z * z)
                self.x /= norm
                self.y /= norm
                self.z /= norm

            def maximize(self, other):
                if False:
                    i = 10
                    return i + 15
                self.x = self.x if self.x > other.x else other.x
                self.y = self.y if self.y > other.y else other.y
                self.z = self.z if self.z > other.z else other.z
                return self

        def maximize(points):
            if False:
                print('Hello World!')
            next = points[0]
            for p in points[1:]:
                next = next.maximize(p)
            return next

        def benchmark():
            if False:
                while True:
                    i = 10
            n = 100
            points = [None] * n
            for i in range(n):
                points[i] = Point(i)
            for p in points:
                p.normalize()
            return maximize(points)
        self.do_one_function(benchmark)

class TestFilterPerformance(BaseTmpl):

    def do_one_function(self, func):
        if False:
            print('Hello World!')
        tracer = VizTracer(verbose=0)
        tracer.start()
        with Timer() as t:
            func()
            baseline = t.get_time()
        tracer.stop()
        tracer.cleanup()
        tracer.include_files = ['/']
        tracer.start()
        with Timer() as t:
            func()
            include_files = t.get_time()
        tracer.stop()
        tracer.cleanup()
        tracer.include_files = []
        tracer.max_stack_depth = 200
        tracer.start()
        with Timer() as t:
            func()
            max_stack_depth = t.get_time()
        tracer.stop()
        tracer.cleanup()
        logging.info('Filter performance:')
        logging.info(f'Baseline:        {baseline:.9f}(1)')
        logging.info(f'Include:         {include_files:.9f}({include_files / baseline:.2f})')
        logging.info(f'Max stack depth: {max_stack_depth:.9f}({max_stack_depth / baseline:.2f})')

    def test_hanoi(self):
        if False:
            for i in range(10):
                print('nop')

        def hanoi():
            if False:
                for i in range(10):
                    print('nop')

            def TowerOfHanoi(n, source, destination, auxiliary):
                if False:
                    i = 10
                    return i + 15
                if n == 1:
                    return
                TowerOfHanoi(n - 1, source, auxiliary, destination)
                TowerOfHanoi(n - 1, auxiliary, destination, source)
            TowerOfHanoi(12, 'A', 'B', 'C')
        self.do_one_function(hanoi)