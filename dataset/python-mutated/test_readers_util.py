from .base import TestCase
from graphite.readers import merge_with_cache
from graphite.wsgi import application
from six.moves import range

class MergeWithCacheTests(TestCase):
    maxDiff = None

    def test_merge_with_cache_with_different_step_no_data(self):
        if False:
            print('Hello World!')
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, None))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(None)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_sum(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum', raw_step=1)
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(60)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_sum_no_raw_step(self):
        if False:
            for i in range(10):
                print('nop')
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(60)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_sum_same_raw_step(self):
        if False:
            while True:
                i = 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(60)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_sum_and_raw_step(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum', raw_step=30)
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(2)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_average(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='average')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(1)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_avg_zero(self):
        if False:
            i = 10
            return i + 15
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1 if i % 2 == 0 else None))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='avg_zero')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(0.5)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_max(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='max')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(1)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_min(self):
        if False:
            for i in range(10):
                print('nop')
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='min')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(1)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_last(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='last')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(1)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_with_different_step_bad(self):
        if False:
            print('Hello World!')
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size // 2, start + window_size, 1):
            cache_results.append((i, 1))
        with self.assertRaisesRegexp(Exception, "Invalid consolidation function: 'bad_function'"):
            values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='bad_function')

    def test_merge_with_cache_beyond_max_range(self):
        if False:
            return 10
        start = 1465844460
        window_size = 7200
        step = 60
        values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            values.append(None)
        cache_results = []
        for i in range(start + window_size, start + window_size * 2, 1):
            cache_results.append((i, None))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values, func='sum')
        expected_values = list(range(0, window_size // 2, step))
        for i in range(0, window_size // 2, step):
            expected_values.append(None)
        self.assertEqual(expected_values, values)

    def test_merge_with_cache_when_previous_window_in_cache(self):
        if False:
            i = 10
            return i + 15
        start = 1465844460
        window_size = 3600
        step = 60
        values = self._create_none_window(step)
        cache_results = []
        prev_window_start = start - window_size
        prev_window_end = prev_window_start + window_size
        for i in range(prev_window_start, prev_window_end, step):
            cache_results.append((i, 1))
        values = merge_with_cache(cached_datapoints=cache_results, start=start, step=step, values=values)
        self.assertEqual(self._create_none_window(step), values)

    @staticmethod
    def _create_none_window(points_per_window):
        if False:
            print('Hello World!')
        return [None for _ in range(0, points_per_window)]