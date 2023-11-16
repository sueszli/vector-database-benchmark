import threading
import unittest
from apache_beam.metrics.cells import CounterCell
from apache_beam.metrics.cells import DistributionCell
from apache_beam.metrics.cells import DistributionData
from apache_beam.metrics.cells import GaugeCell
from apache_beam.metrics.cells import GaugeData
from apache_beam.metrics.metricbase import MetricName

class TestCounterCell(unittest.TestCase):

    @classmethod
    def _modify_counter(cls, d):
        if False:
            return 10
        for i in range(cls.NUM_ITERATIONS):
            d.inc(i)
    NUM_THREADS = 5
    NUM_ITERATIONS = 100

    def test_parallel_access(self):
        if False:
            i = 10
            return i + 15
        threads = []
        c = CounterCell()
        for _ in range(TestCounterCell.NUM_THREADS):
            t = threading.Thread(target=TestCounterCell._modify_counter, args=(c,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total = self.NUM_ITERATIONS * (self.NUM_ITERATIONS - 1) // 2 * self.NUM_THREADS
        self.assertEqual(c.get_cumulative(), total)

    def test_basic_operations(self):
        if False:
            return 10
        c = CounterCell()
        c.inc(2)
        self.assertEqual(c.get_cumulative(), 2)
        c.dec(10)
        self.assertEqual(c.get_cumulative(), -8)
        c.dec()
        self.assertEqual(c.get_cumulative(), -9)
        c.inc()
        self.assertEqual(c.get_cumulative(), -8)

    def test_start_time_set(self):
        if False:
            while True:
                i = 10
        c = CounterCell()
        c.inc(2)
        name = MetricName('namespace', 'name1')
        mi = c.to_runner_api_monitoring_info(name, 'transform_id')
        self.assertGreater(mi.start_time.seconds, 0)

class TestDistributionCell(unittest.TestCase):

    @classmethod
    def _modify_distribution(cls, d):
        if False:
            print('Hello World!')
        for i in range(cls.NUM_ITERATIONS):
            d.update(i)
    NUM_THREADS = 5
    NUM_ITERATIONS = 100

    def test_parallel_access(self):
        if False:
            while True:
                i = 10
        threads = []
        d = DistributionCell()
        for _ in range(TestDistributionCell.NUM_THREADS):
            t = threading.Thread(target=TestDistributionCell._modify_distribution, args=(d,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total = self.NUM_ITERATIONS * (self.NUM_ITERATIONS - 1) // 2 * self.NUM_THREADS
        count = self.NUM_ITERATIONS * self.NUM_THREADS
        self.assertEqual(d.get_cumulative(), DistributionData(total, count, 0, self.NUM_ITERATIONS - 1))

    def test_basic_operations(self):
        if False:
            print('Hello World!')
        d = DistributionCell()
        d.update(10)
        self.assertEqual(d.get_cumulative(), DistributionData(10, 1, 10, 10))
        d.update(2)
        self.assertEqual(d.get_cumulative(), DistributionData(12, 2, 2, 10))
        d.update(900)
        self.assertEqual(d.get_cumulative(), DistributionData(912, 3, 2, 900))

    def test_integer_only(self):
        if False:
            return 10
        d = DistributionCell()
        d.update(3.1)
        d.update(3.2)
        d.update(3.3)
        self.assertEqual(d.get_cumulative(), DistributionData(9, 3, 3, 3))

    def test_start_time_set(self):
        if False:
            for i in range(10):
                print('nop')
        d = DistributionCell()
        d.update(3.1)
        name = MetricName('namespace', 'name1')
        mi = d.to_runner_api_monitoring_info(name, 'transform_id')
        self.assertGreater(mi.start_time.seconds, 0)

class TestGaugeCell(unittest.TestCase):

    def test_basic_operations(self):
        if False:
            print('Hello World!')
        g = GaugeCell()
        g.set(10)
        self.assertEqual(g.get_cumulative().value, GaugeData(10).value)
        g.set(2)
        self.assertEqual(g.get_cumulative().value, 2)

    def test_integer_only(self):
        if False:
            for i in range(10):
                print('nop')
        g = GaugeCell()
        g.set(3.3)
        self.assertEqual(g.get_cumulative().value, 3)

    def test_combine_appropriately(self):
        if False:
            while True:
                i = 10
        g1 = GaugeCell()
        g1.set(3)
        g2 = GaugeCell()
        g2.set(1)
        result = g2.combine(g1)
        self.assertEqual(result.data.value, 1)

    def test_start_time_set(self):
        if False:
            return 10
        g1 = GaugeCell()
        g1.set(3)
        name = MetricName('namespace', 'name1')
        mi = g1.to_runner_api_monitoring_info(name, 'transform_id')
        self.assertGreater(mi.start_time.seconds, 0)
if __name__ == '__main__':
    unittest.main()