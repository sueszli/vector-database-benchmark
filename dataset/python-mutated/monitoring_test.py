"""Tests for monitoring."""
import time
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util

class MonitoringTest(test_util.TensorFlowTestCase):

    def test_counter(self):
        if False:
            for i in range(10):
                print('nop')
        counter = monitoring.Counter('test/counter', 'test counter')
        counter.get_cell().increase_by(1)
        self.assertEqual(counter.get_cell().value(), 1)
        counter.get_cell().increase_by(5)
        self.assertEqual(counter.get_cell().value(), 6)

    def test_multiple_counters(self):
        if False:
            print('Hello World!')
        counter1 = monitoring.Counter('test/counter1', 'test counter', 'label1')
        counter1.get_cell('foo').increase_by(1)
        self.assertEqual(counter1.get_cell('foo').value(), 1)
        counter2 = monitoring.Counter('test/counter2', 'test counter', 'label1', 'label2')
        counter2.get_cell('foo', 'bar').increase_by(5)
        self.assertEqual(counter2.get_cell('foo', 'bar').value(), 5)

    def test_same_counter(self):
        if False:
            i = 10
            return i + 15
        counter1 = monitoring.Counter('test/same_counter', 'test counter')
        counter2 = monitoring.Counter('test/same_counter', 'test counter')

    def test_int_gauge(self):
        if False:
            while True:
                i = 10
        gauge = monitoring.IntGauge('test/gauge', 'test gauge')
        gauge.get_cell().set(1)
        self.assertEqual(gauge.get_cell().value(), 1)
        gauge.get_cell().set(5)
        self.assertEqual(gauge.get_cell().value(), 5)
        gauge1 = monitoring.IntGauge('test/gauge1', 'test gauge1', 'label1')
        gauge1.get_cell('foo').set(2)
        self.assertEqual(gauge1.get_cell('foo').value(), 2)

    def test_string_gauge(self):
        if False:
            print('Hello World!')
        gauge = monitoring.StringGauge('test/gauge', 'test gauge')
        gauge.get_cell().set('left')
        self.assertEqual(gauge.get_cell().value(), 'left')
        gauge.get_cell().set('right')
        self.assertEqual(gauge.get_cell().value(), 'right')
        gauge1 = monitoring.StringGauge('test/gauge1', 'test gauge1', 'label1')
        gauge1.get_cell('foo').set('start')
        self.assertEqual(gauge1.get_cell('foo').value(), 'start')

    def test_bool_gauge(self):
        if False:
            return 10
        gauge = monitoring.BoolGauge('test/gauge', 'test gauge')
        gauge.get_cell().set(True)
        self.assertTrue(gauge.get_cell().value())
        gauge.get_cell().set(False)
        self.assertFalse(gauge.get_cell().value())
        gauge1 = monitoring.BoolGauge('test/gauge1', 'test gauge1', 'label1')
        gauge1.get_cell('foo').set(True)
        self.assertTrue(gauge1.get_cell('foo').value())

    def test_sampler(self):
        if False:
            return 10
        buckets = monitoring.ExponentialBuckets(1.0, 2.0, 2)
        sampler = monitoring.Sampler('test/sampler', buckets, 'test sampler')
        sampler.get_cell().add(1.0)
        sampler.get_cell().add(5.0)
        histogram_proto = sampler.get_cell().value()
        self.assertEqual(histogram_proto.min, 1.0)
        self.assertEqual(histogram_proto.num, 2.0)
        self.assertEqual(histogram_proto.sum, 6.0)
        sampler1 = monitoring.Sampler('test/sampler1', buckets, 'test sampler', 'label1')
        sampler1.get_cell('foo').add(2.0)
        sampler1.get_cell('foo').add(4.0)
        sampler1.get_cell('bar').add(8.0)
        histogram_proto1 = sampler1.get_cell('foo').value()
        self.assertEqual(histogram_proto1.max, 4.0)
        self.assertEqual(histogram_proto1.num, 2.0)
        self.assertEqual(histogram_proto1.sum, 6.0)

    def test_context_manager(self):
        if False:
            print('Hello World!')
        counter = monitoring.Counter('test/ctxmgr', 'test context manager', 'slot')
        with monitoring.MonitoredTimer(counter.get_cell('long')):
            time.sleep(0.01)
            with monitoring.MonitoredTimer(counter.get_cell('short')):
                time.sleep(0.01)
        self.assertGreater(counter.get_cell('long').value(), counter.get_cell('short').value())

    def test_monitored_timer_tracker(self):
        if False:
            print('Hello World!')
        counter = monitoring.Counter('test/ctxmgr', 'test context manager', 'slot')
        counter2 = monitoring.Counter('test/ctxmgr2', 'slot')
        with monitoring.MonitoredTimer(counter.get_cell('long'), 'counter'):
            time.sleep(0.01)
            self.assertIn('counter', monitoring.MonitoredTimerSections)
            with monitoring.MonitoredTimer(counter2.get_cell(), 'counter2'):
                time.sleep(0.01)
                self.assertIn('counter', monitoring.MonitoredTimerSections)
                self.assertIn('counter2', monitoring.MonitoredTimerSections)
                with monitoring.MonitoredTimer(counter.get_cell('long'), 'counter'):
                    time.sleep(0.01)
            self.assertNotIn('counter2', monitoring.MonitoredTimerSections)
        self.assertGreater(counter.get_cell('long').value(), counter.get_cell('short').value())
        self.assertGreater(counter2.get_cell().value(), 0)

    def test_repetitive_monitored_timer(self):
        if False:
            i = 10
            return i + 15
        counter = monitoring.Counter('test/ctxmgr', 'test context manager')
        with monitoring.MonitoredTimer(counter.get_cell(), monitored_section_name='action1', avoid_repetitive_counting=True):
            time.sleep(1)
            with monitoring.MonitoredTimer(counter.get_cell(), monitored_section_name='action1', avoid_repetitive_counting=True):
                time.sleep(1)
            self.assertEqual(counter.get_cell().value(), 0)
        self.assertGreater(counter.get_cell().value(), 0)

    def test_function_decorator(self):
        if False:
            i = 10
            return i + 15
        counter = monitoring.Counter('test/funcdecorator', 'test func decorator')

        @monitoring.monitored_timer(counter.get_cell())
        def timed_function(seconds):
            if False:
                i = 10
                return i + 15
            time.sleep(seconds)
        timed_function(0.001)
        self.assertGreater(counter.get_cell().value(), 1000)
if __name__ == '__main__':
    test.main()