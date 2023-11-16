import logging
import math
import random
import unittest
from apache_beam import coders
from apache_beam import typehints
from apache_beam.runners.worker import opcounters
from apache_beam.runners.worker import statesampler
from apache_beam.runners.worker.opcounters import OperationCounters
from apache_beam.transforms.window import GlobalWindows
from apache_beam.utils import counters
from apache_beam.utils.counters import CounterFactory

class OldClassThatDoesNotImplementLen(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ObjectThatDoesNotImplementLen(object):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class TransformIoCounterTest(unittest.TestCase):

    def test_basic_counters(self):
        if False:
            i = 10
            return i + 15
        counter_factory = CounterFactory()
        sampler = statesampler.StateSampler('stage1', counter_factory)
        sampler.start()
        with sampler.scoped_state('step1', 'stateA'):
            counter = opcounters.SideInputReadCounter(counter_factory, sampler, declaring_step='step1', input_index=1)
        with sampler.scoped_state('step2', 'stateB'):
            with counter:
                counter.add_bytes_read(10)
            counter.update_current_step()
        sampler.stop()
        sampler.commit_counters()
        actual_counter_names = {c.name for c in counter_factory.get_counters()}
        expected_counter_names = set([counters.CounterName('read-sideinput-msecs', stage_name='stage1', step_name='step1', io_target=counters.side_input_id('step1', 1)), counters.CounterName('read-sideinput-byte-count', step_name='step1', io_target=counters.side_input_id('step1', 1)), counters.CounterName('read-sideinput-msecs', stage_name='stage1', step_name='step1', io_target=counters.side_input_id('step2', 1)), counters.CounterName('read-sideinput-byte-count', step_name='step1', io_target=counters.side_input_id('step2', 1))])
        self.assertTrue(actual_counter_names.issuperset(expected_counter_names))

class OperationCountersTest(unittest.TestCase):

    def verify_counters(self, opcounts, expected_elements, expected_size=None):
        if False:
            return 10
        self.assertEqual(expected_elements, opcounts.element_counter.value())
        if expected_size is not None:
            if math.isnan(expected_size):
                self.assertTrue(math.isnan(opcounts.mean_byte_counter.value()[0]))
            else:
                self.assertEqual(expected_size, opcounts.mean_byte_counter.value()[0])

    def test_update_int(self):
        if False:
            print('Hello World!')
        opcounts = OperationCounters(CounterFactory(), 'some-name', coders.PickleCoder(), 0)
        self.verify_counters(opcounts, 0)
        opcounts.update_from(GlobalWindows.windowed_value(1))
        opcounts.update_collect()
        self.verify_counters(opcounts, 1)

    def test_update_str(self):
        if False:
            return 10
        coder = coders.PickleCoder()
        opcounts = OperationCounters(CounterFactory(), 'some-name', coder, 0)
        self.verify_counters(opcounts, 0, float('nan'))
        value = GlobalWindows.windowed_value('abcde')
        opcounts.update_from(value)
        opcounts.update_collect()
        estimated_size = coder.estimate_size(value)
        self.verify_counters(opcounts, 1, estimated_size)

    def test_update_old_object(self):
        if False:
            i = 10
            return i + 15
        coder = coders.PickleCoder()
        opcounts = OperationCounters(CounterFactory(), 'some-name', coder, 0)
        self.verify_counters(opcounts, 0, float('nan'))
        obj = OldClassThatDoesNotImplementLen()
        value = GlobalWindows.windowed_value(obj)
        opcounts.update_from(value)
        opcounts.update_collect()
        estimated_size = coder.estimate_size(value)
        self.verify_counters(opcounts, 1, estimated_size)

    def test_update_new_object(self):
        if False:
            i = 10
            return i + 15
        coder = coders.PickleCoder()
        opcounts = OperationCounters(CounterFactory(), 'some-name', coder, 0)
        self.verify_counters(opcounts, 0, float('nan'))
        obj = ObjectThatDoesNotImplementLen()
        value = GlobalWindows.windowed_value(obj)
        opcounts.update_from(value)
        opcounts.update_collect()
        estimated_size = coder.estimate_size(value)
        self.verify_counters(opcounts, 1, estimated_size)

    def test_update_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        coder = coders.PickleCoder()
        total_size = 0
        opcounts = OperationCounters(CounterFactory(), 'some-name', coder, 0)
        self.verify_counters(opcounts, 0, float('nan'))
        value = GlobalWindows.windowed_value('abcde')
        opcounts.update_from(value)
        opcounts.update_collect()
        total_size += coder.estimate_size(value)
        value = GlobalWindows.windowed_value('defghij')
        opcounts.update_from(value)
        opcounts.update_collect()
        total_size += coder.estimate_size(value)
        self.verify_counters(opcounts, 2, float(total_size) / 2)
        value = GlobalWindows.windowed_value('klmnop')
        opcounts.update_from(value)
        opcounts.update_collect()
        total_size += coder.estimate_size(value)
        self.verify_counters(opcounts, 3, float(total_size) / 3)

    def test_update_batch(self):
        if False:
            for i in range(10):
                print('nop')
        coder = coders.FastPrimitivesCoder()
        opcounts = OperationCounters(CounterFactory(), 'some-name', coder, 0, producer_batch_converter=typehints.batch.BatchConverter.from_typehints(element_type=typehints.Any, batch_type=typehints.List[typehints.Any]))
        size_per_element = coder.estimate_size(50)
        self.verify_counters(opcounts, 0, float('nan'))
        opcounts.update_from_batch(GlobalWindows.windowed_batch(list(range(100))))
        self.verify_counters(opcounts, 100, size_per_element)
        opcounts.update_from_batch(GlobalWindows.windowed_batch(list(range(100, 200))))
        self.verify_counters(opcounts, 200, size_per_element)

    def test_should_sample(self):
        if False:
            for i in range(10):
                print('nop')
        buckets = [0] * 300
        random.seed(1720)
        total_runs = 10 * len(buckets)
        for _ in range(total_runs):
            opcounts = OperationCounters(CounterFactory(), 'some-name', coders.PickleCoder(), 0)
            for i in range(len(buckets)):
                if opcounts.should_sample():
                    buckets[i] += 1
        for i in range(10):
            self.assertEqual(total_runs, buckets[i])
        for i in range(10, len(buckets)):
            self.assertTrue(buckets[i] > 7 * total_runs / i, 'i=%d, buckets[i]=%d, expected=%d, ratio=%f' % (i, buckets[i], 10 * total_runs / i, buckets[i] / (10.0 * total_runs / i)))
            self.assertTrue(buckets[i] < 14 * total_runs / i, 'i=%d, buckets[i]=%d, expected=%d, ratio=%f' % (i, buckets[i], 10 * total_runs / i, buckets[i] / (10.0 * total_runs / i)))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()