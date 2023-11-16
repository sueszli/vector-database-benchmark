"""Unit tests for the PTransform and descendants."""
import inspect
import time
import unittest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.transforms.periodicsequence import PeriodicSequence

class PeriodicSequenceTest(unittest.TestCase):

    def test_periodicsequence_outputs_valid_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        start_offset = 1
        start_time = time.time() + start_offset
        duration = 1
        end_time = start_time + duration
        interval = 0.25
        with TestPipeline() as p:
            result = p | 'ImpulseElement' >> beam.Create([(start_time, end_time, interval)]) | 'ImpulseSeqGen' >> PeriodicSequence()
            k = [start_time + x * interval for x in range(0, int(duration / interval), 1)]
            self.assertEqual(result.is_bounded, False)
            assert_that(result, equal_to(k))

    def test_periodicimpulse_windowing_on_si(self):
        if False:
            i = 10
            return i + 15
        start_offset = -15
        it = time.time() + start_offset
        duration = 15
        et = it + duration
        interval = 5
        with TestPipeline() as p:
            si = p | 'PeriodicImpulse' >> PeriodicImpulse(it, et, interval, True) | 'AddKey' >> beam.Map(lambda v: ('key', v)) | 'GBK' >> beam.GroupByKey() | 'SortGBK' >> beam.MapTuple(lambda k, vs: (k, sorted(vs)))
            actual = si
            k = [('key', [it + x * interval]) for x in range(0, int(duration / interval), 1)]
            assert_that(actual, equal_to(k))

    def test_periodicimpulse_default_start(self):
        if False:
            print('Hello World!')
        default_parameters = inspect.signature(PeriodicImpulse.__init__).parameters
        it = default_parameters['start_timestamp'].default
        duration = 1
        et = it + duration
        interval = 0.5
        is_same_type = isinstance(it, type(default_parameters['stop_timestamp'].default))
        error = "'start_timestamp' and 'stop_timestamp' have different type"
        assert is_same_type, error
        with TestPipeline() as p:
            result = p | 'PeriodicImpulse' >> PeriodicImpulse(it, et, interval)
            k = [it + x * interval for x in range(0, int(duration / interval))]
            self.assertEqual(result.is_bounded, False)
            assert_that(result, equal_to(k))

    def test_periodicsequence_outputs_valid_sequence_in_past(self):
        if False:
            while True:
                i = 10
        start_offset = -10000
        it = time.time() + start_offset
        duration = 5
        et = it + duration
        interval = 1
        with TestPipeline() as p:
            result = p | 'ImpulseElement' >> beam.Create([(it, et, interval)]) | 'ImpulseSeqGen' >> PeriodicSequence()
            k = [it + x * interval for x in range(0, int(duration / interval), 1)]
            self.assertEqual(result.is_bounded, False)
            assert_that(result, equal_to(k))
if __name__ == '__main__':
    unittest.main()