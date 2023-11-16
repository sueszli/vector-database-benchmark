"""Unit tests for deduplicate transform by using TestStream."""
import unittest
import pytest
import apache_beam as beam
from apache_beam.coders import coders
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.testing.util import equal_to_per_window
from apache_beam.transforms import deduplicate
from apache_beam.transforms import window
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp

@pytest.mark.no_sickbay_batch
@pytest.mark.no_sickbay_streaming
@pytest.mark.it_validatesrunner
class DeduplicateTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.runner = None
        self.options = None
        super().__init__(*args, **kwargs)

    def set_runner(self, runner):
        if False:
            i = 10
            return i + 15
        self.runner = runner

    def set_options(self, options):
        if False:
            i = 10
            return i + 15
        self.options = options

    def create_pipeline(self):
        if False:
            while True:
                i = 10
        if self.runner and self.options:
            return TestPipeline(runner=self.runner, options=self.options)
        elif self.runner:
            return TestPipeline(runner=self.runner)
        elif self.options:
            return TestPipeline(options=self.options)
        else:
            return TestPipeline()

    def test_deduplication_in_different_windows(self):
        if False:
            print('Hello World!')
        with self.create_pipeline() as p:
            test_stream = TestStream(coder=coders.StrUtf8Coder()).advance_watermark_to(0).add_elements([window.TimestampedValue('k1', 0), window.TimestampedValue('k2', 10), window.TimestampedValue('k3', 20), window.TimestampedValue('k1', 30), window.TimestampedValue('k2', 40), window.TimestampedValue('k3', 50), window.TimestampedValue('k4', 60), window.TimestampedValue('k5', 70), window.TimestampedValue('k6', 80)]).advance_watermark_to_infinity()
            res = p | test_stream | beam.WindowInto(window.FixedWindows(30)) | deduplicate.Deduplicate(processing_time_duration=10 * 60) | beam.Map(lambda e, ts=beam.DoFn.TimestampParam: (e, ts))
            expect_unique_keys_per_window = {window.IntervalWindow(0, 30): [('k1', Timestamp(0)), ('k2', Timestamp(10)), ('k3', Timestamp(20))], window.IntervalWindow(30, 60): [('k1', Timestamp(30)), ('k2', Timestamp(40)), ('k3', Timestamp(50))], window.IntervalWindow(60, 90): [('k4', Timestamp(60)), ('k5', Timestamp(70)), ('k6', Timestamp(80))]}
            assert_that(res, equal_to_per_window(expect_unique_keys_per_window), use_global_window=False, label='assert per window')

    @unittest.skip('TestStream not yet supported')
    def test_deduplication_with_event_time(self):
        if False:
            while True:
                i = 10
        deduplicate_duration = 60
        with self.create_pipeline() as p:
            test_stream = TestStream(coder=coders.StrUtf8Coder()).with_output_types(str).advance_watermark_to(0).add_elements([window.TimestampedValue('k1', 0), window.TimestampedValue('k2', 20), window.TimestampedValue('k3', 30)]).advance_watermark_to(30).add_elements([window.TimestampedValue('k1', 40), window.TimestampedValue('k2', 50), window.TimestampedValue('k3', 60)]).advance_watermark_to(deduplicate_duration).add_elements([window.TimestampedValue('k1', 70)]).advance_watermark_to_infinity()
            res = p | test_stream | deduplicate.Deduplicate(event_time_duration=Duration(deduplicate_duration)) | beam.Map(lambda e, ts=beam.DoFn.TimestampParam: (e, ts))
            assert_that(res, equal_to([('k1', Timestamp(0)), ('k2', Timestamp(20)), ('k3', Timestamp(30)), ('k1', Timestamp(70))]))

    @unittest.skip('TestStream not yet supported')
    def test_deduplication_with_processing_time(self):
        if False:
            print('Hello World!')
        deduplicate_duration = 60
        with self.create_pipeline() as p:
            test_stream = TestStream(coder=coders.StrUtf8Coder()).with_output_types(str).advance_watermark_to(0).add_elements([window.TimestampedValue('k1', 0), window.TimestampedValue('k2', 20), window.TimestampedValue('k3', 30)]).advance_processing_time(30).add_elements([window.TimestampedValue('k1', 40), window.TimestampedValue('k2', 50), window.TimestampedValue('k3', 60)]).advance_processing_time(deduplicate_duration).add_elements([window.TimestampedValue('k1', 70)]).advance_watermark_to_infinity()
            res = p | test_stream | deduplicate.Deduplicate(processing_time_duration=Duration(deduplicate_duration)) | beam.Map(lambda e, ts=beam.DoFn.TimestampParam: (e, ts))
            assert_that(res, equal_to([('k1', Timestamp(0)), ('k2', Timestamp(20)), ('k3', Timestamp(30)), ('k1', Timestamp(70))]))
if __name__ == '__main__':
    unittest.main()