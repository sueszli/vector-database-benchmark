"""Integration tests for cross-language transform expansion."""
import time
import unittest
import pytest
import apache_beam as beam
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import is_empty
from apache_beam.transforms import trigger
from apache_beam.transforms import window
from apache_beam.transforms.core import DoFn
from apache_beam.transforms.periodicsequence import PeriodicSequence

@unittest.skipIf(not TestPipeline().get_pipeline_options().view_as(StandardOptions).streaming, 'Watermark tests are only valid for streaming jobs.')
class PeriodicSequenceIT(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_pipeline = TestPipeline(is_integration_test=True)

    @pytest.mark.it_postcommit
    @pytest.mark.sickbay_direct
    @pytest.mark.sickbay_spark
    @pytest.mark.timeout(1800)
    def test_periodicsequence_outputs_valid_watermarks_it(self):
        if False:
            for i in range(10):
                print('nop')
        "Tests periodic sequence with watermarks on dataflow.\n    For testing that watermarks are being correctly emitted,\n    we make sure that there's not a long gap between an element being\n    emitted and being correctly aggregated.\n    "

        class FindLongGaps(DoFn):

            def process(self, element):
                if False:
                    print('Hello World!')
                (emitted_at, unused_count) = element
                processed_at = time.time()
                if processed_at - emitted_at > 25:
                    yield ('Elements emitted took too long to process.', emitted_at, processed_at)
        start_time = time.time()
        duration_sec = 540
        end_time = start_time + duration_sec
        interval = 1
        res = self.test_pipeline | 'ImpulseElement' >> beam.Create([(start_time, end_time, interval)]) | 'ImpulseSeqGen' >> PeriodicSequence() | 'MapToCurrentTime' >> beam.Map(lambda element: time.time()) | 'window_into' >> beam.WindowInto(window.FixedWindows(2), accumulation_mode=trigger.AccumulationMode.DISCARDING) | beam.combiners.Count.PerElement() | beam.ParDo(FindLongGaps())
        assert_that(res, is_empty())
        self.test_pipeline.run().wait_until_finish()
if __name__ == '__main__':
    unittest.main()