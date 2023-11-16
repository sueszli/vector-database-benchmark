"""A word-counting workflow."""
import argparse
import unittest
import pytest
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.dataflow import dataflow_exercise_metrics_pipeline
from apache_beam.testing import metric_result_matchers
from apache_beam.testing.test_pipeline import TestPipeline

class ExerciseMetricsPipelineTest(unittest.TestCase):

    def run_pipeline(self, **opts):
        if False:
            print('Hello World!')
        test_pipeline = TestPipeline(is_integration_test=True)
        argv = test_pipeline.get_full_options_as_args(**opts)
        parser = argparse.ArgumentParser()
        (unused_known_args, pipeline_args) = parser.parse_known_args(argv)
        pipeline_options = PipelineOptions(pipeline_args)
        p = beam.Pipeline(options=pipeline_options)
        return dataflow_exercise_metrics_pipeline.apply_and_run(p)

    @pytest.mark.it_postcommit
    @unittest.skip('https://github.com/apache/beam/issues/22605')
    def test_metrics_it(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.run_pipeline()
        errors = metric_result_matchers.verify_all(result.metrics().all_metrics(), dataflow_exercise_metrics_pipeline.metric_matchers())
        self.assertFalse(errors, str(errors))
if __name__ == '__main__':
    unittest.main()