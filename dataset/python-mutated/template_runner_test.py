"""Unit tests for templated pipelines."""
import json
import tempfile
import unittest
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pipeline import Pipeline
from apache_beam.runners.dataflow.dataflow_runner import DataflowRunner
try:
    from apache_beam.runners.dataflow.internal import apiclient
except ImportError:
    apiclient = None

@unittest.skipIf(apiclient is None, 'GCP dependencies are not installed')
class TemplatingDataflowRunnerTest(unittest.TestCase):
    """TemplatingDataflow tests."""

    def test_full_completion(self):
        if False:
            i = 10
            return i + 15
        dummy_file = tempfile.NamedTemporaryFile(delete=False)
        dummy_file_name = dummy_file.name
        dummy_file.close()
        dummy_dir = tempfile.mkdtemp()
        remote_runner = DataflowRunner()
        options = PipelineOptions(['--sdk_location=' + dummy_file_name, '--job_name=test-job', '--project=apache-beam-testing', '--region=us-central1', '--staging_location=gs://apache-beam-testing-stg/stg/', '--temp_location=gs://apache-beam-testing-temp/tmp', '--template_location=' + dummy_file_name])
        with Pipeline(remote_runner, options) as pipeline:
            pipeline | beam.Create([1, 2, 3]) | beam.Map(lambda x: x)
        with open(dummy_file_name) as template_file:
            saved_job_dict = json.load(template_file)
            self.assertEqual(saved_job_dict['environment']['sdkPipelineOptions']['options']['project'], 'apache-beam-testing')
            self.assertEqual(saved_job_dict['environment']['sdkPipelineOptions']['options']['job_name'], 'test-job')
if __name__ == '__main__':
    unittest.main()