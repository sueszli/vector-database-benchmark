import logging
import os
import time
import unittest
import uuid
import pytest
try:
    from apache_beam.examples.ml_transform import vocab_tfidf_processing
    from apache_beam.io.filesystems import FileSystems
    from apache_beam.ml.transforms.utils import ArtifactsFetcher
    from apache_beam.testing.load_tests.load_test_metrics_utils import InfluxDBMetricsPublisherOptions
    from apache_beam.testing.load_tests.load_test_metrics_utils import MetricsReader
    from apache_beam.testing.test_pipeline import TestPipeline
except ImportError:
    raise unittest.SkipTest('tensorflow_transform is not installed.')
_OUTPUT_GCS_BUCKET_ROOT = 'gs://temp-storage-for-end-to-end-tests/tft/'

def _publish_metrics(pipeline, metric_value, metrics_table, metric_name):
    if False:
        print('Hello World!')
    influx_options = InfluxDBMetricsPublisherOptions(metrics_table, pipeline.get_option('influx_db_name'), pipeline.get_option('influx_hostname'), os.getenv('INFLUXDB_USER'), os.getenv('INFLUXDB_USER_PASSWORD'))
    metric_reader = MetricsReader(project_name=pipeline.get_option('project'), bq_table=metrics_table, bq_dataset=pipeline.get_option('metrics_dataset'), publish_to_bq=True, influxdb_options=influx_options)
    metric_reader.publish_values([(metric_name, metric_value)])

@pytest.mark.uses_tft
class LargeMovieReviewDatasetProcessTest(unittest.TestCase):

    def test_process_large_movie_review_dataset(self):
        if False:
            while True:
                i = 10
        input_data_dir = 'gs://apache-beam-ml/datasets/aclImdb'
        artifact_location = os.path.join(_OUTPUT_GCS_BUCKET_ROOT, uuid.uuid4().hex)
        output_dir = os.path.join(_OUTPUT_GCS_BUCKET_ROOT, uuid.uuid4().hex)
        extra_opts = {'input_data_dir': input_data_dir, 'output_dir': output_dir, 'artifact_location': artifact_location}
        extra_opts['job_name'] = 'mltransform-large-movie-review-dataset-{}'.format(uuid.uuid4().hex)
        test_pipeline = TestPipeline(is_integration_test=True)
        start_time = time.time()
        vocab_tfidf_processing.run(test_pipeline.get_full_options_as_args(**extra_opts, save_main_session=False))
        end_time = time.time()
        metrics_table = 'ml_transform_large_movie_review_dataset_process_metrics'
        _publish_metrics(pipeline=test_pipeline, metric_value=end_time - start_time, metrics_table=metrics_table, metric_name='runtime_sec')
        artifacts_fetcher = ArtifactsFetcher(artifact_location=artifact_location)
        actual_vocab_list = artifacts_fetcher.get_vocab_list()
        expected_artifact_filepath = 'gs://apache-beam-ml/testing/expected_outputs/compute_and_apply_vocab'
        with FileSystems.open(expected_artifact_filepath, 'r') as f:
            expected_vocab_list = f.readlines()
        expected_vocab_list = [s.decode('utf-8').rstrip('\n') for s in expected_vocab_list]
        self.assertListEqual(actual_vocab_list, expected_vocab_list)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()