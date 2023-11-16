"""End-to-end test for the wordcount example."""
import logging
import os
import time
import unittest
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples import wordcount
from apache_beam.internal.gcp import auth
from apache_beam.testing.load_tests.load_test_metrics_utils import InfluxDBMetricsPublisherOptions
from apache_beam.testing.load_tests.load_test_metrics_utils import MetricsReader
from apache_beam.testing.pipeline_verifiers import FileChecksumMatcher
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import delete_files

class WordCountIT(unittest.TestCase):
    DEFAULT_CHECKSUM = '33535a832b7db6d78389759577d4ff495980b9c0'

    @pytest.mark.it_postcommit
    @pytest.mark.it_validatescontainer
    def test_wordcount_it(self):
        if False:
            i = 10
            return i + 15
        self._run_wordcount_it(wordcount.run)

    @pytest.mark.it_postcommit
    @pytest.mark.sickbay_direct
    @pytest.mark.sickbay_spark
    @pytest.mark.sickbay_flink
    def test_wordcount_impersonation_it(self):
        if False:
            i = 10
            return i + 15
        'Tests impersonation on dataflow.\n\n    For testing impersonation, we use three ingredients:\n    - a principal to impersonate\n    - a dataflow service account that only that principal is\n      allowed to launch jobs as\n    - a temp root that only the above two accounts have access to\n\n    Jenkins and Dataflow workers both run as GCE default service account.\n    So we remove that account from all the above.\n    '
        with auth._Credentials._credentials_lock:
            auth._Credentials._credentials_init = False
        try:
            ACCOUNT_TO_IMPERSONATE = 'allows-impersonation@apache-beam-testing.iam.gserviceaccount.com'
            RUNNER_ACCOUNT = 'impersonation-dataflow-worker@apache-beam-testing.iam.gserviceaccount.com'
            TEMP_DIR = 'gs://impersonation-test-bucket/temp-it'
            STAGING_LOCATION = 'gs://impersonation-test-bucket/staging-it'
            extra_options = {'impersonate_service_account': ACCOUNT_TO_IMPERSONATE, 'service_account_email': RUNNER_ACCOUNT, 'temp_location': TEMP_DIR, 'staging_location': STAGING_LOCATION}
            self._run_wordcount_it(wordcount.run, **extra_options)
        finally:
            with auth._Credentials._credentials_lock:
                auth._Credentials._credentials_init = False

    @pytest.mark.it_validatescontainer
    def test_wordcount_it_with_prebuilt_sdk_container_local_docker(self):
        if False:
            while True:
                i = 10
        self._run_wordcount_it(wordcount.run, experiment='beam_fn_api', prebuild_sdk_container_engine='local_docker')

    @pytest.mark.it_validatescontainer
    def test_wordcount_it_with_prebuilt_sdk_container_cloud_build(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_wordcount_it(wordcount.run, experiment='beam_fn_api', prebuild_sdk_container_engine='cloud_build')

    def _run_wordcount_it(self, run_wordcount, **opts):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(is_integration_test=True)
        extra_opts = {}
        if test_pipeline.get_option('machine_type') == 't2a-standard-1' and 'prebuild_sdk_container_engine' in opts:
            pytest.skip('prebuild_sdk_container_engine not supported on ARM')
        test_output = '/'.join([test_pipeline.get_option('output'), str(int(time.time() * 1000)), 'results'])
        extra_opts['output'] = test_output
        test_input = test_pipeline.get_option('input')
        if test_input:
            extra_opts['input'] = test_input
        arg_sleep_secs = test_pipeline.get_option('sleep_secs')
        sleep_secs = int(arg_sleep_secs) if arg_sleep_secs is not None else None
        expect_checksum = test_pipeline.get_option('expect_checksum') or self.DEFAULT_CHECKSUM
        pipeline_verifiers = [PipelineStateMatcher(), FileChecksumMatcher(test_output + '*-of-*', expect_checksum, sleep_secs)]
        extra_opts['on_success_matcher'] = all_of(*pipeline_verifiers)
        extra_opts.update(opts)
        self.addCleanup(delete_files, [test_output + '*'])
        publish_to_bq = bool(test_pipeline.get_option('publish_to_big_query'))
        start_time = time.time()
        run_wordcount(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        end_time = time.time()
        run_time = end_time - start_time
        if publish_to_bq:
            self._publish_metrics(test_pipeline, run_time)

    def _publish_metrics(self, pipeline, metric_value):
        if False:
            return 10
        influx_options = InfluxDBMetricsPublisherOptions(pipeline.get_option('influx_measurement'), pipeline.get_option('influx_db_name'), pipeline.get_option('influx_hostname'), os.getenv('INFLUXDB_USER'), os.getenv('INFLUXDB_USER_PASSWORD'))
        metric_reader = MetricsReader(project_name=pipeline.get_option('project'), bq_table=pipeline.get_option('metrics_table'), bq_dataset=pipeline.get_option('metrics_dataset'), publish_to_bq=True, influxdb_options=influx_options)
        metric_reader.publish_values([('runtime', metric_value)])
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()