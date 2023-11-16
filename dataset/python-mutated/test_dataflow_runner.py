"""Wrapper of Beam runners that's built for running and verifying e2e tests."""
import logging
import time
from apache_beam.internal import pickler
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import TestOptions
from apache_beam.runners.dataflow.dataflow_runner import DataflowRunner
from apache_beam.runners.runner import PipelineState
__all__ = ['TestDataflowRunner']
WAIT_IN_STATE_TIMEOUT = 10 * 60
_LOGGER = logging.getLogger(__name__)

class TestDataflowRunner(DataflowRunner):

    def run_pipeline(self, pipeline, options):
        if False:
            print('Hello World!')
        'Execute test pipeline and verify test matcher'
        test_options = options.view_as(TestOptions)
        on_success_matcher = test_options.on_success_matcher
        wait_duration = test_options.wait_until_finish_duration
        is_streaming = options.view_as(StandardOptions).streaming
        test_options.on_success_matcher = None
        self.result = super().run_pipeline(pipeline, options)
        if self.result.has_job:
            print('Worker logs: %s' % self.build_console_url(options))
            _LOGGER.info('Console log: ')
            _LOGGER.info(self.build_console_url(options))
        try:
            self.wait_until_in_state(PipelineState.RUNNING)
            if is_streaming and (not wait_duration):
                _LOGGER.warning('Waiting indefinitely for streaming job.')
            self.result.wait_until_finish(duration=wait_duration)
            if on_success_matcher:
                from hamcrest import assert_that as hc_assert_that
                hc_assert_that(self.result, pickler.loads(on_success_matcher))
        finally:
            if not self.result.is_in_terminal_state():
                self.result.cancel()
                self.wait_until_in_state(PipelineState.CANCELLED)
        return self.result

    def build_console_url(self, options):
        if False:
            while True:
                i = 10
        'Build a console url of Dataflow job.'
        project = options.view_as(GoogleCloudOptions).project
        region_id = options.view_as(GoogleCloudOptions).region
        job_id = self.result.job_id()
        return 'https://console.cloud.google.com/dataflow/jobs/%s/%s?project=%s' % (region_id, job_id, project)

    def wait_until_in_state(self, expected_state, timeout=WAIT_IN_STATE_TIMEOUT):
        if False:
            while True:
                i = 10
        'Wait until Dataflow pipeline enters a certain state.'
        consoleUrl = f'Console URL: https://console.cloud.google.com/dataflow/<regionId>/{self.result.job_id()}?project=<projectId>'
        if not self.result.has_job:
            _LOGGER.error(consoleUrl)
            raise IOError('Failed to get the Dataflow job id.')
        start_time = time.time()
        while time.time() - start_time <= timeout:
            job_state = self.result.state
            if self.result.is_in_terminal_state() or job_state == expected_state:
                return job_state
            time.sleep(5)
        _LOGGER.error(consoleUrl)
        raise RuntimeError('Timeout after %d seconds while waiting for job %s enters expected state %s. Current state is %s.' % (timeout, self.result.job_id(), expected_state, self.result.state))