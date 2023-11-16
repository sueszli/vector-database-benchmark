"""Wrapper of Beam runners that's built for running and verifying e2e tests."""
from apache_beam.internal import pickler
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import TestOptions
from apache_beam.runners.direct.direct_runner import DirectRunner
from apache_beam.runners.runner import PipelineState
__all__ = ['TestDirectRunner']

class TestDirectRunner(DirectRunner):

    def run_pipeline(self, pipeline, options):
        if False:
            while True:
                i = 10
        'Execute test pipeline and verify test matcher'
        test_options = options.view_as(TestOptions)
        on_success_matcher = test_options.on_success_matcher
        is_streaming = options.view_as(StandardOptions).streaming
        test_options.on_success_matcher = None
        self.result = super().run_pipeline(pipeline, options)
        try:
            if not is_streaming:
                self.result.wait_until_finish()
            if on_success_matcher:
                from hamcrest import assert_that as hc_assert_that
                hc_assert_that(self.result, pickler.loads(on_success_matcher))
        finally:
            if not PipelineState.is_terminal(self.result.state):
                self.result.cancel()
                self.result.wait_until_finish()
        return self.result