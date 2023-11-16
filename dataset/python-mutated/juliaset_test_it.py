"""Integration test for the juliaset example."""
import logging
import os
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples.complete.juliaset.juliaset import juliaset
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners.runner import PipelineState
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline

@pytest.mark.it_postcommit
class JuliaSetTestIT(unittest.TestCase):
    GRID_SIZE = 1000

    def test_run_example_with_setup_file(self):
        if False:
            print('Hello World!')
        pipeline = TestPipeline(is_integration_test=True)
        coordinate_output = FileSystems.join(pipeline.get_option('output'), 'juliaset-{}'.format(str(uuid.uuid4())), 'coordinates.txt')
        extra_args = {'coordinate_output': coordinate_output, 'grid_size': self.GRID_SIZE, 'setup_file': os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'setup.py')), 'on_success_matcher': all_of(PipelineStateMatcher(PipelineState.DONE))}
        args = pipeline.get_full_options_as_args(**extra_args)
        juliaset.run(args)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()