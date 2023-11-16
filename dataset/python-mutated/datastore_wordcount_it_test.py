"""End-to-end test for Datastore Wordcount example."""
import logging
import time
import unittest
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.testing.pipeline_verifiers import FileChecksumMatcher
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from apache_beam.examples.cookbook import datastore_wordcount
except ImportError:
    datastore_wordcount = None

class DatastoreWordCountIT(unittest.TestCase):
    DATASTORE_WORDCOUNT_KIND = 'DatastoreWordCount'
    EXPECTED_CHECKSUM = '826f69ed0275858c2e098f1e8407d4e3ba5a4b3f'

    @pytest.mark.it_postcommit
    def test_datastore_wordcount_it(self):
        if False:
            while True:
                i = 10
        test_pipeline = TestPipeline(is_integration_test=True)
        kind = self.DATASTORE_WORDCOUNT_KIND
        output = '/'.join([test_pipeline.get_option('output'), str(int(time.time() * 1000)), 'datastore_wordcount_results'])
        arg_sleep_secs = test_pipeline.get_option('sleep_secs')
        sleep_secs = int(arg_sleep_secs) if arg_sleep_secs is not None else None
        pipeline_verifiers = [PipelineStateMatcher(), FileChecksumMatcher(output + '*-of-*', self.EXPECTED_CHECKSUM, sleep_secs)]
        extra_opts = {'kind': kind, 'output': output, 'read_only': True, 'on_success_matcher': all_of(*pipeline_verifiers)}
        datastore_wordcount.run(test_pipeline.get_full_options_as_args(**extra_opts))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()