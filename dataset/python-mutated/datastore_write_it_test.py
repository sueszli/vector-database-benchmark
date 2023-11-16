"""An integration test for datastore_write_it_pipeline

This test creates entities and writes them to Cloud Datastore. Subsequently,
these entities are read from Cloud Datastore, compared to the expected value
for the entity, and deleted.

There is no output; instead, we use `assert_that` transform to verify the
results in the pipeline.
"""
import logging
import random
import unittest
from datetime import datetime
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from apache_beam.io.gcp.datastore.v1new import datastore_write_it_pipeline
except ImportError:
    datastore_write_it_pipeline = None

class DatastoreWriteIT(unittest.TestCase):
    NUM_ENTITIES = 1001
    LIMIT = 500

    def run_datastore_write(self, limit=None):
        if False:
            while True:
                i = 10
        test_pipeline = TestPipeline(is_integration_test=True)
        current_time = datetime.now().strftime('%m%d%H%M%S')
        seed = random.randint(0, 100000)
        kind = 'testkind%s%d' % (current_time, seed)
        pipeline_verifiers = [PipelineStateMatcher()]
        extra_opts = {'kind': kind, 'num_entities': self.NUM_ENTITIES, 'on_success_matcher': all_of(*pipeline_verifiers)}
        if limit is not None:
            extra_opts['limit'] = limit
        datastore_write_it_pipeline.run(test_pipeline.get_full_options_as_args(**extra_opts))

    @pytest.mark.it_postcommit
    @unittest.skipIf(datastore_write_it_pipeline is None, 'GCP dependencies are not installed')
    def test_datastore_write_limit(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_datastore_write(limit=self.LIMIT)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()