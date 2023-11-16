"""End-to-end test for Bigquery tornadoes example."""
import logging
import time
import unittest
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples.cookbook import bigquery_tornadoes
from apache_beam.io.gcp.tests import utils
from apache_beam.io.gcp.tests.bigquery_matcher import BigqueryMatcher
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline

class BigqueryTornadoesIT(unittest.TestCase):
    DEFAULT_CHECKSUM = 'd860e636050c559a16a791aff40d6ad809d4daf0'

    @pytest.mark.examples_postcommit
    @pytest.mark.it_postcommit
    def test_bigquery_tornadoes_it(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline(is_integration_test=True)
        project = test_pipeline.get_option('project')
        dataset = 'BigQueryTornadoesIT'
        table = 'monthly_tornadoes_%s' % int(round(time.time() * 1000))
        output_table = '.'.join([dataset, table])
        query = 'SELECT month, tornado_count FROM `%s`' % output_table
        pipeline_verifiers = [PipelineStateMatcher(), BigqueryMatcher(project=project, query=query, checksum=self.DEFAULT_CHECKSUM)]
        extra_opts = {'output': output_table, 'on_success_matcher': all_of(*pipeline_verifiers)}
        self.addCleanup(utils.delete_bq_table, project, dataset, table)
        bigquery_tornadoes.run(test_pipeline.get_full_options_as_args(**extra_opts))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()