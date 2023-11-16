"""End-to-end tests for the taxiride examples."""
import logging
import os
import unittest
import uuid
import pandas as pd
import pytest
from apache_beam.examples.dataframe import taxiride
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.testing.test_pipeline import TestPipeline

class TaxirideIT(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.outdir = self.test_pipeline.get_option('temp_location') + '/taxiride_it-' + str(uuid.uuid4())
        self.output_path = os.path.join(self.outdir, 'output.csv')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        FileSystems.delete([self.outdir + '/'])

    @pytest.mark.it_postcommit
    def test_aggregation(self):
        if False:
            print('Hello World!')
        taxiride.run_aggregation_pipeline(self.test_pipeline, 'gs://apache-beam-samples/nyc_taxi/2018/*.csv', self.output_path)
        expected = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'taxiride_2018_aggregation_truth.csv'), comment='#')
        expected = expected.sort_values('DOLocationID').reset_index(drop=True)

        def read_csv(path):
            if False:
                print('Hello World!')
            with FileSystems.open(path) as fp:
                return pd.read_csv(fp)
        result = pd.concat((read_csv(metadata.path) for metadata in FileSystems.match([f'{self.output_path}*'])[0].metadata_list))
        result = result.sort_values('DOLocationID').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, result)

    @pytest.mark.it_postcommit
    def test_enrich(self):
        if False:
            while True:
                i = 10
        self.test_pipeline.get_pipeline_options().view_as(WorkerOptions).machine_type = 'e2-highmem-2'
        taxiride.run_enrich_pipeline(self.test_pipeline, 'gs://apache-beam-samples/nyc_taxi/2018/*.csv', self.output_path)
        expected = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'taxiride_2018_enrich_truth.csv'), comment='#')
        expected = expected.sort_values('Borough').reset_index(drop=True)

        def read_csv(path):
            if False:
                for i in range(10):
                    print('nop')
            with FileSystems.open(path) as fp:
                return pd.read_csv(fp)
        result = pd.concat((read_csv(metadata.path) for metadata in FileSystems.match([f'{self.output_path}*'])[0].metadata_list))
        result = result.sort_values('Borough').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, result)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()