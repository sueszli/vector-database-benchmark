"""End-to-end test for Estimate Pi example."""
import json
import logging
import unittest
import uuid
import pytest
from apache_beam.examples.complete import estimate_pi
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import read_files_from_pattern

class EstimatePiIT(unittest.TestCase):

    @pytest.mark.no_xdist
    @pytest.mark.examples_postcommit
    def test_estimate_pi_output_file(self):
        if False:
            print('Hello World!')
        test_pipeline = TestPipeline(is_integration_test=True)
        OUTPUT_FILE = 'gs://temp-storage-for-end-to-end-tests/py-it-cloud/output'
        output = '/'.join([OUTPUT_FILE, str(uuid.uuid4()), 'result'])
        extra_opts = {'output': output}
        estimate_pi.run(test_pipeline.get_full_options_as_args(**extra_opts))
        result = read_files_from_pattern('%s*' % output)
        [_, _, estimated_pi] = json.loads(result.strip())
        self.assertTrue(3.125 <= estimated_pi <= 3.155)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()