"""Test for the distrib_optimization example."""
import logging
import unittest
import uuid
from ast import literal_eval as make_tuple
import numpy as np
import pytest
from mock import MagicMock
from mock import patch
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import create_file
from apache_beam.testing.test_utils import read_files_from_pattern
FILE_CONTENTS = 'OP01,8,12,0,12\nOP02,30,14,3,12\nOP03,25,7,3,14\nOP04,87,7,2,2\nOP05,19,1,7,10'
EXPECTED_MAPPING = {'OP01': 'A', 'OP02': 'B', 'OP03': 'B', 'OP04': 'C', 'OP05': 'A'}

class DistribOptimizationTest(unittest.TestCase):

    @pytest.mark.sickbay_dataflow
    @pytest.mark.examples_postcommit
    def test_basics(self):
        if False:
            return 10
        test_pipeline = TestPipeline(is_integration_test=True)
        temp_location = test_pipeline.get_option('temp_location')
        input = '/'.join([temp_location, str(uuid.uuid4()), 'input.txt'])
        output = '/'.join([temp_location, str(uuid.uuid4()), 'result'])
        create_file(input, FILE_CONTENTS)
        extra_opts = {'input': input, 'output': output}
        scipy_mock = MagicMock()
        result_mock = MagicMock(x=np.ones(3))
        scipy_mock.optimize.minimize = MagicMock(return_value=result_mock)
        modules = {'scipy': scipy_mock, 'scipy.optimize': scipy_mock.optimize}
        with patch.dict('sys.modules', modules):
            from apache_beam.examples.complete import distribopt
            distribopt.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        lines = read_files_from_pattern('%s*' % output).splitlines()
        self.assertEqual(len(lines), 1)
        optimum = make_tuple(lines[0])
        self.assertAlmostEqual(optimum['cost'], 454.39597, places=3)
        self.assertDictEqual(optimum['mapping'], EXPECTED_MAPPING)
        production = optimum['production']
        for plant in ['A', 'B', 'C']:
            np.testing.assert_almost_equal(production[plant], np.ones(3))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()