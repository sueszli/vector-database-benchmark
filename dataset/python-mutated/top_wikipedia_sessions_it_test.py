"""End-to-end test for Top Wikipedia Sessions example."""
import json
import logging
import unittest
import uuid
import pytest
from apache_beam.examples.complete import top_wikipedia_sessions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import create_file
from apache_beam.testing.test_utils import read_files_from_pattern

class ComputeTopSessionsIT(unittest.TestCase):
    EDITS = [json.dumps({'timestamp': 0.0, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.001, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.002, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.0, 'contributor_username': 'user2'}), json.dumps({'timestamp': 0.001, 'contributor_username': 'user2'}), json.dumps({'timestamp': 3.601, 'contributor_username': 'user2'}), json.dumps({'timestamp': 3.602, 'contributor_username': 'user2'}), json.dumps({'timestamp': 2 * 3600.0, 'contributor_username': 'user2'}), json.dumps({'timestamp': 35 * 24 * 3.6, 'contributor_username': 'user3'})]
    EXPECTED = ['user1 : [0.0, 3600.002) : 3 : [0.0, 2592000.0)', 'user2 : [0.0, 3603.602) : 4 : [0.0, 2592000.0)', 'user2 : [7200.0, 10800.0) : 1 : [0.0, 2592000.0)', 'user3 : [3024.0, 6624.0) : 1 : [0.0, 2592000.0)']

    @pytest.mark.sickbay_dataflow
    @pytest.mark.no_xdist
    @pytest.mark.examples_postcommit
    def test_top_wikipedia_sessions_output_files_on_small_input(self):
        if False:
            return 10
        test_pipeline = TestPipeline(is_integration_test=True)
        OUTPUT_FILE_DIR = 'gs://temp-storage-for-end-to-end-tests/py-it-cloud/output'
        output = '/'.join([OUTPUT_FILE_DIR, str(uuid.uuid4()), 'result'])
        INPUT_FILE_DIR = 'gs://temp-storage-for-end-to-end-tests/py-it-cloud/input'
        input = '/'.join([INPUT_FILE_DIR, str(uuid.uuid4()), 'input.txt'])
        create_file(input, '\n'.join(self.EDITS))
        extra_opts = {'input': input, 'output': output, 'sampling_threshold': '1.0'}
        top_wikipedia_sessions.run(test_pipeline.get_full_options_as_args(**extra_opts))
        result = read_files_from_pattern('%s*' % output).strip().splitlines()
        self.assertEqual(self.EXPECTED, sorted(result, key=lambda x: x.split()[0]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()