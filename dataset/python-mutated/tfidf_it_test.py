"""End-to-end test for  TF-IDF example."""
import logging
import re
import unittest
import uuid
import pytest
from apache_beam.examples.complete import tfidf
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import create_file
from apache_beam.testing.test_utils import read_files_from_pattern
EXPECTED_RESULTS = set([('ghi', '1.txt', 0.3662040962227032), ('abc', '1.txt', 0.0), ('abc', '3.txt', 0.0), ('abc', '2.txt', 0.0), ('def', '1.txt', 0.13515503603605478), ('def', '2.txt', 0.2027325540540822)])
EXPECTED_LINE_RE = "\\(u?\\'([a-z]*)\\', \\(\\'.*([0-9]\\.txt)\\', (.*)\\)\\)"

class TfIdfIT(unittest.TestCase):

    @pytest.mark.examples_postcommit
    @pytest.mark.sickbay_flink
    def test_basics(self):
        if False:
            print('Hello World!')
        test_pipeline = TestPipeline(is_integration_test=True)
        temp_location = test_pipeline.get_option('temp_location')
        input_folder = '/'.join([temp_location, str(uuid.uuid4())])
        create_file('/'.join([input_folder, '1.txt']), 'abc def ghi')
        create_file('/'.join([input_folder, '2.txt']), 'abc def')
        create_file('/'.join([input_folder, '3.txt']), 'abc')
        output = '/'.join([temp_location, str(uuid.uuid4()), 'result'])
        extra_opts = {'uris': '%s/**' % input_folder, 'output': output}
        tfidf.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        results = []
        lines = read_files_from_pattern('%s*' % output).splitlines()
        for line in lines:
            match = re.search(EXPECTED_LINE_RE, line)
            logging.info('Result line: %s', line)
            if match is not None:
                results.append((match.group(1), match.group(2), float(match.group(3))))
        logging.info('Computed results: %s', set(results))
        self.assertEqual(set(results), EXPECTED_RESULTS)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()