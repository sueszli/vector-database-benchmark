"""End-to-end test for Autocomplete example."""
import logging
import re
import unittest
import uuid
import pytest
from apache_beam.examples.complete import autocomplete
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import create_file
from apache_beam.testing.test_utils import read_files_from_pattern

def format_output_file(output_string):
    if False:
        while True:
            i = 10

    def extract_prefix_topk_words_tuples(line):
        if False:
            while True:
                i = 10
        match = re.match('(.*): \\[(.*)\\]', line)
        prefix = match.group(1)
        topK_words_string = extract_top_k_words_tuples(match.group(2))
        return (prefix, topK_words_string)

    def extract_top_k_words_tuples(top_k_words_string):
        if False:
            i = 10
            return i + 15
        top_k_list = top_k_words_string.split('), (')
        return tuple(map(lambda top_k_string: tuple(format_top_k_tuples(top_k_string)), top_k_list))

    def format_top_k_tuples(top_k_string):
        if False:
            while True:
                i = 10
        (frequency, words) = top_k_string.replace('(', '').replace(')', '').replace('"', '').replace("'", '').replace(' ', '').split(',')
        return (int(frequency), words)
    return list(map(lambda line: extract_prefix_topk_words_tuples(line), output_string.split('\n')))

class AutocompleteIT(unittest.TestCase):
    WORDS = ['this', 'this', 'that', 'to', 'to', 'to']
    EXPECTED_PREFIXES = [('t', ((3, 'to'), (2, 'this'), (1, 'that'))), ('to', ((3, 'to'),)), ('th', ((2, 'this'), (1, 'that'))), ('thi', ((2, 'this'),)), ('this', ((2, 'this'),)), ('tha', ((1, 'that'),)), ('that', ((1, 'that'),))]

    @pytest.mark.no_xdist
    @pytest.mark.examples_postcommit
    def test_autocomplete_output_files_on_small_input(self):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(is_integration_test=True)
        OUTPUT_FILE_DIR = 'gs://temp-storage-for-end-to-end-tests/py-it-cloud/output'
        output = '/'.join([OUTPUT_FILE_DIR, str(uuid.uuid4()), 'result'])
        INPUT_FILE_DIR = 'gs://temp-storage-for-end-to-end-tests/py-it-cloud/input'
        input = '/'.join([INPUT_FILE_DIR, str(uuid.uuid4()), 'input.txt'])
        create_file(input, ' '.join(self.WORDS))
        extra_opts = {'input': input, 'output': output}
        autocomplete.run(test_pipeline.get_full_options_as_args(**extra_opts))
        result = read_files_from_pattern('%s*' % output).strip()
        self.assertEqual(sorted(self.EXPECTED_PREFIXES), sorted(format_output_file(result)))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()