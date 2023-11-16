"""Test for the minimal wordcount example."""
import collections
import logging
import re
import unittest
import uuid
import pytest
from apache_beam.examples import wordcount_minimal
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import create_file
from apache_beam.testing.test_utils import read_files_from_pattern

@pytest.mark.examples_postcommit
class WordCountMinimalTest(unittest.TestCase):
    """Unit test for wordcount_minimal example with direct runner."""
    SAMPLE_TEXT = 'a b c a b a\n aa bb cc aa bb aa'

    def test_basics(self):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(is_integration_test=True)
        temp_location = test_pipeline.get_option('temp_location')
        temp_path = '/'.join([temp_location, str(uuid.uuid4())])
        input = create_file('/'.join([temp_path, 'input.txt']), self.SAMPLE_TEXT)
        extra_opts = {'input': input, 'output': '%s.result' % temp_path}
        expected_words = collections.defaultdict(int)
        for word in re.findall('\\w+', self.SAMPLE_TEXT):
            expected_words[word] += 1
        wordcount_minimal.main(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        results = []
        lines = read_files_from_pattern(temp_path + '.result*').splitlines()
        for line in lines:
            match = re.search('([a-z]+): ([0-9]+)', line)
            if match is not None:
                results.append((match.group(1), int(match.group(2))))
        self.assertEqual(sorted(results), sorted(expected_words.items()))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()