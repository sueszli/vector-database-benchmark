import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import max_globally as beam_max_globally
from . import max_per_key as beam_max_per_key

def check_max_element(actual):
    if False:
        return 10
    expected = '[START max_element]\n4\n[END max_element]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_elements_with_max_value_per_key(actual):
    if False:
        while True:
            i = 10
    expected = "[START elements_with_max_value_per_key]\n('🥕', 3)\n('🍆', 1)\n('🍅', 5)\n[END elements_with_max_value_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.max_globally.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.max_per_key.print', str)
class MaxTest(unittest.TestCase):

    def test_max_globally(self):
        if False:
            return 10
        beam_max_globally.max_globally(check_max_element)

    def test_max_per_key(self):
        if False:
            i = 10
            return i + 15
        beam_max_per_key.max_per_key(check_elements_with_max_value_per_key)
if __name__ == '__main__':
    unittest.main()