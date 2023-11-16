import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import min_globally as beam_min_globally
from . import min_per_key as beam_min_per_key

def check_min_element(actual):
    if False:
        print('Hello World!')
    expected = '[START min_element]\n1\n[END min_element]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_elements_with_min_value_per_key(actual):
    if False:
        return 10
    expected = "[START elements_with_min_value_per_key]\n('🥕', 2)\n('🍆', 1)\n('🍅', 3)\n[END elements_with_min_value_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.min_globally.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.min_per_key.print', str)
class MinTest(unittest.TestCase):

    def test_min_globally(self):
        if False:
            print('Hello World!')
        beam_min_globally.min_globally(check_min_element)

    def test_min_per_key(self):
        if False:
            for i in range(10):
                print('nop')
        beam_min_per_key.min_per_key(check_elements_with_min_value_per_key)
if __name__ == '__main__':
    unittest.main()