import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import mean_globally
from . import mean_per_key

def check_mean_element(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '[START mean_element]\n2.5\n[END mean_element]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_elements_with_mean_value_per_key(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START elements_with_mean_value_per_key]\n('🥕', 2.5)\n('🍆', 1.0)\n('🍅', 4.0)\n[END elements_with_mean_value_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.mean_globally.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.mean_per_key.print', str)
class MeanTest(unittest.TestCase):

    def test_mean_globally(self):
        if False:
            for i in range(10):
                print('nop')
        mean_globally.mean_globally(check_mean_element)

    def test_mean_per_key(self):
        if False:
            print('Hello World!')
        mean_per_key.mean_per_key(check_elements_with_mean_value_per_key)
if __name__ == '__main__':
    unittest.main()