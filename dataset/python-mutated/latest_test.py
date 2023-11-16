import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import latest_globally
from . import latest_per_key

def check_latest_element(actual):
    if False:
        i = 10
        return i + 15
    expected = '[START latest_element]\n🍆\n[END latest_element]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_latest_elements_per_key(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START latest_elements_per_key]\n('spring', '🥕')\n('summer', '🍅')\n('autumn', '🍆')\n('winter', '🥬')\n[END latest_elements_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.latest_globally.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.latest_per_key.print', str)
class LatestTest(unittest.TestCase):

    def test_latest_globally(self):
        if False:
            while True:
                i = 10
        latest_globally.latest_globally(check_latest_element)

    def test_latest_per_key(self):
        if False:
            i = 10
            return i + 15
        latest_per_key.latest_per_key(check_latest_elements_per_key)
if __name__ == '__main__':
    unittest.main()