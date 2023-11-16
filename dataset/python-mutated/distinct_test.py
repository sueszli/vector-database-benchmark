import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import distinct

def check_unique_elements(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '[START unique_elements]\n🥕\n🍆\n🍅\n[END unique_elements]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.distinct.print', str)
class DistinctTest(unittest.TestCase):

    def test_distinct(self):
        if False:
            i = 10
            return i + 15
        distinct.distinct(check_unique_elements)
if __name__ == '__main__':
    unittest.main()