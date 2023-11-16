import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import flatten

def check_flatten(actual):
    if False:
        i = 10
        return i + 15
    expected = '[START flatten_result]\n🍓\n🥕\n🍆\n🍅\n🥔\n🍎\n🍐\n🍊\n[END flatten_result]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.other.flatten.print', str)
class FlattenTest(unittest.TestCase):

    def test_flatten(self):
        if False:
            i = 10
            return i + 15
        flatten.flatten(check_flatten)
if __name__ == '__main__':
    unittest.main()