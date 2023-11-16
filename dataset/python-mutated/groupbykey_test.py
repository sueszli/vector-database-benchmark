import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import groupbykey

def check_produce_counts(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START produce_counts]\n('spring', ['🍓', '🥕', '🍆', '🍅'])\n('summer', ['🥕', '🍅', '🌽'])\n('fall', ['🥕', '🍅'])\n('winter', ['🍆'])\n[END produce_counts]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected, lambda pair: (pair[0], sorted(pair[1])))

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.groupbykey.print', str)
class GroupByKeyTest(unittest.TestCase):

    def test_groupbykey(self):
        if False:
            while True:
                i = 10
        groupbykey.groupbykey(check_produce_counts)
if __name__ == '__main__':
    unittest.main()