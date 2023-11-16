import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import groupintobatches

def check_batches_with_keys(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START batches_with_keys]\n('spring', ['🍓', '🥕', '🍆'])\n('summer', ['🥕', '🍅', '🌽'])\n('spring', ['🍅'])\n('fall', ['🥕', '🍅'])\n('winter', ['🍆'])\n[END batches_with_keys]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected, lambda batch: (batch[0], len(batch[1])))

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.groupintobatches.print', str)
class GroupIntoBatchesTest(unittest.TestCase):

    def test_groupintobatches(self):
        if False:
            print('Hello World!')
        groupintobatches.groupintobatches(check_batches_with_keys)
if __name__ == '__main__':
    unittest.main()