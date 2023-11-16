import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import top_largest
from . import top_largest_per_key
from . import top_of
from . import top_per_key
from . import top_smallest
from . import top_smallest_per_key

def check_largest_elements(actual):
    if False:
        while True:
            i = 10
    expected = '[START largest_elements]\n[4, 3]\n[END largest_elements]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_largest_elements_per_key(actual):
    if False:
        return 10
    expected = "[START largest_elements_per_key]\n('🥕', [3, 2])\n('🍆', [1])\n('🍅', [5, 4])\n[END largest_elements_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_smallest_elements(actual):
    if False:
        i = 10
        return i + 15
    expected = '[START smallest_elements]\n[1, 2]\n[END smallest_elements]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_smallest_elements_per_key(actual):
    if False:
        while True:
            i = 10
    expected = "[START smallest_elements_per_key]\n('🥕', [2, 3])\n('🍆', [1])\n('🍅', [3, 4])\n[END smallest_elements_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_shortest_elements(actual):
    if False:
        while True:
            i = 10
    expected = "[START shortest_elements]\n['🌽 Corn', '🥕 Carrot']\n[END shortest_elements]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_shortest_elements_per_key(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START shortest_elements_per_key]\n('spring', ['🥕 Carrot', '🍓 Strawberry'])\n('summer', ['🌽 Corn', '🥕 Carrot'])\n('fall', ['🥕 Carrot', '🍏 Green apple'])\n('winter', ['🍆 Eggplant'])\n[END shortest_elements_per_key]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_largest.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_largest_per_key.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_smallest.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_smallest_per_key.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_of.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.top_per_key.print', str)
class TopTest(unittest.TestCase):

    def test_top_largest(self):
        if False:
            print('Hello World!')
        top_largest.top_largest(check_largest_elements)

    def test_top_largest_per_key(self):
        if False:
            while True:
                i = 10
        top_largest_per_key.top_largest_per_key(check_largest_elements_per_key)

    def test_top_smallest(self):
        if False:
            while True:
                i = 10
        top_smallest.top_smallest(check_smallest_elements)

    def test_top_smallest_per_key(self):
        if False:
            print('Hello World!')
        top_smallest_per_key.top_smallest_per_key(check_smallest_elements_per_key)

    def test_top_of(self):
        if False:
            i = 10
            return i + 15
        top_of.top_of(check_shortest_elements)

    def test_top_per_key(self):
        if False:
            i = 10
            return i + 15
        top_per_key.top_per_key(check_shortest_elements_per_key)
if __name__ == '__main__':
    unittest.main()