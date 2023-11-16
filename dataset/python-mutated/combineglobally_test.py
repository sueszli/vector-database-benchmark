import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import combineglobally_combinefn
from . import combineglobally_function
from . import combineglobally_lambda
from . import combineglobally_multiple_arguments
from . import combineglobally_side_inputs_singleton

def check_common_items(actual):
    if False:
        while True:
            i = 10
    expected = "[START common_items]\n{'🍅', '🥕'}\n[END common_items]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_common_items_with_exceptions(actual):
    if False:
        return 10
    expected = "[START common_items_with_exceptions]\n{'🍅'}\n[END common_items_with_exceptions]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_custom_common_items(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START custom_common_items]\n{'🍅', '🍇', '🌽'}\n[END custom_common_items]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_percentages(actual):
    if False:
        return 10
    expected = "[START percentages]\n{'🥕': 0.3, '🍅': 0.6, '🍆': 0.1}\n[END percentages]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineglobally_function.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineglobally_lambda.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineglobally_multiple_arguments.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineglobally_side_inputs_singleton.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineglobally_combinefn.print', str)
class CombineGloballyTest(unittest.TestCase):

    def test_combineglobally_function(self):
        if False:
            i = 10
            return i + 15
        combineglobally_function.combineglobally_function(check_common_items)

    def test_combineglobally_lambda(self):
        if False:
            return 10
        combineglobally_lambda.combineglobally_lambda(check_common_items)

    def test_combineglobally_multiple_arguments(self):
        if False:
            return 10
        combineglobally_multiple_arguments.combineglobally_multiple_arguments(check_common_items_with_exceptions)

    def test_combineglobally_side_inputs_singleton(self):
        if False:
            while True:
                i = 10
        combineglobally_side_inputs_singleton.combineglobally_side_inputs_singleton(check_common_items_with_exceptions)

    def test_combineglobally_combinefn(self):
        if False:
            while True:
                i = 10
        combineglobally_combinefn.combineglobally_combinefn(check_percentages)
if __name__ == '__main__':
    unittest.main()