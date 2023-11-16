import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import combinevalues_combinefn
from . import combinevalues_function
from . import combinevalues_lambda
from . import combinevalues_multiple_arguments
from . import combinevalues_side_inputs_dict
from . import combinevalues_side_inputs_iter
from . import combinevalues_side_inputs_singleton

def check_total(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START total]\n('🥕', 5)\n('🍆', 1)\n('🍅', 12)\n[END total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_saturated_total(actual):
    if False:
        print('Hello World!')
    expected = "[START saturated_total]\n('🥕', 5)\n('🍆', 1)\n('🍅', 8)\n[END saturated_total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_bounded_total(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START bounded_total]\n('🥕', 5)\n('🍆', 2)\n('🍅', 8)\n[END bounded_total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_percentages_per_season(actual):
    if False:
        return 10
    expected = "[START percentages_per_season]\n('spring', {'🥕': 0.4, '🍅': 0.4, '🍆': 0.2})\n('summer', {'🥕': 0.2, '🍅': 0.6, '🌽': 0.2})\n('fall', {'🥕': 0.5, '🍅': 0.5})\n('winter', {'🍆': 1.0})\n[END percentages_per_season]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_function.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_lambda.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_multiple_arguments.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_side_inputs_singleton.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_side_inputs_iter.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_side_inputs_dict.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combinevalues_combinefn.print', str)
class CombineValuesTest(unittest.TestCase):

    def test_combinevalues_function(self):
        if False:
            while True:
                i = 10
        combinevalues_function.combinevalues_function(check_saturated_total)

    def test_combinevalues_lambda(self):
        if False:
            return 10
        combinevalues_lambda.combinevalues_lambda(check_saturated_total)

    def test_combinevalues_multiple_arguments(self):
        if False:
            i = 10
            return i + 15
        combinevalues_multiple_arguments.combinevalues_multiple_arguments(check_saturated_total)

    def test_combinevalues_side_inputs_singleton(self):
        if False:
            for i in range(10):
                print('nop')
        combinevalues_side_inputs_singleton.combinevalues_side_inputs_singleton(check_saturated_total)

    def test_combinevalues_side_inputs_iter(self):
        if False:
            i = 10
            return i + 15
        combinevalues_side_inputs_iter.combinevalues_side_inputs_iter(check_bounded_total)

    def test_combinevalues_side_inputs_dict(self):
        if False:
            return 10
        combinevalues_side_inputs_dict.combinevalues_side_inputs_dict(check_bounded_total)

    def test_combinevalues_combinefn(self):
        if False:
            return 10
        combinevalues_combinefn.combinevalues_combinefn(check_percentages_per_season)
if __name__ == '__main__':
    unittest.main()