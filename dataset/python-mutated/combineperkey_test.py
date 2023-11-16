import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import combineperkey_combinefn
from . import combineperkey_function
from . import combineperkey_lambda
from . import combineperkey_multiple_arguments
from . import combineperkey_side_inputs_dict
from . import combineperkey_side_inputs_iter
from . import combineperkey_side_inputs_singleton
from . import combineperkey_simple

def check_total(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START total]\n('🥕', 5)\n('🍆', 1)\n('🍅', 12)\n[END total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_saturated_total(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START saturated_total]\n('🥕', 5)\n('🍆', 1)\n('🍅', 8)\n[END saturated_total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_bounded_total(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START bounded_total]\n('🥕', 5)\n('🍆', 2)\n('🍅', 8)\n[END bounded_total]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_average(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START average]\n('🥕', 2.5)\n('🍆', 1.0)\n('🍅', 4.0)\n[END average]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_simple.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_function.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_lambda.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_multiple_arguments.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_multiple_arguments.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_side_inputs_singleton.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_side_inputs_iter.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_side_inputs_dict.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.combineperkey_combinefn.print', str)
class CombinePerKeyTest(unittest.TestCase):

    def test_combineperkey_simple(self):
        if False:
            for i in range(10):
                print('nop')
        combineperkey_simple.combineperkey_simple(check_total)

    def test_combineperkey_function(self):
        if False:
            print('Hello World!')
        combineperkey_function.combineperkey_function(check_saturated_total)

    def test_combineperkey_lambda(self):
        if False:
            i = 10
            return i + 15
        combineperkey_lambda.combineperkey_lambda(check_saturated_total)

    def test_combineperkey_multiple_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        combineperkey_multiple_arguments.combineperkey_multiple_arguments(check_saturated_total)

    def test_combineperkey_side_inputs_singleton(self):
        if False:
            print('Hello World!')
        combineperkey_side_inputs_singleton.combineperkey_side_inputs_singleton(check_saturated_total)

    def test_combineperkey_side_inputs_iter(self):
        if False:
            for i in range(10):
                print('nop')
        combineperkey_side_inputs_iter.combineperkey_side_inputs_iter(check_bounded_total)

    def test_combineperkey_side_inputs_dict(self):
        if False:
            i = 10
            return i + 15
        combineperkey_side_inputs_dict.combineperkey_side_inputs_dict(check_bounded_total)

    def test_combineperkey_combinefn(self):
        if False:
            for i in range(10):
                print('nop')
        combineperkey_combinefn.combineperkey_combinefn(check_average)
if __name__ == '__main__':
    unittest.main()