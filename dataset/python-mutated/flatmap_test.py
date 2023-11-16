import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import flatmap_function
from . import flatmap_generator
from . import flatmap_lambda
from . import flatmap_multiple_arguments
from . import flatmap_side_inputs_dict
from . import flatmap_side_inputs_iter
from . import flatmap_side_inputs_singleton
from . import flatmap_simple
from . import flatmap_tuple

def check_plants(actual):
    if False:
        print('Hello World!')
    expected = '[START plants]\n🍓Strawberry\n🥕Carrot\n🍆Eggplant\n🍅Tomato\n🥔Potato\n[END plants]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_valid_plants(actual):
    if False:
        print('Hello World!')
    expected = "[START valid_plants]\n{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}\n{'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}\n{'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}\n{'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}\n[END valid_plants]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_simple.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_function.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_lambda.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_generator.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_multiple_arguments.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_tuple.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_side_inputs_singleton.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_side_inputs_iter.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.flatmap_side_inputs_dict.print', str)
class FlatMapTest(unittest.TestCase):

    def test_flatmap_simple(self):
        if False:
            while True:
                i = 10
        flatmap_simple.flatmap_simple(check_plants)

    def test_flatmap_function(self):
        if False:
            for i in range(10):
                print('nop')
        flatmap_function.flatmap_function(check_plants)

    def test_flatmap_lambda(self):
        if False:
            return 10
        flatmap_lambda.flatmap_lambda(check_plants)

    def test_flatmap_generator(self):
        if False:
            while True:
                i = 10
        flatmap_generator.flatmap_generator(check_plants)

    def test_flatmap_multiple_arguments(self):
        if False:
            return 10
        flatmap_multiple_arguments.flatmap_multiple_arguments(check_plants)

    def test_flatmap_tuple(self):
        if False:
            while True:
                i = 10
        flatmap_tuple.flatmap_tuple(check_plants)

    def test_flatmap_side_inputs_singleton(self):
        if False:
            print('Hello World!')
        flatmap_side_inputs_singleton.flatmap_side_inputs_singleton(check_plants)

    def test_flatmap_side_inputs_iter(self):
        if False:
            print('Hello World!')
        flatmap_side_inputs_iter.flatmap_side_inputs_iter(check_valid_plants)

    def test_flatmap_side_inputs_dict(self):
        if False:
            return 10
        flatmap_side_inputs_dict.flatmap_side_inputs_dict(check_valid_plants)
if __name__ == '__main__':
    unittest.main()