import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import tostring_element
from . import tostring_iterables
from . import tostring_kvs

def check_plants(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '[START plants]\n🍓,Strawberry\n🥕,Carrot\n🍆,Eggplant\n🍅,Tomato\n🥔,Potato\n[END plants]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_plant_lists(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = "[START plant_lists]\n['🍓', 'Strawberry', 'perennial']\n['🥕', 'Carrot', 'biennial']\n['🍆', 'Eggplant', 'perennial']\n['🍅', 'Tomato', 'annual']\n['🥔', 'Potato', 'perennial']\n[END plant_lists]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_plants_csv(actual):
    if False:
        print('Hello World!')
    expected = '[START plants_csv]\n🍓,Strawberry,perennial\n🥕,Carrot,biennial\n🍆,Eggplant,perennial\n🍅,Tomato,annual\n🥔,Potato,perennial\n[END plants_csv]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.tostring_kvs.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.tostring_element.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.tostring_iterables.print', str)
class ToStringTest(unittest.TestCase):

    def test_tostring_kvs(self):
        if False:
            print('Hello World!')
        tostring_kvs.tostring_kvs(check_plants)

    def test_tostring_element(self):
        if False:
            while True:
                i = 10
        tostring_element.tostring_element(check_plant_lists)

    def test_tostring_iterables(self):
        if False:
            for i in range(10):
                print('nop')
        tostring_iterables.tostring_iterables(check_plants_csv)
if __name__ == '__main__':
    unittest.main()