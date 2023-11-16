import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import regex_all_matches
from . import regex_find
from . import regex_find_all
from . import regex_find_kv
from . import regex_matches
from . import regex_matches_kv
from . import regex_replace_all
from . import regex_replace_first
from . import regex_split

def check_matches(actual):
    if False:
        return 10
    expected = '[START plants_matches]\n🍓, Strawberry, perennial\n🥕, Carrot, biennial\n🍆, Eggplant, perennial\n🍅, Tomato, annual\n🥔, Potato, perennial\n[END plants_matches]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_all_matches(actual):
    if False:
        print('Hello World!')
    expected = "[START plants_all_matches]\n['🍓, Strawberry, perennial', '🍓', 'Strawberry', 'perennial']\n['🥕, Carrot, biennial', '🥕', 'Carrot', 'biennial']\n['🍆, Eggplant, perennial', '🍆', 'Eggplant', 'perennial']\n['🍅, Tomato, annual', '🍅', 'Tomato', 'annual']\n['🥔, Potato, perennial', '🥔', 'Potato', 'perennial']\n[END plants_all_matches]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_matches_kv(actual):
    if False:
        return 10
    expected = "[START plants_matches_kv]\n('🍓', '🍓, Strawberry, perennial')\n('🥕', '🥕, Carrot, biennial')\n('🍆', '🍆, Eggplant, perennial')\n('🍅', '🍅, Tomato, annual')\n('🥔', '🥔, Potato, perennial')\n[END plants_matches_kv]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_find_all(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START plants_find_all]\n['🍓, Strawberry, perennial']\n['🥕, Carrot, biennial']\n['🍆, Eggplant, perennial', '🍌, Banana, perennial']\n['🍅, Tomato, annual', '🍉, Watermelon, annual']\n['🥔, Potato, perennial']\n[END plants_find_all]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_find_kv(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START plants_find_kv]\n('🍓', '🍓, Strawberry, perennial')\n('🥕', '🥕, Carrot, biennial')\n('🍆', '🍆, Eggplant, perennial')\n('🍌', '🍌, Banana, perennial')\n('🍅', '🍅, Tomato, annual')\n('🍉', '🍉, Watermelon, annual')\n('🥔', '🥔, Potato, perennial')\n[END plants_find_kv]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_replace_all(actual):
    if False:
        i = 10
        return i + 15
    expected = '[START plants_replace_all]\n🍓,Strawberry,perennial\n🥕,Carrot,biennial\n🍆,Eggplant,perennial\n🍅,Tomato,annual\n🥔,Potato,perennial\n[END plants_replace_all]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_replace_first(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '[START plants_replace_first]\n🍓: Strawberry, perennial\n🥕: Carrot, biennial\n🍆: Eggplant, perennial\n🍅: Tomato, annual\n🥔: Potato, perennial\n[END plants_replace_first]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_split(actual):
    if False:
        i = 10
        return i + 15
    expected = "[START plants_split]\n['🍓', 'Strawberry', 'perennial']\n['🥕', 'Carrot', 'biennial']\n['🍆', 'Eggplant', 'perennial']\n['🍅', 'Tomato', 'annual']\n['🥔', 'Potato', 'perennial']\n[END plants_split]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_matches.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_all_matches.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_matches_kv.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_find.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_find_all.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_find_kv.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_replace_all.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_replace_first.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.regex_split.print', str)
class RegexTest(unittest.TestCase):

    def test_matches(self):
        if False:
            while True:
                i = 10
        regex_matches.regex_matches(check_matches)

    def test_all_matches(self):
        if False:
            print('Hello World!')
        regex_all_matches.regex_all_matches(check_all_matches)

    def test_matches_kv(self):
        if False:
            while True:
                i = 10
        regex_matches_kv.regex_matches_kv(check_matches_kv)

    def test_find(self):
        if False:
            i = 10
            return i + 15
        regex_find.regex_find(check_matches)

    def test_find_all(self):
        if False:
            return 10
        regex_find_all.regex_find_all(check_find_all)

    def test_find_kv(self):
        if False:
            return 10
        regex_find_kv.regex_find_kv(check_find_kv)

    def test_replace_all(self):
        if False:
            for i in range(10):
                print('nop')
        regex_replace_all.regex_replace_all(check_replace_all)

    def test_replace_first(self):
        if False:
            while True:
                i = 10
        regex_replace_first.regex_replace_first(check_replace_first)

    def test_split(self):
        if False:
            print('Hello World!')
        regex_split.regex_split(check_split)
if __name__ == '__main__':
    unittest.main()