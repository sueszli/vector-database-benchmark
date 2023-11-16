import typing
import pytest
from hamcrest import *
import autokey.iomediator.constants as iomediator_constants
import autokey.model.key

def generate_tests_for_key_split_re():
    if False:
        while True:
            i = 10
    'Yields test_input_str, expected_split_list'
    yield ('<ctrl>+y', ['', '<ctrl>+', 'y'])
    yield ('asdf <ctrl>+y asdf ', ['asdf ', '<ctrl>+', 'y asdf '])
    yield ('<table><ctrl>+y</table>', ['', '<table>', '', '<ctrl>+', 'y', '</table>', ''])
    yield ('<!<alt_gr>+8CDATA<alt_gr>+8', ['<!', '<alt_gr>+', '8CDATA', '<alt_gr>+', '8'])
    yield ('<ctrl>y', ['', '<ctrl>', 'y'])
    yield ('Test<tab>More text', ['Test', '<tab>', 'More text'])

@pytest.mark.parametrize('input_string, expected_split', generate_tests_for_key_split_re())
def test_key_split_re(input_string: str, expected_split: typing.List[str]):
    if False:
        while True:
            i = 10
    assert_that(autokey.model.key.KEY_SPLIT_RE.split(input_string), has_items(*expected_split))