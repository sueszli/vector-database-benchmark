import unittest
from parameterized import parameterized
from streamlit import string_util

class StringUtilTest(unittest.TestCase):

    def test_decode_ascii(self):
        if False:
            print('Hello World!')
        'Test streamlit.string_util.decode_ascii.'
        self.assertEqual('test string.', string_util.decode_ascii(b'test string.'))

    @parameterized.expand([('', False), ('A', False), ('%', False), ('😃', True), ('👨\u200d👨\u200d👧\u200d👦', True), ('😃😃', False), ('😃X', False), ('X😃', False), ('️🚨', True), ('️⛔️', True), ('️👍🏽', True)])
    def test_is_emoji(self, text: str, expected: bool):
        if False:
            for i in range(10):
                print('nop')
        'Test streamlit.string_util.is_emoji.'
        self.assertEqual(string_util.is_emoji(text), expected)

    @parameterized.expand([('', ('', '')), ('A', ('', 'A')), ('%', ('', '%')), ('😃', ('😃', '')), ('😃 page name', ('😃', 'page name')), ('😃-page name', ('😃', 'page name')), ('😃_page name', ('😃', 'page name')), ('😃 _- page name', ('😃', 'page name')), ('👨\u200d👨\u200d👧\u200d👦_page name', ('👨\u200d👨\u200d👧\u200d👦', 'page name')), ('😃😃', ('😃', '😃')), ('1️⃣X', ('1️⃣', 'X')), ('X😃', ('', 'X😃')), ('何_is_this', ('', '何_is_this'))])
    def test_extract_leading_emoji(self, text, expected):
        if False:
            while True:
                i = 10
        self.assertEqual(string_util.extract_leading_emoji(text), expected)

    def test_simplify_number(self):
        if False:
            while True:
                i = 10
        'Test streamlit.string_util.simplify_number.'
        self.assertEqual(string_util.simplify_number(100), '100')
        self.assertEqual(string_util.simplify_number(10000), '10k')
        self.assertEqual(string_util.simplify_number(1000000), '1m')
        self.assertEqual(string_util.simplify_number(1000000000), '1b')
        self.assertEqual(string_util.simplify_number(1000000000000), '1t')

    @parameterized.expand([('<br/>', True), ('<p>foo</p>', True), ('bar <div>baz</div>', True), ('Hello, World', False), ('<a>', False), ('<<a>>', False), ('a < 3 && b > 3', False)])
    def test_probably_contains_html_tags(self, text, expected):
        if False:
            return 10
        self.assertEqual(string_util.probably_contains_html_tags(text), expected)