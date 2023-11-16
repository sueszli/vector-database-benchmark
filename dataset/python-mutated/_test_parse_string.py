from typing import Literal
from posthog.hogql.errors import SyntaxException
from posthog.hogql.parse_string import parse_string as parse_string_py
from hogql_parser import unquote_string as unquote_string_cpp
from posthog.test.base import BaseTest

def parse_string_test_factory(backend: Literal['python', 'cpp']):
    if False:
        return 10
    parse_string = parse_string_py if backend == 'python' else unquote_string_cpp

    class TestParseString(BaseTest):

        def test_quote_types(self):
            if False:
                while True:
                    i = 10
            self.assertEqual(parse_string('`asd`'), 'asd')
            self.assertEqual(parse_string("'asd'"), 'asd')
            self.assertEqual(parse_string('"asd"'), 'asd')
            self.assertEqual(parse_string('{asd}'), 'asd')

        def test_escaped_quotes(self):
            if False:
                print('Hello World!')
            self.assertEqual(parse_string('`a``sd`'), 'a`sd')
            self.assertEqual(parse_string("'a''sd'"), "a'sd")
            self.assertEqual(parse_string('"a""sd"'), 'a"sd')
            self.assertEqual(parse_string('{a{{sd}'), 'a{sd')
            self.assertEqual(parse_string('{a}sd}'), 'a}sd')

        def test_escaped_quotes_slash(self):
            if False:
                print('Hello World!')
            self.assertEqual(parse_string('`a\\`sd`'), 'a`sd')
            self.assertEqual(parse_string("'a\\'sd'"), "a'sd")
            self.assertEqual(parse_string('"a\\"sd"'), 'a"sd')
            self.assertEqual(parse_string('{a\\{sd}'), 'a{sd')

        def test_slash_escape(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(parse_string('`a\nsd`'), 'a\nsd')
            self.assertEqual(parse_string('`a\\bsd`'), 'a\x08sd')
            self.assertEqual(parse_string('`a\\fsd`'), 'a\x0csd')
            self.assertEqual(parse_string('`a\\rsd`'), 'a\rsd')
            self.assertEqual(parse_string('`a\\nsd`'), 'a\nsd')
            self.assertEqual(parse_string('`a\\tsd`'), 'a\tsd')
            self.assertEqual(parse_string('`a\\asd`'), 'a\x07sd')
            self.assertEqual(parse_string('`a\\vsd`'), 'a\x0bsd')
            self.assertEqual(parse_string('`a\\\\sd`'), 'a\\sd')
            self.assertEqual(parse_string('`a\\0sd`'), 'asd')

        def test_slash_escape_not_escaped(self):
            if False:
                print('Hello World!')
            self.assertEqual(parse_string('`a\\xsd`'), 'a\\xsd')
            self.assertEqual(parse_string('`a\\ysd`'), 'a\\ysd')
            self.assertEqual(parse_string('`a\\osd`'), 'a\\osd')

        def test_slash_escape_slash_multiple(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(parse_string('`a\\\\nsd`'), 'a\\\nsd')
            self.assertEqual(parse_string('`a\\\\n\\sd`'), 'a\\\n\\sd')
            self.assertEqual(parse_string('`a\\\\n\\\\tsd`'), 'a\\\n\\\tsd')

        def test_raises_on_mismatched_quotes(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertRaisesMessage(SyntaxException, "Invalid string literal, must start and end with the same quote type: `asd'", parse_string, "`asd'")
    return TestParseString