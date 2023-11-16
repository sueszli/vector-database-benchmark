from __future__ import absolute_import
from sre_parse import parse, AT, AT_BEGINNING, AT_BEGINNING_STRING, AT_END, AT_END_STRING
from mock import Mock
from unittest2 import TestCase
from st2common.exceptions.content import ParseException
from st2common.models.utils.action_alias_utils import ActionAliasFormatParser, search_regex_tokens, inject_immutable_parameters

class TestActionAliasParser(TestCase):

    def test_empty_string(self):
        if False:
            print('Hello World!')
        alias_format = ''
        param_stream = ''
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {})

    def test_arbitrary_pairs(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = ''
        param_stream = 'a=foobar1'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'foobar1'})
        alias_format = 'foo'
        param_stream = 'foo a="foobar2 poonies bar"'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'foobar2 poonies bar'})
        alias_format = 'foo'
        param_stream = "foo a='foobar2 poonies bar'"
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'foobar2 poonies bar'})
        alias_format = 'foo'
        param_stream = 'foo a={"foobar2": "poonies"}'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': '{"foobar2": "poonies"}'})
        alias_format = ''
        param_stream = 'a=foobar1 b="boobar2 3 4" c=\'coobar3 4\' d={"a": "b"}'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'foobar1', 'b': 'boobar2 3 4', 'c': 'coobar3 4', 'd': '{"a": "b"}'})
        alias_format = '{{ captain }} is my captain'
        param_stream = 'Malcolm Reynolds is my captain weirdo="River Tam"'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'captain': 'Malcolm Reynolds', 'weirdo': 'River Tam'})

    def test_simple_parsing(self):
        if False:
            return 10
        alias_format = 'skip {{a}} more skip {{b}} and skip more.'
        param_stream = 'skip a1 more skip b1 and skip more.'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'a1', 'b': 'b1'})

    def test_end_string_parsing(self):
        if False:
            while True:
                i = 10
        alias_format = 'skip {{a}} more skip {{b}}'
        param_stream = 'skip a1 more skip b1'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'a1', 'b': 'b1'})

    def test_spaced_parsing(self):
        if False:
            print('Hello World!')
        alias_format = 'skip {{a}} more skip {{b}} and skip more.'
        param_stream = 'skip "a1 a2" more skip b1 and skip more.'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'a1 a2', 'b': 'b1'})

    def test_default_values(self):
        if False:
            return 10
        alias_format = 'acl {{a}} {{b}} {{c}} {{d=1}}'
        param_stream = 'acl "a1 a2" "b1" "c1"'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'a1 a2', 'b': 'b1', 'c': 'c1', 'd': '1'})

    def test_spacing(self):
        if False:
            i = 10
            return i + 15
        alias_format = 'acl {{a=test}}'
        param_stream = 'acl'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'test'})

    def test_json_parsing(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = 'skip {{a}} more skip.'
        param_stream = 'skip {"a": "b", "c": "d"} more skip.'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': '{"a": "b", "c": "d"}'})

    def test_mixed_parsing(self):
        if False:
            return 10
        alias_format = 'skip {{a}} more skip {{b}}.'
        param_stream = 'skip {"a": "b", "c": "d"} more skip x.'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': '{"a": "b", "c": "d"}', 'b': 'x'})

    def test_param_spaces(self):
        if False:
            while True:
                i = 10
        alias_format = 's {{a}} more {{ b }} more {{ c=99 }} more {{ d = 99 }}'
        param_stream = 's one more two more three more'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'one', 'b': 'two', 'c': 'three', 'd': '99'})

    def test_enclosed_defaults(self):
        if False:
            while True:
                i = 10
        alias_format = 'skip {{ a = value }} more'
        param_stream = 'skip one more'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'one'})
        alias_format = 'skip {{ a = value }} more'
        param_stream = 'skip more'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'value'})

    def test_template_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = 'two by two hands of {{ color = {{ colors.default_color }} }}'
        param_stream = 'two by two hands of'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'color': '{{ colors.default_color }}'})

    def test_key_value_combinations(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = 'testing {{ a }}'
        param_stream = 'testing value b=value2'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'value', 'b': 'value2'})
        alias_format = 'testing {{ a=new }}'
        param_stream = 'testing b="another value"'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'a': 'new', 'b': 'another value'})
        alias_format = 'testing {{ b=abc }} {{ c=xyz }}'
        param_stream = 'testing newvalue d={"1": "2"} e="long value"'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'b': 'newvalue', 'c': 'xyz', 'd': '{"1": "2"}', 'e': 'long value'})

    def test_stream_is_none_with_all_default_values(self):
        if False:
            i = 10
            return i + 15
        alias_format = 'skip {{d=test1}} more skip {{e=test1}}.'
        param_stream = 'skip more skip'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'d': 'test1', 'e': 'test1'})

    def test_stream_is_not_none_some_default_values(self):
        if False:
            print('Hello World!')
        alias_format = 'skip {{d=test}} more skip {{e=test}}'
        param_stream = 'skip ponies more skip'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'d': 'ponies', 'e': 'test'})

    def test_stream_is_none_no_default_values(self):
        if False:
            i = 10
            return i + 15
        alias_format = 'skip {{d}} more skip {{e}}.'
        param_stream = None
        parser = ActionAliasFormatParser(alias_format, param_stream)
        expected_msg = 'Command "" doesn\'t match format string "skip {{d}} more skip {{e}}."'
        self.assertRaisesRegexp(ParseException, expected_msg, parser.get_extracted_param_value)

    def test_all_the_things(self):
        if False:
            return 10
        alias_format = "{{ p0='http' }} g {{ p1=p }} a " + "{{ url }} {{ p2={'a':'b'} }} {{ p3={{ e.i }} }}"
        param_stream = "g a http://google.com {{ execution.id }} p4='testing' p5={'a':'c'}"
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'p0': 'http', 'p1': 'p', 'url': 'http://google.com', 'p2': '{{ execution.id }}', 'p3': '{{ e.i }}', 'p4': 'testing', 'p5': "{'a':'c'}"})

    def test_command_doesnt_match_format_string(self):
        if False:
            i = 10
            return i + 15
        alias_format = 'foo bar ponies'
        param_stream = 'foo lulz ponies'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        expected_msg = 'Command "foo lulz ponies" doesn\'t match format string "foo bar ponies"'
        self.assertRaisesRegexp(ParseException, expected_msg, parser.get_extracted_param_value)

    def test_ending_parameters_matching(self):
        if False:
            return 10
        alias_format = 'foo bar'
        param_stream = 'foo bar pony1=foo pony2=bar'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'pony1': 'foo', 'pony2': 'bar'})

    def test_regex_beginning_anchors(self):
        if False:
            while True:
                i = 10
        alias_format = '^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)'
        param_stream = 'foo ASDF-1234'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'issue_key': 'ASDF-1234'})

    def test_regex_beginning_anchors_dont_match(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = '^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)'
        param_stream = 'bar foo ASDF-1234'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        expected_msg = 'Command "bar foo ASDF-1234" doesn\'t match format string "^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)"'
        with self.assertRaises(ParseException) as e:
            parser.get_extracted_param_value()
            self.assertEqual(e.msg, expected_msg)

    def test_regex_ending_anchors(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = 'foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)\\s*$'
        param_stream = 'foo ASDF-1234'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'issue_key': 'ASDF-1234'})

    def test_regex_ending_anchors_dont_match(self):
        if False:
            print('Hello World!')
        alias_format = 'foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)\\s*$'
        param_stream = 'foo ASDF-1234 bar'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        expected_msg = 'Command "foo ASDF-1234 bar" doesn\'t match format string "foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)\\s*$"'
        with self.assertRaises(ParseException) as e:
            parser.get_extracted_param_value()
            self.assertEqual(e.msg, expected_msg)

    def test_regex_beginning_and_ending_anchors(self):
        if False:
            for i in range(10):
                print('nop')
        alias_format = '^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+) bar\\s*$'
        param_stream = 'foo ASDF-1234 bar'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        extracted_values = parser.get_extracted_param_value()
        self.assertEqual(extracted_values, {'issue_key': 'ASDF-1234'})

    def test_regex_beginning_and_ending_anchors_dont_match(self):
        if False:
            i = 10
            return i + 15
        alias_format = '^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)\\s*$'
        param_stream = 'bar ASDF-1234'
        parser = ActionAliasFormatParser(alias_format, param_stream)
        expected_msg = 'Command "bar ASDF-1234" doesn\'t match format string "^\\s*foo (?P<issue_key>[A-Z][A-Z0-9]+-[0-9]+)\\s*$"'
        with self.assertRaises(ParseException) as e:
            parser.get_extracted_param_value()
            self.assertEqual(e.msg, expected_msg)

class TestSearchRegexTokens(TestCase):
    beginning_tokens = ((AT, AT_BEGINNING), (AT, AT_BEGINNING_STRING))
    end_tokens = ((AT, AT_END), (AT, AT_END_STRING))

    def test_beginning_tokens(self):
        if False:
            i = 10
            return i + 15
        tokens = parse('^asdf')
        self.assertTrue(search_regex_tokens(self.beginning_tokens, tokens))

    def test_no_ending_tokens(self):
        if False:
            i = 10
            return i + 15
        tokens = parse('^asdf')
        self.assertFalse(search_regex_tokens(self.end_tokens, tokens))

    def test_no_beginning_or_ending_tokens(self):
        if False:
            return 10
        tokens = parse('asdf')
        self.assertFalse(search_regex_tokens(self.beginning_tokens, tokens))
        self.assertFalse(search_regex_tokens(self.end_tokens, tokens))

    def test_backwards(self):
        if False:
            i = 10
            return i + 15
        tokens = parse('^asdf$')
        self.assertTrue(search_regex_tokens(self.end_tokens, tokens, backwards=True))

    def test_branches(self):
        if False:
            print('Hello World!')
        tokens = parse('^asdf|fdsa$')
        self.assertTrue(search_regex_tokens(self.end_tokens, tokens))

    def test_subpatterns(self):
        if False:
            i = 10
            return i + 15
        tokens = parse('^(?:asdf|fdsa$)')
        self.assertTrue(search_regex_tokens(self.end_tokens, tokens))

class TestInjectImmutableParameters(TestCase):

    def test_immutable_parameters_are_injected(self):
        if False:
            for i in range(10):
                print('nop')
        action_alias_db = Mock()
        action_alias_db.immutable_parameters = {'env': 'dev'}
        exec_params = [{'param1': 'value1', 'param2': 'value2'}]
        inject_immutable_parameters(action_alias_db, exec_params, {})
        self.assertEqual(exec_params, [{'param1': 'value1', 'param2': 'value2', 'env': 'dev'}])

    def test_immutable_parameters_with_jinja(self):
        if False:
            print('Hello World!')
        action_alias_db = Mock()
        action_alias_db.immutable_parameters = {'env': '{{ "dev" + "1" }}'}
        exec_params = [{'param1': 'value1', 'param2': 'value2'}]
        inject_immutable_parameters(action_alias_db, exec_params, {})
        self.assertEqual(exec_params, [{'param1': 'value1', 'param2': 'value2', 'env': 'dev1'}])

    def test_override_raises_error(self):
        if False:
            return 10
        action_alias_db = Mock()
        action_alias_db.immutable_parameters = {'env': 'dev'}
        exec_params = [{'param1': 'value1', 'env': 'prod'}]
        with self.assertRaises(ValueError):
            inject_immutable_parameters(action_alias_db, exec_params, {})