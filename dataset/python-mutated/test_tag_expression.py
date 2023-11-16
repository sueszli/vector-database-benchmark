"""
Functional tests for ``TagExpressionParser.parse()`` and ``Expression.evaluate()``.
"""
from __future__ import absolute_import, print_function
from behave.tag_expression.parser import TagExpressionParser, TagExpressionError
import pytest

class TestTagExpression(object):

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('', True, [], 'no_tags'), ('', True, ['a'], 'one tag: a'), ('', True, ['other'], 'one tag: other')])
    def test_empty_expression_is_true(self, tag_expression_text, expected, tags, case):
        if False:
            while True:
                i = 10
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('not a', False, ['a', 'other'], 'two tags: a, other'), ('not a', False, ['a'], 'one tag: a'), ('not a', True, ['other'], 'one tag: other'), ('not a', True, [], 'no_tags')])
    def test_not_operation(self, tag_expression_text, expected, tags, case):
        if False:
            while True:
                i = 10
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    def test_complex_example(self):
        if False:
            i = 10
            return i + 15
        tag_expression_text = 'not @a or @b and not @c or not @d or @e and @f'
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert False == tag_expression.evaluate('@a @c @d'.split())

    def test_with_escaped_chars(self):
        if False:
            for i in range(10):
                print('nop')
        print('NOT-SUPPORTED-YET')

    @pytest.mark.parametrize('tag_part', ['not', 'and', 'or'])
    def test_fails_when_only_operators_are_used(self, tag_part):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TagExpressionError):
            text = '{part} {part}'.format(part=tag_part)
            TagExpressionParser.parse(text)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('a and b', True, ['a', 'b'], 'both tags'), ('a and b', True, ['a', 'b', 'other'], 'both tags and more'), ('a and b', False, ['a'], 'one tag: a'), ('a and b', False, ['b'], 'one tag: b'), ('a and b', False, ['other'], 'one tag: other'), ('a and b', False, [], 'no_tags')])
    def test_and_operation(self, tag_expression_text, expected, tags, case):
        if False:
            i = 10
            return i + 15
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('a or b', True, ['a', 'b'], 'both tags'), ('a or b', True, ['a', 'b', 'other'], 'both tags and more'), ('a or b', True, ['a'], 'one tag: a'), ('a or b', True, ['b'], 'one tag: b'), ('a or b', False, ['other'], 'one tag: other'), ('a or b', False, [], 'no_tags')])
    def test_or_operation(self, tag_expression_text, expected, tags, case):
        if False:
            print('Hello World!')
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('a', True, ['a', 'other'], 'two tags: a, other'), ('a', True, ['a'], 'one tag: a'), ('a', False, ['other'], 'one tag: other'), ('a', False, [], 'no_tags')])
    def test_literal(self, tag_expression_text, expected, tags, case):
        if False:
            while True:
                i = 10
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('a and b', True, ['a', 'b'], 'two tags: a, b'), ('a and b', False, ['a'], 'one tag: a'), ('a and b', False, [], 'no_tags'), ('a or b', True, ['a', 'b'], 'two tags: a, b'), ('a or b', True, ['b'], 'one tag: b'), ('a or b', False, [], 'no_tags'), ('a and b or c', True, ['a', 'b', 'c'], 'three tags: a, b, c'), ('a and b or c', True, ['a', 'other', 'c'], 'three tags: a, other, c'), ('a and b or c', True, ['a', 'b', 'other'], 'three tags: a, b, other'), ('a and b or c', True, ['a', 'b'], 'two tags: a, b'), ('a and b or c', True, ['a', 'c'], 'two tags: a, c'), ('a and b or c', False, ['a'], 'one tag: a'), ('a and b or c', True, ['c'], 'one tag: c'), ('a and b or c', False, [], 'not tags')])
    def test_not_not_expression_sameas_expression(self, tag_expression_text, expected, tags, case):
        if False:
            i = 10
            return i + 15
        not2_tag_expression_text = 'not not ' + tag_expression_text
        tag_expression1 = TagExpressionParser.parse(tag_expression_text)
        tag_expression2 = TagExpressionParser.parse(not2_tag_expression_text)
        value1 = tag_expression1.evaluate(tags)
        value2 = tag_expression2.evaluate(tags)
        assert value1 == value2
        assert expected == value1

class TestTagExpressionExtension(object):
    """Extension of cucumber-tag-expressions to support tag-name-matching."""

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('a.*', False, [], 'no tags'), ('a.*', True, ['a.bar'], 'matching_tag'), ('a.*', False, ['a_bar'], 'similar_not_matching_tag'), ('a.*', False, ['A.bar'], 'case_insensitive'), ('a.*', False, ['other'], 'other tag'), ('*.a', False, [], 'no tags'), ('*.a', True, ['bar.a'], 'matching_tag'), ('*.a', False, ['bar_a'], 'similar_not_matching_tag'), ('*.a', False, ['other'], 'other tag'), ('*.a.*', False, [], 'no tags'), ('*.a.*', True, ['bar.a.baz'], 'matching_tag'), ('*.a.*', False, ['bar_a.baz'], 'similar_not_matching_tag'), ('*.a.*', False, ['other'], 'other tag')])
    def test_matcher(self, tag_expression_text, expected, tags, case):
        if False:
            while True:
                i = 10
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)

    @pytest.mark.parametrize('tag_expression_text, expected, tags, case', [('not a.*', True, [], 'no tags'), ('not a.*', False, ['a.bar'], 'matching_tag'), ('not a.*', True, ['a_bar'], 'similar_not_matching_tag'), ('not a.*', True, ['A.bar'], 'case_insensitive'), ('not a.*', True, ['other'], 'other tag')])
    def test_not_matcher(self, tag_expression_text, expected, tags, case):
        if False:
            while True:
                i = 10
        tag_expression = TagExpressionParser.parse(tag_expression_text)
        assert expected == tag_expression.evaluate(tags)