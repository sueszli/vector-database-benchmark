"""
Alternative approach to test TagExpression by testing all possible combinations.

REQUIRES: Python >= 2.6, because itertools.combinations() is used.
"""
from __future__ import absolute_import
import itertools
from six.moves import range
import pytest
from behave.tag_expression import TagExpression
has_combinations = hasattr(itertools, 'combinations')
if has_combinations:

    def all_combinations(items):
        if False:
            return 10
        variants = []
        for n in range(len(items) + 1):
            variants.extend(itertools.combinations(items, n))
        return variants
    NO_TAGS = '__NO_TAGS__'

    def make_tags_line(tags):
        if False:
            i = 10
            return i + 15
        '\n        Convert into tags-line as in feature file.\n        '
        if tags:
            return '@' + ' @'.join(tags)
        return NO_TAGS
    TestCase = object

    class TestAllCombinations(TestCase):

        def test_all_combinations_with_2values(self):
            if False:
                i = 10
                return i + 15
            items = '@one @two'.split()
            expected = [(), ('@one',), ('@two',), ('@one', '@two')]
            actual = all_combinations(items)
            assert actual == expected
            assert len(actual) == 4

        def test_all_combinations_with_3values(self):
            if False:
                while True:
                    i = 10
            items = '@one @two @three'.split()
            expected = [(), ('@one',), ('@two',), ('@three',), ('@one', '@two'), ('@one', '@three'), ('@two', '@three'), ('@one', '@two', '@three')]
            actual = all_combinations(items)
            assert actual == expected
            assert len(actual) == 8

    class TagExpressionTestCase(TestCase):

        def assert_tag_expression_matches(self, tag_expression, tag_combinations, expected):
            if False:
                print('Hello World!')
            matched = [make_tags_line(c) for c in tag_combinations if tag_expression.check(c)]
            assert matched == expected

        def assert_tag_expression_mismatches(self, tag_expression, tag_combinations, expected):
            if False:
                for i in range(10):
                    print('nop')
            mismatched = [make_tags_line(c) for c in tag_combinations if not tag_expression.check(c)]
            assert mismatched == expected

    class TestTagExpressionWith1Term(TagExpressionTestCase):
        """
        ALL_COMBINATIONS[4] with: @foo @other
            self.NO_TAGS,
            "@foo", "@other",
            "@foo @other",
        """
        tags = ('foo', 'other')
        tag_combinations = all_combinations(tags)

        def test_matches__foo(self):
            if False:
                for i in range(10):
                    print('nop')
            tag_expression = TagExpression(['@foo'])
            expected = ['@foo', '@foo @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__not_foo(self):
            if False:
                while True:
                    i = 10
            tag_expression = TagExpression(['-@foo'])
            expected = [NO_TAGS, '@other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

    class TestTagExpressionWith2Terms(TagExpressionTestCase):
        """
        ALL_COMBINATIONS[8] with: @foo @bar @other
            self.NO_TAGS,
            "@foo", "@bar", "@other",
            "@foo @bar", "@foo @other", "@bar @other",
            "@foo @bar @other",
        """
        tags = ('foo', 'bar', 'other')
        tag_combinations = all_combinations(tags)

        def test_matches__foo_or_bar(self):
            if False:
                for i in range(10):
                    print('nop')
            tag_expression = TagExpression(['@foo,@bar'])
            expected = ['@foo', '@bar', '@foo @bar', '@foo @other', '@bar @other', '@foo @bar @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__foo_or_not_bar(self):
            if False:
                print('Hello World!')
            tag_expression = TagExpression(['@foo,-@bar'])
            expected = [NO_TAGS, '@foo', '@other', '@foo @bar', '@foo @other', '@foo @bar @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__not_foo_or_not_bar(self):
            if False:
                while True:
                    i = 10
            tag_expression = TagExpression(['-@foo,-@bar'])
            expected = [NO_TAGS, '@foo', '@bar', '@other', '@foo @other', '@bar @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__foo_and_bar(self):
            if False:
                i = 10
                return i + 15
            tag_expression = TagExpression(['@foo', '@bar'])
            expected = ['@foo @bar', '@foo @bar @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__foo_and_not_bar(self):
            if False:
                return 10
            tag_expression = TagExpression(['@foo', '-@bar'])
            expected = ['@foo', '@foo @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

        def test_matches__not_foo_and_not_bar(self):
            if False:
                i = 10
                return i + 15
            tag_expression = TagExpression(['-@foo', '-@bar'])
            expected = [NO_TAGS, '@other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, expected)

    class TestTagExpressionWith3Terms(TagExpressionTestCase):
        """
        ALL_COMBINATIONS[16] with: @foo @bar @zap @other
            self.NO_TAGS,
            "@foo", "@bar", "@zap", "@other",
            "@foo @bar", "@foo @zap", "@foo @other",
            "@bar @zap", "@bar @other",
            "@zap @other",
            "@foo @bar @zap", "@foo @bar @other", "@foo @zap @other",
            "@bar @zap @other",
            "@foo @bar @zap @other",
        """
        tags = ('foo', 'bar', 'zap', 'other')
        tag_combinations = all_combinations(tags)

        def test_matches__foo_or_bar_or_zap(self):
            if False:
                for i in range(10):
                    print('nop')
            tag_expression = TagExpression(['@foo,@bar,@zap'])
            matched = ['@foo', '@bar', '@zap', '@foo @bar', '@foo @zap', '@foo @other', '@bar @zap', '@bar @other', '@zap @other', '@foo @bar @zap', '@foo @bar @other', '@foo @zap @other', '@bar @zap @other', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = [NO_TAGS, '@other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__foo_or_not_bar_or_zap(self):
            if False:
                i = 10
                return i + 15
            tag_expression = TagExpression(['@foo,-@bar,@zap'])
            matched = [NO_TAGS, '@foo', '@zap', '@other', '@foo @bar', '@foo @zap', '@foo @other', '@bar @zap', '@zap @other', '@foo @bar @zap', '@foo @bar @other', '@foo @zap @other', '@bar @zap @other', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = ['@bar', '@bar @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__foo_or_not_bar_or_not_zap(self):
            if False:
                print('Hello World!')
            tag_expression = TagExpression(['foo,-@bar,-@zap'])
            matched = [NO_TAGS, '@foo', '@bar', '@zap', '@other', '@foo @bar', '@foo @zap', '@foo @other', '@bar @other', '@zap @other', '@foo @bar @zap', '@foo @bar @other', '@foo @zap @other', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = ['@bar @zap', '@bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__not_foo_or_not_bar_or_not_zap(self):
            if False:
                return 10
            tag_expression = TagExpression(['-@foo,-@bar,-@zap'])
            matched = [NO_TAGS, '@foo', '@bar', '@zap', '@other', '@foo @bar', '@foo @zap', '@foo @other', '@bar @zap', '@bar @other', '@zap @other', '@foo @bar @other', '@foo @zap @other', '@bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = ['@foo @bar @zap', '@foo @bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__foo_and_bar_or_zap(self):
            if False:
                print('Hello World!')
            tag_expression = TagExpression(['@foo', '@bar,@zap'])
            matched = ['@foo @bar', '@foo @zap', '@foo @bar @zap', '@foo @bar @other', '@foo @zap @other', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = [NO_TAGS, '@foo', '@bar', '@zap', '@other', '@foo @other', '@bar @zap', '@bar @other', '@zap @other', '@bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__foo_and_bar_or_not_zap(self):
            if False:
                print('Hello World!')
            tag_expression = TagExpression(['@foo', '@bar,-@zap'])
            matched = ['@foo', '@foo @bar', '@foo @other', '@foo @bar @zap', '@foo @bar @other', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = [NO_TAGS, '@bar', '@zap', '@other', '@foo @zap', '@bar @zap', '@bar @other', '@zap @other', '@foo @zap @other', '@bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__foo_and_bar_and_zap(self):
            if False:
                return 10
            tag_expression = TagExpression(['@foo', '@bar', '@zap'])
            matched = ['@foo @bar @zap', '@foo @bar @zap @other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = [NO_TAGS, '@foo', '@bar', '@zap', '@other', '@foo @bar', '@foo @zap', '@foo @other', '@bar @zap', '@bar @other', '@zap @other', '@foo @bar @other', '@foo @zap @other', '@bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)

        def test_matches__not_foo_and_not_bar_and_not_zap(self):
            if False:
                print('Hello World!')
            tag_expression = TagExpression(['-@foo', '-@bar', '-@zap'])
            matched = [NO_TAGS, '@other']
            self.assert_tag_expression_matches(tag_expression, self.tag_combinations, matched)
            mismatched = ['@foo', '@bar', '@zap', '@foo @bar', '@foo @zap', '@foo @other', '@bar @zap', '@bar @other', '@zap @other', '@foo @bar @zap', '@foo @bar @other', '@foo @zap @other', '@bar @zap @other', '@foo @bar @zap @other']
            self.assert_tag_expression_mismatches(tag_expression, self.tag_combinations, mismatched)