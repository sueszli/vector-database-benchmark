from __future__ import absolute_import
from behave.tag_expression import TagExpression
import pytest
import unittest

class TestTagExpressionNoTags(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.e = TagExpression([])

    def test_should_match_empty_tags(self):
        if False:
            print('Hello World!')
        assert self.e.check([])

    def test_should_match_foo(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo'])

class TestTagExpressionFoo(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.e = TagExpression(['foo'])

    def test_should_not_match_no_tags(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check([])

    def test_should_match_foo(self):
        if False:
            return 10
        assert self.e.check(['foo'])

    def test_should_not_match_bar(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['bar'])

class TestTagExpressionNotFoo(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.e = TagExpression(['-foo'])

    def test_should_match_no_tags(self):
        if False:
            print('Hello World!')
        assert self.e.check([])

    def test_should_not_match_foo(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['foo'])

    def test_should_match_bar(self):
        if False:
            print('Hello World!')
        assert self.e.check(['bar'])

class TestTagExpressionFooAndBar(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.e = TagExpression(['foo', 'bar'])

    def test_should_not_match_no_tags(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check([])

    def test_should_not_match_foo(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['foo'])

    def test_should_not_match_bar(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['bar'])

    def test_should_not_match_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['other'])

    def test_should_match_foo_bar(self):
        if False:
            return 10
        assert self.e.check(['foo', 'bar'])
        assert self.e.check(['bar', 'foo'])

    def test_should_not_match_foo_other(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['foo', 'other'])
        assert not self.e.check(['other', 'foo'])

    def test_should_not_match_bar_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['bar', 'other'])
        assert not self.e.check(['other', 'bar'])

    def test_should_not_match_zap_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['zap', 'other'])
        assert not self.e.check(['other', 'zap'])

    def test_should_match_foo_bar_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo', 'bar', 'other'])
        assert self.e.check(['bar', 'other', 'foo'])
        assert self.e.check(['other', 'bar', 'foo'])

    def test_should_not_match_foo_zap_other(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['foo', 'zap', 'other'])
        assert not self.e.check(['other', 'zap', 'foo'])

    def test_should_not_match_bar_zap_other(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['bar', 'zap', 'other'])
        assert not self.e.check(['other', 'bar', 'zap'])

    def test_should_not_match_zap_baz_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['zap', 'baz', 'other'])
        assert not self.e.check(['baz', 'other', 'baz'])
        assert not self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionFooAndNotBar(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.e = TagExpression(['foo', '-bar'])

    def test_should_not_match_no_tags(self):
        if False:
            print('Hello World!')
        assert not self.e.check([])

    def test_should_match_foo(self):
        if False:
            return 10
        assert self.e.check(['foo'])

    def test_should_not_match_bar(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['bar'])

    def test_should_not_match_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['other'])

    def test_should_not_match_foo_bar(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['foo', 'bar'])
        assert not self.e.check(['bar', 'foo'])

    def test_should_match_foo_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['foo', 'other'])
        assert self.e.check(['other', 'foo'])

    def test_should_not_match_bar_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['bar', 'other'])
        assert not self.e.check(['other', 'bar'])

    def test_should_not_match_zap_other(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['bar', 'other'])
        assert not self.e.check(['other', 'bar'])

    def test_should_not_match_foo_bar_other(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['foo', 'bar', 'other'])
        assert not self.e.check(['bar', 'other', 'foo'])
        assert not self.e.check(['other', 'bar', 'foo'])

    def test_should_match_foo_zap_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['foo', 'zap', 'other'])
        assert self.e.check(['other', 'zap', 'foo'])

    def test_should_not_match_bar_zap_other(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['bar', 'zap', 'other'])
        assert not self.e.check(['other', 'bar', 'zap'])

    def test_should_not_match_zap_baz_other(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['zap', 'baz', 'other'])
        assert not self.e.check(['baz', 'other', 'baz'])
        assert not self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionNotBarAndFoo(TestTagExpressionFooAndNotBar):

    def setUp(self):
        if False:
            print('Hello World!')
        self.e = TagExpression(['-bar', 'foo'])

class TestTagExpressionNotFooAndNotBar(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.e = TagExpression(['-foo', '-bar'])

    def test_should_match_no_tags(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check([])

    def test_should_not_match_foo(self):
        if False:
            return 10
        assert not self.e.check(['foo'])

    def test_should_not_match_bar(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['bar'])

    def test_should_match_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['other'])

    def test_should_not_match_foo_bar(self):
        if False:
            return 10
        assert not self.e.check(['foo', 'bar'])
        assert not self.e.check(['bar', 'foo'])

    def test_should_not_match_foo_other(self):
        if False:
            return 10
        assert not self.e.check(['foo', 'other'])
        assert not self.e.check(['other', 'foo'])

    def test_should_not_match_bar_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['bar', 'other'])
        assert not self.e.check(['other', 'bar'])

    def test_should_match_zap_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['zap', 'other'])
        assert self.e.check(['other', 'zap'])

    def test_should_not_match_foo_bar_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['foo', 'bar', 'other'])
        assert not self.e.check(['bar', 'other', 'foo'])
        assert not self.e.check(['other', 'bar', 'foo'])

    def test_should_not_match_foo_zap_other(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['foo', 'zap', 'other'])
        assert not self.e.check(['other', 'zap', 'foo'])

    def test_should_not_match_bar_zap_other(self):
        if False:
            return 10
        assert not self.e.check(['bar', 'zap', 'other'])
        assert not self.e.check(['other', 'bar', 'zap'])

    def test_should_match_zap_baz_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['zap', 'baz', 'other'])
        assert self.e.check(['baz', 'other', 'baz'])
        assert self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionNotBarAndNotFoo(TestTagExpressionNotFooAndNotBar):

    def setUp(self):
        if False:
            return 10
        self.e = TagExpression(['-bar', '-foo'])

class TestTagExpressionFooOrBar(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.e = TagExpression(['foo,bar'])

    def test_should_not_match_no_tags(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check([])

    def test_should_match_foo(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo'])

    def test_should_match_bar(self):
        if False:
            print('Hello World!')
        assert self.e.check(['bar'])

    def test_should_not_match_other(self):
        if False:
            return 10
        assert not self.e.check(['other'])

    def test_should_match_foo_bar(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'bar'])
        assert self.e.check(['bar', 'foo'])

    def test_should_match_foo_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'other'])
        assert self.e.check(['other', 'foo'])

    def test_should_match_bar_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['bar', 'other'])
        assert self.e.check(['other', 'bar'])

    def test_should_not_match_zap_other(self):
        if False:
            return 10
        assert not self.e.check(['zap', 'other'])
        assert not self.e.check(['other', 'zap'])

    def test_should_match_foo_bar_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo', 'bar', 'other'])
        assert self.e.check(['bar', 'other', 'foo'])
        assert self.e.check(['other', 'bar', 'foo'])

    def test_should_match_foo_zap_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['foo', 'zap', 'other'])
        assert self.e.check(['other', 'zap', 'foo'])

    def test_should_match_bar_zap_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['bar', 'zap', 'other'])
        assert self.e.check(['other', 'bar', 'zap'])

    def test_should_not_match_zap_baz_other(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['zap', 'baz', 'other'])
        assert not self.e.check(['baz', 'other', 'baz'])
        assert not self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionBarOrFoo(TestTagExpressionFooOrBar):

    def setUp(self):
        if False:
            return 10
        self.e = TagExpression(['bar,foo'])

class TestTagExpressionFooOrNotBar(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.e = TagExpression(['foo,-bar'])

    def test_should_match_no_tags(self):
        if False:
            return 10
        assert self.e.check([])

    def test_should_match_foo(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo'])

    def test_should_not_match_bar(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['bar'])

    def test_should_match_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['other'])

    def test_should_match_foo_bar(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'bar'])
        assert self.e.check(['bar', 'foo'])

    def test_should_match_foo_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'other'])
        assert self.e.check(['other', 'foo'])

    def test_should_not_match_bar_other(self):
        if False:
            return 10
        assert not self.e.check(['bar', 'other'])
        assert not self.e.check(['other', 'bar'])

    def test_should_match_zap_other(self):
        if False:
            return 10
        assert self.e.check(['zap', 'other'])
        assert self.e.check(['other', 'zap'])

    def test_should_match_foo_bar_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'bar', 'other'])
        assert self.e.check(['bar', 'other', 'foo'])
        assert self.e.check(['other', 'bar', 'foo'])

    def test_should_match_foo_zap_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['foo', 'zap', 'other'])
        assert self.e.check(['other', 'zap', 'foo'])

    def test_should_not_match_bar_zap_other(self):
        if False:
            return 10
        assert not self.e.check(['bar', 'zap', 'other'])
        assert not self.e.check(['other', 'bar', 'zap'])

    def test_should_match_zap_baz_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['zap', 'baz', 'other'])
        assert self.e.check(['baz', 'other', 'baz'])
        assert self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionNotBarOrFoo(TestTagExpressionFooOrNotBar):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.e = TagExpression(['-bar,foo'])

class TestTagExpressionNotFooOrNotBar(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.e = TagExpression(['-foo,-bar'])

    def test_should_match_no_tags(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check([])

    def test_should_match_foo(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo'])

    def test_should_match_bar(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['bar'])

    def test_should_match_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['other'])

    def test_should_not_match_foo_bar(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['foo', 'bar'])
        assert not self.e.check(['bar', 'foo'])

    def test_should_match_foo_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo', 'other'])
        assert self.e.check(['other', 'foo'])

    def test_should_match_bar_other(self):
        if False:
            return 10
        assert self.e.check(['bar', 'other'])
        assert self.e.check(['other', 'bar'])

    def test_should_match_zap_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['zap', 'other'])
        assert self.e.check(['other', 'zap'])

    def test_should_not_match_foo_bar_other(self):
        if False:
            return 10
        assert not self.e.check(['foo', 'bar', 'other'])
        assert not self.e.check(['bar', 'other', 'foo'])
        assert not self.e.check(['other', 'bar', 'foo'])

    def test_should_match_foo_zap_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo', 'zap', 'other'])
        assert self.e.check(['other', 'zap', 'foo'])

    def test_should_match_bar_zap_other(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['bar', 'zap', 'other'])
        assert self.e.check(['other', 'bar', 'zap'])

    def test_should_match_zap_baz_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['zap', 'baz', 'other'])
        assert self.e.check(['baz', 'other', 'baz'])
        assert self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionNotBarOrNotFoo(TestTagExpressionNotFooOrNotBar):

    def setUp(self):
        if False:
            return 10
        self.e = TagExpression(['-bar,-foo'])

class TestTagExpressionFooOrBarAndNotZap(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.e = TagExpression(['foo,bar', '-zap'])

    def test_should_match_foo(self):
        if False:
            while True:
                i = 10
        assert self.e.check(['foo'])

    def test_should_not_match_foo_zap(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['foo', 'zap'])

    def test_should_not_match_tags(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check([])

    def test_should_match_foo(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['foo'])

    def test_should_match_bar(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.check(['bar'])

    def test_should_not_match_other(self):
        if False:
            return 10
        assert not self.e.check(['other'])

    def test_should_match_foo_bar(self):
        if False:
            print('Hello World!')
        assert self.e.check(['foo', 'bar'])
        assert self.e.check(['bar', 'foo'])

    def test_should_match_foo_other(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['foo', 'other'])
        assert self.e.check(['other', 'foo'])

    def test_should_match_bar_other(self):
        if False:
            print('Hello World!')
        assert self.e.check(['bar', 'other'])
        assert self.e.check(['other', 'bar'])

    def test_should_not_match_zap_other(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.e.check(['zap', 'other'])
        assert not self.e.check(['other', 'zap'])

    def test_should_match_foo_bar_other(self):
        if False:
            return 10
        assert self.e.check(['foo', 'bar', 'other'])
        assert self.e.check(['bar', 'other', 'foo'])
        assert self.e.check(['other', 'bar', 'foo'])

    def test_should_not_match_foo_bar_zap(self):
        if False:
            i = 10
            return i + 15
        assert not self.e.check(['foo', 'bar', 'zap'])
        assert not self.e.check(['bar', 'zap', 'foo'])
        assert not self.e.check(['zap', 'bar', 'foo'])

    def test_should_not_match_foo_zap_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['foo', 'zap', 'other'])
        assert not self.e.check(['other', 'zap', 'foo'])

    def test_should_not_match_bar_zap_other(self):
        if False:
            while True:
                i = 10
        assert not self.e.check(['bar', 'zap', 'other'])
        assert not self.e.check(['other', 'bar', 'zap'])

    def test_should_not_match_zap_baz_other(self):
        if False:
            print('Hello World!')
        assert not self.e.check(['zap', 'baz', 'other'])
        assert not self.e.check(['baz', 'other', 'baz'])
        assert not self.e.check(['other', 'baz', 'zap'])

class TestTagExpressionFoo3OrNotBar4AndZap5(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.e = TagExpression(['foo:3,-bar', 'zap:5'])

    def test_should_count_tags_for_positive_tags(self):
        if False:
            return 10
        assert self.e.limits == {'foo': 3, 'zap': 5}

    def test_should_match_foo_zap(self):
        if False:
            i = 10
            return i + 15
        assert self.e.check(['foo', 'zap'])

class TestTagExpressionParsing(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.e = TagExpression([' foo:3 , -bar ', ' zap:5 '])

    def test_should_have_limits(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.e.limits == {'zap': 5, 'foo': 3}

class TestTagExpressionTagLimits(unittest.TestCase):

    def test_should_be_counted_for_negative_tags(self):
        if False:
            while True:
                i = 10
        e = TagExpression(['-todo:3'])
        assert e.limits == {'todo': 3}

    def test_should_be_counted_for_positive_tags(self):
        if False:
            return 10
        e = TagExpression(['todo:3'])
        assert e.limits == {'todo': 3}

    def test_should_raise_an_error_for_inconsistent_limits(self):
        if False:
            return 10
        with pytest.raises(Exception):
            _ = TagExpression(['todo:3', '-todo:4'])

    def test_should_allow_duplicate_consistent_limits(self):
        if False:
            i = 10
            return i + 15
        e = TagExpression(['todo:3', '-todo:3'])
        assert e.limits == {'todo': 3}