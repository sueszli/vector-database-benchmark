import collections
import re
import textwrap
import unittest
import pytest
from cupy import testing

@testing.parameterize({'actual': {'a': [1, 2], 'b': [3, 4, 5]}, 'expect': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]}, {'actual': {'a': [1, 2]}, 'expect': [{'a': 1}, {'a': 2}]}, {'actual': {'a': [1, 2], 'b': []}, 'expect': []}, {'actual': {'a': []}, 'expect': []}, {'actual': {}, 'expect': [{}]})
class TestProduct(unittest.TestCase):

    def test_product(self):
        if False:
            print('Hello World!')
        assert testing.product(self.actual) == self.expect

@testing.parameterize({'actual': [[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}], [{'c': 5}, {'c': 6}]], 'expect': [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6}, {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]}, {'actual': [[{'a': 1}, {'a': 2}], [{'b': 3}, {'b': 4}, {'b': 5}]], 'expect': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]}, {'actual': [[{'a': 1}, {'a': 2}]], 'expect': [{'a': 1}, {'a': 2}]}, {'actual': [[{'a': 1}, {'a': 2}], []], 'expect': []}, {'actual': [[]], 'expect': []}, {'actual': [], 'expect': [{}]})
class TestProductDict(unittest.TestCase):

    def test_product_dict(self):
        if False:
            for i in range(10):
                print('nop')
        assert testing.product_dict(*self.actual) == self.expect

def f(x):
    if False:
        i = 10
        return i + 15
    return x

class C(object):

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<C object>'

    def __call__(self, x):
        if False:
            print('Hello World!')
        return x

    def method(self, x):
        if False:
            while True:
                i = 10
        return x

@testing.parameterize({'callable': f}, {'callable': lambda x: x}, {'callable': C()}, {'callable': C().method})
class TestParameterize(unittest.TestCase):

    def test_callable(self):
        if False:
            return 10
        y = self.callable(1)
        assert y == 1

    def test_skip(self):
        if False:
            return 10
        self.skipTest('skip')

@testing.parameterize({'callable': f}, {'callable': lambda x: x}, {'callable': C()}, {'callable': C().method})
class TestParameterizePytestImpl:

    def test_callable(self):
        if False:
            while True:
                i = 10
        y = self.callable(1)
        assert y == 1

    def test_skip(self):
        if False:
            print('Hello World!')
        pytest.skip('skip')

@pytest.mark.parametrize(('src', 'outcomes'), [(textwrap.dedent("        @testing.parameterize({'a': 1}, {'a': 2})\n        class TestA:\n            def test_a(self):\n                assert self.a > 0\n        "), [('::TestA::test_a[_param_0_{a=1}]', 'PASSED'), ('::TestA::test_a[_param_1_{a=2}]', 'PASSED')]), (textwrap.dedent("        @testing.parameterize({'a': 1}, {'a': 2})\n        class TestA:\n            def test_a(self):\n                assert self.a == 1\n        "), [('::TestA::test_a[_param_0_{a=1}]', 'PASSED'), ('::TestA::test_a[_param_1_{a=2}]', 'FAILED')]), (textwrap.dedent("        @testing.parameterize({'a': 1}, {'b': 2})\n        class TestA:\n            def test_a(self):\n                a = getattr(self, 'a', 3)\n                b = getattr(self, 'b', 4)\n                assert (a, b) in [(1, 4), (3, 2)]\n        "), [('::TestA::test_a[_param_0_{a=1}]', 'PASSED'), ('::TestA::test_a[_param_1_{b=2}]', 'PASSED')]), (textwrap.dedent("        import numpy\n        @testing.parameterize({'a': numpy.array(1)}, {'a': 1})\n        class TestA:\n            def test_first(self):\n                assert self.a == 1\n                self.a += 2\n            def test_second(self):\n                assert self.a == 1\n                self.a += 2\n        "), [('::TestA::test_first[_param_0_{a=array(1)}]', 'PASSED'), ('::TestA::test_first[_param_1_{a=1}]', 'PASSED'), ('::TestA::test_second[_param_0_{a=array(1)}]', 'FAILED'), ('::TestA::test_second[_param_1_{a=1}]', 'PASSED')]), (textwrap.dedent("        @testing.parameterize({'a': 1, 'b': 4}, {'a': 2, 'b': 3})\n        class TestA:\n            c = 5\n            def test_a(self):\n                assert self.a + self.b == self.c\n        "), [('::TestA::test_a[_param_0_{a=1, b=4}]', 'PASSED'), ('::TestA::test_a[_param_1_{a=2, b=3}]', 'PASSED')]), (textwrap.dedent('        import pytest\n        @pytest.mark.parametrize("outer", ["E", "e"])\n        @testing.parameterize({"x": "D"}, {"x": "d"})\n        @pytest.mark.parametrize("inner", ["c", "C"])\n        class TestA:\n            @pytest.mark.parametrize(\n                ("fn1", "fn2"), [("A", "b"), ("a", "B")])\n            def test_a(self, fn2, inner, outer, fn1):\n                assert (\n                    (fn1 + fn2 + inner + self.x + outer).lower()\n                    == "abcde")\n            @pytest.mark.parametrize(\n                "fn", ["A", "a"])\n            def test_b(self, outer, fn, inner):\n                assert sum(\n                    c.isupper() for c in [fn, inner, self.x, outer]\n                ) != 2\n        '), [("::TestA::test_a[A-b-_param_0_{x='D'}-E-c]", 'PASSED'), ("::TestA::test_a[A-b-_param_0_{x='D'}-E-C]", 'PASSED'), ("::TestA::test_a[A-b-_param_0_{x='D'}-e-c]", 'PASSED'), ("::TestA::test_a[A-b-_param_0_{x='D'}-e-C]", 'PASSED'), ("::TestA::test_a[A-b-_param_1_{x='d'}-E-c]", 'PASSED'), ("::TestA::test_a[A-b-_param_1_{x='d'}-E-C]", 'PASSED'), ("::TestA::test_a[A-b-_param_1_{x='d'}-e-c]", 'PASSED'), ("::TestA::test_a[A-b-_param_1_{x='d'}-e-C]", 'PASSED'), ("::TestA::test_a[a-B-_param_0_{x='D'}-E-c]", 'PASSED'), ("::TestA::test_a[a-B-_param_0_{x='D'}-E-C]", 'PASSED'), ("::TestA::test_a[a-B-_param_0_{x='D'}-e-c]", 'PASSED'), ("::TestA::test_a[a-B-_param_0_{x='D'}-e-C]", 'PASSED'), ("::TestA::test_a[a-B-_param_1_{x='d'}-E-c]", 'PASSED'), ("::TestA::test_a[a-B-_param_1_{x='d'}-E-C]", 'PASSED'), ("::TestA::test_a[a-B-_param_1_{x='d'}-e-c]", 'PASSED'), ("::TestA::test_a[a-B-_param_1_{x='d'}-e-C]", 'PASSED'), ("::TestA::test_b[A-_param_0_{x='D'}-E-c]", 'PASSED'), ("::TestA::test_b[A-_param_0_{x='D'}-E-C]", 'PASSED'), ("::TestA::test_b[A-_param_0_{x='D'}-e-c]", 'FAILED'), ("::TestA::test_b[A-_param_0_{x='D'}-e-C]", 'PASSED'), ("::TestA::test_b[A-_param_1_{x='d'}-E-c]", 'FAILED'), ("::TestA::test_b[A-_param_1_{x='d'}-E-C]", 'PASSED'), ("::TestA::test_b[A-_param_1_{x='d'}-e-c]", 'PASSED'), ("::TestA::test_b[A-_param_1_{x='d'}-e-C]", 'FAILED'), ("::TestA::test_b[a-_param_0_{x='D'}-E-c]", 'FAILED'), ("::TestA::test_b[a-_param_0_{x='D'}-E-C]", 'PASSED'), ("::TestA::test_b[a-_param_0_{x='D'}-e-c]", 'PASSED'), ("::TestA::test_b[a-_param_0_{x='D'}-e-C]", 'FAILED'), ("::TestA::test_b[a-_param_1_{x='d'}-E-c]", 'PASSED'), ("::TestA::test_b[a-_param_1_{x='d'}-E-C]", 'FAILED'), ("::TestA::test_b[a-_param_1_{x='d'}-e-c]", 'PASSED'), ("::TestA::test_b[a-_param_1_{x='d'}-e-C]", 'PASSED')])])
@pytest.mark.skipif(pytest.__version__ > '7.4.2', reason='test name not compatible')
def test_parameterize_pytest_impl(testdir, src, outcomes):
    if False:
        for i in range(10):
            print('nop')
    testdir.makepyfile('from cupy import testing\n' + src)
    result = testdir.runpytest('-v', '--tb=no')
    expected_lines = ['.*{} {}.*'.format(re.escape(name), res) for (name, res) in outcomes]
    print('Result', pytest.__version__)
    print('---')
    print('Expected', '\n'.join(expected_lines))
    result.stdout.re_match_lines(expected_lines)
    expected_count = collections.Counter([res.lower() for (_, res) in outcomes])
    result.assert_outcomes(**expected_count)