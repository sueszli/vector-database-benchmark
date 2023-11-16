from hypothesis import given, strategies as st
from hypothesis.internal.reflection import get_pretty_function_description, proxies

def test_can_copy_signature_of_unicode_args():
    if False:
        while True:
            i = 10

    def foo(μ):
        if False:
            print('Hello World!')
        return μ

    @proxies(foo)
    def bar(μ):
        if False:
            i = 10
            return i + 15
        return foo(μ)
    assert bar(1) == 1

def test_can_copy_signature_of_unicode_name():
    if False:
        i = 10
        return i + 15

    def ā():
        if False:
            while True:
                i = 10
        return 1

    @proxies(ā)
    def bar():
        if False:
            for i in range(10):
                print('nop')
        return 2
    assert bar() == 2
is_approx_π = lambda x: x == 3.1415

def test_can_handle_unicode_identifier_in_same_line_as_lambda_def():
    if False:
        i = 10
        return i + 15
    assert get_pretty_function_description(is_approx_π) == 'lambda x: x == 3.1415'

def test_regression_issue_1700():
    if False:
        return 10
    π = 3.1415

    @given(st.floats(min_value=-π, max_value=π).filter(lambda x: abs(x) > 1e-05))
    def test_nonzero(x):
        if False:
            for i in range(10):
                print('nop')
        assert x != 0
    test_nonzero()