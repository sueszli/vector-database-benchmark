from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st

@settings(max_examples=1, database=None)
@given(st.integers())
def test_single_example(n):
    if False:
        for i in range(10):
            print('nop')
    pass

@settings(max_examples=1, database=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], verbosity=Verbosity.debug)
@given(st.integers())
def test_hard_to_find_single_example(n):
    if False:
        i = 10
        return i + 15
    assume(n % 50 == 11)