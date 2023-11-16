import re
import pytest
from hypothesis import HealthCheck, given, reject, settings, strategies as st
from hypothesis.errors import InvalidArgument, Unsatisfiable

def test_contains_the_test_function_name_in_the_exception_string():
    if False:
        for i in range(10):
            print('nop')
    look_for_one = settings(max_examples=1, suppress_health_check=list(HealthCheck))

    @given(st.integers())
    @look_for_one
    def this_has_a_totally_unique_name(x):
        if False:
            return 10
        reject()
    with pytest.raises(Unsatisfiable, match=re.escape(this_has_a_totally_unique_name.__name__)):
        this_has_a_totally_unique_name()

    class Foo:

        @given(st.integers())
        @look_for_one
        def this_has_a_unique_name_and_lives_on_a_class(self, x):
            if False:
                return 10
            reject()
    with pytest.raises(Unsatisfiable, match=re.escape(Foo.this_has_a_unique_name_and_lives_on_a_class.__name__)):
        Foo().this_has_a_unique_name_and_lives_on_a_class()

def test_signature_mismatch_error_message():
    if False:
        return 10

    @settings(max_examples=2)
    @given(x=st.integers())
    def bad_test():
        if False:
            return 10
        pass
    with pytest.raises(InvalidArgument, match="bad_test\\(\\) got an unexpected keyword argument 'x', from `x=integers\\(\\)` in @given"):
        bad_test()

@given(data=st.data(), keys=st.lists(st.integers(), unique=True))
def test_fixed_dict_preserves_iteration_order(data, keys):
    if False:
        for i in range(10):
            print('nop')
    d = data.draw(st.fixed_dictionaries({k: st.none() for k in keys}))
    assert all((a == b for (a, b) in zip(keys, d))), f'keys={keys}, d.keys()={d.keys()}'