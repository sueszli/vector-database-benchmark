from random import Random
from hypothesis.internal.conjecture.shrinking import Integer
from tests.common.utils import capture_out

def test_debug_output():
    if False:
        return 10
    with capture_out() as o:
        Integer.shrink(10, lambda x: True, debug=True, random=Random(0))
    assert 'initial=10' in o.getvalue()
    assert 'shrinking to 0' in o.getvalue()

def test_includes_name_in_repr_if_set():
    if False:
        print('Hello World!')
    assert repr(Integer(10, lambda x: True, name='hi there', random=Random(0))) == "Integer('hi there', initial=10, current=10)"

def test_normally_contains_no_space_for_name():
    if False:
        print('Hello World!')
    assert repr(Integer(10, lambda x: True, random=Random(0))) == 'Integer(initial=10, current=10)'