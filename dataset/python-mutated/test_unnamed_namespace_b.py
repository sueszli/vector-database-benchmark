from pybind11_tests import unnamed_namespace_b as m

def test_have_attr_any_struct():
    if False:
        for i in range(10):
            print('nop')
    assert hasattr(m, 'unnamed_namespace_b_any_struct')