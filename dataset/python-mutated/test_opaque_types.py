import pytest

def test_string_list():
    if False:
        print('Hello World!')
    from pybind11_tests import StringList, ClassWithSTLVecProperty, print_opaque_list
    l = StringList()
    l.push_back('Element 1')
    l.push_back('Element 2')
    assert print_opaque_list(l) == 'Opaque list: [Element 1, Element 2]'
    assert l.back() == 'Element 2'
    for (i, k) in enumerate(l, start=1):
        assert k == 'Element {}'.format(i)
    l.pop_back()
    assert print_opaque_list(l) == 'Opaque list: [Element 1]'
    cvp = ClassWithSTLVecProperty()
    assert print_opaque_list(cvp.stringList) == 'Opaque list: []'
    cvp.stringList = l
    cvp.stringList.push_back('Element 3')
    assert print_opaque_list(cvp.stringList) == 'Opaque list: [Element 1, Element 3]'

def test_pointers(msg):
    if False:
        i = 10
        return i + 15
    from pybind11_tests import return_void_ptr, get_void_ptr_value, ExampleMandA, print_opaque_list, return_null_str, get_null_str_value, return_unique_ptr, ConstructorStats
    living_before = ConstructorStats.get(ExampleMandA).alive()
    assert get_void_ptr_value(return_void_ptr()) == 4660
    assert get_void_ptr_value(ExampleMandA())
    assert ConstructorStats.get(ExampleMandA).alive() == living_before
    with pytest.raises(TypeError) as excinfo:
        get_void_ptr_value([1, 2, 3])
    assert msg(excinfo.value) == '\n        get_void_ptr_value(): incompatible function arguments. The following argument types are supported:\n            1. (arg0: capsule) -> int\n\n        Invoked with: [1, 2, 3]\n    '
    assert return_null_str() is None
    assert get_null_str_value(return_null_str()) is not None
    ptr = return_unique_ptr()
    assert 'StringList' in repr(ptr)
    assert print_opaque_list(ptr) == 'Opaque list: [some value]'