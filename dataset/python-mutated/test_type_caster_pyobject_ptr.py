import pytest
from pybind11_tests import type_caster_pyobject_ptr as m

class ValueHolder:

    def __init__(self, value):
        if False:
            return 10
        self.value = value

def test_cast_from_pyobject_ptr():
    if False:
        for i in range(10):
            print('nop')
    assert m.cast_from_pyobject_ptr() == 6758

def test_cast_handle_to_pyobject_ptr():
    if False:
        return 10
    assert m.cast_handle_to_pyobject_ptr(ValueHolder(24)) == 76

def test_cast_object_to_pyobject_ptr():
    if False:
        print('Hello World!')
    assert m.cast_object_to_pyobject_ptr(ValueHolder(43)) == 257

def test_cast_list_to_pyobject_ptr():
    if False:
        return 10
    assert m.cast_list_to_pyobject_ptr([1, 2, 3, 4, 5]) == 395

def test_return_pyobject_ptr():
    if False:
        return 10
    assert m.return_pyobject_ptr() == 2314

def test_pass_pyobject_ptr():
    if False:
        return 10
    assert m.pass_pyobject_ptr(ValueHolder(82)) == 118

@pytest.mark.parametrize('call_callback', [m.call_callback_with_object_return, m.call_callback_with_pyobject_ptr_return])
def test_call_callback_with_object_return(call_callback):
    if False:
        print('Hello World!')

    def cb(value):
        if False:
            for i in range(10):
                print('nop')
        if value < 0:
            raise ValueError('Raised from cb')
        return ValueHolder(1000 - value)
    assert call_callback(cb, 287).value == 713
    with pytest.raises(ValueError, match='^Raised from cb$'):
        call_callback(cb, -1)

def test_call_callback_with_pyobject_ptr_arg():
    if False:
        i = 10
        return i + 15

    def cb(obj):
        if False:
            while True:
                i = 10
        return 300 - obj.value
    assert m.call_callback_with_pyobject_ptr_arg(cb, ValueHolder(39)) == 261

@pytest.mark.parametrize('set_error', [True, False])
def test_cast_to_python_nullptr(set_error):
    if False:
        while True:
            i = 10
    expected = {True: '^Reflective of healthy error handling\\.$', False: '^Internal error: pybind11::error_already_set called while Python error indicator not set\\.$'}[set_error]
    with pytest.raises(RuntimeError, match=expected):
        m.cast_to_pyobject_ptr_nullptr(set_error)

def test_cast_to_python_non_nullptr_with_error_set():
    if False:
        return 10
    with pytest.raises(SystemError) as excinfo:
        m.cast_to_pyobject_ptr_non_nullptr_with_error_set()
    assert str(excinfo.value) == 'src != nullptr but PyErr_Occurred()'
    assert str(excinfo.value.__cause__) == 'Reflective of unhealthy error handling.'

def test_pass_list_pyobject_ptr():
    if False:
        while True:
            i = 10
    acc = m.pass_list_pyobject_ptr([ValueHolder(842), ValueHolder(452)])
    assert acc == 842452

def test_return_list_pyobject_ptr_take_ownership():
    if False:
        print('Hello World!')
    vec_obj = m.return_list_pyobject_ptr_take_ownership(ValueHolder)
    assert [e.value for e in vec_obj] == [93, 186]

def test_return_list_pyobject_ptr_reference():
    if False:
        for i in range(10):
            print('nop')
    vec_obj = m.return_list_pyobject_ptr_reference(ValueHolder)
    assert [e.value for e in vec_obj] == [93, 186]
    assert m.dec_ref_each_pyobject_ptr(vec_obj) == 2

def test_type_caster_name_via_incompatible_function_arguments_type_error():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError, match='1\\. \\(arg0: object, arg1: int\\) -> None'):
        m.pass_pyobject_ptr_and_int(ValueHolder(101), ValueHolder(202))