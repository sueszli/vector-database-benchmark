import pytest
from dagster import DagsterTypeCheckDidNotPass, Dict, In, Out, op
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_typed_python_dict():
    if False:
        print('Hello World!')
    int_to_int = Dict[int, int]
    int_to_int.type_check(None, {1: 1})

def test_typed_python_dict_failure():
    if False:
        print('Hello World!')
    int_to_int = Dict[int, int]
    res = int_to_int.type_check(None, {1: '1'})
    assert not res.success

def test_basic_op_dict_int_int_output():
    if False:
        i = 10
        return i + 15

    @op(out=Out(Dict[int, int]))
    def emit_dict_int_int():
        if False:
            return 10
        return {1: 1}
    assert wrap_op_in_graph_and_execute(emit_dict_int_int).output_value() == {1: 1}

def test_basic_op_dict_int_int_output_faile():
    if False:
        i = 10
        return i + 15

    @op(out=Out(Dict[int, int]))
    def emit_dict_int_int():
        if False:
            while True:
                i = 10
        return {1: '1'}
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict_int_int)

def test_basic_op_dict_int_int_input_pass():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'ddict': In(Dict[int, int])})
    def emit_dict_int_int(ddict):
        if False:
            print('Hello World!')
        return ddict
    assert wrap_op_in_graph_and_execute(emit_dict_int_int, input_values={'ddict': {1: 2}}).output_value() == {1: 2}

def test_basic_op_dict_int_int_input_fails():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'ddict': In(Dict[int, int])})
    def emit_dict_int_int(ddict):
        if False:
            return 10
        return ddict
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict_int_int, input_values={'ddict': {'1': 2}})