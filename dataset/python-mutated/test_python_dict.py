import typing
import pytest
from dagster import DagsterInvalidConfigError, DagsterInvalidDefinitionError, DagsterTypeCheckDidNotPass, Dict, In, Out, op, usable_as_dagster_type
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_basic_python_dictionary_output():
    if False:
        i = 10
        return i + 15

    @op(out=Out(dict))
    def emit_dict():
        if False:
            return 10
        return {'key': 'value'}
    assert wrap_op_in_graph_and_execute(emit_dict).output_value() == {'key': 'value'}

def test_basic_python_dictionary_input():
    if False:
        return 10

    @op(ins={'data': In(dict)}, out=Out(str))
    def input_dict(data):
        if False:
            while True:
                i = 10
        return data['key']
    assert wrap_op_in_graph_and_execute(input_dict, input_values={'data': {'key': 'value'}}).output_value() == 'value'

def test_basic_python_dictionary_wrong():
    if False:
        while True:
            i = 10

    @op(out=Out(dict))
    def emit_dict():
        if False:
            print('Hello World!')
        return 1
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict)

def test_basic_python_dictionary_input_wrong():
    if False:
        print('Hello World!')

    @op(ins={'data': In(dict)}, out=Out(str))
    def input_dict(data):
        if False:
            i = 10
            return i + 15
        return data['key']
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(input_dict, input_values={'data': 123})

def test_execute_dict_from_config():
    if False:
        print('Hello World!')

    @op(ins={'data': In(dict)}, out=Out(str))
    def input_dict(data):
        if False:
            for i in range(10):
                print('nop')
        return data['key']
    assert wrap_op_in_graph_and_execute(input_dict, run_config={'ops': {'input_dict': {'inputs': {'data': {'key': 'in_config'}}}}}, do_input_mapping=False).output_value() == 'in_config'

def test_dagster_dictionary_output():
    if False:
        print('Hello World!')

    @op(out=Out(dict))
    def emit_dict():
        if False:
            print('Hello World!')
        return {'key': 'value'}
    assert wrap_op_in_graph_and_execute(emit_dict).output_value() == {'key': 'value'}

def test_basic_dagster_dictionary_input():
    if False:
        i = 10
        return i + 15

    @op(ins={'data': In(Dict)}, out=Out(str))
    def input_dict(data):
        if False:
            i = 10
            return i + 15
        return data['key']
    assert wrap_op_in_graph_and_execute(input_dict, input_values={'data': {'key': 'value'}}).output_value() == 'value'

def test_basic_typing_dictionary_output():
    if False:
        print('Hello World!')

    @op(out=Out(typing.Dict))
    def emit_dict():
        if False:
            while True:
                i = 10
        return {'key': 'value'}
    assert wrap_op_in_graph_and_execute(emit_dict).output_value() == {'key': 'value'}

def test_basic_typing_dictionary_input():
    if False:
        i = 10
        return i + 15

    @op(ins={'data': In(typing.Dict)}, out=Out(str))
    def input_dict(data):
        if False:
            print('Hello World!')
        return data['key']
    assert wrap_op_in_graph_and_execute(input_dict, input_values={'data': {'key': 'value'}}).output_value() == 'value'

def test_basic_closed_typing_dictionary_wrong():
    if False:
        print('Hello World!')

    @op(out=Out(typing.Dict[int, int]))
    def emit_dict():
        if False:
            i = 10
            return i + 15
        return 1
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict)

def test_basic_closed_typing_dictionary_output():
    if False:
        while True:
            i = 10

    @op(out=Out(typing.Dict[str, str]))
    def emit_dict():
        if False:
            return 10
        return {'key': 'value'}
    assert wrap_op_in_graph_and_execute(emit_dict).output_value() == {'key': 'value'}
    assert emit_dict.output_defs[0].dagster_type.key == 'TypedPythonDict.String.String'
    assert emit_dict.output_defs[0].dagster_type.key_type.unique_name == 'String'
    assert emit_dict.output_defs[0].dagster_type.value_type.unique_name == 'String'

def test_basic_closed_typing_dictionary_input():
    if False:
        while True:
            i = 10

    @op(ins={'data': In(typing.Dict[str, str])}, out=Out(str))
    def input_dict(data):
        if False:
            print('Hello World!')
        return data['key']
    assert wrap_op_in_graph_and_execute(input_dict, input_values={'data': {'key': 'value'}}).output_value() == 'value'

def test_basic_closed_typing_dictionary_key_wrong():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(typing.Dict[str, str]))
    def emit_dict():
        if False:
            while True:
                i = 10
        return {1: 'foo'}
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict)

def test_basic_closed_typing_dictionary_value_wrong():
    if False:
        print('Hello World!')

    @op(out=Out(typing.Dict[str, str]))
    def emit_dict():
        if False:
            i = 10
            return i + 15
        return {'foo': 1}
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict)

def test_complicated_dictionary_typing_pass():
    if False:
        i = 10
        return i + 15

    @op(out=Out(typing.Dict[str, typing.List[typing.Dict[int, int]]]))
    def emit_dict():
        if False:
            for i in range(10):
                print('nop')
        return {'foo': [{1: 1, 2: 2}]}
    assert wrap_op_in_graph_and_execute(emit_dict).output_value() == {'foo': [{1: 1, 2: 2}]}

def test_complicated_dictionary_typing_fail():
    if False:
        print('Hello World!')

    @op(out=Out(typing.Dict[str, typing.List[typing.Dict[int, int]]]))
    def emit_dict():
        if False:
            return 10
        return {'foo': [{1: 1, '2': 2}]}
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_dict)

def test_dict_type_loader():
    if False:
        return 10
    test_input = {'hello': 5, 'goodbye': 42}

    @op(ins={'dict_input': In(dagster_type=typing.Dict[str, int])})
    def emit_dict(dict_input):
        if False:
            i = 10
            return i + 15
        return dict_input
    result = wrap_op_in_graph_and_execute(emit_dict, run_config={'ops': {'emit_dict': {'inputs': {'dict_input': test_input}}}}, do_input_mapping=False)
    assert result.success
    assert result.output_value() == test_input

def test_dict_type_loader_typing_fail():
    if False:
        return 10

    @usable_as_dagster_type
    class CustomType(str):
        pass
    test_input = {'hello': 'foo', 'goodbye': 'bar'}

    @op(ins={'dict_input': In(dagster_type=typing.Dict[str, CustomType])})
    def emit_dict(dict_input):
        if False:
            print('Hello World!')
        return dict_input
    with pytest.raises(DagsterInvalidDefinitionError, match="Input 'dict_input' of op 'emit_dict' has no way of being resolved."):
        wrap_op_in_graph_and_execute(emit_dict, run_config={'ops': {'emit_dict': {'inputs': {'dict_input': test_input}}}}, do_input_mapping=False)

def test_dict_type_loader_inner_type_mismatch():
    if False:
        for i in range(10):
            print('nop')
    test_input = {'hello': 'foo', 'goodbye': 'bar'}

    @op(ins={'dict_input': In(dagster_type=typing.Dict[str, int])})
    def emit_dict(dict_input):
        if False:
            return 10
        return dict_input
    with pytest.raises(DagsterInvalidConfigError):
        wrap_op_in_graph_and_execute(emit_dict, run_config={'ops': {'emit_dict': {'inputs': {'dict_input': test_input}}}})