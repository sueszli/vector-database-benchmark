import typing
import pytest
from dagster import DagsterTypeCheckDidNotPass, In, Out, op
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_basic_list_output_pass():
    if False:
        i = 10
        return i + 15

    @op(out=Out(list))
    def emit_list():
        if False:
            return 10
        return [1]
    assert wrap_op_in_graph_and_execute(emit_list).output_value() == [1]

def test_basic_list_output_fail():
    if False:
        i = 10
        return i + 15

    @op(out=Out(list))
    def emit_list():
        if False:
            while True:
                i = 10
        return 'foo'
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_list).output_value()

def test_basic_list_input_pass():
    if False:
        return 10

    @op(ins={'alist': In(list)})
    def ingest_list(alist):
        if False:
            i = 10
            return i + 15
        return alist
    assert wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': [2]}).output_value() == [2]

def test_basic_list_input_fail():
    if False:
        i = 10
        return i + 15

    @op(ins={'alist': In(list)})
    def ingest_list(alist):
        if False:
            print('Hello World!')
        return alist
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': 'foobar'})

def test_typing_list_output_pass():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(typing.List))
    def emit_list():
        if False:
            while True:
                i = 10
        return [1]
    assert wrap_op_in_graph_and_execute(emit_list).output_value() == [1]

def test_typing_list_output_fail():
    if False:
        print('Hello World!')

    @op(out=Out(typing.List))
    def emit_list():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_list).output_value()

def test_typing_list_input_pass():
    if False:
        return 10

    @op(ins={'alist': In(typing.List)})
    def ingest_list(alist):
        if False:
            i = 10
            return i + 15
        return alist
    assert wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': [2]}).output_value() == [2]

def test_typing_list_input_fail():
    if False:
        print('Hello World!')

    @op(ins={'alist': In(typing.List)})
    def ingest_list(alist):
        if False:
            print('Hello World!')
        return alist
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': 'foobar'})

def test_typing_list_of_int_output_pass():
    if False:
        return 10

    @op(out=Out(typing.List[int]))
    def emit_list():
        if False:
            print('Hello World!')
        return [1]
    assert wrap_op_in_graph_and_execute(emit_list).output_value() == [1]

def test_typing_list_of_int_output_fail():
    if False:
        print('Hello World!')

    @op(out=Out(typing.List[int]))
    def emit_list():
        if False:
            for i in range(10):
                print('nop')
        return ['foo']
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_list).output_value()

def test_typing_list_of_int_input_pass():
    if False:
        i = 10
        return i + 15

    @op(ins={'alist': In(typing.List[int])})
    def ingest_list(alist):
        if False:
            i = 10
            return i + 15
        return alist
    assert wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': [2]}).output_value() == [2]

def test_typing_list_of_int_input_fail():
    if False:
        print('Hello World!')

    @op(ins={'alist': In(typing.List[int])})
    def ingest_list(alist):
        if False:
            i = 10
            return i + 15
        return alist
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': ['foobar']})
LIST_LIST_INT = typing.List[typing.List[int]]

def test_typing_list_of_list_of_int_output_pass():
    if False:
        return 10

    @op(out=Out(LIST_LIST_INT))
    def emit_list():
        if False:
            return 10
        return [[1, 2], [3, 4]]
    assert wrap_op_in_graph_and_execute(emit_list).output_value() == [[1, 2], [3, 4]]

def test_typing_list_of_list_of_int_output_fail():
    if False:
        return 10

    @op(out=Out(LIST_LIST_INT))
    def emit_list():
        if False:
            i = 10
            return i + 15
        return [[1, 2], [3, '4']]
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(emit_list).output_value()

def test_typing_list_of_list_of_int_input_pass():
    if False:
        return 10

    @op(ins={'alist': In(LIST_LIST_INT)})
    def ingest_list(alist):
        if False:
            return 10
        return alist
    assert wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': [[1, 2], [3, 4]]}).output_value() == [[1, 2], [3, 4]]

def test_typing_list_of_list_of_int_input_fail():
    if False:
        while True:
            i = 10

    @op(ins={'alist': In(LIST_LIST_INT)})
    def ingest_list(alist):
        if False:
            print('Hello World!')
        return alist
    with pytest.raises(DagsterTypeCheckDidNotPass):
        wrap_op_in_graph_and_execute(ingest_list, input_values={'alist': [[1, 2], [3, '4']]})