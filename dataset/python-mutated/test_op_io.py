import sys
from typing import Generator, Iterable, Iterator
import pytest
from dagster import AssetKey, DynamicOut, DynamicOutput, In, Out, Output, op
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._legacy import InputDefinition, OutputDefinition

def test_flex_inputs():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'arg_b': In(metadata={'explicit': True})})
    def partial(_context, arg_a, arg_b):
        if False:
            while True:
                i = 10
        return arg_a + arg_b
    assert partial.input_defs[0].name == 'arg_b'
    assert partial.input_defs[0].metadata['explicit']
    assert partial.input_defs[1].name == 'arg_a'

def test_merge_type():
    if False:
        return 10

    @op(ins={'arg_b': In(metadata={'explicit': True})})
    def merged(_context, arg_b: int):
        if False:
            while True:
                i = 10
        return arg_b
    assert merged.input_defs[0].dagster_type == InputDefinition('test', dagster_type=int).dagster_type
    assert merged.input_defs[0].metadata['explicit']

def test_merge_desc():
    if False:
        i = 10
        return i + 15

    @op(ins={'arg_b': In(metadata={'explicit': True})})
    def merged(_context, arg_a, arg_b, arg_c):
        if False:
            while True:
                i = 10
        'Testing.\n\n        Args:\n            arg_b: described\n        '
        return arg_a + arg_b + arg_c
    assert merged.input_defs[0].name == 'arg_b'
    assert merged.input_defs[0].description == 'described'
    assert merged.input_defs[0].metadata['explicit']

def test_merge_default_val():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'arg_b': In(dagster_type=int, metadata={'explicit': True})})
    def merged(_context, arg_a: int, arg_b=3, arg_c=0):
        if False:
            return 10
        return arg_a + arg_b + arg_c
    assert merged.input_defs[0].name == 'arg_b'
    assert merged.input_defs[0].default_value == 3
    assert merged.input_defs[0].dagster_type == InputDefinition('test', dagster_type=int).dagster_type

def test_precedence():
    if False:
        while True:
            i = 10

    @op(ins={'arg_b': In(dagster_type=str, default_value='hi', description='legit', metadata={'explicit': True}, input_manager_key='rudy', asset_key=AssetKey('table_1'), asset_partitions={'0'})})
    def precedence(_context, arg_a: int, arg_b: int, arg_c: int):
        if False:
            print('Hello World!')
        'Testing.\n\n        Args:\n            arg_b: boo\n        '
        return arg_a + arg_b + arg_c
    assert precedence.input_defs[0].name == 'arg_b'
    assert precedence.input_defs[0].dagster_type == InputDefinition('test', dagster_type=str).dagster_type
    assert precedence.input_defs[0].description == 'legit'
    assert precedence.input_defs[0].default_value == 'hi'
    assert precedence.input_defs[0].metadata['explicit']
    assert precedence.input_defs[0].input_manager_key == 'rudy'
    assert precedence.input_defs[0].get_asset_key(None) is not None
    assert precedence.input_defs[0].get_asset_partitions(None) is not None

def test_output_merge():
    if False:
        for i in range(10):
            print('nop')

    @op(out={'four': Out()})
    def foo(_) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 4
    assert foo.output_defs[0].name == 'four'
    assert foo.output_defs[0].dagster_type == OutputDefinition(int).dagster_type

def test_iter_out():
    if False:
        while True:
            i = 10

    @op(out={'A': Out()})
    def _ok(_) -> Iterator[Output]:
        if False:
            return 10
        yield Output('a', output_name='A')

    @op
    def _also_ok(_) -> Iterator[Output]:
        if False:
            while True:
                i = 10
        yield Output('a', output_name='A')

    @op
    def _gen_too(_) -> Generator[Output, None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield Output('a', output_name='A')

    @op(out={'A': Out(), 'B': Out()})
    def _multi_fine(_) -> Iterator[Output]:
        if False:
            i = 10
            return i + 15
        yield Output('a', output_name='A')
        yield Output('b', output_name='B')

def test_dynamic():
    if False:
        while True:
            i = 10

    @op(out=DynamicOut(dagster_type=int))
    def dyn_desc(_) -> Iterator[DynamicOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            numbers.\n        '
        yield DynamicOutput(4, '4')
    assert dyn_desc.output_defs[0].description == 'numbers.'
    assert dyn_desc.output_defs[0].is_dynamic

@pytest.mark.skipif(sys.version_info < (3, 7), reason='typing types isinstance of type in py3.6, https://github.com/dagster-io/dagster/issues/4077')
def test_not_type_input():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidDefinitionError, match="Problem using type '.*' from type annotation for argument 'arg_b', correct the issue or explicitly set the dagster_type"):

        @op
        def _create(_context, arg_b: Iterator[int]):
            if False:
                return 10
            return arg_b
    with pytest.raises(DagsterInvalidDefinitionError, match="Problem using type '.*' from type annotation for argument 'arg_b', correct the issue or explicitly set the dagster_type"):

        @op(ins={'arg_b': In()})
        def _combine(_context, arg_b: Iterator[int]):
            if False:
                print('Hello World!')
            return arg_b
    with pytest.raises(DagsterInvalidDefinitionError, match="Problem using type '.*' from return type annotation, correct the issue or explicitly set the dagster_type"):

        @op
        def _out(_context) -> Iterable[int]:
            if False:
                for i in range(10):
                    print('nop')
            return [1]