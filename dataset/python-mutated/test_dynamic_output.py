import gc
from typing import NamedTuple
import objgraph
import pytest
from dagster import DynamicOut, DynamicOutput, Out, build_op_context, execute_job, graph, job, op, reconstructable
from dagster._core.definitions.events import Output
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvariantViolationError
from dagster._core.events import DagsterEventType
from dagster._core.test_utils import instance_for_test

def test_basic():
    if False:
        print('Hello World!')

    @op(out=DynamicOut())
    def should_work():
        if False:
            for i in range(10):
                print('nop')
        yield DynamicOutput(1, mapping_key='1')
        yield DynamicOutput(2, mapping_key='2')
    assert [do.value for do in should_work()] == [1, 2]

def test_fails_without_def():
    if False:
        i = 10
        return i + 15

    @op
    def should_fail():
        if False:
            for i in range(10):
                print('nop')
        yield DynamicOutput(True, mapping_key='foo')

    @graph
    def wrap():
        if False:
            print('Hello World!')
        should_fail()
    with pytest.raises(DagsterInvariantViolationError, match='did not use DynamicOutputDefinition'):
        wrap.execute_in_process(raise_on_error=True)

def test_fails_with_wrong_output():
    if False:
        print('Hello World!')

    @op(out=DynamicOut())
    def should_fail():
        if False:
            for i in range(10):
                print('nop')
        yield Output(1)

    @graph
    def wrap():
        if False:
            i = 10
            return i + 15
        should_fail()
    with pytest.raises(DagsterInvariantViolationError, match='must yield DynamicOutput'):
        wrap.execute_in_process(raise_on_error=True)

    @op(out=DynamicOut())
    def should_also_fail():
        if False:
            while True:
                i = 10
        return 1

    @graph
    def wrap_also():
        if False:
            i = 10
            return i + 15
        should_also_fail()
    with pytest.raises(DagsterInvariantViolationError, match="dynamic output 'result' expected a list of DynamicOutput objects"):
        wrap_also.execute_in_process(raise_on_error=True)

def test_fails_dupe_keys():
    if False:
        for i in range(10):
            print('nop')

    @op(out=DynamicOut())
    def should_fail():
        if False:
            return 10
        yield DynamicOutput(True, mapping_key='dunk')
        yield DynamicOutput(True, mapping_key='dunk')

    @graph
    def wrap():
        if False:
            i = 10
            return i + 15
        should_fail()
    with pytest.raises(DagsterInvariantViolationError, match='mapping_key "dunk" multiple times'):
        wrap.execute_in_process(raise_on_error=True)

def test_invalid_mapping_keys():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError):
        DynamicOutput(True, mapping_key='')
    with pytest.raises(DagsterInvalidDefinitionError):
        DynamicOutput(True, mapping_key='?')
    with pytest.raises(DagsterInvalidDefinitionError):
        DynamicOutput(True, mapping_key='foo.baz')

def test_multi_output():
    if False:
        return 10

    @op(out={'numbers': DynamicOut(int), 'letters': DynamicOut(str), 'wildcard': Out(str)})
    def multiout():
        if False:
            return 10
        yield DynamicOutput(1, output_name='numbers', mapping_key='1')
        yield DynamicOutput(2, output_name='numbers', mapping_key='2')
        yield DynamicOutput('a', output_name='letters', mapping_key='a')
        yield DynamicOutput('b', output_name='letters', mapping_key='b')
        yield DynamicOutput('c', output_name='letters', mapping_key='c')
        yield Output('*', 'wildcard')

    @op
    def double(n):
        if False:
            return 10
        return n * 2

    @job
    def multi_dyn():
        if False:
            for i in range(10):
                print('nop')
        (numbers, _, _) = multiout()
        numbers.map(double)
    pipe_result = multi_dyn.execute_in_process()
    assert pipe_result.success
    assert pipe_result.output_for_node('multiout', 'numbers') == {'1': 1, '2': 2}
    assert pipe_result.output_for_node('multiout', 'letters') == {'a': 'a', 'b': 'b', 'c': 'c'}
    assert pipe_result.output_for_node('multiout', 'wildcard') == '*'
    assert pipe_result.output_for_node('double') == {'1': 2, '2': 4}

def test_multi_out_map():
    if False:
        for i in range(10):
            print('nop')

    @op(out=DynamicOut())
    def emit():
        if False:
            for i in range(10):
                print('nop')
        yield DynamicOutput(1, mapping_key='1')
        yield DynamicOutput(2, mapping_key='2')
        yield DynamicOutput(3, mapping_key='3')

    @op(out={'a': Out(is_required=False), 'b': Out(is_required=False), 'c': Out(is_required=False)})
    def multiout(inp: int):
        if False:
            print('Hello World!')
        if inp == 1:
            yield Output(inp, output_name='a')
        else:
            yield Output(inp, output_name='b')

    @op
    def echo(a):
        if False:
            print('Hello World!')
        return a

    @job
    def destructure():
        if False:
            i = 10
            return i + 15
        (a, b, c) = emit().map(multiout)
        echo.alias('echo_a')(a.collect())
        echo.alias('echo_b')(b.collect())
        echo.alias('echo_c')(c.collect())
    result = destructure.execute_in_process()
    assert result.output_for_node('echo_a') == [1]
    assert result.output_for_node('echo_b') == [2, 3]
    assert DagsterEventType.STEP_SKIPPED in [event.event_type for event in result.all_events if event.step_key == 'echo_c']

def test_context_mapping_key():
    if False:
        return 10
    _observed = []

    @op
    def observe_key(context, _dep=None):
        if False:
            i = 10
            return i + 15
        _observed.append(context.get_mapping_key())

    @op(out=DynamicOut())
    def emit():
        if False:
            print('Hello World!')
        yield DynamicOutput(1, mapping_key='key_1')
        yield DynamicOutput(2, mapping_key='key_2')

    @job
    def test():
        if False:
            for i in range(10):
                print('nop')
        observe_key()
        emit().map(observe_key)
    result = test.execute_in_process()
    assert result.success
    assert _observed == [None, 'key_1', 'key_2']
    _observed = []
    observe_key(build_op_context())
    assert _observed == [None]

def test_dynamic_with_op():
    if False:
        return 10

    @op
    def passthrough(_ctx, _dep=None):
        if False:
            i = 10
            return i + 15
        pass

    @op(out=DynamicOut())
    def emit():
        if False:
            return 10
        yield DynamicOutput(1, mapping_key='key_1')
        yield DynamicOutput(2, mapping_key='key_2')

    @graph
    def test_graph():
        if False:
            while True:
                i = 10
        emit().map(passthrough)
    assert test_graph.execute_in_process().success

class DangerNoodle(NamedTuple):
    x: int

@op(out={'items': DynamicOut(), 'refs': Out()})
def spawn():
    if False:
        i = 10
        return i + 15
    for i in range(10):
        yield DynamicOutput(DangerNoodle(i), output_name='items', mapping_key=f'num_{i}')
    gc.collect()
    yield Output(len(objgraph.by_type('DangerNoodle')), output_name='refs')

@job
def no_leaks_plz():
    if False:
        print('Hello World!')
    spawn()

def test_dealloc_prev_outputs():
    if False:
        return 10
    with instance_for_test() as inst:
        with execute_job(reconstructable(no_leaks_plz), instance=inst) as result:
            assert result.success
            assert result.output_for_node('spawn', 'refs') <= 1

def test_collect_and_map():
    if False:
        return 10

    @op(out=DynamicOut())
    def dyn_vals():
        if False:
            i = 10
            return i + 15
        for i in range(3):
            yield DynamicOutput(i, mapping_key=f'num_{i}')

    @op
    def echo(x):
        if False:
            for i in range(10):
                print('nop')
        return x

    @op
    def add_each(vals, x):
        if False:
            print('Hello World!')
        return [v + x for v in vals]

    @graph
    def both_w_echo():
        if False:
            print('Hello World!')
        d1 = dyn_vals()
        r = d1.map(lambda x: add_each(echo(d1.collect()), x))
        echo.alias('final')(r.collect())
    result = both_w_echo.execute_in_process()
    assert result.output_for_node('final') == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]