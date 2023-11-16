from typing import Optional
import pytest
from dagster import DagsterEventType, DagsterInvalidDefinitionError, GraphIn, In, Nothing, graph, job, op

def execute_in_graph(an_op, raise_on_error=True, run_config=None):
    if False:
        return 10

    @graph
    def my_graph():
        if False:
            while True:
                i = 10
        an_op()
    result = my_graph.execute_in_process(raise_on_error=raise_on_error, run_config=run_config)
    return result

def test_none():
    if False:
        while True:
            i = 10

    @op(ins={'x': In(Optional[int], default_value=None)})
    def none_x(x):
        if False:
            i = 10
            return i + 15
        return x
    result = execute_in_graph(none_x)
    assert result.output_for_node('none_x') is None

def test_none_infer():
    if False:
        print('Hello World!')

    @op
    def none_x(x=None):
        if False:
            i = 10
            return i + 15
        return x
    result = execute_in_graph(none_x)
    assert result.output_for_node('none_x') is None

def test_int():
    if False:
        while True:
            i = 10

    @op(ins={'x': In(Optional[int], default_value=1337)})
    def int_x(x):
        if False:
            i = 10
            return i + 15
        return x
    result = execute_in_graph(int_x)
    assert result.output_for_node('int_x') == 1337

def test_int_infer():
    if False:
        for i in range(10):
            print('nop')

    @op
    def int_x(x=1337):
        if False:
            i = 10
            return i + 15
        return x
    result = execute_in_graph(int_x)
    assert result.output_for_node('int_x') == 1337

def test_early_fail():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidDefinitionError, match='Type check failed for the default_value of InputDefinition x of type Int'):

        @op(ins={'x': In(int, default_value='foo')})
        def _int_x(x):
            if False:
                return 10
            return x
    with pytest.raises(DagsterInvalidDefinitionError, match='Type check failed for the default_value of InputDefinition x of type String'):

        @op(ins={'x': In(str, default_value=1337)})
        def _int_x(x):
            if False:
                i = 10
                return i + 15
            return x

@op(ins={'x': In(Optional[int], default_value='number')})
def bad_default(x):
    if False:
        print('Hello World!')
    return x

def test_mismatch():
    if False:
        for i in range(10):
            print('nop')
    result = execute_in_graph(bad_default, raise_on_error=False)
    assert result.success is False
    input_event = result.filter_events(lambda event: event.event_type == DagsterEventType.STEP_INPUT)[0]
    assert input_event.step_input_data.type_check_data.success is False

def test_env_precedence():
    if False:
        print('Hello World!')
    result = execute_in_graph(bad_default, run_config={'ops': {'bad_default': {'inputs': {'x': 1}}}}, raise_on_error=False)
    assert result.success is True
    assert result.output_for_node('bad_default') == 1

def test_input_precedence():
    if False:
        print('Hello World!')

    @op
    def emit_one():
        if False:
            i = 10
            return i + 15
        return 1

    @job
    def the_job():
        if False:
            i = 10
            return i + 15
        bad_default(emit_one())
    result = the_job.execute_in_process()
    assert result.success
    assert result.output_for_node('bad_default') == 1

def test_nothing():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError):

        @op(ins={'x': In(Nothing, default_value=None)})
        def _nothing():
            if False:
                while True:
                    i = 10
            pass

def test_composite_inner_default():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'x': In(Optional[int], default_value=1337)})
    def int_x(x):
        if False:
            while True:
                i = 10
        return x

    @graph(ins={'y': GraphIn()})
    def wrap(y):
        if False:
            for i in range(10):
                print('nop')
        return int_x(y)
    result = execute_in_graph(wrap)
    assert result.success
    assert result.output_for_node('wrap') == 1337

def test_custom_type_default():
    if False:
        i = 10
        return i + 15

    class CustomType:
        pass

    @op
    def test_op(_inp: Optional[CustomType]=None):
        if False:
            return 10
        return 1

    @job
    def test_job():
        if False:
            print('Hello World!')
        test_op()
    result = test_job.execute_in_process()
    assert result.output_for_node('test_op') == 1