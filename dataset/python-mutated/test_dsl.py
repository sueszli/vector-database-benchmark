import pytest
from dagster import DagsterInvalidDefinitionError, DynamicOut, DynamicOutput, GraphOut, graph, job, op

@op(out=DynamicOut())
def dynamic_numbers(_):
    if False:
        i = 10
        return i + 15
    yield DynamicOutput(1, mapping_key='1')
    yield DynamicOutput(2, mapping_key='2')

@op
def emit_one(_):
    if False:
        i = 10
        return i + 15
    return 1

@op
def echo(_, x):
    if False:
        i = 10
        return i + 15
    return x

@op
def add_one(_, x):
    if False:
        while True:
            i = 10
    return x + 1

def test_must_unpack():
    if False:
        return 10
    with pytest.raises(DagsterInvalidDefinitionError, match='Dynamic output must be unpacked by invoking map or collect'):

        @job
        def _should_fail():
            if False:
                for i in range(10):
                    print('nop')
            echo(dynamic_numbers())

def test_must_unpack_composite():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError, match='Dynamic output must be unpacked by invoking map or collect'):

        @graph
        def composed():
            if False:
                i = 10
                return i + 15
            return dynamic_numbers()

        @job
        def _should_fail():
            if False:
                while True:
                    i = 10
            echo(composed())

def test_mapping():
    if False:
        i = 10
        return i + 15

    @job
    def mapping():
        if False:
            i = 10
            return i + 15
        dynamic_numbers().map(add_one).map(echo)
    result = mapping.execute_in_process()
    assert result.success

def test_mapping_multi():
    if False:
        print('Hello World!')

    def _multi(item):
        if False:
            print('Hello World!')
        a = add_one(item)
        b = add_one(a)
        c = add_one(b)
        return (a, b, c)

    @job
    def multi_map():
        if False:
            i = 10
            return i + 15
        (a, b, c) = dynamic_numbers().map(_multi)
        a.map(echo)
        b.map(echo)
        c.map(echo)
    result = multi_map.execute_in_process()
    assert result.success

def test_composite_multi_out():
    if False:
        i = 10
        return i + 15

    @graph(out={'one': GraphOut(), 'numbers': GraphOut()})
    def multi_out():
        if False:
            return 10
        one = emit_one()
        numbers = dynamic_numbers()
        return {'one': one, 'numbers': numbers}

    @job
    def composite_multi():
        if False:
            while True:
                i = 10
        (one, numbers) = multi_out()
        echo(one)
        numbers.map(echo)
    result = composite_multi.execute_in_process()
    assert result.success