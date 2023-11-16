import pytest
from dagster import DagsterInvalidDefinitionError, DynamicOut, DynamicOutput, graph, job, op

@op(out=DynamicOut())
def dynamic_op():
    if False:
        while True:
            i = 10
    yield DynamicOutput(1, mapping_key='1')
    yield DynamicOutput(2, mapping_key='2')

@op(out=DynamicOut())
def dynamic_echo(x):
    if False:
        return 10
    yield DynamicOutput(x, mapping_key='echo')

@op
def echo(x):
    if False:
        i = 10
        return i + 15
    return x

@op
def add(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x + y

def test_fan_in():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidDefinitionError, match='Problematic dependency on dynamic output "dynamic_op:result"'):

        @job
        def _should_fail():
            if False:
                while True:
                    i = 10
            numbers = []
            dynamic_op().map(numbers.append)
            echo(numbers)

def test_multi_direct():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of more than one dynamic output'):

        @job
        def _should_fail():
            if False:
                print('Hello World!')

            def _add(x):
                if False:
                    return 10
                dynamic_op().map(lambda y: add(x, y))
            dynamic_op().map(_add)

def test_multi_indirect():
    if False:
        return 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of more than one dynamic output'):

        @job
        def _should_fail():
            if False:
                print('Hello World!')

            def _add(x):
                if False:
                    while True:
                        i = 10
                dynamic_op().map(lambda y: add(x, y))
            dynamic_op().map(lambda z: _add(echo(z)))

def test_multi_composite_out():
    if False:
        return 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of more than one dynamic output'):

        @graph
        def composed_echo():
            if False:
                i = 10
                return i + 15
            return dynamic_op().map(echo)

        @job
        def _should_fail():
            if False:
                for i in range(10):
                    print('nop')

            def _complex(item):
                if False:
                    while True:
                        i = 10
                composed_echo().map(lambda y: add(y, item))
            dynamic_op().map(_complex)

def test_multi_composite_in():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of dynamic output "dynamic_op:result" since input "a" maps to a node that is already downstream of another dynamic output'):

        @graph
        def composed_add(a):
            if False:
                return 10
            dynamic_op().map(lambda b: add(a, b))

        @job
        def _should_fail():
            if False:
                while True:
                    i = 10
            dynamic_op().map(lambda x: composed_add(echo(x)))

def test_multi_composite_in_2():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of dynamic output "dynamic_op:result" since input "a" maps to a node that is already downstream of another dynamic output'):

        @graph
        def composed_add(a):
            if False:
                for i in range(10):
                    print('nop')
            dynamic_op().map(lambda b: add(a, b))

        @graph
        def indirect(a):
            if False:
                for i in range(10):
                    print('nop')
            composed_add(a)

        @job
        def _should_fail():
            if False:
                i = 10
                return i + 15
            dynamic_op().map(lambda x: indirect(echo(x)))

def test_multi_composite_in_3():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of dynamic output "dynamic_op:result" since input "a" maps to a node that is already downstream of another dynamic output'):

        @graph
        def composed(a):
            if False:
                i = 10
                return i + 15
            dynamic_echo(a).map(echo)

        @job
        def _should_fail():
            if False:
                print('Hello World!')
            dynamic_op().map(composed)

def test_multi_composite_in_4():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of dynamic output "dynamic_op:result" since input "a" maps to a node that is already downstream of another dynamic output'):

        @graph
        def composed(a):
            if False:
                while True:
                    i = 10
            dynamic_echo(a).map(echo)

        @graph
        def indirect(a):
            if False:
                for i in range(10):
                    print('nop')
            composed(a)

        @job
        def _should_fail():
            if False:
                while True:
                    i = 10
            dynamic_op().map(indirect)

def test_direct_dep():
    if False:
        return 10

    @op(out=DynamicOut())
    def dynamic_add(_, x):
        if False:
            print('Hello World!')
        yield DynamicOutput(x + 1, mapping_key='1')
        yield DynamicOutput(x + 2, mapping_key='2')

    @job
    def _is_fine_1():
        if False:
            return 10

        def _add(item):
            if False:
                while True:
                    i = 10
            dynamic_add(item)
        dynamic_op().map(_add)
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of more than one dynamic output'):

        @job
        def _should_fail():
            if False:
                while True:
                    i = 10

            def _add_echo(item):
                if False:
                    while True:
                        i = 10
                dynamic_add(item).map(echo)
            dynamic_op().map(_add_echo)

    @job
    def _is_fine_2():
        if False:
            for i in range(10):
                print('nop')
        dynamic_op().map(dynamic_add)
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be downstream of more than one dynamic output'):

        @job
        def _should_fail():
            if False:
                print('Hello World!')
            echo(dynamic_op().map(dynamic_add).collect())

def test_collect_and_dep():
    if False:
        return 10
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot both collect over dynamic output'):

        @job
        def _bad():
            if False:
                while True:
                    i = 10
            x = dynamic_op()
            x.map(lambda y: add(y, x.collect()))
    with pytest.raises(DagsterInvalidDefinitionError, match='cannot be both downstream of dynamic output'):

        @job
        def _bad_other():
            if False:
                print('Hello World!')
            x = dynamic_op()
            x.map(lambda y: add(x.collect(), y))