from __future__ import annotations
from datetime import timedelta
import pendulum
import pytest
from airflow.decorators import dag, task_group
from airflow.models.expandinput import DictOfListsExpandInput, ListOfDictsExpandInput, MappedArgument
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import MappedTaskGroup

def test_task_group_with_overridden_kwargs():
    if False:
        while True:
            i = 10

    @task_group(default_args={'params': {'x': 5, 'y': 5}}, add_suffix_on_collision=True)
    def simple_tg():
        if False:
            print('Hello World!')
        ...
    tg_with_overridden_kwargs = simple_tg.override(group_id='custom_group_id', default_args={'params': {'x': 10}})
    assert tg_with_overridden_kwargs.tg_kwargs == {'group_id': 'custom_group_id', 'default_args': {'params': {'x': 10}}, 'add_suffix_on_collision': True}

def test_tooltip_derived_from_function_docstring():
    if False:
        for i in range(10):
            print('nop')
    "Test that the tooltip for TaskGroup is the decorated-function's docstring."

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            print('Hello World!')

        @task_group()
        def tg():
            if False:
                while True:
                    i = 10
            'Function docstring.'
        tg()
    _ = pipeline()
    assert _.task_group_dict['tg'].tooltip == 'Function docstring.'

def test_tooltip_not_overridden_by_function_docstring():
    if False:
        while True:
            i = 10
    '\n    Test that the tooltip for TaskGroup is the explicitly set value even if the decorated function has a\n    docstring.\n    '

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            i = 10
            return i + 15

        @task_group(tooltip='tooltip for the TaskGroup')
        def tg():
            if False:
                return 10
            'Function docstring.'
        tg()
    _ = pipeline()
    assert _.task_group_dict['tg'].tooltip == 'tooltip for the TaskGroup'

def test_partial_evolves_factory():
    if False:
        return 10
    tgp = None

    @task_group()
    def tg(a, b):
        if False:
            i = 10
            return i + 15
        pass

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            while True:
                i = 10
        nonlocal tgp
        tgp = tg.partial(a=1)
    d = pipeline()
    assert d.task_group_dict == {}
    assert type(tgp) == type(tg)
    assert tgp.partial_kwargs == {'a': 1}
    with pytest.warns(UserWarning, match="Partial task group 'tg' was never mapped!"):
        del tgp

def test_expand_fail_empty():
    if False:
        i = 10
        return i + 15

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            for i in range(10):
                print('nop')

        @task_group()
        def tg():
            if False:
                for i in range(10):
                    print('nop')
            pass
        tg.expand()
    with pytest.raises(TypeError) as ctx:
        pipeline()
    assert str(ctx.value) == 'no arguments to expand against'

def test_expand_create_mapped():
    if False:
        return 10
    saved = {}

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            while True:
                i = 10

        @task_group()
        def tg(a, b):
            if False:
                while True:
                    i = 10
            saved['a'] = a
            saved['b'] = b
        tg.partial(a=1).expand(b=['x', 'y'])
    d = pipeline()
    tg = d.task_group_dict['tg']
    assert isinstance(tg, MappedTaskGroup)
    assert tg._expand_input == DictOfListsExpandInput({'b': ['x', 'y']})
    assert saved == {'a': 1, 'b': MappedArgument(input=tg._expand_input, key='b')}

def test_expand_kwargs_no_wildcard():
    if False:
        for i in range(10):
            print('nop')

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            for i in range(10):
                print('nop')

        @task_group()
        def tg(**kwargs):
            if False:
                print('Hello World!')
            pass
        tg.expand_kwargs([])
    with pytest.raises(TypeError) as ctx:
        pipeline()
    assert str(ctx.value) == 'calling expand_kwargs() on task group function with * or ** is not supported'

def test_expand_kwargs_create_mapped():
    if False:
        return 10
    saved = {}

    @dag(start_date=pendulum.datetime(2022, 1, 1))
    def pipeline():
        if False:
            print('Hello World!')

        @task_group()
        def tg(a, b):
            if False:
                i = 10
                return i + 15
            saved['a'] = a
            saved['b'] = b
        tg.partial(a=1).expand_kwargs([{'b': 'x'}, {'b': None}])
    d = pipeline()
    tg = d.task_group_dict['tg']
    assert isinstance(tg, MappedTaskGroup)
    assert tg._expand_input == ListOfDictsExpandInput([{'b': 'x'}, {'b': None}])
    assert saved == {'a': 1, 'b': MappedArgument(input=tg._expand_input, key='b')}

@pytest.mark.db_test
def test_task_group_expand_kwargs_with_upstream(dag_maker, session, caplog):
    if False:
        print('Hello World!')
    with dag_maker() as dag:

        @dag.task
        def t1():
            if False:
                i = 10
                return i + 15
            return [{'a': 1}, {'a': 2}]

        @task_group('tg1')
        def tg1(a, b):
            if False:
                print('Hello World!')

            @dag.task()
            def t2():
                if False:
                    for i in range(10):
                        print('nop')
                return [a, b]
            t2()
        tg1.expand_kwargs(t1())
    dr = dag_maker.create_dagrun()
    dr.task_instance_scheduling_decisions()
    assert 'Cannot expand' not in caplog.text
    assert "missing upstream values: ['expand_kwargs() argument']" not in caplog.text

@pytest.mark.db_test
def test_task_group_expand_with_upstream(dag_maker, session, caplog):
    if False:
        return 10
    with dag_maker() as dag:

        @dag.task
        def t1():
            if False:
                print('Hello World!')
            return [1, 2, 3]

        @task_group('tg1')
        def tg1(a, b):
            if False:
                return 10

            @dag.task()
            def t2():
                if False:
                    return 10
                return [a, b]
            t2()
        tg1.partial(a=1).expand(b=t1())
    dr = dag_maker.create_dagrun()
    dr.task_instance_scheduling_decisions()
    assert 'Cannot expand' not in caplog.text
    assert "missing upstream values: ['b']" not in caplog.text

def test_override_dag_default_args():
    if False:
        while True:
            i = 10

    @dag(dag_id='test_dag', start_date=pendulum.parse('20200101'), default_args={'retries': 1, 'owner': 'x'})
    def pipeline():
        if False:
            return 10

        @task_group(group_id='task_group', default_args={'owner': 'y', 'execution_timeout': timedelta(seconds=10)})
        def tg():
            if False:
                for i in range(10):
                    print('nop')
            EmptyOperator(task_id='task')
        tg()
    test_dag = pipeline()
    test_task = test_dag.task_group_dict['task_group'].children['task_group.task']
    assert test_task.retries == 1
    assert test_task.owner == 'y'
    assert test_task.execution_timeout == timedelta(seconds=10)

def test_override_dag_default_args_nested_tg():
    if False:
        i = 10
        return i + 15

    @dag(dag_id='test_dag', start_date=pendulum.parse('20200101'), default_args={'retries': 1, 'owner': 'x'})
    def pipeline():
        if False:
            return 10

        @task_group(group_id='task_group', default_args={'owner': 'y', 'execution_timeout': timedelta(seconds=10)})
        def tg():
            if False:
                i = 10
                return i + 15

            @task_group(group_id='nested_task_group')
            def nested_tg():
                if False:
                    print('Hello World!')

                @task_group(group_id='another_task_group')
                def another_tg():
                    if False:
                        while True:
                            i = 10
                    EmptyOperator(task_id='task')
                another_tg()
            nested_tg()
        tg()
    test_dag = pipeline()
    test_task = test_dag.task_group_dict['task_group'].children['task_group.nested_task_group'].children['task_group.nested_task_group.another_task_group'].children['task_group.nested_task_group.another_task_group.task']
    assert test_task.retries == 1
    assert test_task.owner == 'y'
    assert test_task.execution_timeout == timedelta(seconds=10)