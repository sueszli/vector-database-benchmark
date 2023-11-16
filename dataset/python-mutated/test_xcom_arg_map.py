from __future__ import annotations
import pytest
from airflow.exceptions import AirflowSkipException
from airflow.utils.state import TaskInstanceState
from airflow.utils.trigger_rule import TriggerRule
pytestmark = pytest.mark.db_test

def test_xcom_map(dag_maker, session):
    if False:
        return 10
    results = set()
    with dag_maker(session=session) as dag:

        @dag.task
        def push():
            if False:
                while True:
                    i = 10
            return ['a', 'b', 'c']

        @dag.task
        def pull(value):
            if False:
                i = 10
                return i + 15
            results.add(value)
        pull.expand_kwargs(push().map(lambda v: {'value': v * 2}))
    assert set(dag.task_dict) == {'push', 'pull'}
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    session.commit()
    decision = dr.task_instance_scheduling_decisions(session=session)
    tis = {(ti.task_id, ti.map_index): ti for ti in decision.schedulable_tis}
    assert sorted(tis) == [('pull', 0), ('pull', 1), ('pull', 2)]
    for ti in tis.values():
        ti.run(session=session)
    assert results == {'aa', 'bb', 'cc'}

def test_xcom_map_transform_to_none(dag_maker, session):
    if False:
        for i in range(10):
            print('nop')
    results = set()
    with dag_maker(session=session) as dag:

        @dag.task()
        def push():
            if False:
                for i in range(10):
                    print('nop')
            return ['a', 'b', 'c']

        @dag.task()
        def pull(value):
            if False:
                print('Hello World!')
            results.add(value)

        def c_to_none(v):
            if False:
                for i in range(10):
                    print('nop')
            if v == 'c':
                return None
            return v
        pull.expand(value=push().map(c_to_none))
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    assert results == {'a', 'b', None}

def test_xcom_convert_to_kwargs_fails_task(dag_maker, session):
    if False:
        for i in range(10):
            print('nop')
    results = set()
    with dag_maker(session=session) as dag:

        @dag.task()
        def push():
            if False:
                while True:
                    i = 10
            return ['a', 'b', 'c']

        @dag.task()
        def pull(value):
            if False:
                while True:
                    i = 10
            results.add(value)

        def c_to_none(v):
            if False:
                for i in range(10):
                    print('nop')
            if v == 'c':
                return None
            return {'value': v}
        pull.expand_kwargs(push().map(c_to_none))
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    tis = {(ti.task_id, ti.map_index): ti for ti in decision.schedulable_tis}
    tis['pull', 0].run(session=session)
    tis['pull', 1].run(session=session)
    with pytest.raises(ValueError) as ctx:
        tis['pull', 2].run(session=session)
    assert str(ctx.value) == 'expand_kwargs() expects a list[dict], not list[None]'
    assert [tis['pull', i].state for i in range(3)] == [TaskInstanceState.SUCCESS, TaskInstanceState.SUCCESS, TaskInstanceState.FAILED]

def test_xcom_map_error_fails_task(dag_maker, session):
    if False:
        print('Hello World!')
    with dag_maker(session=session) as dag:

        @dag.task()
        def push():
            if False:
                return 10
            return ['a', 'b', 'c']

        @dag.task()
        def pull(value):
            if False:
                print('Hello World!')
            print(value)

        def does_not_work_with_c(v):
            if False:
                while True:
                    i = 10
            if v == 'c':
                raise ValueError('nope')
            return {'value': v * 2}
        pull.expand_kwargs(push().map(does_not_work_with_c))
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    assert [ti.state for ti in decision.schedulable_tis] == [TaskInstanceState.SUCCESS]
    decision = dr.task_instance_scheduling_decisions(session=session)
    tis = {(ti.task_id, ti.map_index): ti for ti in decision.schedulable_tis}
    tis['pull', 0].run(session=session)
    tis['pull', 1].run(session=session)
    with pytest.raises(ValueError) as ctx:
        tis['pull', 2].run(session=session)
    assert str(ctx.value) == 'nope'
    assert [tis['pull', i].state for i in range(3)] == [TaskInstanceState.SUCCESS, TaskInstanceState.SUCCESS, TaskInstanceState.FAILED]

def test_xcom_map_raise_to_skip(dag_maker, session):
    if False:
        i = 10
        return i + 15
    result = None
    with dag_maker(session=session) as dag:

        @dag.task()
        def push():
            if False:
                while True:
                    i = 10
            return ['a', 'b', 'c']

        @dag.task()
        def forward(value):
            if False:
                print('Hello World!')
            return value

        @dag.task(trigger_rule=TriggerRule.ALL_DONE)
        def collect(value):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal result
            result = list(value)

        def skip_c(v):
            if False:
                for i in range(10):
                    print('nop')
            if v == 'c':
                raise AirflowSkipException
            return {'value': v}
        collect(value=forward.expand_kwargs(push().map(skip_c)))
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    assert result == ['a', 'b']

def test_xcom_map_nest(dag_maker, session):
    if False:
        for i in range(10):
            print('nop')
    results = set()
    with dag_maker(session=session) as dag:

        @dag.task()
        def push():
            if False:
                i = 10
                return i + 15
            return ['a', 'b', 'c']

        @dag.task()
        def pull(value):
            if False:
                i = 10
                return i + 15
            results.add(value)
        converted = push().map(lambda v: v * 2).map(lambda v: {'value': v})
        pull.expand_kwargs(converted)
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    session.flush()
    session.commit()
    decision = dr.task_instance_scheduling_decisions(session=session)
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    assert results == {'aa', 'bb', 'cc'}

def test_xcom_map_zip_nest(dag_maker, session):
    if False:
        for i in range(10):
            print('nop')
    results = set()
    with dag_maker(session=session) as dag:

        @dag.task
        def push_letters():
            if False:
                print('Hello World!')
            return ['a', 'b', 'c', 'd']

        @dag.task
        def push_numbers():
            if False:
                while True:
                    i = 10
            return [1, 2, 3, 4]

        @dag.task
        def pull(value):
            if False:
                i = 10
                return i + 15
            results.add(value)
        doubled = push_numbers().map(lambda v: v * 2)
        combined = doubled.zip(push_letters())

        def convert_zipped(zipped):
            if False:
                for i in range(10):
                    print('nop')
            (letter, number) = zipped
            return letter * number
        pull.expand(value=combined.map(convert_zipped))
    dr = dag_maker.create_dagrun(session=session)
    decision = dr.task_instance_scheduling_decisions(session=session)
    assert decision.schedulable_tis and all((ti.task_id.startswith('push_') for ti in decision.schedulable_tis))
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    session.commit()
    decision = dr.task_instance_scheduling_decisions(session=session)
    assert decision.schedulable_tis and all((ti.task_id == 'pull' for ti in decision.schedulable_tis))
    for ti in decision.schedulable_tis:
        ti.run(session=session)
    assert results == {'aa', 'bbbb', 'cccccc', 'dddddddd'}