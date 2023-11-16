from __future__ import annotations
import itertools
import pytest
from airflow.decorators import setup, task, teardown
from airflow.models.baseoperator import BaseOperator
pytestmark = pytest.mark.db_test

def cleared_tasks(dag, task_id):
    if False:
        for i in range(10):
            print('nop')
    dag_ = dag.partial_subset(task_id, include_downstream=True, include_upstream=False)
    return {x.task_id for x in dag_.tasks}

def get_task_attr(task_like, attr):
    if False:
        return 10
    try:
        return getattr(task_like, attr)
    except AttributeError:
        return getattr(task_like.operator, attr)

def make_task(name, type_, setup_=False, teardown_=False):
    if False:
        for i in range(10):
            print('nop')
    if type_ == 'classic' and setup_:
        return BaseOperator(task_id=name).as_setup()
    elif type_ == 'classic' and teardown_:
        return BaseOperator(task_id=name).as_teardown()
    elif type_ == 'classic':
        return BaseOperator(task_id=name)
    elif setup_:

        @setup
        def setuptask():
            if False:
                i = 10
                return i + 15
            pass
        return setuptask.override(task_id=name)()
    elif teardown_:

        @teardown
        def teardowntask():
            if False:
                for i in range(10):
                    print('nop')
            pass
        return teardowntask.override(task_id=name)()
    else:

        @task
        def my_task():
            if False:
                for i in range(10):
                    print('nop')
            pass
        return my_task.override(task_id=name)()

@pytest.mark.parametrize('setup_type, work_type, teardown_type', itertools.product(['classic', 'taskflow'], repeat=3))
def test_as_teardown(dag_maker, setup_type, work_type, teardown_type):
    if False:
        print('Hello World!')
    '\n    Check that as_teardown works properly as implemented in PlainXComArg\n\n    It should mark the teardown as teardown, and if a task is provided, it should mark that as setup\n    and set it as a direct upstream.\n    '
    with dag_maker() as dag:
        s1 = make_task(name='s1', type_=setup_type)
        w1 = make_task(name='w1', type_=work_type)
        t1 = make_task(name='t1', type_=teardown_type)
    assert cleared_tasks(dag, 'w1') == {'w1'}
    s1 >> w1 >> t1
    assert cleared_tasks(dag, 'w1') == {'w1', 't1'}
    assert get_task_attr(t1, 'is_teardown') is False
    assert get_task_attr(s1, 'is_setup') is False
    assert get_task_attr(t1, 'upstream_task_ids') == {'w1'}
    t1.as_teardown(setups=s1)
    assert cleared_tasks(dag, 'w1') == {'s1', 'w1', 't1'}
    assert get_task_attr(t1, 'is_teardown') is True
    assert get_task_attr(s1, 'is_setup') is True
    assert get_task_attr(t1, 'upstream_task_ids') == {'w1', 's1'}

@pytest.mark.parametrize('setup_type, work_type, teardown_type', itertools.product(['classic', 'taskflow'], repeat=3))
def test_as_teardown_oneline(dag_maker, setup_type, work_type, teardown_type):
    if False:
        return 10
    '\n    Check that as_teardown implementations work properly. Tests all combinations of taskflow and classic.\n\n    It should mark the teardown as teardown, and if a task is provided, it should mark that as setup\n    and set it as a direct upstream.\n    '
    with dag_maker() as dag:
        s1 = make_task(name='s1', type_=setup_type)
        w1 = make_task(name='w1', type_=work_type)
        t1 = make_task(name='t1', type_=teardown_type)
    for task_ in (s1, w1, t1):
        assert get_task_attr(task_, 'upstream_list') == []
        assert get_task_attr(task_, 'downstream_list') == []
        assert get_task_attr(task_, 'is_setup') is False
        assert get_task_attr(task_, 'is_teardown') is False
        assert cleared_tasks(dag, get_task_attr(task_, 'task_id')) == {get_task_attr(task_, 'task_id')}
    s1 >> w1 >> t1.as_teardown(setups=s1)
    for (task_, exp_up, exp_down) in [(s1, set(), {'w1', 't1'}), (w1, {'s1'}, {'t1'}), (t1, {'s1', 'w1'}, set())]:
        assert get_task_attr(task_, 'upstream_task_ids') == exp_up
        assert get_task_attr(task_, 'downstream_task_ids') == exp_down
    assert cleared_tasks(dag, 's1') == {'s1', 'w1', 't1'}
    assert cleared_tasks(dag, 'w1') == {'s1', 'w1', 't1'}
    assert cleared_tasks(dag, 't1') == {'t1'}
    for (task_, exp_is_setup, exp_is_teardown) in [(s1, True, False), (w1, False, False), (t1, False, True)]:
        assert get_task_attr(task_, 'is_setup') is exp_is_setup
        assert get_task_attr(task_, 'is_teardown') is exp_is_teardown

@pytest.mark.parametrize('type_', ['classic', 'taskflow'])
def test_cannot_be_both_setup_and_teardown(dag_maker, type_):
    if False:
        while True:
            i = 10
    for (first, second) in [('setup', 'teardown'), ('teardown', 'setup')]:
        with dag_maker():
            s1 = make_task(name='s1', type_=type_)
            getattr(s1, f'as_{first}')()
            with pytest.raises(ValueError, match=f"Cannot mark task 's1' as {second}; task is already a {first}."):
                getattr(s1, f'as_{second}')()
                s1.as_teardown()

def test_cannot_set_on_failure_fail_dagrun_unless_teardown_classic(dag_maker):
    if False:
        print('Hello World!')
    with dag_maker():
        t = make_task(name='t', type_='classic')
        assert t.is_teardown is False
        with pytest.raises(ValueError, match="Cannot set task on_failure_fail_dagrun for 't' because it is not a teardown task"):
            t.on_failure_fail_dagrun = True

def test_cannot_set_on_failure_fail_dagrun_unless_teardown_taskflow(dag_maker):
    if False:
        i = 10
        return i + 15

    @task(on_failure_fail_dagrun=True)
    def my_bad_task():
        if False:
            for i in range(10):
                print('nop')
        pass

    @task
    def my_ok_task():
        if False:
            while True:
                i = 10
        pass
    with dag_maker():
        with pytest.raises(ValueError, match="Cannot set task on_failure_fail_dagrun for 'my_bad_task' because it is not a teardown task"):
            my_bad_task()
        m = my_ok_task()
        assert m.operator.is_teardown is False
        m = my_ok_task().as_teardown()
        assert m.operator.is_teardown is True
        assert m.operator.on_failure_fail_dagrun is False
        m = my_ok_task().as_teardown(on_failure_fail_dagrun=True)
        assert m.operator.is_teardown is True
        assert m.operator.on_failure_fail_dagrun is True
        with pytest.raises(ValueError, match="Cannot mark task 'my_ok_task__2' as setup; task is already a teardown."):
            m.as_setup()
        with pytest.raises(ValueError, match="Cannot mark task 'my_ok_task__2' as setup; task is already a teardown."):
            m.operator.is_setup = True