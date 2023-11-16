from __future__ import annotations
import logging
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import patch
import pendulum
import pytest
from airflow.decorators import setup, task, task_group, teardown
from airflow.exceptions import AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.models.dag import DAG
from airflow.models.mappedoperator import MappedOperator
from airflow.models.param import ParamsDict
from airflow.models.taskinstance import TaskInstance
from airflow.models.taskmap import TaskMap
from airflow.models.xcom_arg import XComArg
from airflow.operators.python import PythonOperator
from airflow.utils.state import TaskInstanceState
from airflow.utils.task_group import TaskGroup
from airflow.utils.task_instance_session import set_current_task_instance_session
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.xcom import XCOM_RETURN_KEY
from tests.models import DEFAULT_DATE
from tests.test_utils.mapping import expand_mapped_task
from tests.test_utils.mock_operators import MockOperator, MockOperatorWithNestedFields, NestedFields
pytestmark = pytest.mark.db_test
if TYPE_CHECKING:
    from airflow.utils.context import Context

def test_task_mapping_with_dag():
    if False:
        print('Hello World!')
    with DAG('test-dag', start_date=DEFAULT_DATE) as dag:
        task1 = BaseOperator(task_id='op1')
        literal = ['a', 'b', 'c']
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=literal)
        finish = MockOperator(task_id='finish')
        task1 >> mapped >> finish
    assert task1.downstream_list == [mapped]
    assert mapped in dag.tasks
    assert mapped.task_group == dag.task_group
    assert len(dag.tasks) == 3
    assert finish.upstream_list == [mapped]
    assert mapped.downstream_list == [finish]

@patch('airflow.models.abstractoperator.AbstractOperator.render_template')
def test_task_mapping_with_dag_and_list_of_pandas_dataframe(mock_render_template, caplog):
    if False:
        i = 10
        return i + 15
    caplog.set_level(logging.INFO)

    class UnrenderableClass:

        def __bool__(self):
            if False:
                return 10
            raise ValueError('Similar to Pandas DataFrames, this class raises an exception.')

    class CustomOperator(BaseOperator):
        template_fields = ('arg',)

        def __init__(self, arg, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)
            self.arg = arg

        def execute(self, context: Context):
            if False:
                i = 10
                return i + 15
            pass
    with DAG('test-dag', start_date=DEFAULT_DATE) as dag:
        task1 = CustomOperator(task_id='op1', arg=None)
        unrenderable_values = [UnrenderableClass(), UnrenderableClass()]
        mapped = CustomOperator.partial(task_id='task_2').expand(arg=unrenderable_values)
        task1 >> mapped
    dag.test()
    assert caplog.text.count('task_2 ran successfully') == 2
    assert "Unable to check if the value of type 'UnrenderableClass' is False for task 'task_2', field 'arg'" in caplog.text
    mock_render_template.assert_called()

def test_task_mapping_without_dag_context():
    if False:
        return 10
    with DAG('test-dag', start_date=DEFAULT_DATE) as dag:
        task1 = BaseOperator(task_id='op1')
    literal = ['a', 'b', 'c']
    mapped = MockOperator.partial(task_id='task_2').expand(arg2=literal)
    task1 >> mapped
    assert isinstance(mapped, MappedOperator)
    assert mapped in dag.tasks
    assert task1.downstream_list == [mapped]
    assert mapped in dag.tasks
    assert len(dag.tasks) == 2

def test_task_mapping_default_args():
    if False:
        for i in range(10):
            print('nop')
    default_args = {'start_date': DEFAULT_DATE.now(), 'owner': 'test'}
    with DAG('test-dag', start_date=DEFAULT_DATE, default_args=default_args):
        task1 = BaseOperator(task_id='op1')
        literal = ['a', 'b', 'c']
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=literal)
        task1 >> mapped
    assert mapped.partial_kwargs['owner'] == 'test'
    assert mapped.start_date == pendulum.instance(default_args['start_date'])

def test_task_mapping_override_default_args():
    if False:
        return 10
    default_args = {'retries': 2, 'start_date': DEFAULT_DATE.now()}
    with DAG('test-dag', start_date=DEFAULT_DATE, default_args=default_args):
        literal = ['a', 'b', 'c']
        mapped = MockOperator.partial(task_id='task', retries=1).expand(arg2=literal)
    assert mapped.partial_kwargs['retries'] == 1
    assert mapped.start_date == pendulum.instance(default_args['start_date'])
    assert mapped.owner == 'airflow'

def test_map_unknown_arg_raises():
    if False:
        print('Hello World!')
    with pytest.raises(TypeError, match="argument 'file'"):
        BaseOperator.partial(task_id='a').expand(file=[1, 2, {'a': 'b'}])

def test_map_xcom_arg():
    if False:
        print('Hello World!')
    'Test that dependencies are correct when mapping with an XComArg'
    with DAG('test-dag', start_date=DEFAULT_DATE):
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=task1.output)
        finish = MockOperator(task_id='finish')
        mapped >> finish
    assert task1.downstream_list == [mapped]

def test_map_xcom_arg_multiple_upstream_xcoms(dag_maker, session):
    if False:
        for i in range(10):
            print('nop')
    'Test that the correct number of downstream tasks are generated when mapping with an XComArg'

    class PushExtraXComOperator(BaseOperator):
        """Push an extra XCom value along with the default return value."""

        def __init__(self, return_value, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)
            self.return_value = return_value

        def execute(self, context):
            if False:
                return 10
            context['task_instance'].xcom_push(key='extra_key', value='extra_value')
            return self.return_value
    with dag_maker('test-dag', session=session, start_date=DEFAULT_DATE) as dag:
        upstream_return = [1, 2, 3]
        task1 = PushExtraXComOperator(return_value=upstream_return, task_id='task_1')
        task2 = PushExtraXComOperator.partial(task_id='task_2').expand(return_value=task1.output)
        task3 = PushExtraXComOperator.partial(task_id='task_3').expand(return_value=task2.output)
    dr = dag_maker.create_dagrun()
    ti_1 = dr.get_task_instance('task_1', session)
    ti_1.run()
    (ti_2s, _) = task2.expand_mapped_task(dr.run_id, session=session)
    for ti in ti_2s:
        ti.refresh_from_task(dag.get_task('task_2'))
        ti.run()
    (ti_3s, _) = task3.expand_mapped_task(dr.run_id, session=session)
    for ti in ti_3s:
        ti.refresh_from_task(dag.get_task('task_3'))
        ti.run()
    assert len(ti_3s) == len(ti_2s) == len(upstream_return)

def test_partial_on_instance() -> None:
    if False:
        return 10
    "`.partial` on an instance should fail -- it's only designed to be called on classes"
    with pytest.raises(TypeError):
        MockOperator(task_id='a').partial()

def test_partial_on_class() -> None:
    if False:
        i = 10
        return i + 15
    op = MockOperator.partial(task_id='a', arg1='a', trigger_rule=TriggerRule.ONE_FAILED)
    assert op.kwargs['arg1'] == 'a'
    assert op.kwargs['trigger_rule'] == TriggerRule.ONE_FAILED

def test_partial_on_class_invalid_ctor_args() -> None:
    if False:
        return 10
    'Test that when we pass invalid args to partial().\n\n    I.e. if an arg is not known on the class or any of its parent classes we error at parse time\n    '
    with pytest.raises(TypeError, match="arguments 'foo', 'bar'"):
        MockOperator.partial(task_id='a', foo='bar', bar=2)

@pytest.mark.parametrize(['num_existing_tis', 'expected'], (pytest.param(0, [(0, None), (1, None), (2, None)], id='only-unmapped-ti-exists'), pytest.param(3, [(0, 'success'), (1, 'success'), (2, 'success')], id='all-tis-exist'), pytest.param(5, [(0, 'success'), (1, 'success'), (2, 'success'), (3, TaskInstanceState.REMOVED), (4, TaskInstanceState.REMOVED)], id='tis-to-be-removed')))
def test_expand_mapped_task_instance(dag_maker, session, num_existing_tis, expected):
    if False:
        return 10
    literal = [1, 2, {'a': 'b'}]
    with dag_maker(session=session):
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=task1.output)
    dr = dag_maker.create_dagrun()
    session.add(TaskMap(dag_id=dr.dag_id, task_id=task1.task_id, run_id=dr.run_id, map_index=-1, length=len(literal), keys=None))
    if num_existing_tis:
        session.query(TaskInstance).filter(TaskInstance.dag_id == mapped.dag_id, TaskInstance.task_id == mapped.task_id, TaskInstance.run_id == dr.run_id).delete()
    for index in range(num_existing_tis):
        ti = TaskInstance(mapped, run_id=dr.run_id, map_index=index, state=TaskInstanceState.SUCCESS)
        session.add(ti)
    session.flush()
    mapped.expand_mapped_task(dr.run_id, session=session)
    indices = session.query(TaskInstance.map_index, TaskInstance.state).filter_by(task_id=mapped.task_id, dag_id=mapped.dag_id, run_id=dr.run_id).order_by(TaskInstance.map_index).all()
    assert indices == expected

def test_expand_mapped_task_failed_state_in_db(dag_maker, session):
    if False:
        while True:
            i = 10
    '\n    This test tries to recreate a faulty state in the database and checks if we can recover from it.\n    The state that happens is that there exists mapped task instances and the unmapped task instance.\n    So we have instances with map_index [-1, 0, 1]. The -1 task instances should be removed in this case.\n    '
    literal = [1, 2]
    with dag_maker(session=session):
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=task1.output)
    dr = dag_maker.create_dagrun()
    session.add(TaskMap(dag_id=dr.dag_id, task_id=task1.task_id, run_id=dr.run_id, map_index=-1, length=len(literal), keys=None))
    for index in range(2):
        ti = TaskInstance(mapped, run_id=dr.run_id, map_index=index, state=TaskInstanceState.SUCCESS)
        session.add(ti)
    session.flush()
    indices = session.query(TaskInstance.map_index, TaskInstance.state).filter_by(task_id=mapped.task_id, dag_id=mapped.dag_id, run_id=dr.run_id).order_by(TaskInstance.map_index).all()
    assert indices == [(-1, None), (0, 'success'), (1, 'success')]
    mapped.expand_mapped_task(dr.run_id, session=session)
    indices = session.query(TaskInstance.map_index, TaskInstance.state).filter_by(task_id=mapped.task_id, dag_id=mapped.dag_id, run_id=dr.run_id).order_by(TaskInstance.map_index).all()
    assert indices == [(0, 'success'), (1, 'success')]

def test_expand_mapped_task_instance_skipped_on_zero(dag_maker, session):
    if False:
        i = 10
        return i + 15
    with dag_maker(session=session):
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=task1.output)
    dr = dag_maker.create_dagrun()
    expand_mapped_task(mapped, dr.run_id, task1.task_id, length=0, session=session)
    indices = session.query(TaskInstance.map_index, TaskInstance.state).filter_by(task_id=mapped.task_id, dag_id=mapped.dag_id, run_id=dr.run_id).order_by(TaskInstance.map_index).all()
    assert indices == [(-1, TaskInstanceState.SKIPPED)]

def test_mapped_task_applies_default_args_classic(dag_maker):
    if False:
        while True:
            i = 10
    with dag_maker(default_args={'execution_timeout': timedelta(minutes=30)}) as dag:
        MockOperator(task_id='simple', arg1=None, arg2=0)
        MockOperator.partial(task_id='mapped').expand(arg1=[1], arg2=[2, 3])
    assert dag.get_task('simple').execution_timeout == timedelta(minutes=30)
    assert dag.get_task('mapped').execution_timeout == timedelta(minutes=30)

def test_mapped_task_applies_default_args_taskflow(dag_maker):
    if False:
        return 10
    with dag_maker(default_args={'execution_timeout': timedelta(minutes=30)}) as dag:

        @dag.task
        def simple(arg):
            if False:
                print('Hello World!')
            pass

        @dag.task
        def mapped(arg):
            if False:
                i = 10
                return i + 15
            pass
        simple(arg=0)
        mapped.expand(arg=[1, 2])
    assert dag.get_task('simple').execution_timeout == timedelta(minutes=30)
    assert dag.get_task('mapped').execution_timeout == timedelta(minutes=30)

@pytest.mark.parametrize('dag_params, task_params, expected_partial_params', [pytest.param(None, None, ParamsDict(), id='none'), pytest.param({'a': -1}, None, ParamsDict({'a': -1}), id='dag'), pytest.param(None, {'b': -2}, ParamsDict({'b': -2}), id='task'), pytest.param({'a': -1}, {'b': -2}, ParamsDict({'a': -1, 'b': -2}), id='merge')])
def test_mapped_expand_against_params(dag_maker, dag_params, task_params, expected_partial_params):
    if False:
        print('Hello World!')
    with dag_maker(params=dag_params) as dag:
        MockOperator.partial(task_id='t', params=task_params).expand(params=[{'c': 'x'}, {'d': 1}])
    t = dag.get_task('t')
    assert isinstance(t, MappedOperator)
    assert t.params == expected_partial_params
    assert t.expand_input.value == {'params': [{'c': 'x'}, {'d': 1}]}

def test_mapped_render_template_fields_validating_operator(dag_maker, session):
    if False:
        i = 10
        return i + 15
    with set_current_task_instance_session(session=session):

        class MyOperator(BaseOperator):
            template_fields = ('partial_template', 'map_template', 'file_template')
            template_ext = ('.ext',)

            def __init__(self, partial_template, partial_static, map_template, map_static, file_template, **kwargs):
                if False:
                    return 10
                for value in [partial_template, partial_static, map_template, map_static, file_template]:
                    assert isinstance(value, str), 'value should have been resolved before unmapping'
                    super().__init__(**kwargs)
                    self.partial_template = partial_template
                self.partial_static = partial_static
                self.map_template = map_template
                self.map_static = map_static
                self.file_template = file_template

        def execute(self, context):
            if False:
                for i in range(10):
                    print('nop')
            pass
        with dag_maker(session=session):
            task1 = BaseOperator(task_id='op1')
            output1 = task1.output
            mapped = MyOperator.partial(task_id='a', partial_template='{{ ti.task_id }}', partial_static='{{ ti.task_id }}').expand(map_template=output1, map_static=output1, file_template=['/path/to/file.ext'])
        dr = dag_maker.create_dagrun()
        ti: TaskInstance = dr.get_task_instance(task1.task_id, session=session)
        ti.xcom_push(key=XCOM_RETURN_KEY, value=['{{ ds }}'], session=session)
        session.add(TaskMap(dag_id=dr.dag_id, task_id=task1.task_id, run_id=dr.run_id, map_index=-1, length=1, keys=None))
        session.flush()
        mapped_ti: TaskInstance = dr.get_task_instance(mapped.task_id, session=session)
        mapped_ti.map_index = 0
        assert isinstance(mapped_ti.task, MappedOperator)
        with patch('builtins.open', mock.mock_open(read_data=b'loaded data')), patch('os.path.isfile', return_value=True), patch('os.path.getmtime', return_value=0):
            mapped.render_template_fields(context=mapped_ti.get_template_context(session=session))
        assert isinstance(mapped_ti.task, MyOperator)
        assert mapped_ti.task.partial_template == 'a', 'Should be templated!'
        assert mapped_ti.task.partial_static == '{{ ti.task_id }}', 'Should not be templated!'
        assert mapped_ti.task.map_template == '{{ ds }}', 'Should not be templated!'
        assert mapped_ti.task.map_static == '{{ ds }}', 'Should not be templated!'
        assert mapped_ti.task.file_template == 'loaded data', 'Should be templated!'

def test_mapped_expand_kwargs_render_template_fields_validating_operator(dag_maker, session):
    if False:
        return 10
    with set_current_task_instance_session(session=session):

        class MyOperator(BaseOperator):
            template_fields = ('partial_template', 'map_template', 'file_template')
            template_ext = ('.ext',)

            def __init__(self, partial_template, partial_static, map_template, map_static, file_template, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                for value in [partial_template, partial_static, map_template, map_static, file_template]:
                    assert isinstance(value, str), 'value should have been resolved before unmapping'
                super().__init__(**kwargs)
                self.partial_template = partial_template
                self.partial_static = partial_static
                self.map_template = map_template
                self.map_static = map_static
                self.file_template = file_template

            def execute(self, context):
                if False:
                    i = 10
                    return i + 15
                pass
        with dag_maker(session=session):
            mapped = MyOperator.partial(task_id='a', partial_template='{{ ti.task_id }}', partial_static='{{ ti.task_id }}').expand_kwargs([{'map_template': '{{ ds }}', 'map_static': '{{ ds }}', 'file_template': '/path/to/file.ext'}])
        dr = dag_maker.create_dagrun()
        mapped_ti: TaskInstance = dr.get_task_instance(mapped.task_id, session=session, map_index=0)
        assert isinstance(mapped_ti.task, MappedOperator)
        with patch('builtins.open', mock.mock_open(read_data=b'loaded data')), patch('os.path.isfile', return_value=True), patch('os.path.getmtime', return_value=0):
            mapped.render_template_fields(context=mapped_ti.get_template_context(session=session))
        assert isinstance(mapped_ti.task, MyOperator)
        assert mapped_ti.task.partial_template == 'a', 'Should be templated!'
        assert mapped_ti.task.partial_static == '{{ ti.task_id }}', 'Should not be templated!'
        assert mapped_ti.task.map_template == '2016-01-01', 'Should be templated!'
        assert mapped_ti.task.map_static == '{{ ds }}', 'Should not be templated!'
        assert mapped_ti.task.file_template == 'loaded data', 'Should be templated!'

def test_mapped_render_nested_template_fields(dag_maker, session):
    if False:
        print('Hello World!')
    with dag_maker(session=session):
        MockOperatorWithNestedFields.partial(task_id='t', arg2=NestedFields(field_1='{{ ti.task_id }}', field_2='value_2')).expand(arg1=['{{ ti.task_id }}', ['s', '{{ ti.task_id }}']])
    dr = dag_maker.create_dagrun()
    decision = dr.task_instance_scheduling_decisions()
    tis = {(ti.task_id, ti.map_index): ti for ti in decision.schedulable_tis}
    assert len(tis) == 2
    ti = tis['t', 0]
    ti.run(session=session)
    assert ti.task.arg1 == 't'
    assert ti.task.arg2.field_1 == 't'
    assert ti.task.arg2.field_2 == 'value_2'
    ti = tis['t', 1]
    ti.run(session=session)
    assert ti.task.arg1 == ['s', 't']
    assert ti.task.arg2.field_1 == 't'
    assert ti.task.arg2.field_2 == 'value_2'

@pytest.mark.parametrize(['num_existing_tis', 'expected'], (pytest.param(0, [(0, None), (1, None), (2, None)], id='only-unmapped-ti-exists'), pytest.param(3, [(0, 'success'), (1, 'success'), (2, 'success')], id='all-tis-exist'), pytest.param(5, [(0, 'success'), (1, 'success'), (2, 'success'), (3, TaskInstanceState.REMOVED), (4, TaskInstanceState.REMOVED)], id='tis-to-be-removed')))
def test_expand_kwargs_mapped_task_instance(dag_maker, session, num_existing_tis, expected):
    if False:
        return 10
    literal = [{'arg1': 'a'}, {'arg1': 'b'}, {'arg1': 'c'}]
    with dag_maker(session=session):
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand_kwargs(task1.output)
    dr = dag_maker.create_dagrun()
    session.add(TaskMap(dag_id=dr.dag_id, task_id=task1.task_id, run_id=dr.run_id, map_index=-1, length=len(literal), keys=None))
    if num_existing_tis:
        session.query(TaskInstance).filter(TaskInstance.dag_id == mapped.dag_id, TaskInstance.task_id == mapped.task_id, TaskInstance.run_id == dr.run_id).delete()
    for index in range(num_existing_tis):
        ti = TaskInstance(mapped, run_id=dr.run_id, map_index=index, state=TaskInstanceState.SUCCESS)
        session.add(ti)
    session.flush()
    mapped.expand_mapped_task(dr.run_id, session=session)
    indices = session.query(TaskInstance.map_index, TaskInstance.state).filter_by(task_id=mapped.task_id, dag_id=mapped.dag_id, run_id=dr.run_id).order_by(TaskInstance.map_index).all()
    assert indices == expected

@pytest.mark.parametrize('map_index, expected', [pytest.param(0, '2016-01-01', id='0'), pytest.param(1, 2, id='1')])
def test_expand_kwargs_render_template_fields_validating_operator(dag_maker, session, map_index, expected):
    if False:
        for i in range(10):
            print('nop')
    with set_current_task_instance_session(session=session):
        with dag_maker(session=session):
            task1 = BaseOperator(task_id='op1')
            mapped = MockOperator.partial(task_id='a', arg2='{{ ti.task_id }}').expand_kwargs(task1.output)
        dr = dag_maker.create_dagrun()
        ti: TaskInstance = dr.get_task_instance(task1.task_id, session=session)
        ti.xcom_push(key=XCOM_RETURN_KEY, value=[{'arg1': '{{ ds }}'}, {'arg1': 2}], session=session)
        session.add(TaskMap(dag_id=dr.dag_id, task_id=task1.task_id, run_id=dr.run_id, map_index=-1, length=2, keys=None))
        session.flush()
        ti: TaskInstance = dr.get_task_instance(mapped.task_id, session=session)
        ti.refresh_from_task(mapped)
        ti.map_index = map_index
        assert isinstance(ti.task, MappedOperator)
        mapped.render_template_fields(context=ti.get_template_context(session=session))
        assert isinstance(ti.task, MockOperator)
        assert ti.task.arg1 == expected
        assert ti.task.arg2 == 'a'

def test_xcomarg_property_of_mapped_operator(dag_maker):
    if False:
        return 10
    with dag_maker('test_xcomarg_property_of_mapped_operator'):
        op_a = MockOperator.partial(task_id='a').expand(arg1=['x', 'y', 'z'])
    dag_maker.create_dagrun()
    assert op_a.output == XComArg(op_a)

def test_set_xcomarg_dependencies_with_mapped_operator(dag_maker):
    if False:
        return 10
    with dag_maker('test_set_xcomargs_dependencies_with_mapped_operator'):
        op1 = MockOperator.partial(task_id='op1').expand(arg1=[1, 2, 3])
        op2 = MockOperator.partial(task_id='op2').expand(arg2=['a', 'b', 'c'])
        op3 = MockOperator(task_id='op3', arg1=op1.output)
        op4 = MockOperator(task_id='op4', arg1=[op1.output, op2.output])
        op5 = MockOperator(task_id='op5', arg1={'op1': op1.output, 'op2': op2.output})
    assert op1 in op3.upstream_list
    assert op1 in op4.upstream_list
    assert op2 in op4.upstream_list
    assert op1 in op5.upstream_list
    assert op2 in op5.upstream_list

def test_all_xcomargs_from_mapped_tasks_are_consumable(dag_maker, session):
    if False:
        print('Hello World!')

    class PushXcomOperator(MockOperator):

        def __init__(self, arg1, **kwargs):
            if False:
                return 10
            super().__init__(arg1=arg1, **kwargs)

        def execute(self, context):
            if False:
                return 10
            return self.arg1

    class ConsumeXcomOperator(PushXcomOperator):

        def execute(self, context):
            if False:
                for i in range(10):
                    print('nop')
            assert set(self.arg1) == {1, 2, 3}
    with dag_maker('test_all_xcomargs_from_mapped_tasks_are_consumable'):
        op1 = PushXcomOperator.partial(task_id='op1').expand(arg1=[1, 2, 3])
        ConsumeXcomOperator(task_id='op2', arg1=op1.output)
    dr = dag_maker.create_dagrun()
    tis = dr.get_task_instances(session=session)
    for ti in tis:
        ti.run()

def test_task_mapping_with_task_group_context():
    if False:
        while True:
            i = 10
    with DAG('test-dag', start_date=DEFAULT_DATE) as dag:
        task1 = BaseOperator(task_id='op1')
        finish = MockOperator(task_id='finish')
        with TaskGroup('test-group') as group:
            literal = ['a', 'b', 'c']
            mapped = MockOperator.partial(task_id='task_2').expand(arg2=literal)
            task1 >> group >> finish
    assert task1.downstream_list == [mapped]
    assert mapped.upstream_list == [task1]
    assert mapped in dag.tasks
    assert mapped.task_group == group
    assert finish.upstream_list == [mapped]
    assert mapped.downstream_list == [finish]

def test_task_mapping_with_explicit_task_group():
    if False:
        for i in range(10):
            print('nop')
    with DAG('test-dag', start_date=DEFAULT_DATE) as dag:
        task1 = BaseOperator(task_id='op1')
        finish = MockOperator(task_id='finish')
        group = TaskGroup('test-group')
        literal = ['a', 'b', 'c']
        mapped = MockOperator.partial(task_id='task_2', task_group=group).expand(arg2=literal)
        task1 >> group >> finish
    assert task1.downstream_list == [mapped]
    assert mapped.upstream_list == [task1]
    assert mapped in dag.tasks
    assert mapped.task_group == group
    assert finish.upstream_list == [mapped]
    assert mapped.downstream_list == [finish]

class TestMappedSetupTeardown:

    @staticmethod
    def get_states(dr):
        if False:
            return 10
        ti_dict = defaultdict(dict)
        for ti in dr.get_task_instances():
            if ti.map_index == -1:
                ti_dict[ti.task_id] = ti.state
            else:
                ti_dict[ti.task_id][ti.map_index] = ti.state
        return dict(ti_dict)

    def classic_operator(self, task_id, ret=None, partial=False, fail=False):
        if False:
            return 10

        def success_callable(ret=None):
            if False:
                while True:
                    i = 10

            def inner(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                print(args)
                print(kwargs)
                if ret:
                    return ret
            return inner

        def failure_callable():
            if False:
                return 10

            def inner(*args, **kwargs):
                if False:
                    return 10
                print(args)
                print(kwargs)
                raise ValueError('fail')
            return inner
        kwargs = dict(task_id=task_id)
        if not fail:
            kwargs.update(python_callable=success_callable(ret=ret))
        else:
            kwargs.update(python_callable=failure_callable())
        if partial:
            return PythonOperator.partial(**kwargs)
        else:
            return PythonOperator(**kwargs)

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_one_to_many_work_failed(self, type_, dag_maker):
        if False:
            print('Hello World!')
        '\n        Work task failed.  Setup maps to teardown.  Should have 3 teardowns all successful even\n        though the work task has failed.\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @setup
                def my_setup():
                    if False:
                        while True:
                            i = 10
                    print('setting up multiple things')
                    return [1, 2, 3]

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'doing work with multiple things: {val}')
                    raise ValueError('fail!')

                @teardown
                def my_teardown(val):
                    if False:
                        print('Hello World!')
                    print(f'teardown: {val}')
                s = my_setup()
                t = my_teardown.expand(val=s)
                with t:
                    my_work(s)
        else:

            @task
            def my_work(val):
                if False:
                    return 10
                print(f'work: {val}')
                raise ValueError('i fail')
            with dag_maker() as dag:
                my_setup = self.classic_operator('my_setup', [[1], [2], [3]])
                my_teardown = self.classic_operator('my_teardown', partial=True)
                t = my_teardown.expand(op_args=my_setup.output)
                with t.as_teardown(setups=my_setup):
                    my_work(my_setup.output)
            return dag
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': 'success', 'my_work': 'failed', 'my_teardown': {0: 'success', 1: 'success', 2: 'success'}}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_many_one_explicit_odd_setup_mapped_setups_fail(self, type_, dag_maker):
        if False:
            print('Hello World!')
        '\n        one unmapped setup goes to two different teardowns\n        one mapped setup goes to same teardown\n        mapped setups fail\n        teardowns should still run\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def other_setup():
                    if False:
                        return 10
                    print('other setup')
                    return 'other setup'

                @task
                def other_work():
                    if False:
                        return 10
                    print('other work')
                    return 'other work'

                @task
                def other_teardown():
                    if False:
                        print('Hello World!')
                    print('other teardown')
                    return 'other teardown'

                @task
                def my_setup(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'setup: {val}')
                    raise ValueError('fail')
                    return val

                @task
                def my_work(val):
                    if False:
                        return 10
                    print(f'work: {val}')

                @task
                def my_teardown(val):
                    if False:
                        while True:
                            i = 10
                    print(f'teardown: {val}')
                s = my_setup.expand(val=['data1.json', 'data2.json', 'data3.json'])
                o_setup = other_setup()
                o_teardown = other_teardown()
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                t = my_teardown(s).as_teardown(setups=s)
                with t:
                    my_work(s)
                o_setup >> t
        else:
            with dag_maker() as dag:

                @task
                def other_work():
                    if False:
                        return 10
                    print('other work')
                    return 'other work'

                @task
                def my_work(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'work: {val}')
                my_teardown = self.classic_operator('my_teardown')
                my_setup = self.classic_operator('my_setup', partial=True, fail=True)
                s = my_setup.expand(op_args=[['data1.json'], ['data2.json'], ['data3.json']])
                o_setup = self.classic_operator('other_setup')
                o_teardown = self.classic_operator('other_teardown')
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                t = my_teardown.as_teardown(setups=s)
                with t:
                    my_work(s.output)
                o_setup >> t
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': {0: 'failed', 1: 'failed', 2: 'failed'}, 'other_setup': 'success', 'other_teardown': 'success', 'other_work': 'success', 'my_teardown': 'success', 'my_work': 'upstream_failed'}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_many_one_explicit_odd_setup_all_setups_fail(self, type_, dag_maker):
        if False:
            return 10
        '\n        one unmapped setup goes to two different teardowns\n        one mapped setup goes to same teardown\n        all setups fail\n        teardowns should not run\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def other_setup():
                    if False:
                        i = 10
                        return i + 15
                    print('other setup')
                    raise ValueError('fail')
                    return 'other setup'

                @task
                def other_work():
                    if False:
                        for i in range(10):
                            print('nop')
                    print('other work')
                    return 'other work'

                @task
                def other_teardown():
                    if False:
                        return 10
                    print('other teardown')
                    return 'other teardown'

                @task
                def my_setup(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'setup: {val}')
                    raise ValueError('fail')
                    return val

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'work: {val}')

                @task
                def my_teardown(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'teardown: {val}')
                s = my_setup.expand(val=['data1.json', 'data2.json', 'data3.json'])
                o_setup = other_setup()
                o_teardown = other_teardown()
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                t = my_teardown(s).as_teardown(setups=s)
                with t:
                    my_work(s)
                o_setup >> t
        else:
            with dag_maker() as dag:

                @task
                def other_setup():
                    if False:
                        i = 10
                        return i + 15
                    print('other setup')
                    raise ValueError('fail')
                    return 'other setup'

                @task
                def other_work():
                    if False:
                        print('Hello World!')
                    print('other work')
                    return 'other work'

                @task
                def other_teardown():
                    if False:
                        return 10
                    print('other teardown')
                    return 'other teardown'

                @task
                def my_work(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'work: {val}')
                my_setup = self.classic_operator('my_setup', partial=True, fail=True)
                s = my_setup.expand(op_args=[['data1.json'], ['data2.json'], ['data3.json']])
                o_setup = other_setup()
                o_teardown = other_teardown()
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                my_teardown = self.classic_operator('my_teardown')
                t = my_teardown.as_teardown(setups=s)
                with t:
                    my_work(s.output)
                o_setup >> t
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_teardown': 'upstream_failed', 'other_setup': 'failed', 'other_work': 'upstream_failed', 'other_teardown': 'upstream_failed', 'my_setup': {0: 'failed', 1: 'failed', 2: 'failed'}, 'my_work': 'upstream_failed'}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_many_one_explicit_odd_setup_one_mapped_fails(self, type_, dag_maker):
        if False:
            print('Hello World!')
        '\n        one unmapped setup goes to two different teardowns\n        one mapped setup goes to same teardown\n        one of the mapped setup instances fails\n        teardowns should all run\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def other_setup():
                    if False:
                        for i in range(10):
                            print('nop')
                    print('other setup')
                    return 'other setup'

                @task
                def other_work():
                    if False:
                        print('Hello World!')
                    print('other work')
                    return 'other work'

                @task
                def other_teardown():
                    if False:
                        while True:
                            i = 10
                    print('other teardown')
                    return 'other teardown'

                @task
                def my_setup(val):
                    if False:
                        return 10
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')
                    return val

                @task
                def my_work(val):
                    if False:
                        while True:
                            i = 10
                    print(f'work: {val}')

                @task
                def my_teardown(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'teardown: {val}')
                s = my_setup.expand(val=['data1.json', 'data2.json', 'data3.json'])
                o_setup = other_setup()
                o_teardown = other_teardown()
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                t = my_teardown(s).as_teardown(setups=s)
                with t:
                    my_work(s)
                o_setup >> t
        else:
            with dag_maker() as dag:

                @task
                def other_setup():
                    if False:
                        while True:
                            i = 10
                    print('other setup')
                    return 'other setup'

                @task
                def other_work():
                    if False:
                        while True:
                            i = 10
                    print('other work')
                    return 'other work'

                @task
                def other_teardown():
                    if False:
                        i = 10
                        return i + 15
                    print('other teardown')
                    return 'other teardown'

                def my_setup_callable(val):
                    if False:
                        return 10
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')
                    return val
                my_setup = PythonOperator.partial(task_id='my_setup', python_callable=my_setup_callable)

                @task
                def my_work(val):
                    if False:
                        print('Hello World!')
                    print(f'work: {val}')

                def my_teardown_callable(val):
                    if False:
                        print('Hello World!')
                    print(f'teardown: {val}')
                s = my_setup.expand(op_args=[['data1.json'], ['data2.json'], ['data3.json']])
                o_setup = other_setup()
                o_teardown = other_teardown()
                with o_teardown.as_teardown(setups=o_setup):
                    other_work()
                my_teardown = PythonOperator(task_id='my_teardown', op_args=[s.output], python_callable=my_teardown_callable)
                t = my_teardown.as_teardown(setups=s)
                with t:
                    my_work(s.output)
                o_setup >> t
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': {0: 'success', 1: 'failed', 2: 'skipped'}, 'other_setup': 'success', 'other_teardown': 'success', 'other_work': 'success', 'my_teardown': 'success', 'my_work': 'upstream_failed'}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_one_to_many_as_teardown(self, type_, dag_maker):
        if False:
            while True:
                i = 10
        '\n        1 setup mapping to 3 teardowns\n        1 work task\n        work fails\n        teardowns succeed\n        dagrun should be failure\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def my_setup():
                    if False:
                        i = 10
                        return i + 15
                    print('setting up multiple things')
                    return [1, 2, 3]

                @task
                def my_work(val):
                    if False:
                        print('Hello World!')
                    print(f'doing work with multiple things: {val}')
                    raise ValueError('this fails')
                    return val

                @task
                def my_teardown(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'teardown: {val}')
                s = my_setup()
                t = my_teardown.expand(val=s).as_teardown(setups=s)
                with t:
                    my_work(s)
        else:
            with dag_maker() as dag:

                @task
                def my_work(val):
                    if False:
                        while True:
                            i = 10
                    print(f'doing work with multiple things: {val}')
                    raise ValueError('this fails')
                    return val
                my_teardown = self.classic_operator(task_id='my_teardown', partial=True)
                s = self.classic_operator(task_id='my_setup', ret=[[1], [2], [3]])
                t = my_teardown.expand(op_args=s.output).as_teardown(setups=s)
                with t:
                    my_work(s)
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': 'success', 'my_teardown': {0: 'success', 1: 'success', 2: 'success'}, 'my_work': 'failed'}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_one_to_many_as_teardown_on_failure_fail_dagrun(self, type_, dag_maker):
        if False:
            return 10
        '\n        1 setup mapping to 3 teardowns\n        1 work task\n        work succeeds\n        all but one teardown succeed\n        on_failure_fail_dagrun=True\n        dagrun should be success\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def my_setup():
                    if False:
                        print('Hello World!')
                    print('setting up multiple things')
                    return [1, 2, 3]

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'doing work with multiple things: {val}')
                    return val

                @task
                def my_teardown(val):
                    if False:
                        print('Hello World!')
                    print(f'teardown: {val}')
                    if val == 2:
                        raise ValueError('failure')
                s = my_setup()
                t = my_teardown.expand(val=s).as_teardown(setups=s, on_failure_fail_dagrun=True)
                with t:
                    my_work(s)
        else:
            with dag_maker() as dag:

                @task
                def my_work(val):
                    if False:
                        while True:
                            i = 10
                    print(f'doing work with multiple things: {val}')
                    return val

                def my_teardown_callable(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'teardown: {val}')
                    if val == 2:
                        raise ValueError('failure')
                s = self.classic_operator(task_id='my_setup', ret=[[1], [2], [3]])
                my_teardown = PythonOperator.partial(task_id='my_teardown', python_callable=my_teardown_callable).expand(op_args=s.output)
                t = my_teardown.as_teardown(setups=s, on_failure_fail_dagrun=True)
                with t:
                    my_work(s.output)
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': 'success', 'my_teardown': {0: 'success', 1: 'failed', 2: 'success'}, 'my_work': 'success'}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_mapped_task_group_simple(self, type_, dag_maker, session):
        if False:
            i = 10
            return i + 15
        "\n        Mapped task group wherein there's a simple s >> w >> t pipeline.\n        When s is skipped, all should be skipped\n        When s is failed, all should be upstream failed\n        "
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @setup
                def my_setup(val):
                    if False:
                        i = 10
                        return i + 15
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')

                @task
                def my_work(val):
                    if False:
                        while True:
                            i = 10
                    print(f'work: {val}')

                @teardown
                def my_teardown(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'teardown: {val}')

                @task_group
                def file_transforms(filename):
                    if False:
                        while True:
                            i = 10
                    s = my_setup(filename)
                    t = my_teardown(filename)
                    s >> t
                    with t:
                        my_work(filename)
                file_transforms.expand(filename=['data1.json', 'data2.json', 'data3.json'])
        else:
            with dag_maker() as dag:

                def my_setup_callable(val):
                    if False:
                        i = 10
                        return i + 15
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')

                @task
                def my_work(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'work: {val}')

                def my_teardown_callable(val):
                    if False:
                        print('Hello World!')
                    print(f'teardown: {val}')

                @task_group
                def file_transforms(filename):
                    if False:
                        return 10
                    s = PythonOperator(task_id='my_setup', python_callable=my_setup_callable, op_args=filename)
                    t = PythonOperator(task_id='my_teardown', python_callable=my_teardown_callable, op_args=filename)
                    with t.as_teardown(setups=s):
                        my_work(filename)
                file_transforms.expand(filename=[['data1.json'], ['data2.json'], ['data3.json']])
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'file_transforms.my_setup': {0: 'success', 1: 'failed', 2: 'skipped'}, 'file_transforms.my_work': {0: 'success', 1: 'upstream_failed', 2: 'skipped'}, 'file_transforms.my_teardown': {0: 'success', 1: 'upstream_failed', 2: 'skipped'}}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_mapped_task_group_work_fail_or_skip(self, type_, dag_maker):
        if False:
            return 10
        "\n        Mapped task group wherein there's a simple s >> w >> t pipeline.\n        When w is skipped, teardown should still run\n        When w is failed, teardown should still run\n        "
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @setup
                def my_setup(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'setup: {val}')

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'work: {val}')

                @teardown
                def my_teardown(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'teardown: {val}')

                @task_group
                def file_transforms(filename):
                    if False:
                        for i in range(10):
                            print('nop')
                    s = my_setup(filename)
                    t = my_teardown(filename).as_teardown(setups=s)
                    with t:
                        my_work(filename)
                file_transforms.expand(filename=['data1.json', 'data2.json', 'data3.json'])
        else:
            with dag_maker() as dag:

                @task
                def my_work(vals):
                    if False:
                        for i in range(10):
                            print('nop')
                    val = vals[0]
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'work: {val}')

                @teardown
                def my_teardown(val):
                    if False:
                        i = 10
                        return i + 15
                    print(f'teardown: {val}')

                def null_callable(val):
                    if False:
                        while True:
                            i = 10
                    pass

                @task_group
                def file_transforms(filename):
                    if False:
                        for i in range(10):
                            print('nop')
                    s = PythonOperator(task_id='my_setup', python_callable=null_callable, op_args=filename)
                    t = PythonOperator(task_id='my_teardown', python_callable=null_callable, op_args=filename)
                    t = t.as_teardown(setups=s)
                    with t:
                        my_work(filename)
                file_transforms.expand(filename=[['data1.json'], ['data2.json'], ['data3.json']])
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'file_transforms.my_setup': {0: 'success', 1: 'success', 2: 'success'}, 'file_transforms.my_teardown': {0: 'success', 1: 'success', 2: 'success'}, 'file_transforms.my_work': {0: 'success', 1: 'failed', 2: 'skipped'}}
        assert states == expected

    @pytest.mark.parametrize('type_', ['taskflow', 'classic'])
    def test_teardown_many_one_explicit(self, type_, dag_maker):
        if False:
            return 10
        '-- passing\n        one mapped setup going to one unmapped work\n        3 diff states for setup: success / failed / skipped\n        teardown still runs, and receives the xcom from the single successful setup\n        '
        if type_ == 'taskflow':
            with dag_maker() as dag:

                @task
                def my_setup(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')
                    return val

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'work: {val}')

                @task
                def my_teardown(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'teardown: {val}')
                s = my_setup.expand(val=['data1.json', 'data2.json', 'data3.json'])
                with my_teardown(s).as_teardown(setups=s):
                    my_work(s)
        else:
            with dag_maker() as dag:

                def my_setup_callable(val):
                    if False:
                        while True:
                            i = 10
                    if val == 'data2.json':
                        raise ValueError('fail!')
                    elif val == 'data3.json':
                        raise AirflowSkipException('skip!')
                    print(f'setup: {val}')
                    return val

                @task
                def my_work(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    print(f'work: {val}')
                s = PythonOperator.partial(task_id='my_setup', python_callable=my_setup_callable)
                s = s.expand(op_args=[['data1.json'], ['data2.json'], ['data3.json']])
                t = self.classic_operator('my_teardown')
                with t.as_teardown(setups=s):
                    my_work(s.output)
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': {0: 'success', 1: 'failed', 2: 'skipped'}, 'my_teardown': 'success', 'my_work': 'upstream_failed'}
        assert states == expected

    def test_one_to_many_with_teardown_and_fail_stop(self, dag_maker):
        if False:
            return 10
        '\n        With fail_stop enabled, the teardown for an already-completed setup\n        should not be skipped.\n        '
        with dag_maker(fail_stop=True) as dag:

            @task
            def my_setup():
                if False:
                    for i in range(10):
                        print('nop')
                print('setting up multiple things')
                return [1, 2, 3]

            @task
            def my_work(val):
                if False:
                    return 10
                print(f'doing work with multiple things: {val}')
                raise ValueError('this fails')
                return val

            @task
            def my_teardown(val):
                if False:
                    while True:
                        i = 10
                print(f'teardown: {val}')
            s = my_setup()
            t = my_teardown.expand(val=s).as_teardown(setups=s)
            with t:
                my_work(s)
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'my_setup': 'success', 'my_teardown': {0: 'success', 1: 'success', 2: 'success'}, 'my_work': 'failed'}
        assert states == expected

    def test_one_to_many_with_teardown_and_fail_stop_more_tasks(self, dag_maker):
        if False:
            return 10
        '\n        when fail_stop enabled, teardowns should run according to their setups.\n        in this case, the second teardown skips because its setup skips.\n        '
        with dag_maker(fail_stop=True) as dag:
            for num in (1, 2):
                with TaskGroup(f'tg_{num}'):

                    @task
                    def my_setup():
                        if False:
                            for i in range(10):
                                print('nop')
                        print('setting up multiple things')
                        return [1, 2, 3]

                    @task
                    def my_work(val):
                        if False:
                            i = 10
                            return i + 15
                        print(f'doing work with multiple things: {val}')
                        raise ValueError('this fails')
                        return val

                    @task
                    def my_teardown(val):
                        if False:
                            print('Hello World!')
                        print(f'teardown: {val}')
                    s = my_setup()
                    t = my_teardown.expand(val=s).as_teardown(setups=s)
                    with t:
                        my_work(s)
        (tg1, tg2) = dag.task_group.children.values()
        tg1 >> tg2
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'tg_1.my_setup': 'success', 'tg_1.my_teardown': {0: 'success', 1: 'success', 2: 'success'}, 'tg_1.my_work': 'failed', 'tg_2.my_setup': 'skipped', 'tg_2.my_teardown': 'skipped', 'tg_2.my_work': 'skipped'}
        assert states == expected

    def test_one_to_many_with_teardown_and_fail_stop_more_tasks_mapped_setup(self, dag_maker):
        if False:
            print('Hello World!')
        '\n        when fail_stop enabled, teardowns should run according to their setups.\n        in this case, the second teardown skips because its setup skips.\n        '
        with dag_maker(fail_stop=True) as dag:
            for num in (1, 2):
                with TaskGroup(f'tg_{num}'):

                    @task
                    def my_pre_setup():
                        if False:
                            print('Hello World!')
                        print('input to the setup')
                        return [1, 2, 3]

                    @task
                    def my_setup(val):
                        if False:
                            i = 10
                            return i + 15
                        print('setting up multiple things')
                        return val

                    @task
                    def my_work(val):
                        if False:
                            print('Hello World!')
                        print(f'doing work with multiple things: {val}')
                        raise ValueError('this fails')
                        return val

                    @task
                    def my_teardown(val):
                        if False:
                            for i in range(10):
                                print('nop')
                        print(f'teardown: {val}')
                    s = my_setup.expand(val=my_pre_setup())
                    t = my_teardown.expand(val=s).as_teardown(setups=s)
                    with t:
                        my_work(s)
        (tg1, tg2) = dag.task_group.children.values()
        tg1 >> tg2
        dr = dag.test()
        states = self.get_states(dr)
        expected = {'tg_1.my_pre_setup': 'success', 'tg_1.my_setup': {0: 'success', 1: 'success', 2: 'success'}, 'tg_1.my_teardown': {0: 'success', 1: 'success', 2: 'success'}, 'tg_1.my_work': 'failed', 'tg_2.my_pre_setup': 'skipped', 'tg_2.my_setup': 'skipped', 'tg_2.my_teardown': 'skipped', 'tg_2.my_work': 'skipped'}
        assert states == expected