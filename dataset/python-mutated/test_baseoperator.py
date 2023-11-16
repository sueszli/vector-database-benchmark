from __future__ import annotations
import copy
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple
from unittest import mock
import jinja2
import pytest
from airflow.decorators import task as task_decorator
from airflow.exceptions import AirflowException, FailStopDagInvalidTriggerRule, RemovedInAirflow3Warning
from airflow.lineage.entities import File
from airflow.models.baseoperator import BaseOperator, BaseOperatorMeta, chain, chain_linear, cross_downstream
from airflow.models.dag import DAG
from airflow.models.dagrun import DagRun
from airflow.models.taskinstance import TaskInstance
from airflow.utils.edgemodifier import Label
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.types import DagRunType
from airflow.utils.weight_rule import WeightRule
from tests.models import DEFAULT_DATE
from tests.test_utils.config import conf_vars
from tests.test_utils.mock_operators import DeprecatedOperator, MockOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class ClassWithCustomAttributes:
    """Class for testing purpose: allows to create objects with custom attributes in one single statement."""

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        for (key, value) in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{ClassWithCustomAttributes.__name__}({str(self.__dict__)})'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)
object1 = ClassWithCustomAttributes(attr='{{ foo }}_1', template_fields=['ref'])
object2 = ClassWithCustomAttributes(attr='{{ foo }}_2', ref=object1, template_fields=['ref'])
setattr(object1, 'ref', object2)

class DummyClass(metaclass=BaseOperatorMeta):

    def __init__(self, test_param, params=None, default_args=None):
        if False:
            print('Hello World!')
        self.test_param = test_param

    def set_xcomargs_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        ...

class DummySubClass(DummyClass):

    def __init__(self, test_sub_param, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.test_sub_param = test_sub_param

class MockNamedTuple(NamedTuple):
    var1: str
    var2: str

class TestBaseOperator:

    def test_expand(self):
        if False:
            for i in range(10):
                print('nop')
        dummy = DummyClass(test_param=True)
        assert dummy.test_param
        with pytest.raises(AirflowException, match="missing keyword argument 'test_param'"):
            DummySubClass(test_sub_param=True)

    def test_default_args(self):
        if False:
            while True:
                i = 10
        default_args = {'test_param': True}
        dummy_class = DummyClass(default_args=default_args)
        assert dummy_class.test_param
        default_args = {'test_param': True, 'test_sub_param': True}
        dummy_subclass = DummySubClass(default_args=default_args)
        assert dummy_class.test_param
        assert dummy_subclass.test_sub_param
        default_args = {'test_param': True}
        dummy_subclass = DummySubClass(default_args=default_args, test_sub_param=True)
        assert dummy_class.test_param
        assert dummy_subclass.test_sub_param
        with pytest.raises(AirflowException, match="missing keyword argument 'test_sub_param'"):
            DummySubClass(default_args=default_args)

    def test_execution_timeout_type(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match="execution_timeout must be timedelta object but passed as type: <class 'str'>"):
            BaseOperator(task_id='test', execution_timeout='1')
        with pytest.raises(ValueError, match="execution_timeout must be timedelta object but passed as type: <class 'int'>"):
            BaseOperator(task_id='test', execution_timeout=1)

    def test_incorrect_default_args(self):
        if False:
            for i in range(10):
                print('nop')
        default_args = {'test_param': True, 'extra_param': True}
        dummy_class = DummyClass(default_args=default_args)
        assert dummy_class.test_param
        default_args = {'random_params': True}
        with pytest.raises(AirflowException, match="missing keyword argument 'test_param'"):
            DummyClass(default_args=default_args)

    def test_incorrect_priority_weight(self):
        if False:
            while True:
                i = 10
        error_msg = "`priority_weight` for task 'test_op' only accepts integers, received '<class 'str'>'."
        with pytest.raises(AirflowException, match=error_msg):
            BaseOperator(task_id='test_op', priority_weight='2')

    def test_illegal_args(self):
        if False:
            return 10
        '\n        Tests that Operators reject illegal arguments\n        '
        msg = 'Invalid arguments were passed to BaseOperator \\(task_id: test_illegal_args\\)'
        with conf_vars({('operators', 'allow_illegal_arguments'): 'True'}):
            with pytest.warns(RemovedInAirflow3Warning, match=msg):
                BaseOperator(task_id='test_illegal_args', illegal_argument_1234='hello?')

    def test_illegal_args_forbidden(self):
        if False:
            print('Hello World!')
        '\n        Tests that operators raise exceptions on illegal arguments when\n        illegal arguments are not allowed.\n        '
        msg = 'Invalid arguments were passed to BaseOperator \\(task_id: test_illegal_args\\)'
        with pytest.raises(AirflowException, match=msg):
            BaseOperator(task_id='test_illegal_args', illegal_argument_1234='hello?')

    def test_trigger_rule_validation(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow.models.abstractoperator import DEFAULT_TRIGGER_RULE
        fail_stop_dag = DAG(dag_id='test_dag_trigger_rule_validation', start_date=DEFAULT_DATE, fail_stop=True)
        non_fail_stop_dag = DAG(dag_id='test_dag_trigger_rule_validation', start_date=DEFAULT_DATE, fail_stop=False)
        try:
            BaseOperator(task_id='test_valid_trigger_rule', dag=fail_stop_dag, trigger_rule=DEFAULT_TRIGGER_RULE)
        except FailStopDagInvalidTriggerRule as exception:
            assert False, f'BaseOperator raises exception with fail-stop dag & default trigger rule: {exception}'
        try:
            BaseOperator(task_id='test_valid_trigger_rule', dag=non_fail_stop_dag, trigger_rule=TriggerRule.DUMMY)
        except FailStopDagInvalidTriggerRule as exception:
            assert False, f'BaseOperator raises exception with non fail-stop dag & non-default trigger rule: {exception}'
        with pytest.raises(FailStopDagInvalidTriggerRule):
            BaseOperator(task_id='test_invalid_trigger_rule', dag=fail_stop_dag, trigger_rule=TriggerRule.DUMMY)

    @pytest.mark.db_test
    @pytest.mark.parametrize(('content', 'context', 'expected_output'), [('{{ foo }}', {'foo': 'bar'}, 'bar'), (['{{ foo }}_1', '{{ foo }}_2'], {'foo': 'bar'}, ['bar_1', 'bar_2']), (('{{ foo }}_1', '{{ foo }}_2'), {'foo': 'bar'}, ('bar_1', 'bar_2')), ({'key1': '{{ foo }}_1', 'key2': '{{ foo }}_2'}, {'foo': 'bar'}, {'key1': 'bar_1', 'key2': 'bar_2'}), ({'key_{{ foo }}_1': 1, 'key_2': '{{ foo }}_2'}, {'foo': 'bar'}, {'key_{{ foo }}_1': 1, 'key_2': 'bar_2'}), (date(2018, 12, 6), {'foo': 'bar'}, date(2018, 12, 6)), (datetime(2018, 12, 6, 10, 55), {'foo': 'bar'}, datetime(2018, 12, 6, 10, 55)), (MockNamedTuple('{{ foo }}_1', '{{ foo }}_2'), {'foo': 'bar'}, MockNamedTuple('bar_1', 'bar_2')), ({'{{ foo }}_1', '{{ foo }}_2'}, {'foo': 'bar'}, {'bar_1', 'bar_2'}), (None, {}, None), ([], {}, []), ({}, {}, {}), (ClassWithCustomAttributes(att1='{{ foo }}_1', att2='{{ foo }}_2', template_fields=['att1']), {'foo': 'bar'}, ClassWithCustomAttributes(att1='bar_1', att2='{{ foo }}_2', template_fields=['att1'])), (ClassWithCustomAttributes(nested1=ClassWithCustomAttributes(att1='{{ foo }}_1', att2='{{ foo }}_2', template_fields=['att1']), nested2=ClassWithCustomAttributes(att3='{{ foo }}_3', att4='{{ foo }}_4', template_fields=['att3']), template_fields=['nested1']), {'foo': 'bar'}, ClassWithCustomAttributes(nested1=ClassWithCustomAttributes(att1='bar_1', att2='{{ foo }}_2', template_fields=['att1']), nested2=ClassWithCustomAttributes(att3='{{ foo }}_3', att4='{{ foo }}_4', template_fields=['att3']), template_fields=['nested1'])), (ClassWithCustomAttributes(att1=None, template_fields=['att1']), {}, ClassWithCustomAttributes(att1=None, template_fields=['att1'])), (object1, {'foo': 'bar'}, object1), ('{{ foo }}\n\n', {'foo': 'bar'}, 'bar\n')])
    def test_render_template(self, content, context, expected_output):
        if False:
            while True:
                i = 10
        'Test render_template given various input types.'
        task = BaseOperator(task_id='op1')
        result = task.render_template(content, context)
        assert result == expected_output

    @pytest.mark.parametrize(('content', 'context', 'expected_output'), [('{{ foo }}', {'foo': 'bar'}, 'bar'), ('{{ foo }}', {'foo': ['bar1', 'bar2']}, ['bar1', 'bar2']), (['{{ foo }}', '{{ foo | length}}'], {'foo': ['bar1', 'bar2']}, [['bar1', 'bar2'], 2]), (('{{ foo }}_1', '{{ foo }}_2'), {'foo': 'bar'}, ('bar_1', 'bar_2')), ('{{ ds }}', {'ds': date(2018, 12, 6)}, date(2018, 12, 6)), (datetime(2018, 12, 6, 10, 55), {'foo': 'bar'}, datetime(2018, 12, 6, 10, 55)), ('{{ ds }}', {'ds': datetime(2018, 12, 6, 10, 55)}, datetime(2018, 12, 6, 10, 55)), (MockNamedTuple('{{ foo }}_1', '{{ foo }}_2'), {'foo': 'bar'}, MockNamedTuple('bar_1', 'bar_2')), (('{{ foo }}', '{{ foo.isoformat() }}'), {'foo': datetime(2018, 12, 6, 10, 55)}, (datetime(2018, 12, 6, 10, 55), '2018-12-06T10:55:00')), (None, {}, None), ([], {}, []), ({}, {}, {})])
    def test_render_template_with_native_envs(self, content, context, expected_output):
        if False:
            return 10
        'Test render_template given various input types with Native Python types'
        with DAG('test-dag', start_date=DEFAULT_DATE, render_template_as_native_obj=True):
            task = BaseOperator(task_id='op1')
        result = task.render_template(content, context)
        assert result == expected_output

    def test_mapped_dag_slas_disabled_classic(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AirflowException, match='SLAs are unsupported with mapped tasks'):
            with DAG('test-dag', start_date=DEFAULT_DATE, default_args=dict(sla=timedelta(minutes=30))) as dag:

                @dag.task
                def get_values():
                    if False:
                        i = 10
                        return i + 15
                    return [0, 1, 2]
                task1 = get_values()

                class MyOp(BaseOperator):

                    def __init__(self, x, **kwargs):
                        if False:
                            while True:
                                i = 10
                        self.x = x
                        super().__init__(**kwargs)

                    def execute(self, context):
                        if False:
                            print('Hello World!')
                        print(self.x)
                MyOp.partial(task_id='hi').expand(x=task1)

    def test_mapped_dag_slas_disabled_taskflow(self):
        if False:
            while True:
                i = 10
        with pytest.raises(AirflowException, match='SLAs are unsupported with mapped tasks'):
            with DAG('test-dag', start_date=DEFAULT_DATE, default_args=dict(sla=timedelta(minutes=30))) as dag:

                @dag.task
                def get_values():
                    if False:
                        for i in range(10):
                            print('nop')
                    return [0, 1, 2]
                task1 = get_values()

                @dag.task
                def print_val(x):
                    if False:
                        while True:
                            i = 10
                    print(x)
                print_val.expand(x=task1)

    @pytest.mark.db_test
    def test_render_template_fields(self):
        if False:
            while True:
                i = 10
        'Verify if operator attributes are correctly templated.'
        task = MockOperator(task_id='op1', arg1='{{ foo }}', arg2='{{ bar }}')
        assert task.arg1 == '{{ foo }}'
        assert task.arg2 == '{{ bar }}'
        task.render_template_fields(context={'foo': 'footemplated', 'bar': 'bartemplated'})
        assert task.arg1 == 'footemplated'
        assert task.arg2 == 'bartemplated'

    @pytest.mark.parametrize(('content',), [(object(),), (uuid.uuid4(),)])
    def test_render_template_fields_no_change(self, content):
        if False:
            while True:
                i = 10
        'Tests if non-templatable types remain unchanged.'
        task = BaseOperator(task_id='op1')
        result = task.render_template(content, {'foo': 'bar'})
        assert content is result

    @pytest.mark.db_test
    def test_nested_template_fields_declared_must_exist(self):
        if False:
            while True:
                i = 10
        'Test render_template when a nested template field is missing.'
        task = BaseOperator(task_id='op1')
        error_message = "'missing_field' is configured as a template field but ClassWithCustomAttributes does not have this attribute."
        with pytest.raises(AttributeError, match=error_message):
            task.render_template(ClassWithCustomAttributes(template_fields=['missing_field'], task_type='ClassWithCustomAttributes'), {})

    def test_string_template_field_attr_is_converted_to_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify template_fields attribute is converted to a list if declared as a string.'

        class StringTemplateFieldsOperator(BaseOperator):
            template_fields = 'a_string'
        warning_message = 'The `template_fields` value for StringTemplateFieldsOperator is a string but should be a list or tuple of string. Wrapping it in a list for execution. Please update StringTemplateFieldsOperator accordingly.'
        with pytest.warns(UserWarning, match=warning_message) as warnings:
            task = StringTemplateFieldsOperator(task_id='op1')
            assert len(warnings) == 1
            assert isinstance(task.template_fields, list)

    def test_jinja_invalid_expression_is_just_propagated(self):
        if False:
            for i in range(10):
                print('nop')
        'Test render_template propagates Jinja invalid expression errors.'
        task = BaseOperator(task_id='op1')
        with pytest.raises(jinja2.exceptions.TemplateSyntaxError):
            task.render_template('{{ invalid expression }}', {})

    @pytest.mark.db_test
    @mock.patch('airflow.templates.SandboxedEnvironment', autospec=True)
    def test_jinja_env_creation(self, mock_jinja_env):
        if False:
            print('Hello World!')
        'Verify if a Jinja environment is created only once when templating.'
        task = MockOperator(task_id='op1', arg1='{{ foo }}', arg2='{{ bar }}')
        task.render_template_fields(context={'foo': 'whatever', 'bar': 'whatever'})
        assert mock_jinja_env.call_count == 1

    def test_default_resources(self):
        if False:
            for i in range(10):
                print('nop')
        task = BaseOperator(task_id='default-resources')
        assert task.resources is None

    def test_custom_resources(self):
        if False:
            return 10
        task = BaseOperator(task_id='custom-resources', resources={'cpus': 1, 'ram': 1024})
        assert task.resources.cpus.qty == 1
        assert task.resources.ram.qty == 1024

    def test_default_email_on_actions(self):
        if False:
            while True:
                i = 10
        test_task = BaseOperator(task_id='test_default_email_on_actions')
        assert test_task.email_on_retry is True
        assert test_task.email_on_failure is True

    def test_email_on_actions(self):
        if False:
            return 10
        test_task = BaseOperator(task_id='test_default_email_on_actions', email_on_retry=False, email_on_failure=True)
        assert test_task.email_on_retry is False
        assert test_task.email_on_failure is True

    def test_cross_downstream(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if all dependencies between tasks are all set correctly.'
        dag = DAG(dag_id='test_dag', start_date=datetime.now())
        start_tasks = [BaseOperator(task_id=f't{i}', dag=dag) for i in range(1, 4)]
        end_tasks = [BaseOperator(task_id=f't{i}', dag=dag) for i in range(4, 7)]
        cross_downstream(from_tasks=start_tasks, to_tasks=end_tasks)
        for start_task in start_tasks:
            assert set(start_task.get_direct_relatives(upstream=False)) == set(end_tasks)
        xstart_tasks = [task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(1, 4)]
        xend_tasks = [task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(4, 7)]
        cross_downstream(from_tasks=xstart_tasks, to_tasks=xend_tasks)
        for xstart_task in xstart_tasks:
            assert set(xstart_task.operator.get_direct_relatives(upstream=False)) == {xend_task.operator for xend_task in xend_tasks}

    def test_chain(self):
        if False:
            return 10
        dag = DAG(dag_id='test_chain', start_date=datetime.now())
        [label1, label2] = [Label(label=f'label{i}') for i in range(1, 3)]
        [op1, op2, op3, op4, op5, op6] = [BaseOperator(task_id=f't{i}', dag=dag) for i in range(1, 7)]
        chain(op1, [label1, label2], [op2, op3], [op4, op5], op6)
        assert {op2, op3} == set(op1.get_direct_relatives(upstream=False))
        assert [op4] == op2.get_direct_relatives(upstream=False)
        assert [op5] == op3.get_direct_relatives(upstream=False)
        assert {op4, op5} == set(op6.get_direct_relatives(upstream=True))
        assert {'label': 'label1'} == dag.get_edge_info(upstream_task_id=op1.task_id, downstream_task_id=op2.task_id)
        assert {'label': 'label2'} == dag.get_edge_info(upstream_task_id=op1.task_id, downstream_task_id=op3.task_id)
        [xlabel1, xlabel2] = [Label(label=f'xcomarg_label{i}') for i in range(1, 3)]
        [xop1, xop2, xop3, xop4, xop5, xop6] = [task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(1, 7)]
        chain(xop1, [xlabel1, xlabel2], [xop2, xop3], [xop4, xop5], xop6)
        assert {xop2.operator, xop3.operator} == set(xop1.operator.get_direct_relatives(upstream=False))
        assert [xop4.operator] == xop2.operator.get_direct_relatives(upstream=False)
        assert [xop5.operator] == xop3.operator.get_direct_relatives(upstream=False)
        assert {xop4.operator, xop5.operator} == set(xop6.operator.get_direct_relatives(upstream=True))
        assert {'label': 'xcomarg_label1'} == dag.get_edge_info(upstream_task_id=xop1.operator.task_id, downstream_task_id=xop2.operator.task_id)
        assert {'label': 'xcomarg_label2'} == dag.get_edge_info(upstream_task_id=xop1.operator.task_id, downstream_task_id=xop3.operator.task_id)
        [tg1, tg2] = [TaskGroup(group_id=f'tg{i}', dag=dag) for i in range(1, 3)]
        [op1, op2] = [BaseOperator(task_id=f'task{i}', dag=dag) for i in range(1, 3)]
        [tgop1, tgop2] = [BaseOperator(task_id=f'task_group_task{i}', task_group=tg1, dag=dag) for i in range(1, 3)]
        [tgop3, tgop4] = [BaseOperator(task_id=f'task_group_task{i}', task_group=tg2, dag=dag) for i in range(1, 3)]
        chain(op1, tg1, tg2, op2)
        assert {tgop1, tgop2} == set(op1.get_direct_relatives(upstream=False))
        assert {tgop3, tgop4} == set(tgop1.get_direct_relatives(upstream=False))
        assert {tgop3, tgop4} == set(tgop2.get_direct_relatives(upstream=False))
        assert [op2] == tgop3.get_direct_relatives(upstream=False)
        assert [op2] == tgop4.get_direct_relatives(upstream=False)

    def test_chain_linear(self):
        if False:
            i = 10
            return i + 15
        dag = DAG(dag_id='test_chain_linear', start_date=datetime.now())
        (t1, t2, t3, t4, t5, t6, t7) = (BaseOperator(task_id=f't{i}', dag=dag) for i in range(1, 8))
        chain_linear(t1, [t2, t3, t4], [t5, t6], t7)
        assert set(t1.get_direct_relatives(upstream=False)) == {t2, t3, t4}
        assert set(t2.get_direct_relatives(upstream=False)) == {t5, t6}
        assert set(t3.get_direct_relatives(upstream=False)) == {t5, t6}
        assert set(t7.get_direct_relatives(upstream=True)) == {t5, t6}
        (t1, t2, t3, t4, t5, t6) = (task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(1, 7))
        chain_linear(t1, [t2, t3], [t4, t5], t6)
        assert set(t1.operator.get_direct_relatives(upstream=False)) == {t2.operator, t3.operator}
        assert set(t2.operator.get_direct_relatives(upstream=False)) == {t4.operator, t5.operator}
        assert set(t3.operator.get_direct_relatives(upstream=False)) == {t4.operator, t5.operator}
        assert set(t6.operator.get_direct_relatives(upstream=True)) == {t4.operator, t5.operator}
        (tg1, tg2) = (TaskGroup(group_id=f'tg{i}', dag=dag) for i in range(1, 3))
        (op1, op2) = (BaseOperator(task_id=f'task{i}', dag=dag) for i in range(1, 3))
        (tgop1, tgop2) = (BaseOperator(task_id=f'task_group_task{i}', task_group=tg1, dag=dag) for i in range(1, 3))
        (tgop3, tgop4) = (BaseOperator(task_id=f'task_group_task{i}', task_group=tg2, dag=dag) for i in range(1, 3))
        chain_linear(op1, tg1, tg2, op2)
        assert set(op1.get_direct_relatives(upstream=False)) == {tgop1, tgop2}
        assert set(tgop1.get_direct_relatives(upstream=False)) == {tgop3, tgop4}
        assert set(tgop2.get_direct_relatives(upstream=False)) == {tgop3, tgop4}
        assert set(tgop3.get_direct_relatives(upstream=False)) == {op2}
        assert set(tgop4.get_direct_relatives(upstream=False)) == {op2}
        (t1, t2) = (BaseOperator(task_id=f't-{i}', dag=dag) for i in range(1, 3))
        with pytest.raises(ValueError, match='Labels are not supported'):
            chain_linear(t1, Label('hi'), t2)
        with pytest.raises(ValueError, match='nothing to do'):
            chain_linear()
        with pytest.raises(ValueError, match='Did you forget to expand'):
            chain_linear(t1)

    def test_chain_not_support_type(self):
        if False:
            while True:
                i = 10
        dag = DAG(dag_id='test_chain', start_date=datetime.now())
        [op1, op2] = [BaseOperator(task_id=f't{i}', dag=dag) for i in range(1, 3)]
        with pytest.raises(TypeError):
            chain([op1, op2], 1)
        [xop1, xop2] = [task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(1, 3)]
        with pytest.raises(TypeError):
            chain([xop1, xop2], 1)
        with pytest.raises(TypeError):
            chain([Label('labe1'), Label('label2')], 1)
        [tg1, tg2] = [TaskGroup(group_id=f'tg{i}', dag=dag) for i in range(1, 3)]
        with pytest.raises(TypeError):
            chain([tg1, tg2], 1)

    def test_chain_different_length_iterable(self):
        if False:
            i = 10
            return i + 15
        dag = DAG(dag_id='test_chain', start_date=datetime.now())
        [label1, label2] = [Label(label=f'label{i}') for i in range(1, 3)]
        [op1, op2, op3, op4, op5] = [BaseOperator(task_id=f't{i}', dag=dag) for i in range(1, 6)]
        with pytest.raises(AirflowException):
            chain([op1, op2], [op3, op4, op5])
        with pytest.raises(AirflowException):
            chain([op1, op2, op3], [label1, label2])
        [label3, label4] = [Label(label=f'xcomarg_label{i}') for i in range(1, 3)]
        [xop1, xop2, xop3, xop4, xop5] = [task_decorator(task_id=f'xcomarg_task{i}', python_callable=lambda : None, dag=dag)() for i in range(1, 6)]
        with pytest.raises(AirflowException):
            chain([xop1, xop2], [xop3, xop4, xop5])
        with pytest.raises(AirflowException):
            chain([xop1, xop2, xop3], [label1, label2])
        [tg1, tg2, tg3, tg4, tg5] = [TaskGroup(group_id=f'tg{i}', dag=dag) for i in range(1, 6)]
        with pytest.raises(AirflowException):
            chain([tg1, tg2], [tg3, tg4, tg5])

    def test_lineage_composition(self):
        if False:
            print('Hello World!')
        '\n        Test composition with lineage\n        '
        inlet = File(url='in')
        outlet = File(url='out')
        dag = DAG('test-dag', start_date=DEFAULT_DATE)
        task1 = BaseOperator(task_id='op1', dag=dag)
        task2 = BaseOperator(task_id='op2', dag=dag)
        task1.supports_lineage = True
        inlet > task1 | (task2 > outlet)
        assert task1.get_inlet_defs() == [inlet]
        assert task2.get_inlet_defs() == [task1.task_id]
        assert task2.get_outlet_defs() == [outlet]
        fail = ClassWithCustomAttributes()
        with pytest.raises(TypeError):
            fail > task1
        with pytest.raises(TypeError):
            task1 > fail
        with pytest.raises(TypeError):
            fail | task1
        with pytest.raises(TypeError):
            task1 | fail
        task3 = BaseOperator(task_id='op3', dag=dag)
        extra = File(url='extra')
        [inlet, extra] > task3
        assert task3.get_inlet_defs() == [inlet, extra]
        task1.supports_lineage = False
        with pytest.raises(ValueError):
            task1 | task3
        assert task2.supports_lineage is False
        task2 | task3
        assert len(task3.get_inlet_defs()) == 3
        task4 = BaseOperator(task_id='op4', dag=dag)
        task4 > [inlet, outlet, extra]
        assert task4.get_outlet_defs() == [inlet, outlet, extra]

    def test_warnings_are_properly_propagated(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(DeprecationWarning) as warnings:
            DeprecatedOperator(task_id='test')
            assert len(warnings) == 1
            warning = warnings[0]
            assert warning.filename == __file__

    def test_pre_execute_hook(self):
        if False:
            i = 10
            return i + 15
        hook = mock.MagicMock()
        op = BaseOperator(task_id='test_task', pre_execute=hook)
        op_copy = op.prepare_for_execution()
        op_copy.pre_execute({})
        assert hook.called

    def test_post_execute_hook(self):
        if False:
            i = 10
            return i + 15
        hook = mock.MagicMock()
        op = BaseOperator(task_id='test_task', post_execute=hook)
        op_copy = op.prepare_for_execution()
        op_copy.post_execute({})
        assert hook.called

    def test_task_naive_datetime(self):
        if False:
            print('Hello World!')
        naive_datetime = DEFAULT_DATE.replace(tzinfo=None)
        op_no_dag = BaseOperator(task_id='test_task_naive_datetime', start_date=naive_datetime, end_date=naive_datetime)
        assert op_no_dag.start_date.tzinfo
        assert op_no_dag.end_date.tzinfo

    def test_setattr_performs_no_custom_action_at_execute_time(self):
        if False:
            i = 10
            return i + 15
        op = MockOperator(task_id='test_task')
        op_copy = op.prepare_for_execution()
        with mock.patch('airflow.models.baseoperator.BaseOperator.set_xcomargs_dependencies') as method_mock:
            op_copy.execute({})
        assert method_mock.call_count == 0

    def test_upstream_is_set_when_template_field_is_xcomarg(self):
        if False:
            while True:
                i = 10
        with DAG('xcomargs_test', default_args={'start_date': datetime.today()}):
            op1 = BaseOperator(task_id='op1')
            op2 = MockOperator(task_id='op2', arg1=op1.output)
        assert op1 in op2.upstream_list
        assert op2 in op1.downstream_list

    def test_set_xcomargs_dependencies_works_recursively(self):
        if False:
            while True:
                i = 10
        with DAG('xcomargs_test', default_args={'start_date': datetime.today()}):
            op1 = BaseOperator(task_id='op1')
            op2 = BaseOperator(task_id='op2')
            op3 = MockOperator(task_id='op3', arg1=[op1.output, op2.output])
            op4 = MockOperator(task_id='op4', arg1={'op1': op1.output, 'op2': op2.output})
        assert op1 in op3.upstream_list
        assert op2 in op3.upstream_list
        assert op1 in op4.upstream_list
        assert op2 in op4.upstream_list

    def test_set_xcomargs_dependencies_works_when_set_after_init(self):
        if False:
            i = 10
            return i + 15
        with DAG(dag_id='xcomargs_test', default_args={'start_date': datetime.today()}):
            op1 = BaseOperator(task_id='op1')
            op2 = MockOperator(task_id='op2')
            op2.arg1 = op1.output
        assert op1 in op2.upstream_list

    def test_set_xcomargs_dependencies_error_when_outside_dag(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AirflowException):
            op1 = BaseOperator(task_id='op1')
            MockOperator(task_id='op2', arg1=op1.output)

    def test_invalid_trigger_rule(self):
        if False:
            return 10
        with pytest.raises(AirflowException, match=f"The trigger_rule must be one of {TriggerRule.all_triggers()},'.op1'; received 'some_rule'."):
            BaseOperator(task_id='op1', trigger_rule='some_rule')

    @pytest.mark.parametrize('rule', ['dummy', TriggerRule.DUMMY])
    def test_replace_dummy_trigger_rule(self, rule):
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(DeprecationWarning, match='dummy Trigger Rule is deprecated. Please use `TriggerRule.ALWAYS`.'):
            op1 = BaseOperator(task_id='op1', trigger_rule=rule)
            assert op1.trigger_rule == TriggerRule.ALWAYS

    def test_weight_rule_default(self):
        if False:
            i = 10
            return i + 15
        op = BaseOperator(task_id='test_task')
        assert WeightRule.DOWNSTREAM == op.weight_rule

    def test_weight_rule_override(self):
        if False:
            while True:
                i = 10
        op = BaseOperator(task_id='test_task', weight_rule='upstream')
        assert WeightRule.UPSTREAM == op.weight_rule

    @pytest.mark.usefixtures('reset_logging_config')
    def test_logging_propogated_by_default(self, caplog):
        if False:
            print('Hello World!')
        "Test that when set_context hasn't been called that log records are emitted"
        BaseOperator(task_id='test').log.warning('test')
        assert caplog.messages == ['test']

def test_init_subclass_args():
    if False:
        for i in range(10):
            print('nop')

    class InitSubclassOp(BaseOperator):
        _class_arg: Any

        def __init_subclass__(cls, class_arg=None, **kwargs) -> None:
            if False:
                while True:
                    i = 10
            cls._class_arg = class_arg
            super().__init_subclass__()

        def execute(self, context: Context):
            if False:
                return 10
            self.context_arg = context
    class_arg = 'foo'
    context = {'key': 'value'}

    class ConcreteSubclassOp(InitSubclassOp, class_arg=class_arg):
        pass
    task = ConcreteSubclassOp(task_id='op1')
    task_copy = task.prepare_for_execution()
    task_copy.execute(context)
    assert task_copy._class_arg == class_arg
    assert task_copy.context_arg == context

@pytest.mark.db_test
def test_operator_retries_invalid(dag_maker):
    if False:
        while True:
            i = 10
    with pytest.raises(AirflowException) as ctx:
        with dag_maker():
            BaseOperator(task_id='test_illegal_args', retries='foo')
    assert str(ctx.value) == "'retries' type must be int, not str"

@pytest.mark.db_test
@pytest.mark.parametrize(('retries', 'expected'), [pytest.param(None, [], id='None'), pytest.param(5, [], id='5'), pytest.param('1', [('airflow.models.baseoperator.BaseOperator', logging.WARNING, "Implicitly converting 'retries' from '1' to int")], id='str')])
def test_operator_retries(caplog, dag_maker, retries, expected):
    if False:
        print('Hello World!')
    with caplog.at_level(logging.WARNING):
        with dag_maker():
            BaseOperator(task_id='test_illegal_args', retries=retries)
    assert caplog.record_tuples == expected

@pytest.mark.db_test
def test_default_retry_delay(dag_maker):
    if False:
        i = 10
        return i + 15
    with dag_maker(dag_id='test_default_retry_delay'):
        task1 = BaseOperator(task_id='test_no_explicit_retry_delay')
        assert task1.retry_delay == timedelta(seconds=300)

@pytest.mark.db_test
def test_dag_level_retry_delay(dag_maker):
    if False:
        i = 10
        return i + 15
    with dag_maker(dag_id='test_dag_level_retry_delay', default_args={'retry_delay': timedelta(seconds=100)}):
        task1 = BaseOperator(task_id='test_no_explicit_retry_delay')
        assert task1.retry_delay == timedelta(seconds=100)

@pytest.mark.db_test
def test_task_level_retry_delay(dag_maker):
    if False:
        return 10
    with dag_maker(dag_id='test_task_level_retry_delay', default_args={'retry_delay': timedelta(seconds=100)}):
        task1 = BaseOperator(task_id='test_no_explicit_retry_delay', retry_delay=timedelta(seconds=200))
        assert task1.retry_delay == timedelta(seconds=200)

def test_deepcopy():
    if False:
        while True:
            i = 10
    with DAG('dag0', start_date=DEFAULT_DATE) as dag:

        @dag.task
        def task0():
            if False:
                while True:
                    i = 10
            pass
        MockOperator(task_id='task1', arg1=task0())
    copy.deepcopy(dag)

@pytest.mark.db_test
@pytest.mark.parametrize(('task', 'context', 'expected_exception', 'expected_rendering', 'expected_log', 'not_expected_log'), [(MockOperator(task_id='op1', arg1='{{ foo }}'), dict(foo='footemplated'), None, dict(arg1='footemplated'), None, 'Exception rendering Jinja template'), (MockOperator(task_id='op1', arg1='{{ foo'), dict(), jinja2.TemplateSyntaxError, None, "Exception rendering Jinja template for task 'op1', field 'arg1'. Template: '{{ foo'", None), (MockOperator(task_id='op1', arg1='{{ foo + 1 }}'), dict(foo='footemplated'), TypeError, None, "Exception rendering Jinja template for task 'op1', field 'arg1'. Template: '{{ foo + 1 }}'", None)])
def test_render_template_fields_logging(caplog, monkeypatch, task, context, expected_exception, expected_rendering, expected_log, not_expected_log):
    if False:
        i = 10
        return i + 15
    'Verify if operator attributes are correctly templated.'

    def _do_render():
        if False:
            return 10
        task.render_template_fields(context=context)
    logger = logging.getLogger('airflow.task')
    monkeypatch.setattr(logger, 'propagate', True)
    if expected_exception:
        with pytest.raises(expected_exception):
            _do_render()
    else:
        _do_render()
        for (k, v) in expected_rendering.items():
            assert getattr(task, k) == v
    if expected_log:
        assert expected_log in caplog.text
    if not_expected_log:
        assert not_expected_log not in caplog.text

@pytest.mark.db_test
def test_find_mapped_dependants_in_another_group(dag_maker):
    if False:
        for i in range(10):
            print('nop')
    from airflow.utils.task_group import TaskGroup

    @task_decorator
    def gen(x):
        if False:
            for i in range(10):
                print('nop')
        return list(range(x))

    @task_decorator
    def add(x, y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    with dag_maker():
        with TaskGroup(group_id='g1'):
            gen_result = gen(3)
        with TaskGroup(group_id='g2'):
            add_result = add.partial(y=1).expand(x=gen_result)
    dependants = list(gen_result.operator.iter_mapped_dependants())
    assert dependants == [add_result.operator]

def get_states(dr):
    if False:
        i = 10
        return i + 15
    '\n    For a given dag run, get a dict of states.\n\n    Example::\n        {\n            "my_setup": "success",\n            "my_teardown": {0: "success", 1: "success", 2: "success"},\n            "my_work": "failed",\n        }\n    '
    ti_dict = defaultdict(dict)
    for ti in dr.get_task_instances():
        if ti.map_index == -1:
            ti_dict[ti.task_id] = ti.state
        else:
            ti_dict[ti.task_id][ti.map_index] = ti.state
    return dict(ti_dict)

@pytest.mark.db_test
def test_teardown_and_fail_stop(dag_maker):
    if False:
        return 10
    '\n    when fail_stop enabled, teardowns should run according to their setups.\n    in this case, the second teardown skips because its setup skips.\n    '
    with dag_maker(fail_stop=True) as dag:
        for num in (1, 2):
            with TaskGroup(f'tg_{num}'):

                @task_decorator
                def my_setup():
                    if False:
                        for i in range(10):
                            print('nop')
                    print('setting up multiple things')
                    return [1, 2, 3]

                @task_decorator
                def my_work(val):
                    if False:
                        while True:
                            i = 10
                    print(f'doing work with multiple things: {val}')
                    raise ValueError('this fails')
                    return val

                @task_decorator
                def my_teardown():
                    if False:
                        for i in range(10):
                            print('nop')
                    print('teardown')
                s = my_setup()
                t = my_teardown().as_teardown(setups=s)
                with t:
                    my_work(s)
    (tg1, tg2) = dag.task_group.children.values()
    tg1 >> tg2
    dr = dag.test()
    states = get_states(dr)
    expected = {'tg_1.my_setup': 'success', 'tg_1.my_teardown': 'success', 'tg_1.my_work': 'failed', 'tg_2.my_setup': 'skipped', 'tg_2.my_teardown': 'skipped', 'tg_2.my_work': 'skipped'}
    assert states == expected

@pytest.mark.db_test
def test_get_task_instances(session):
    if False:
        print('Hello World!')
    import pendulum
    first_execution_date = pendulum.datetime(2023, 1, 1)
    second_execution_date = pendulum.datetime(2023, 1, 2)
    third_execution_date = pendulum.datetime(2023, 1, 3)
    test_dag = DAG(dag_id='test_dag', start_date=first_execution_date)
    task = BaseOperator(task_id='test_task', dag=test_dag)
    common_dr_kwargs = {'dag_id': test_dag.dag_id, 'run_type': DagRunType.MANUAL}
    dr1 = DagRun(execution_date=first_execution_date, run_id='test_run_id_1', **common_dr_kwargs)
    ti_1 = TaskInstance(run_id=dr1.run_id, task=task, execution_date=first_execution_date)
    dr2 = DagRun(execution_date=second_execution_date, run_id='test_run_id_2', **common_dr_kwargs)
    ti_2 = TaskInstance(run_id=dr2.run_id, task=task, execution_date=second_execution_date)
    dr3 = DagRun(execution_date=third_execution_date, run_id='test_run_id_3', **common_dr_kwargs)
    ti_3 = TaskInstance(run_id=dr3.run_id, task=task, execution_date=third_execution_date)
    session.add_all([dr1, dr2, dr3, ti_1, ti_2, ti_3])
    session.commit()
    assert task.get_task_instances(session=session) == [ti_1, ti_2, ti_3]
    assert task.get_task_instances(session=session, start_date=second_execution_date) == [ti_2, ti_3]
    assert task.get_task_instances(session=session, end_date=second_execution_date) == [ti_1, ti_2]
    assert task.get_task_instances(session=session, start_date=second_execution_date, end_date=second_execution_date) == [ti_2]