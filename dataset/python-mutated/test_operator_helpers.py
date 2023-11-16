from __future__ import annotations
from datetime import datetime
from unittest import mock
import pytest
from airflow.utils import operator_helpers

class TestOperatorHelpers:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.dag_id = 'dag_id'
        self.task_id = 'task_id'
        self.try_number = 1
        self.execution_date = '2017-05-21T00:00:00'
        self.dag_run_id = 'dag_run_id'
        self.owner = ['owner1', 'owner2']
        self.email = ['email1@test.com']
        self.context = {'dag_run': mock.MagicMock(name='dag_run', run_id=self.dag_run_id, execution_date=datetime.strptime(self.execution_date, '%Y-%m-%dT%H:%M:%S')), 'task_instance': mock.MagicMock(name='task_instance', task_id=self.task_id, dag_id=self.dag_id, try_number=self.try_number, execution_date=datetime.strptime(self.execution_date, '%Y-%m-%dT%H:%M:%S')), 'task': mock.MagicMock(name='task', owner=self.owner, email=self.email)}

    def test_context_to_airflow_vars_empty_context(self):
        if False:
            while True:
                i = 10
        assert operator_helpers.context_to_airflow_vars({}) == {}

    def test_context_to_airflow_vars_all_context(self):
        if False:
            return 10
        assert operator_helpers.context_to_airflow_vars(self.context) == {'airflow.ctx.dag_id': self.dag_id, 'airflow.ctx.execution_date': self.execution_date, 'airflow.ctx.task_id': self.task_id, 'airflow.ctx.dag_run_id': self.dag_run_id, 'airflow.ctx.try_number': str(self.try_number), 'airflow.ctx.dag_owner': 'owner1,owner2', 'airflow.ctx.dag_email': 'email1@test.com'}
        assert operator_helpers.context_to_airflow_vars(self.context, in_env_var_format=True) == {'AIRFLOW_CTX_DAG_ID': self.dag_id, 'AIRFLOW_CTX_EXECUTION_DATE': self.execution_date, 'AIRFLOW_CTX_TASK_ID': self.task_id, 'AIRFLOW_CTX_TRY_NUMBER': str(self.try_number), 'AIRFLOW_CTX_DAG_RUN_ID': self.dag_run_id, 'AIRFLOW_CTX_DAG_OWNER': 'owner1,owner2', 'AIRFLOW_CTX_DAG_EMAIL': 'email1@test.com'}

    def test_context_to_airflow_vars_with_default_context_vars(self):
        if False:
            return 10
        with mock.patch('airflow.settings.get_airflow_context_vars') as mock_method:
            airflow_cluster = 'cluster-a'
            mock_method.return_value = {'airflow_cluster': airflow_cluster}
            context_vars = operator_helpers.context_to_airflow_vars(self.context)
            assert context_vars['airflow.ctx.airflow_cluster'] == airflow_cluster
            context_vars = operator_helpers.context_to_airflow_vars(self.context, in_env_var_format=True)
            assert context_vars['AIRFLOW_CTX_AIRFLOW_CLUSTER'] == airflow_cluster
        with mock.patch('airflow.settings.get_airflow_context_vars') as mock_method:
            mock_method.return_value = {'airflow_cluster': [1, 2]}
            with pytest.raises(TypeError) as error:
                operator_helpers.context_to_airflow_vars(self.context)
            assert "value of key <airflow_cluster> must be string, not <class 'list'>" == str(error.value)
        with mock.patch('airflow.settings.get_airflow_context_vars') as mock_method:
            mock_method.return_value = {1: 'value'}
            with pytest.raises(TypeError) as error:
                operator_helpers.context_to_airflow_vars(self.context)
            assert 'key <1> must be string' == str(error.value)

def callable1(ds_nodash):
    if False:
        return 10
    return (ds_nodash,)

def callable2(ds_nodash, prev_ds_nodash):
    if False:
        while True:
            i = 10
    return (ds_nodash, prev_ds_nodash)

def callable3(ds_nodash, prev_ds_nodash, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    return (ds_nodash, prev_ds_nodash, args, kwargs)

def callable4(ds_nodash, prev_ds_nodash, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return (ds_nodash, prev_ds_nodash, kwargs)

def callable5(**kwargs):
    if False:
        i = 10
        return i + 15
    return (kwargs,)

def callable6(arg1, ds_nodash):
    if False:
        i = 10
        return i + 15
    return (arg1, ds_nodash)

def callable7(arg1, **kwargs):
    if False:
        i = 10
        return i + 15
    return (arg1, kwargs)

def callable8(arg1, *args, **kwargs):
    if False:
        return 10
    return (arg1, args, kwargs)

def callable9(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return (args, kwargs)

def callable10(arg1, *, ds_nodash='20200201'):
    if False:
        i = 10
        return i + 15
    return (arg1, ds_nodash)

def callable11(*, ds_nodash, **kwargs):
    if False:
        while True:
            i = 10
    return (ds_nodash, kwargs)
KWARGS = {'prev_ds_nodash': '20191231', 'ds_nodash': '20200101', 'tomorrow_ds_nodash': '20200102'}

@pytest.mark.parametrize('func,args,kwargs,expected', [(callable1, (), KWARGS, ('20200101',)), (callable2, (), KWARGS, ('20200101', '20191231')), (callable3, (), KWARGS, ('20200101', '20191231', (), {'tomorrow_ds_nodash': '20200102'})), (callable4, (), KWARGS, ('20200101', '20191231', {'tomorrow_ds_nodash': '20200102'})), (callable5, (), KWARGS, (KWARGS,)), (callable6, (1,), KWARGS, (1, '20200101')), (callable7, (1,), KWARGS, (1, KWARGS)), (callable8, (1, 2), KWARGS, (1, (2,), KWARGS)), (callable9, (1, 2), KWARGS, ((1, 2), KWARGS)), (callable10, (1,), KWARGS, (1, '20200101')), (callable11, (), KWARGS, ('20200101', {'prev_ds_nodash': '20191231', 'tomorrow_ds_nodash': '20200102'}))])
def test_make_kwargs_callable(func, args, kwargs, expected):
    if False:
        for i in range(10):
            print('nop')
    kwargs_callable = operator_helpers.make_kwargs_callable(func)
    ret = kwargs_callable(*args, **kwargs)
    assert ret == expected

def test_make_kwargs_callable_conflict():
    if False:
        while True:
            i = 10

    def func(ds_nodash):
        if False:
            return 10
        pytest.fail(f'Should not reach here: {ds_nodash}')
    kwargs_callable = operator_helpers.make_kwargs_callable(func)
    args = ['20200101']
    kwargs = {'ds_nodash': '20200101', 'tomorrow_ds_nodash': '20200102'}
    with pytest.raises(ValueError) as exc_info:
        kwargs_callable(*args, **kwargs)
    assert 'ds_nodash' in str(exc_info)