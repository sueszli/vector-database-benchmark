from __future__ import annotations
from unittest import mock
import pytest
from airflow.jobs.job import Job
from airflow.jobs.local_task_job_runner import LocalTaskJobRunner
from airflow.models.baseoperator import BaseOperator
from airflow.task.task_runner.base_task_runner import BaseTaskRunner
pytestmark = pytest.mark.db_test

@pytest.mark.parametrize(['impersonation'], (('nobody',), (None,)))
@mock.patch('subprocess.check_call')
@mock.patch('airflow.task.task_runner.base_task_runner.tmp_configuration_copy')
def test_config_copy_mode(tmp_configuration_copy, subprocess_call, dag_maker, impersonation):
    if False:
        print('Hello World!')
    tmp_configuration_copy.return_value = '/tmp/some-string'
    with dag_maker('test'):
        BaseOperator(task_id='task_1', run_as_user=impersonation)
    dr = dag_maker.create_dagrun()
    ti = dr.task_instances[0]
    job = Job(dag_id=ti.dag_id)
    job_runner = LocalTaskJobRunner(job=job, task_instance=ti)
    runner = BaseTaskRunner(job_runner)
    del runner._cfg_path
    includes = bool(impersonation)
    tmp_configuration_copy.assert_called_with(chmod=384, include_env=includes, include_cmds=includes)
    if impersonation:
        subprocess_call.assert_called_with(['sudo', 'chown', impersonation, '/tmp/some-string'], close_fds=True)
    else:
        subprocess_call.not_assert_called()