from __future__ import annotations
from unittest import mock
import pytest
from airflow.jobs.local_task_job_runner import LocalTaskJobRunner
from airflow.task.task_runner import CORE_TASK_RUNNERS, get_task_runner
from airflow.utils.module_loading import import_string
pytestmark = pytest.mark.db_test
custom_task_runner = mock.MagicMock()

class TestGetTaskRunner:

    @pytest.mark.parametrize('import_path', CORE_TASK_RUNNERS.values())
    def test_should_have_valid_imports(self, import_path):
        if False:
            while True:
                i = 10
        assert import_string(import_path) is not None

    @mock.patch('airflow.task.task_runner.base_task_runner.subprocess')
    @mock.patch('airflow.task.task_runner._TASK_RUNNER_NAME', 'StandardTaskRunner')
    def test_should_support_core_task_runner(self, mock_subprocess):
        if False:
            for i in range(10):
                print('nop')
        ti = mock.MagicMock(map_index=-1, run_as_user=None)
        ti.get_template_context.return_value = {'ti': ti}
        ti.get_dagrun.return_value.get_log_template.return_value.filename = 'blah'
        Job = mock.MagicMock(task_instance=ti)
        Job.job_type = None
        job_runner = LocalTaskJobRunner(job=Job, task_instance=ti)
        task_runner = get_task_runner(job_runner)
        assert 'StandardTaskRunner' == task_runner.__class__.__name__

    @mock.patch('airflow.task.task_runner._TASK_RUNNER_NAME', 'tests.task.task_runner.test_task_runner.custom_task_runner')
    def test_should_support_custom_legacy_task_runner(self):
        if False:
            return 10
        mock.MagicMock(**{'task_instance.get_template_context.return_value': {'ti': mock.MagicMock()}})
        custom_task_runner.reset_mock()
        task_runner = get_task_runner(custom_task_runner)
        custom_task_runner.assert_called_once_with(custom_task_runner)
        assert custom_task_runner.return_value == task_runner