from __future__ import annotations
from unittest import mock
from airflow.task.task_runner.cgroup_task_runner import CgroupTaskRunner

class TestCgroupTaskRunner:

    @mock.patch('airflow.task.task_runner.base_task_runner.BaseTaskRunner.__init__')
    @mock.patch('airflow.task.task_runner.base_task_runner.BaseTaskRunner.on_finish')
    def test_cgroup_task_runner_super_calls(self, mock_super_on_finish, mock_super_init):
        if False:
            while True:
                i = 10
        '\n        This test ensures that initiating CgroupTaskRunner object\n        calls init method of BaseTaskRunner,\n        and when task finishes, CgroupTaskRunner.on_finish() calls\n        super().on_finish() to delete the temp cfg file.\n        '
        Job = mock.Mock()
        Job.job_type = None
        Job.task_instance = mock.MagicMock()
        Job.task_instance.run_as_user = None
        Job.task_instance.command_as_list.return_value = ['sleep', '1000']
        runner = CgroupTaskRunner(Job)
        assert mock_super_init.called
        runner.on_finish()
        assert mock_super_on_finish.called