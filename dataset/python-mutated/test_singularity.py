from __future__ import annotations
from unittest import mock
import pytest
from spython.instance import Instance
from airflow.exceptions import AirflowException
from airflow.providers.singularity.operators.singularity import SingularityOperator

class TestSingularityOperator:

    @mock.patch('airflow.providers.singularity.operators.singularity.Client')
    def test_execute(self, client_mock):
        if False:
            return 10
        instance = mock.Mock(autospec=Instance, **{'start.return_value': 0, 'stop.return_value': 0})
        client_mock.instance.return_value = instance
        client_mock.execute.return_value = {'return_code': 0, 'message': 'message'}
        task = SingularityOperator(task_id='task-id', image='docker://busybox', command='echo hello')
        task.execute({})
        client_mock.instance.assert_called_once_with('docker://busybox', options=[], args=None, start=False)
        client_mock.execute.assert_called_once_with(mock.ANY, 'echo hello', return_result=True)
        (execute_args, _) = client_mock.execute.call_args
        assert execute_args[0] is instance
        instance.start.assert_called_once_with()
        instance.stop.assert_called_once_with()

    @pytest.mark.parametrize('command', [pytest.param('', id='empty'), pytest.param(None, id='none')])
    def test_command_is_required(self, command):
        if False:
            return 10
        task = SingularityOperator(task_id='task-id', image='docker://busybox', command=command)
        with pytest.raises(AirflowException, match='You must define a command.'):
            task.execute({})

    @mock.patch('airflow.providers.singularity.operators.singularity.Client')
    def test_image_should_be_pulled_when_not_exists(self, client_mock):
        if False:
            for i in range(10):
                print('nop')
        instance = mock.Mock(autospec=Instance, **{'start.return_value': 0, 'stop.return_value': 0})
        client_mock.pull.return_value = '/tmp/busybox_latest.sif'
        client_mock.instance.return_value = instance
        client_mock.execute.return_value = {'return_code': 0, 'message': 'message'}
        task = SingularityOperator(task_id='task-id', image='docker://busybox', command='echo hello', pull_folder='/tmp', force_pull=True)
        task.execute({})
        client_mock.instance.assert_called_once_with('/tmp/busybox_latest.sif', options=[], args=None, start=False)
        client_mock.pull.assert_called_once_with('docker://busybox', stream=True, pull_folder='/tmp')
        client_mock.execute.assert_called_once_with(mock.ANY, 'echo hello', return_result=True)

    @pytest.mark.parametrize('volumes, expected_options', [(None, []), ([], []), (['AAA'], ['--bind', 'AAA']), (['AAA', 'BBB'], ['--bind', 'AAA', '--bind', 'BBB']), (['AAA', 'BBB', 'CCC'], ['--bind', 'AAA', '--bind', 'BBB', '--bind', 'CCC'])])
    @mock.patch('airflow.providers.singularity.operators.singularity.Client')
    def test_bind_options(self, client_mock, volumes, expected_options):
        if False:
            return 10
        instance = mock.Mock(autospec=Instance, **{'start.return_value': 0, 'stop.return_value': 0})
        client_mock.pull.return_value = 'docker://busybox'
        client_mock.instance.return_value = instance
        client_mock.execute.return_value = {'return_code': 0, 'message': 'message'}
        task = SingularityOperator(task_id='task-id', image='docker://busybox', command='echo hello', force_pull=True, volumes=volumes)
        task.execute({})
        client_mock.instance.assert_called_once_with('docker://busybox', options=expected_options, args=None, start=False)

    @pytest.mark.parametrize('working_dir, expected_working_dir', [(None, []), ('', ['--workdir', '']), ('/work-dir/', ['--workdir', '/work-dir/'])])
    @mock.patch('airflow.providers.singularity.operators.singularity.Client')
    def test_working_dir(self, client_mock, working_dir, expected_working_dir):
        if False:
            return 10
        instance = mock.Mock(autospec=Instance, **{'start.return_value': 0, 'stop.return_value': 0})
        client_mock.pull.return_value = 'docker://busybox'
        client_mock.instance.return_value = instance
        client_mock.execute.return_value = {'return_code': 0, 'message': 'message'}
        task = SingularityOperator(task_id='task-id', image='docker://busybox', command='echo hello', force_pull=True, working_dir=working_dir)
        task.execute({})
        client_mock.instance.assert_called_once_with('docker://busybox', options=expected_working_dir, args=None, start=False)