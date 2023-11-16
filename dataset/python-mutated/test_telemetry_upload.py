import logging
import os
import mock
import pytest
import responses
from click.testing import CliRunner
from dagster._cli.job import job_execute_command
from dagster._core.telemetry import cleanup_telemetry_logger
from dagster._core.telemetry_upload import get_dagster_telemetry_url, upload_logs
from dagster._core.test_utils import environ, instance_for_test
from dagster._utils import pushd, script_relative_path

def path_to_file(path):
    if False:
        return 10
    return script_relative_path(os.path.join('./', path))

@pytest.mark.parametrize('env', [{'BUILDKITE': None, 'TF_BUILD': None, 'DAGSTER_DISABLE_TELEMETRY': None}])
@responses.activate
def test_dagster_telemetry_upload(env):
    if False:
        return 10
    logger = logging.getLogger('dagster_telemetry_logger')
    for handler in logger.handlers:
        logger.removeHandler(handler)
    responses.add(responses.POST, get_dagster_telemetry_url())
    with instance_for_test(overrides={'telemetry': {'enabled': True}}):
        with environ(env):
            runner = CliRunner()
            with pushd(path_to_file('')):
                job_attribute = 'qux_job'
                runner.invoke(job_execute_command, ['-f', path_to_file('test_cli_commands.py'), '-a', job_attribute])
            mock_stop_event = mock.MagicMock()
            mock_stop_event.is_set.return_value = False

            def side_effect(_):
                if False:
                    print('Hello World!')
                mock_stop_event.is_set.return_value = True
            mock_stop_event.wait.side_effect = side_effect
            cleanup_telemetry_logger()
            upload_logs(mock_stop_event, raise_errors=True)
            assert responses.assert_call_count(get_dagster_telemetry_url(), 1)

@pytest.mark.parametrize('env', [{'BUILDKITE': 'True', 'DAGSTER_DISABLE_TELEMETRY': None}, {'TF_BUILD': 'True', 'DAGSTER_DISABLE_TELEMETRY': None}, {'DAGSTER_DISABLE_TELEMETRY': 'True'}])
@responses.activate
def test_dagster_telemetry_no_test_env_upload(env):
    if False:
        while True:
            i = 10
    with instance_for_test():
        with environ(env):
            runner = CliRunner()
            with pushd(path_to_file('')):
                job_attribute = 'qux_job'
                runner.invoke(job_execute_command, ['-f', path_to_file('test_cli_commands.py'), '-a', job_attribute])
            upload_logs(mock.MagicMock())
            assert responses.assert_call_count(get_dagster_telemetry_url(), 0)