from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from kylinpy.exceptions import KylinCubeError
from airflow.exceptions import AirflowException
from airflow.providers.apache.kylin.hooks.kylin import KylinHook
pytestmark = pytest.mark.db_test

class TestKylinHook:

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        self.hook = KylinHook(kylin_conn_id='kylin_default', project='learn_kylin')

    @patch('kylinpy.Kylin.get_job')
    def test_get_job_status(self, mock_job):
        if False:
            for i in range(10):
                print('nop')
        job = MagicMock()
        job.status = 'ERROR'
        mock_job.return_value = job
        assert self.hook.get_job_status('123') == 'ERROR'

    @patch('kylinpy.Kylin.get_datasource')
    def test_cube_run(self, cube_source):
        if False:
            return 10

        class MockCubeSource:

            def invoke_command(self, command, **kwargs):
                if False:
                    while True:
                        i = 10
                invoke_command_list = ['fullbuild', 'build', 'merge', 'refresh', 'delete', 'build_streaming', 'merge_streaming', 'refresh_streaming', 'disable', 'enable', 'purge', 'clone', 'drop']
                if command in invoke_command_list:
                    return {'code': '000', 'data': {}}
                else:
                    raise KylinCubeError(f'Unsupported invoke command for datasource: {command}')
        cube_source.return_value = MockCubeSource()
        response_data = {'code': '000', 'data': {}}
        assert self.hook.cube_run('kylin_sales_cube', 'build') == response_data
        assert self.hook.cube_run('kylin_sales_cube', 'refresh') == response_data
        assert self.hook.cube_run('kylin_sales_cube', 'merge') == response_data
        assert self.hook.cube_run('kylin_sales_cube', 'build_streaming') == response_data
        with pytest.raises(AirflowException):
            self.hook.cube_run('kylin_sales_cube', 'build123')