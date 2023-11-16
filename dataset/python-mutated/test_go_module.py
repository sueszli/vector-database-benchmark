from __future__ import annotations
from unittest import mock
from airflow.providers.google.go_module_utils import init_module, install_dependencies

class TestGoModule:

    @mock.patch('airflow.providers.google.go_module_utils.execute_in_subprocess')
    def test_should_init_go_module(self, mock_execute_in_subprocess):
        if False:
            print('Hello World!')
        init_module(go_module_name='example.com/main', go_module_path='/home/example/go')
        mock_execute_in_subprocess.assert_called_once_with(['go', 'mod', 'init', 'example.com/main'], cwd='/home/example/go')

    @mock.patch('airflow.providers.google.go_module_utils.execute_in_subprocess')
    def test_should_install_module_dependencies(self, mock_execute_in_subprocess):
        if False:
            while True:
                i = 10
        install_dependencies(go_module_path='/home/example/go')
        mock_execute_in_subprocess.assert_called_once_with(['go', 'mod', 'tidy'], cwd='/home/example/go')