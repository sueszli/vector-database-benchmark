import pytest
import sys
import os
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args, parse_cluster_info

class TestParseRuntimeEnvArgs:

    @pytest.mark.skipif(sys.platform == 'win32', reason='File path incorrect on Windows.')
    def test_runtime_env_valid(self):
        if False:
            i = 10
            return i + 15
        config_file_name = os.path.join(os.path.dirname(__file__), 'test_config_files', 'basic_runtime_env.yaml')
        assert parse_runtime_env_args(runtime_env=config_file_name) == {'py_modules': ['pm1', 'pm2'], 'working_dir': 'wd'}

    def test_runtime_env_json_valid(self):
        if False:
            while True:
                i = 10
        runtime_env = '{"py_modules": ["pm1", "pm2"], "working_dir": "wd"}'
        assert parse_runtime_env_args(runtime_env_json=runtime_env) == {'py_modules': ['pm1', 'pm2'], 'working_dir': 'wd'}

    @pytest.mark.skipif(sys.platform == 'win32', reason='File path incorrect on Windows.')
    def test_runtime_env_and_json(self):
        if False:
            return 10
        config_file_name = os.path.join(os.path.dirname(__file__), 'test_config_files', 'basic_runtime_env.yaml')
        runtime_env_json = '{"py_modules": ["pm1", "pm2"], "working_dir": "wd"}'
        with pytest.raises(ValueError):
            parse_runtime_env_args(runtime_env=config_file_name, runtime_env_json=runtime_env_json)

    def test_working_dir_valid(self):
        if False:
            for i in range(10):
                print('nop')
        assert parse_runtime_env_args(working_dir='wd') == {'working_dir': 'wd'}

    @pytest.mark.skipif(sys.platform == 'win32', reason='File path incorrect on Windows.')
    def test_working_dir_override(self):
        if False:
            while True:
                i = 10
        config_file_name = os.path.join(os.path.dirname(__file__), 'test_config_files', 'basic_runtime_env.yaml')
        assert parse_runtime_env_args(runtime_env=config_file_name, working_dir='wd2') == {'py_modules': ['pm1', 'pm2'], 'working_dir': 'wd2'}
        runtime_env = '{"py_modules": ["pm1", "pm2"], "working_dir": "wd2"}'
        assert parse_runtime_env_args(runtime_env_json=runtime_env, working_dir='wd2') == {'py_modules': ['pm1', 'pm2'], 'working_dir': 'wd2'}

    def test_all_none(self):
        if False:
            while True:
                i = 10
        assert parse_runtime_env_args() == {}

def test_get_job_submission_client_cluster_info():
    if False:
        while True:
            i = 10
    from ray.dashboard.modules.dashboard_sdk import get_job_submission_client_cluster_info

def test_parse_cluster_address_validation():
    if False:
        i = 10
        return i + 15
    'Test that parse_cluster_info validates address schemes.'
    with pytest.raises(ValueError):
        parse_cluster_info('auto')
    invalid_schemes = ['ray']
    for scheme in invalid_schemes:
        with pytest.raises(ValueError):
            parse_cluster_info(f'{scheme}://localhost:10001')
    valid_schemes = ['http', 'https']
    for scheme in valid_schemes:
        parse_cluster_info(f'{scheme}://localhost:10001')
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))