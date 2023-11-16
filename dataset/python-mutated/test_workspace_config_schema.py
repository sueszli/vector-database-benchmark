import yaml
from dagster._core.test_utils import environ
from dagster._core.workspace.config_schema import process_workspace_config

def _validate_yaml_contents(yaml_contents):
    if False:
        i = 10
        return i + 15
    return process_workspace_config(yaml.safe_load(yaml_contents))

def test_python_file():
    if False:
        for i in range(10):
            print('nop')
    terse_workspace_yaml = '\nload_from:\n    - python_file: a_file.py\n'
    assert _validate_yaml_contents(terse_workspace_yaml).success
    nested_workspace_yaml = '\nload_from:\n    - python_file:\n        relative_path: a_file.py\n'
    assert _validate_yaml_contents(nested_workspace_yaml).success
    nested_workspace_yaml_with_def_name = '\nload_from:\n    - python_file:\n        relative_path: a_file.py\n        attribute: repo_symbol\n'
    assert _validate_yaml_contents(nested_workspace_yaml_with_def_name).success
    nested_workspace_yaml_with_def_name_and_location = '\nload_from:\n    - python_file:\n        relative_path: a_file.py\n        attribute: repo_symbol\n        location_name: some_location\n'
    assert _validate_yaml_contents(nested_workspace_yaml_with_def_name_and_location).success
    workspace_yaml_with_executable_path = '\nload_from:\n    - python_file:\n        relative_path: a_file.py\n        executable_path: /path/to/venv/bin/python\n'
    assert _validate_yaml_contents(workspace_yaml_with_executable_path).success

def test_python_module():
    if False:
        print('Hello World!')
    terse_workspace_yaml = '\nload_from:\n    - python_module: a_module\n'
    assert _validate_yaml_contents(terse_workspace_yaml).success
    nested_workspace_yaml = '\nload_from:\n    - python_module:\n        module_name: a_module\n'
    assert _validate_yaml_contents(nested_workspace_yaml).success
    nested_workspace_yaml_with_def_name = '\nload_from:\n    - python_module:\n        module_name: a_module\n        attribute: repo_symbol\n'
    assert _validate_yaml_contents(nested_workspace_yaml_with_def_name).success
    nested_workspace_yaml_with_def_name_and_location = '\nload_from:\n    - python_module:\n        module_name: a_module\n        attribute: repo_symbol\n        location_name: some_location\n'
    assert _validate_yaml_contents(nested_workspace_yaml_with_def_name_and_location).success
    workspace_yaml_with_executable_path = '\nload_from:\n    - python_module:\n        module_name: a_module\n        executable_path: /path/to/venv/bin/python\n'
    assert _validate_yaml_contents(workspace_yaml_with_executable_path).success

def test_python_package():
    if False:
        while True:
            i = 10
    workspace_yaml = '\nload_from:\n    - python_package: a_package\n'
    assert _validate_yaml_contents(workspace_yaml).success
    nested_workspace_yaml = '\nload_from:\n    - python_package:\n        package_name: a_package\n'
    assert _validate_yaml_contents(nested_workspace_yaml).success
    workspace_yaml_with_executable_path = '\nload_from:\n    - python_package:\n        package_name: a_package\n        executable_path: /path/to/venv/bin/python\n'
    assert _validate_yaml_contents(workspace_yaml_with_executable_path).success

def test_cannot_do_both():
    if False:
        i = 10
        return i + 15
    both_yaml = '\nload_from:\n    - python_module: a_module\n      python_file: a_file.py\n'
    assert not _validate_yaml_contents(both_yaml).success

def test_load_both():
    if False:
        print('Hello World!')
    both_yaml = '\nload_from:\n    - python_module: a_module\n    - python_file: a_file.py\n'
    assert _validate_yaml_contents(both_yaml).success

def test_load_python_file_with_env_var():
    if False:
        while True:
            i = 10
    with environ({'TEST_EXECUTABLE_PATH': 'executable/path/bin/python'}):
        workspace_yaml = '\n    load_from:\n        - python_file:\n            relative_path: file_valid_in_that_env.py\n            executable_path:\n                env: TEST_EXECUTABLE_PATH\n    '
        validation_result = _validate_yaml_contents(workspace_yaml)
        assert validation_result.success

def test_load_python_module_with_env_var():
    if False:
        for i in range(10):
            print('nop')
    with environ({'TEST_EXECUTABLE_PATH': 'executable/path/bin/python'}):
        workspace_yaml = '\n    load_from:\n        - python_module:\n            module_name: module_valid_in_that_env\n            executable_path:\n                env: TEST_EXECUTABLE_PATH\n    '
        validation_result = _validate_yaml_contents(workspace_yaml)
        assert validation_result.success

def test_load_python_package_with_env_var():
    if False:
        i = 10
        return i + 15
    with environ({'TEST_EXECUTABLE_PATH': 'executable/path/bin/python'}):
        workspace_yaml = '\n    load_from:\n        - python_package:\n            package_name: package_valid_in_that_env\n            executable_path:\n                env: TEST_EXECUTABLE_PATH\n    '
        validation_result = _validate_yaml_contents(workspace_yaml)
        assert validation_result.success

def test_load_from_grpc_server():
    if False:
        for i in range(10):
            print('nop')
    with environ({'TEST_EXECUTABLE_PATH': 'executable/path/bin/python'}):
        valid_yaml = "\n    load_from:\n        - grpc_server:\n            host: remotehost\n            port: 4266\n            location_name: 'my_grpc_server'\n    "
        validation_result = _validate_yaml_contents(valid_yaml)
        assert validation_result.success
        valid_yaml = "\n    load_from:\n        - grpc_server:\n            host: remotehost\n            port: 4266\n            location_name: 'my_grpc_server'\n            ssl: true\n    "
        validation_result = _validate_yaml_contents(valid_yaml)
        assert validation_result.success
        valid_yaml = "\n    load_from:\n        - grpc_server:\n            host: remotehost\n            port: 4266\n            location_name: 'my_grpc_server'\n            ssl: false\n    "
        validation_result = _validate_yaml_contents(valid_yaml)
        assert validation_result.success

def test_load_from_grpc_server_env():
    if False:
        print('Hello World!')
    with environ({'TEST_EXECUTABLE_PATH': 'executable/path/bin/python', 'FOO_PORT': '1234', 'FOO_SOCKET': 'barsocket', 'FOO_HOST': 'barhost'}):
        valid_yaml = "\n    load_from:\n        - grpc_server:\n            host:\n              env: FOO_HOST\n            port:\n              env: FOO_PORT\n            location_name: 'my_grpc_server'\n    "
        assert _validate_yaml_contents(valid_yaml).success
        valid_socket_yaml = "\n    load_from:\n        - grpc_server:\n            host:\n              env: FOO_HOST\n            socket:\n              env: FOO_SOCKET\n            location_name: 'my_grpc_server'\n    "
        assert _validate_yaml_contents(valid_socket_yaml).success