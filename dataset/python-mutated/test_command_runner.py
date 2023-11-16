import pytest
from ray.tests.test_autoscaler import MockProvider, MockProcessRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler._private.command_runner import SSHCommandRunner, DockerCommandRunner, _with_environment_variables
from ray.autoscaler.sdk import get_docker_host_mount_location
from getpass import getuser
import hashlib
auth_config = {'ssh_user': 'ray', 'ssh_private_key': '8265.pem'}

def test_environment_variable_encoder_strings():
    if False:
        return 10
    env_vars = {'var1': 'quote between this " and this', 'var2': '123'}
    res = _with_environment_variables('echo hello', env_vars)
    expected = 'export var1=\'"quote between this \\" and this"\';export var2=\'"123"\';echo hello'
    assert res == expected

def test_environment_variable_encoder_dict():
    if False:
        print('Hello World!')
    env_vars = {'value1': 'string1', 'value2': {'a': 'b', 'c': 2}}
    res = _with_environment_variables('echo hello', env_vars)
    expected = 'export value1=\'"string1"\';export value2=\'{"a":"b","c":2}\';echo hello'
    assert res == expected

def test_command_runner_interface_abstraction_violation():
    if False:
        for i in range(10):
            print('nop')
    'Enforces the CommandRunnerInterface functions on the subclasses.\n\n    This is important to make sure the subclasses do not violate the\n    function abstractions. If you need to add a new function to one of\n    the CommandRunnerInterface subclasses, you have to add it to\n    CommandRunnerInterface and all of its subclasses.\n    '
    cmd_runner_interface_public_functions = dir(CommandRunnerInterface)
    allowed_public_interface_functions = {func for func in cmd_runner_interface_public_functions if not func.startswith('_')}
    for subcls in [SSHCommandRunner, DockerCommandRunner]:
        subclass_available_functions = dir(subcls)
        subclass_public_functions = {func for func in subclass_available_functions if not func.startswith('_')}
        assert allowed_public_interface_functions == subclass_public_functions

def test_ssh_command_runner():
    if False:
        i = 10
        return i + 15
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
    ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
    ssh_control_path = '/tmp/ray_ssh_{}/{}'.format(ssh_user_hash[:10], ssh_control_hash[:10])
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False}
    cmd_runner = SSHCommandRunner(**args)
    env_vars = {'var1': 'quote between this " and this', 'var2': '123'}
    cmd_runner.run('echo helloo', port_forward=[(8265, 8265)], environment_variables=env_vars)
    expected = ['ssh', '-tt', '-L', '8265:localhost:8265', '-i', '8265.pem', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'IdentitiesOnly=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', '-o', 'ControlMaster=auto', '-o', 'ControlPath={}/%C'.format(ssh_control_path), '-o', 'ControlPersist=10s', '-o', 'ConnectTimeout=120s', 'ray@1.2.3.4', 'bash', '--login', '-c', '-i', '\'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (export var1=\'"\'"\'"quote between this \\" and this"\'"\'"\';export var2=\'"\'"\'"123"\'"\'"\';echo helloo)\'']
    for (x, y) in zip(process_runner.calls[0], expected):
        assert x == y
    process_runner.assert_has_call('1.2.3.4', exact=expected)

def test_docker_command_runner():
    if False:
        return 10
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
    ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
    ssh_control_path = '/tmp/ray_ssh_{}/{}'.format(ssh_user_hash[:10], ssh_control_hash[:10])
    docker_config = {'container_name': 'container'}
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False, 'docker_config': docker_config}
    cmd_runner = DockerCommandRunner(**args)
    assert len(process_runner.calls) == 0, 'No calls should be made in ctor'
    env_vars = {'var1': 'quote between this " and this', 'var2': '123'}
    cmd_runner.run('echo hello', environment_variables=env_vars)
    cmd = '\'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (docker exec -it  container /bin/bash -c \'"\'"\'bash --login -c -i \'"\'"\'"\'"\'"\'"\'"\'"\'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (export var1=\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"quote between this \\" and this"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\';export var2=\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"123"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\'"\';echo hello)\'"\'"\'"\'"\'"\'"\'"\'"\'\'"\'"\' )\''
    expected = ['ssh', '-tt', '-i', '8265.pem', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'IdentitiesOnly=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', '-o', 'ControlMaster=auto', '-o', 'ControlPath={}/%C'.format(ssh_control_path), '-o', 'ControlPersist=10s', '-o', 'ConnectTimeout=120s', 'ray@1.2.3.4', 'bash', '--login', '-c', '-i', cmd]
    for (x, y) in zip(process_runner.calls[0], expected):
        print(f'expeted:\t{y}')
        print(f'actual: \t{x}')
        assert x == y
    process_runner.assert_has_call('1.2.3.4', exact=expected)

def test_docker_rsync():
    if False:
        while True:
            i = 10
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    docker_config = {'container_name': 'container'}
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False, 'docker_config': docker_config}
    cmd_runner = DockerCommandRunner(**args)
    local_mount = '/home/ubuntu/base/mount/'
    remote_mount = '/root/protected_mount/'
    docker_mount_prefix = get_docker_host_mount_location(cluster_name)
    remote_host_mount = f'{docker_mount_prefix}{remote_mount}'
    local_file = '/home/ubuntu/base-file'
    remote_file = '/root/protected-file'
    remote_host_file = f'{docker_mount_prefix}{remote_file}'
    process_runner.respond_to_call('docker inspect -f', ['true'])
    cmd_runner.run_rsync_up(local_mount, remote_mount, options={'docker_mount_if_possible': True})
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'-avz {local_mount} ray@1.2.3.4:{remote_mount}')
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'mkdir -p {remote_mount}')
    process_runner.assert_not_has_call('1.2.3.4', pattern='rsync -e.*docker exec -i')
    process_runner.assert_has_call('1.2.3.4', pattern=f'-avz {local_mount} ray@1.2.3.4:{remote_host_mount}')
    process_runner.clear_history()
    process_runner.respond_to_call('docker inspect -f', ['true'])
    cmd_runner.run_rsync_up(local_file, remote_file, options={'docker_mount_if_possible': False})
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'-avz {local_file} ray@1.2.3.4:{remote_file}')
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'mkdir -p {remote_file}')
    process_runner.assert_has_call('1.2.3.4', pattern='rsync -e.*docker exec -i')
    process_runner.assert_has_call('1.2.3.4', pattern=f'-avz {local_file} ray@1.2.3.4:{remote_host_file}')
    process_runner.clear_history()
    cmd_runner.run_rsync_down(remote_mount, local_mount, options={'docker_mount_if_possible': True})
    process_runner.assert_not_has_call('1.2.3.4', pattern='rsync -e.*docker exec -i')
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'-avz ray@1.2.3.4:{remote_mount} {local_mount}')
    process_runner.assert_has_call('1.2.3.4', pattern=f'-avz ray@1.2.3.4:{remote_host_mount} {local_mount}')
    process_runner.clear_history()
    cmd_runner.run_rsync_down(remote_file, local_file, options={'docker_mount_if_possible': False})
    process_runner.assert_has_call('1.2.3.4', pattern='rsync -e.*docker exec -i')
    process_runner.assert_not_has_call('1.2.3.4', pattern=f'-avz ray@1.2.3.4:{remote_file} {local_file}')
    process_runner.assert_has_call('1.2.3.4', pattern=f'-avz ray@1.2.3.4:{remote_host_file} {local_file}')

def test_rsync_exclude_and_filter():
    if False:
        return 10
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False}
    cmd_runner = SSHCommandRunner(**args)
    local_mount = '/home/ubuntu/base/mount/'
    remote_mount = '/root/protected_mount/'
    process_runner.respond_to_call('docker inspect -f', ['true'])
    cmd_runner.run_rsync_up(local_mount, remote_mount, options={'docker_mount_if_possible': True, 'rsync_exclude': ['test'], 'rsync_filter': ['.ignore']})
    process_runner.assert_has_call('1.2.3.4', pattern='--exclude test --filter dir-merge,- .ignore')

def test_rsync_without_exclude_and_filter():
    if False:
        return 10
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False}
    cmd_runner = SSHCommandRunner(**args)
    local_mount = '/home/ubuntu/base/mount/'
    remote_mount = '/root/protected_mount/'
    process_runner.respond_to_call('docker inspect -f', ['true'])
    cmd_runner.run_rsync_up(local_mount, remote_mount, options={'docker_mount_if_possible': True})
    process_runner.assert_not_has_call('1.2.3.4', pattern='--exclude test')
    process_runner.assert_not_has_call('1.2.3.4', pattern='--filter dir-merge,- .ignore')

@pytest.mark.parametrize('run_option_type', ['run_options', 'head_run_options'])
def test_docker_shm_override(run_option_type):
    if False:
        i = 10
        return i + 15
    process_runner = MockProcessRunner()
    provider = MockProvider()
    provider.create_node({}, {}, 1)
    cluster_name = 'cluster'
    docker_config = {'container_name': 'container', 'image': 'rayproject/ray:latest', run_option_type: ['--shm-size=80g']}
    args = {'log_prefix': 'prefix', 'node_id': '0', 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': False, 'docker_config': docker_config}
    cmd_runner = DockerCommandRunner(**args)
    process_runner.respond_to_call('json .Config.Env', 2 * ['[]'])
    cmd_runner.run_init(as_head=True, file_mounts={}, sync_run_yet=True)
    process_runner.assert_has_call('1.2.3.4', pattern='--shm-size=80g')
    process_runner.assert_not_has_call('1.2.3.4', pattern='/proc/meminfo')
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))