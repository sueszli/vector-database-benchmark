import os
import platform
from pathlib import Path
import pytest
import subprocess
import sys
import tempfile
import time
from typing import List
from unittest import mock
import yaml
import ray
from ray.runtime_env import RuntimeEnv
from ray._private.runtime_env.conda import inject_dependencies, _inject_ray_to_conda_site, _resolve_install_from_source_ray_dependencies, _current_py_version
from ray._private.runtime_env.conda_utils import get_conda_env_list
from ray._private.test_utils import run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition, chdir
from ray._private.utils import get_conda_env_dir, get_conda_bin_executable, try_to_create_directory
if not os.environ.get('CI'):
    os.environ['RAY_RUNTIME_ENV_LOCAL_DEV_MODE'] = '1'
EMOJI_VERSIONS = ['2.1.0', '2.2.0']
_WIN32 = os.name == 'nt'

@pytest.fixture(scope='session')
def conda_envs(tmp_path_factory):
    if False:
        return 10
    'Creates two conda env with different `emoji` package versions.'
    conda_path = get_conda_bin_executable('conda')
    init_cmd = f'. {os.path.dirname(conda_path)}/../etc/profile.d/conda.sh'

    def delete_env(env_name):
        if False:
            for i in range(10):
                print('nop')
        subprocess.run(['conda', 'remove', '--name', env_name, '--all', '-y'])

    def create_package_env(env_name, package_version: str):
        if False:
            return 10
        delete_env(env_name)
        proc = subprocess.run(['conda', 'create', '-n', env_name, '-y', f'python={_current_py_version()}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print('conda create failed, returned %d' % proc.returncode)
            print(proc.stdout.decode())
            print(proc.stderr.decode())
            assert False
        _inject_ray_to_conda_site(get_conda_env_dir(env_name))
        ray_deps: List[str] = _resolve_install_from_source_ray_dependencies()
        ray_deps.append(f'emoji=={package_version}')
        reqs = tmp_path_factory.mktemp('reqs') / 'requirements.txt'
        with reqs.open('wt') as fid:
            for line in ray_deps:
                fid.write(line)
                fid.write('\n')
        commands = [f'conda activate {env_name}', f'python -m pip install -r {str(reqs)}', 'conda deactivate']
        if _WIN32:
            command = ' && '.join(commands)
        else:
            commands.insert(0, init_cmd)
            command = [' && '.join(commands)]
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print('conda/pip install failed, returned %d' % proc.returncode)
            print('command', command)
            print(proc.stdout.decode())
            print(proc.stderr.decode())
            assert False
    for package_version in EMOJI_VERSIONS:
        create_package_env(env_name=f'package-{package_version}', package_version=package_version)
    yield
    for package_version in EMOJI_VERSIONS:
        delete_env(env_name=f'package-{package_version}')

@ray.remote
def get_emoji_version():
    if False:
        for i in range(10):
            print('nop')
    import emoji
    return emoji.__version__

@ray.remote
class VersionActor:

    def get_emoji_version(self):
        if False:
            while True:
                i = 10
        import emoji
        return emoji.__version__
check_remote_client_conda = '\nimport ray\ncontext = (ray.client("localhost:24001")\n              .env({{"conda" : "package-{package_version}"}})\n              .connect())\n@ray.remote\ndef get_package_version():\n    import emoji\n    return emoji.__version__\n\nassert ray.get(get_package_version.remote()) == "{package_version}"\ncontext.disconnect()\n'

@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV') is None, reason='must be run from within a conda environment')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 24001 --port 0'], indirect=True)
def test_client_tasks_and_actors_inherit_from_driver(conda_envs, call_ray_start):
    if False:
        i = 10
        return i + 15
    for (i, package_version) in enumerate(EMOJI_VERSIONS):
        runtime_env = {'conda': f'package-{package_version}'}
        with ray.client('localhost:24001').env(runtime_env).connect():
            assert ray.get(get_emoji_version.remote()) == package_version
            actor_handle = VersionActor.remote()
            assert ray.get(actor_handle.get_emoji_version.remote()) == package_version
            other_package_version = EMOJI_VERSIONS[(i + 1) % 2]
            run_string_as_driver(check_remote_client_conda.format(package_version=other_package_version))

@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV') is None, reason='must be run from within a conda environment')
def test_task_actor_conda_env(conda_envs, shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init()
    for package_version in EMOJI_VERSIONS:
        runtime_env = {'conda': f'package-{package_version}'}
        task = get_emoji_version.options(runtime_env=runtime_env)
        assert ray.get(task.remote()) == package_version
        actor = VersionActor.options(runtime_env=runtime_env).remote()
        assert ray.get(actor.get_emoji_version.remote()) == package_version

    @ray.remote
    def wrapped_version():
        if False:
            i = 10
            return i + 15
        return ray.get(get_emoji_version.remote())

    @ray.remote
    class Wrapper:

        def wrapped_version(self):
            if False:
                while True:
                    i = 10
            return ray.get(get_emoji_version.remote())
    for package_version in EMOJI_VERSIONS:
        runtime_env = {'conda': f'package-{package_version}'}
        task = wrapped_version.options(runtime_env=runtime_env)
        assert ray.get(task.remote()) == package_version
        actor = Wrapper.options(runtime_env=runtime_env).remote()
        assert ray.get(actor.wrapped_version.remote()) == package_version

@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV') is None, reason='must be run from within a conda environment')
def test_task_conda_env_validation_cached(conda_envs, shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    "Verify that when a task is running with the same conda env\n    it doesn't validate if env exists.\n    "
    ray.init()
    version = EMOJI_VERSIONS[0]
    runtime_env = {'conda': f'package-{version}'}
    task = get_emoji_version.options(runtime_env=runtime_env)
    s = time.time()
    ray.get(task.remote())
    first_run = time.time() - s
    print('First run took', first_run)
    s = time.time()
    for _ in range(10):
        ray.get(task.remote())
    second_10_runs = time.time() - s
    print('second 10 runs took', second_10_runs)
    assert second_10_runs < first_run

@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV') is None, reason='must be run from within a conda environment')
def test_job_config_conda_env(conda_envs, shutdown_only):
    if False:
        return 10
    for package_version in EMOJI_VERSIONS:
        runtime_env = {'conda': f'package-{package_version}'}
        ray.init(runtime_env=runtime_env)
        assert ray.get(get_emoji_version.remote()) == package_version
        ray.shutdown()

@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV') is None, reason='must be run from within a conda environment')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('runtime_env_class', [dict, RuntimeEnv])
def test_job_eager_install(shutdown_only, runtime_env_class):
    if False:
        print('Hello World!')
    runtime_env = {'conda': {'dependencies': ['toolz']}}
    env_count = len(get_conda_env_list())
    ray.init(runtime_env=runtime_env_class(**runtime_env))
    wait_for_condition(lambda : len(get_conda_env_list()) == env_count + 1, timeout=60)
    ray.shutdown()
    runtime_env = {'conda': {'dependencies': ['toolz']}, 'config': {'eager_install': False}}
    ray.init(runtime_env=runtime_env_class(**runtime_env))
    with pytest.raises(RuntimeError):
        wait_for_condition(lambda : len(get_conda_env_list()) == env_count + 2, timeout=5)
    ray.shutdown()
    runtime_env = {'conda': {'dependencies': ['toolz']}, 'config': {'eager_install': 123}}
    with pytest.raises(TypeError):
        ray.init(runtime_env=runtime_env_class(**runtime_env))
    ray.shutdown()

def test_get_conda_env_dir(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Typical output of `conda env list`, for context:\n\n    base                 /Users/scaly/anaconda3\n    my_env_1             /Users/scaly/anaconda3/envs/my_env_1\n\n    For this test, `tmp_path` is a stand-in for `Users/scaly/anaconda3`.\n    '
    d = tmp_path / 'envs' / 'tf1'
    Path.mkdir(d, parents=True)
    with mock.patch.dict(os.environ, {'CONDA_PREFIX': str(d), 'CONDA_DEFAULT_ENV': 'tf1'}):
        with pytest.raises(ValueError):
            env_dir = get_conda_env_dir('tf2')
        tf2_dir = tmp_path / 'envs' / 'tf2'
        Path.mkdir(tf2_dir, parents=True)
        env_dir = get_conda_env_dir('tf2')
        assert env_dir == str(tmp_path / 'envs' / 'tf2')
    with mock.patch.dict(os.environ, {'CONDA_PREFIX': str(tmp_path), 'CONDA_DEFAULT_ENV': 'base'}):
        with pytest.raises(ValueError):
            env_dir = get_conda_env_dir('tf3')
        env_dir = get_conda_env_dir('tf2')
        assert env_dir == str(tmp_path / 'envs' / 'tf2')

@pytest.mark.skipif(os.environ.get('CONDA_EXE') is None, reason='Requires properly set-up conda shell')
def test_conda_create_task(shutdown_only):
    if False:
        return 10
    "Tests dynamic creation of a conda env in a task's runtime env. Assumes\n    `conda init` has been successfully called."
    ray.init()
    runtime_env = {'conda': {'dependencies': ['pip', {'pip': ['pip-install-test==0.5']}]}}

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        import pip_install_test
        return True
    with pytest.raises(ModuleNotFoundError):
        import pip_install_test
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(f.remote())
    assert 'ModuleNotFoundError' in str(excinfo.value)
    assert ray.get(f.options(runtime_env=runtime_env).remote())

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.skipif(os.environ.get('CONDA_EXE') is None, reason='Requires properly set-up conda shell')
def test_conda_create_job_config(shutdown_only):
    if False:
        print('Hello World!')
    'Tests dynamic conda env creation in a runtime env in the JobConfig.'
    runtime_env = {'conda': {'dependencies': ['pip', {'pip': ['pip-install-test==0.5']}]}}
    ray.init(runtime_env=runtime_env)

    @ray.remote
    def f():
        if False:
            return 10
        import pip_install_test
        return True
    with pytest.raises(ModuleNotFoundError):
        import pip_install_test
    assert ray.get(f.remote())

def test_inject_dependencies():
    if False:
        return 10
    num_tests = 4
    conda_dicts = [None] * num_tests
    outputs = [None] * num_tests
    conda_dicts[0] = {}
    outputs[0] = {'dependencies': ['python=7.8', 'pip', {'pip': ['ray==1.2.3']}]}
    conda_dicts[1] = {'dependencies': ['blah']}
    outputs[1] = {'dependencies': ['blah', 'python=7.8', 'pip', {'pip': ['ray==1.2.3']}]}
    conda_dicts[2] = {'dependencies': ['blah', 'pip']}
    outputs[2] = {'dependencies': ['blah', 'pip', 'python=7.8', {'pip': ['ray==1.2.3']}]}
    conda_dicts[3] = {'dependencies': ['blah', 'pip', {'pip': ['some_pkg']}]}
    outputs[3] = {'dependencies': ['blah', 'pip', {'pip': ['ray==1.2.3', 'some_pkg']}, 'python=7.8']}
    for i in range(num_tests):
        output = inject_dependencies(conda_dicts[i], '7.8', ['ray==1.2.3'])
        error_msg = f'failed on input {i}.Output: {output} \nExpected output: {outputs[i]}'
        assert output == outputs[i], error_msg

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 24001 --port 0'], indirect=True)
def test_conda_create_ray_client(call_ray_start):
    if False:
        print('Hello World!')
    'Tests dynamic conda env creation in RayClient.'
    runtime_env = {'conda': {'dependencies': ['pip', {'pip': ['pip-install-test==0.5']}]}}

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        import pip_install_test
        return True
    with ray.client('localhost:24001').env(runtime_env).connect():
        with pytest.raises(ModuleNotFoundError):
            import pip_install_test
        assert ray.get(f.remote())
    with ray.client('localhost:24001').connect():
        with pytest.raises(ModuleNotFoundError):
            ray.get(f.remote())

@pytest.mark.parametrize('pip_as_str', [True, False])
def test_pip_task(shutdown_only, pip_as_str, tmp_path):
    if False:
        i = 10
        return i + 15
    'Tests pip installs in the runtime env specified in f.options().'
    ray.init()
    if pip_as_str:
        d = tmp_path / 'pip_requirements'
        d.mkdir()
        p = d / 'requirements.txt'
        requirements_txt = '\n        pip-install-test==0.5\n        '
        p.write_text(requirements_txt)
        runtime_env = {'pip': str(p)}
    else:
        runtime_env = {'pip': ['pip-install-test==0.5']}

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        import pip_install_test
        return True
    with pytest.raises(ModuleNotFoundError):
        import pip_install_test
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(f.remote())
    assert 'ModuleNotFoundError' in str(excinfo.value)
    assert ray.get(f.options(runtime_env=runtime_env).remote())

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('option', ['conda', 'pip'])
def test_conda_pip_extras_ray_serve(shutdown_only, option):
    if False:
        print('Hello World!')
    'Tests that ray[extras] can be included as a conda/pip dependency.'
    ray.init()
    pip = ['pip-install-test==0.5', 'ray[serve]']
    if option == 'conda':
        runtime_env = {'conda': {'dependencies': ['pip', {'pip': pip}]}}
    elif option == 'pip':
        runtime_env = {'pip': pip}
    else:
        assert False, f'Unknown option: {option}'

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        import pip_install_test
        return True
    with pytest.raises(ModuleNotFoundError):
        import pip_install_test
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(f.remote())
    assert 'ModuleNotFoundError' in str(excinfo.value)
    assert ray.get(f.options(runtime_env=runtime_env).remote())

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('pip_as_str', [True, False])
def test_pip_job_config(shutdown_only, pip_as_str, tmp_path):
    if False:
        return 10
    "Tests dynamic installation of pip packages in a task's runtime env."
    if pip_as_str:
        d = tmp_path / 'pip_requirements'
        d.mkdir()
        p = d / 'requirements.txt'
        requirements_txt = '\n        pip-install-test==0.5\n        '
        p.write_text(requirements_txt)
        runtime_env = {'pip': str(p)}
    else:
        runtime_env = {'pip': ['pip-install-test==0.5']}
    ray.init(runtime_env=runtime_env)

    @ray.remote
    def f():
        if False:
            return 10
        import pip_install_test
        return True
    with pytest.raises(ModuleNotFoundError):
        import pip_install_test
    assert ray.get(f.remote())

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='dirname(__file__) returns an invalid path')
def test_experimental_package(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=2)
    pkg = ray.experimental.load_package(os.path.join(os.path.dirname(__file__), '../experimental/packaging/example_pkg/ray_pkg.yaml'))
    a = pkg.MyActor.remote()
    assert ray.get(a.f.remote()) == 'hello world'
    assert ray.get(pkg.my_func.remote()) == 'hello world'

@pytest.mark.skipif(os.environ.get('CI') and sys.platform == 'win32', reason='dirname(__file__) returns an invalid path')
def test_experimental_package_lazy(shutdown_only):
    if False:
        print('Hello World!')
    pkg = ray.experimental.load_package(os.path.join(os.path.dirname(__file__), '../experimental/packaging/example_pkg/ray_pkg.yaml'))
    ray.init(num_cpus=2)
    a = pkg.MyActor.remote()
    assert ray.get(a.f.remote()) == 'hello world'
    assert ray.get(pkg.my_func.remote()) == 'hello world'

@pytest.mark.skipif(_WIN32, reason='requires tar cli command')
def test_experimental_package_github(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2)
    pkg = ray.experimental.load_package('http://raw.githubusercontent.com/ray-project/ray/master/python/ray/experimental/packaging/example_pkg/ray_pkg.yaml')
    a = pkg.MyActor.remote()
    assert ray.get(a.f.remote()) == 'hello world'
    assert ray.get(pkg.my_func.remote()) == 'hello world'

@pytest.mark.skipif(_WIN32, reason='Fails on windows')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 24001 --port 0'], indirect=True)
def test_client_working_dir_filepath(call_ray_start, tmp_path):
    if False:
        i = 10
        return i + 15
    'Test that pip and conda filepaths work with working_dir.'
    working_dir = tmp_path / 'requirements'
    working_dir.mkdir()
    pip_file = working_dir / 'requirements.txt'
    requirements_txt = '\n    pip-install-test==0.5\n    '
    pip_file.write_text(requirements_txt)
    runtime_env_pip = {'working_dir': str(working_dir), 'pip': str(pip_file)}
    conda_file = working_dir / 'environment.yml'
    conda_dict = {'dependencies': ['pip', {'pip': ['pip-install-test==0.5']}]}
    conda_str = yaml.dump(conda_dict)
    conda_file.write_text(conda_str)
    runtime_env_conda = {'working_dir': str(working_dir), 'conda': str(conda_file)}

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        import pip_install_test
        return True
    with ray.client('localhost:24001').connect():
        with pytest.raises(ModuleNotFoundError):
            ray.get(f.remote())
    for runtime_env in [runtime_env_pip, runtime_env_conda]:
        with ray.client('localhost:24001').env(runtime_env).connect():
            with pytest.raises(ModuleNotFoundError):
                import pip_install_test
            assert ray.get(f.remote())

@pytest.mark.skipif(_WIN32, reason='Hangs on windows')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 24001 --port 0'], indirect=True)
def test_conda_pip_filepaths_remote(call_ray_start, tmp_path):
    if False:
        return 10
    'Test that pip and conda filepaths work, simulating a remote cluster.'
    working_dir = tmp_path / 'requirements'
    working_dir.mkdir()
    pip_file = working_dir / 'requirements.txt'
    requirements_txt = '\n    pip-install-test==0.5\n    '
    pip_file.write_text(requirements_txt)
    runtime_env_pip = {'pip': str(pip_file)}
    conda_file = working_dir / 'environment.yml'
    conda_dict = {'dependencies': ['pip', {'pip': ['pip-install-test==0.5']}]}
    conda_str = yaml.dump(conda_dict)
    conda_file.write_text(conda_str)
    runtime_env_conda = {'conda': str(conda_file)}

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        import pip_install_test
        return True
    with ray.client('localhost:24001').connect():
        with pytest.raises(ModuleNotFoundError):
            ray.get(f.remote())
    f_pip = f.options(runtime_env=runtime_env_pip)
    f_conda = f.options(runtime_env=runtime_env_conda)
    os.remove(pip_file)
    os.remove(conda_file)
    client_envs = [{}, {'working_dir': str(working_dir)}]
    for runtime_env in client_envs:
        with ray.client('localhost:24001').env(runtime_env).connect():
            with pytest.raises(ModuleNotFoundError):
                import pip_install_test
            assert ray.get(f_pip.remote()), str(runtime_env)
            assert ray.get(f_conda.remote()), str(runtime_env)
install_env_script = '\nimport ray\nimport time\nray.init(address="auto", runtime_env={env})\n@ray.remote\ndef f():\n    return "hello"\nf.remote()\n# Give the env 5 seconds to begin installing in a new worker.\ntime.sleep(5)\n'

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
def test_env_installation_nonblocking(shutdown_only):
    if False:
        print('Hello World!')
    'Test fix for https://github.com/ray-project/ray/issues/16226.'
    env1 = {'pip': ['pip-install-test==0.5']}
    ray.init(runtime_env=env1)

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        return 'hello'
    ray.get(f.remote())

    def assert_tasks_finish_quickly(total_sleep_s=0.1):
        if False:
            while True:
                i = 10
        'Call f every 0.01 seconds for total time total_sleep_s.'
        gap_s = 0.01
        for i in range(int(total_sleep_s / gap_s)):
            start = time.time()
            ray.get(f.remote())
            assert time.time() - start < 2.0
            time.sleep(gap_s)
    assert_tasks_finish_quickly()
    env2 = {'pip': ['pip-install-test==0.5', 'requests']}
    f.options(runtime_env=env2).remote()
    assert_tasks_finish_quickly()
    proc = run_string_as_driver_nonblocking(install_env_script.format(env=env1))
    assert_tasks_finish_quickly(total_sleep_s=5)
    proc.kill()
    proc.wait()

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
def test_simultaneous_install(shutdown_only):
    if False:
        return 10
    'Test that two envs can be installed without affecting each other.'
    ray.init()

    @ray.remote
    class VersionWorker:

        def __init__(self, key):
            if False:
                print('Hello World!')
            self.key = key

        def get(self):
            if False:
                print('Hello World!')
            import emoji
            return (self.key, emoji.__version__)
    worker_1 = VersionWorker.options(runtime_env={'pip': {'packages': ['emoji==2.1.0'], 'pip_check': False}}).remote(key=1)
    worker_2 = VersionWorker.options(runtime_env={'pip': {'packages': ['emoji==2.2.0'], 'pip_check': False}}).remote(key=2)
    assert ray.get(worker_1.get.remote()) == (1, '2.1.0')
    assert ray.get(worker_2.get.remote()) == (2, '2.2.0')
CLIENT_SERVER_PORT = 24001

@pytest.mark.skipif(_WIN32, reason='Fails on windows')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
@pytest.mark.skipif(sys.platform == 'linux' and platform.processor() == 'aarch64', reason='This test is currently not supported on Linux ARM64')
@pytest.mark.skipif(sys.version_info.major >= 3 and sys.version_info.minor >= 11, reason='Some dependencies are not available with python 3.11.')
@pytest.mark.parametrize('call_ray_start', [f'ray start --head --ray-client-server-port {CLIENT_SERVER_PORT} --port 0'], indirect=True)
def test_e2e_complex(call_ray_start, tmp_path):
    if False:
        i = 10
        return i + 15
    "Test multiple runtime_env options across multiple client connections.\n\n    1.  Run a Ray Client job with both working_dir and pip specified. Check the\n        environment using imports and file reads in tasks and actors.\n    2.  On the same cluster, run a job as above but using the Ray Summit\n        2021 demo's pip requirements.txt.  Also, check that per-task and\n        per-actor pip requirements work, all using the job's working_dir.\n    "
    specific_path = tmp_path / 'test'
    specific_path.write_text('Hello')
    with ray.client(f'localhost:{CLIENT_SERVER_PORT}').env({'working_dir': str(tmp_path), 'pip': ['pip-install-test']}).connect():

        @ray.remote
        def test_read():
            if False:
                print('Hello World!')
            return Path('./test').read_text()
        assert ray.get(test_read.remote()) == 'Hello'

        @ray.remote
        def test_pip():
            if False:
                i = 10
                return i + 15
            import pip_install_test
            import ray
            return Path('./test').read_text()
        assert ray.get(test_pip.remote()) == 'Hello'

        @ray.remote
        class TestActor:

            def test(self):
                if False:
                    while True:
                        i = 10
                import pip_install_test
                return Path('./test').read_text()
        a = TestActor.remote()
        assert ray.get(a.test.remote()) == 'Hello'
    requirement_path = tmp_path / 'requirements.txt'
    requirement_path.write_text('\n'.join(['ray[serve, tune]', 'texthero', 'PyGithub', 'xgboost_ray', 'pandas==1.1', 'typer', 'aiofiles']))
    with ray.client(f'localhost:{CLIENT_SERVER_PORT}').env({'working_dir': str(tmp_path), 'pip': str(requirement_path)}).connect():

        @ray.remote
        def test_read():
            if False:
                return 10
            return Path('./test').read_text()
        assert ray.get(test_read.remote()) == 'Hello'

        @ray.remote
        def test_import():
            if False:
                print('Hello World!')
            import ray
            from ray import serve
            from ray import tune
            import typer
            import xgboost_ray
            return Path('./test').read_text()
        assert ray.get(test_import.remote()) == 'Hello'

        @ray.remote
        class TestActor:

            def test(self):
                if False:
                    while True:
                        i = 10
                import ray
                from ray import serve
                from ray import tune
                import typer
                import xgboost_ray
                return Path('./test').read_text()
        a = TestActor.options(runtime_env={'pip': str(requirement_path)}).remote()
        assert ray.get(a.test.remote()) == 'Hello'

        @ray.remote
        def test_pip():
            if False:
                i = 10
                return i + 15
            import pip_install_test
            return Path('./test').read_text()
        assert ray.get(test_pip.options(runtime_env={'pip': ['pip-install-test']}).remote()) == 'Hello'
        with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
            ray.get(test_pip.remote())
        assert 'ModuleNotFoundError' in str(excinfo.value)

        @ray.remote
        class TestActor:

            def test(self):
                if False:
                    return 10
                import pip_install_test
                return Path('./test').read_text()
        a = TestActor.options(runtime_env={'pip': ['pip-install-test']}).remote()
        assert ray.get(a.test.remote()) == 'Hello'

@pytest.mark.skipif(_WIN32, reason='Fails on windows')
@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
def test_runtime_env_override(call_ray_start):
    if False:
        return 10
    with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
        ray.init(address='auto', namespace='test')

        @ray.remote
        class Child:

            def getcwd(self):
                if False:
                    return 10
                import os
                return os.getcwd()

            def read(self, path):
                if False:
                    for i in range(10):
                        print('nop')
                return open(path).read()

            def ready(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        @ray.remote
        class Parent:

            def spawn_child(self, name, runtime_env):
                if False:
                    i = 10
                    return i + 15
                child = Child.options(lifetime='detached', name=name, runtime_env=runtime_env).remote()
                ray.get(child.ready.remote())
        Parent.options(lifetime='detached', name='parent').remote()
        ray.shutdown()
        with open('hello', 'w') as f:
            f.write('world')
        job_config = ray.job_config.JobConfig(runtime_env={'working_dir': '.'})
        ray.init(address='auto', namespace='test', job_config=job_config)
        os.remove('hello')
        parent = ray.get_actor('parent')
        env = ray.get_runtime_context().runtime_env
        print('Spawning with env:', env)
        ray.get(parent.spawn_child.remote('child', env))
        child = ray.get_actor('child')
        child_cwd = ray.get(child.getcwd.remote())
        assert child_cwd != os.getcwd(), (child_cwd, os.getcwd())
        assert ray.get(child.read.remote('hello')) == 'world'
        ray.shutdown()

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='This test is only run on linux CI machines.')
def test_pip_with_env_vars(start_cluster, tmp_path):
    if False:
        print('Hello World!')
    with chdir(tmp_path):
        TEST_ENV_NAME = 'TEST_ENV_VARS'
        TEST_ENV_VALUE = 'TEST'
        package_name = 'test_package'
        package_dir = os.path.join(tmp_path, package_name)
        try_to_create_directory(package_dir)
        setup_filename = os.path.join(package_dir, 'setup.py')
        setup_code = 'import os\nfrom setuptools import setup, find_packages\nfrom setuptools.command.install import install\n\nclass InstallTestPackage(install):\n    # this function will be called when pip install this package\n    def run(self):\n        assert os.environ.get(\'{TEST_ENV_NAME}\') == \'{TEST_ENV_VALUE}\'\n\nsetup(\n    name=\'test_package\',\n    version=\'0.0.1\',\n    packages=find_packages(),\n    cmdclass=dict(install=InstallTestPackage),\n    license="MIT",\n    zip_safe=False,\n)\n'.format(TEST_ENV_NAME=TEST_ENV_NAME, TEST_ENV_VALUE=TEST_ENV_VALUE)
        with open(setup_filename, 'wt') as f:
            f.writelines(setup_code)
        python_filename = os.path.join(package_dir, 'test.py')
        python_code = 'import os; print(os.environ)'
        with open(python_filename, 'wt') as f:
            f.writelines(python_code)
        gz_filename = os.path.join(tmp_path, package_name + '.tar.gz')
        subprocess.check_call(['tar', '-zcvf', gz_filename, package_name])
        with pytest.raises(ray.exceptions.RuntimeEnvSetupError):

            @ray.remote(runtime_env={'env_vars': {TEST_ENV_NAME: 'failed'}, 'pip': [gz_filename]})
            def f1():
                if False:
                    return 10
                return True
            ray.get(f1.remote())

        @ray.remote(runtime_env={'env_vars': {TEST_ENV_NAME: TEST_ENV_VALUE}, 'pip': [gz_filename]})
        def f2():
            if False:
                return 10
            return True
        assert ray.get(f2.remote())
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))