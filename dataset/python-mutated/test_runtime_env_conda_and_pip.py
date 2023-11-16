import os
import pytest
import sys
import platform
import time
from ray._private.test_utils import wait_for_condition, chdir, check_local_files_gced, generate_runtime_env_dict
from ray._private.runtime_env.conda import _get_conda_dict_with_ray_inserted
from ray._private.runtime_env.pip import INTERNAL_PIP_FILENAME, MAX_INTERNAL_PIP_FILENAME_TRIES, _PathHelper
from ray.runtime_env import RuntimeEnv
import yaml
import tempfile
from pathlib import Path
import subprocess
import ray
if not os.environ.get('CI'):
    os.environ['RAY_RUNTIME_ENV_LOCAL_DEV_MODE'] = '1'

def test_get_conda_dict_with_ray_inserted_m1_wheel(monkeypatch):
    if False:
        i = 10
        return i + 15
    if os.environ.get('RAY_RUNTIME_ENV_LOCAL_DEV_MODE') is not None:
        monkeypatch.delenv('RAY_RUNTIME_ENV_LOCAL_DEV_MODE')
    if os.environ.get('RAY_CI_POST_WHEEL_TESTS') is not None:
        monkeypatch.delenv('RAY_CI_POST_WHEEL_TESTS')
    monkeypatch.setattr(ray, '__version__', '1.9.0')
    monkeypatch.setattr(ray, '__commit__', '92599d9127e228fe8d0a2d94ca75754ec21c4ae4')
    monkeypatch.setattr(sys, 'version_info', (3, 9, 7, 'final', 0))
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(platform, 'machine', lambda : 'arm64')
    input_conda = {'dependencies': ['blah', 'pip', {'pip': ['pip_pkg']}]}
    runtime_env = RuntimeEnv(conda=input_conda)
    output_conda = _get_conda_dict_with_ray_inserted(runtime_env)
    assert output_conda == {'dependencies': ['blah', 'pip', {'pip': ['ray==1.9.0', 'ray[default]', 'pip_pkg']}, 'python=3.9.7']}

@pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Requires PR wheels built in CI, so only run on linux CI machines.')
@pytest.mark.parametrize('field', ['conda', 'pip'])
def test_requirements_files(start_cluster, field):
    if False:
        while True:
            i = 10
    'Test the use of requirements.txt and environment.yaml.\n\n    Tests that requirements files are parsed on the driver, not the cluster.\n    This is the desired behavior because the file paths only make sense on the\n    driver machine. The files do not exist on the remote cluster.\n\n    Also tests the common use case of specifying the option --extra-index-url\n    in a pip requirements.txt file.\n    '
    (cluster, address) = start_cluster
    with tempfile.TemporaryDirectory() as tmpdir, chdir(tmpdir):
        pip_list = ['--extra-index-url https://pypi.org/simple', 'pip-install-test==0.5']
        if field == 'conda':
            conda_dict = {'dependencies': ['pip', {'pip': pip_list}]}
            relative_filepath = 'environment.yml'
            conda_file = Path(relative_filepath)
            conda_file.write_text(yaml.dump(conda_dict))
            runtime_env = {'conda': relative_filepath}
        elif field == 'pip':
            relative_filepath = 'requirements.txt'
            pip_file = Path(relative_filepath)
            pip_file.write_text('\n'.join(pip_list))
            runtime_env = {'pip': relative_filepath}
        ray.init(address, runtime_env=runtime_env)

        @ray.remote
        def f():
            if False:
                while True:
                    i = 10
            import pip_install_test
            return True
        assert ray.get(f.remote())

class TestGC:

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Needs PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('field', ['conda', 'pip'])
    @pytest.mark.parametrize('spec_format', ['file', 'python_object'])
    def test_job_level_gc(self, runtime_env_disable_URI_cache, start_cluster, field, spec_format, tmp_path):
        if False:
            while True:
                i = 10
        "Tests that job-level conda env is GC'd when the job exits."
        (cluster, address) = start_cluster
        ray.init(address, runtime_env=generate_runtime_env_dict(field, spec_format, tmp_path))

        @ray.remote
        def f():
            if False:
                i = 10
                return i + 15
            import pip_install_test
            return True
        assert ray.get(f.remote())
        time.sleep(2)
        assert not check_local_files_gced(cluster)
        ray.shutdown()
        wait_for_condition(lambda : check_local_files_gced(cluster), timeout=30)
        ray.init(address, runtime_env=generate_runtime_env_dict(field, spec_format, tmp_path))
        assert ray.get(f.remote())

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Requires PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('field', ['conda', 'pip'])
    @pytest.mark.parametrize('spec_format', ['file', 'python_object'])
    def test_detached_actor_gc(self, runtime_env_disable_URI_cache, start_cluster, field, spec_format, tmp_path):
        if False:
            while True:
                i = 10
        "Tests that detached actor's conda env is GC'd only when it exits."
        (cluster, address) = start_cluster
        ray.init(address, namespace='test', runtime_env=generate_runtime_env_dict(field, spec_format, tmp_path))

        @ray.remote
        class A:

            def test_import(self):
                if False:
                    i = 10
                    return i + 15
                import pip_install_test
                return True
        a = A.options(name='test', lifetime='detached').remote()
        ray.get(a.test_import.remote())
        assert not check_local_files_gced(cluster)
        ray.shutdown()
        ray.init(address, namespace='test')
        assert not check_local_files_gced(cluster)
        a = ray.get_actor('test')
        assert ray.get(a.test_import.remote())
        ray.kill(a)
        wait_for_condition(lambda : check_local_files_gced(cluster), timeout=30)

def test_import_in_subprocess(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init()

    @ray.remote(runtime_env={'pip': ['pip-install-test==0.5']})
    def f():
        if False:
            while True:
                i = 10
        return subprocess.run(['python', '-c', 'import pip_install_test']).returncode
    assert ray.get(f.remote()) == 0

def test_runtime_env_conda_not_exists_not_hang(shutdown_only):
    if False:
        while True:
            i = 10
    "Verify when the conda env doesn't exist, it doesn't hang Ray."
    ray.init(runtime_env={'conda': 'env_which_does_not_exist'})

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        return 1
    refs = [f.remote() for _ in range(5)]
    for ref in refs:
        with pytest.raises(ray.exceptions.RuntimeEnvSetupError) as exc_info:
            ray.get(ref)
        assert "doesn't exist from the output of `conda env list --json`" in str(exc_info.value)

def test_get_requirements_file():
    if False:
        i = 10
        return i + 15
    'Unit test for _PathHelper.get_requirements_file.'
    with tempfile.TemporaryDirectory() as tmpdir:
        path_helper = _PathHelper()
        assert path_helper.get_requirements_file(tmpdir, pip_list=None) == os.path.join(tmpdir, INTERNAL_PIP_FILENAME)
        assert path_helper.get_requirements_file(tmpdir, pip_list=['foo', 'bar']) == os.path.join(tmpdir, INTERNAL_PIP_FILENAME)
        assert path_helper.get_requirements_file(tmpdir, pip_list=['foo', 'bar', f'-r {INTERNAL_PIP_FILENAME}']) == os.path.join(tmpdir, f'{INTERNAL_PIP_FILENAME}.1')
        assert path_helper.get_requirements_file(tmpdir, pip_list=['foo', 'bar', f'{INTERNAL_PIP_FILENAME}.1', f'{INTERNAL_PIP_FILENAME}.2']) == os.path.join(tmpdir, f'{INTERNAL_PIP_FILENAME}.3')
        with pytest.raises(RuntimeError) as excinfo:
            path_helper.get_requirements_file(tmpdir, pip_list=['foo', 'bar', *[f'{INTERNAL_PIP_FILENAME}.{i}' for i in range(MAX_INTERNAL_PIP_FILENAME_TRIES)]])
        assert 'Could not find a valid filename for the internal ' in str(excinfo.value)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))