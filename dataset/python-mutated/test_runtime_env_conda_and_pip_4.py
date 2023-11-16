import os
import pytest
import sys
from ray._private.runtime_env.pip import PipProcessor
import ray
if not os.environ.get('CI'):
    os.environ['RAY_RUNTIME_ENV_LOCAL_DEV_MODE'] = '1'

def test_in_virtualenv(start_cluster):
    if False:
        while True:
            i = 10
    assert PipProcessor._is_in_virtualenv() is False and 'IN_VIRTUALENV' not in os.environ or (PipProcessor._is_in_virtualenv() is True and 'IN_VIRTUALENV' in os.environ)
    (cluster, address) = start_cluster
    runtime_env = {'pip': ['pip-install-test==0.5']}
    ray.init(address, runtime_env=runtime_env)

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        import pip_install_test
        return PipProcessor._is_in_virtualenv()
    assert ray.get(f.remote())

def test_multiple_pip_installs(start_cluster, monkeypatch):
    if False:
        print('Hello World!')
    "Test that multiple pip installs don't interfere with each other."
    monkeypatch.setenv('RUNTIME_ENV_RETRY_TIMES', '0')
    (cluster, address) = start_cluster
    if sys.platform == 'win32' and 'ray' not in address:
        pytest.skip('Failing on windows, as python.exe is in use during deletion attempt.')
    ray.init(address, runtime_env={'pip': ['pip-install-test'], 'env_vars': {'TEST_VAR_1': 'test_1'}})

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        return True

    @ray.remote(runtime_env={'pip': ['pip-install-test'], 'env_vars': {'TEST_VAR_2': 'test_2'}})
    def f2():
        if False:
            return 10
        return True

    @ray.remote(runtime_env={'pip': ['pip-install-test'], 'env_vars': {'TEST_VAR_3': 'test_3'}})
    def f3():
        if False:
            return 10
        return True
    assert all(ray.get([f.remote(), f2.remote(), f3.remote()]))

class TestGC:

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Requires PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('field', ['pip'])
    def test_pip_ray_is_overwritten(self, start_cluster, field):
        if False:
            return 10
        (cluster, address) = start_cluster
        ray.init(address, runtime_env={'pip': ['pip-install-test==0.5', 'ray']})

        @ray.remote
        def f():
            if False:
                for i in range(10):
                    print('nop')
            import pip_install_test
            return True
        assert ray.get(f.remote())
        ray.shutdown()
        ray.init(address, runtime_env={'pip': ['pip-install-test==0.5', 'ray>=1.12.0']})

        @ray.remote
        def f():
            if False:
                while True:
                    i = 10
            import pip_install_test
            return True
        assert ray.get(f.remote())
        ray.shutdown()
        with pytest.raises(Exception):
            ray.init(address, runtime_env={'pip': ['pip-install-test==0.5', 'ray<=1.6.0']})

            @ray.remote
            def f():
                if False:
                    i = 10
                    return i + 15
                import pip_install_test
                return True
            assert ray.get(f.remote())
        ray.shutdown()

@pytest.mark.skipif('IN_VIRTUALENV' in os.environ or sys.platform != 'linux', reason='Requires PR wheels built in CI, so only run on linux CI machines.')
def test_run_in_virtualenv(cloned_virtualenv):
    if False:
        for i in range(10):
            print('nop')
    python_exe_path = cloned_virtualenv.python
    print(python_exe_path)
    cloned_virtualenv.run(f"{python_exe_path} -c 'from ray._private.runtime_env.pip import PipProcessor;assert PipProcessor._is_in_virtualenv()'", capture=True)
    cloned_virtualenv.run(f'IN_VIRTUALENV=1 python -m pytest {__file__}', capture=True)

@pytest.mark.skipif('IN_VIRTUALENV' in os.environ, reason='Pip option not supported in virtual env.')
def test_runtime_env_with_pip_config(start_cluster):
    if False:
        return 10

    @ray.remote(runtime_env={'pip': {'packages': ['pip-install-test==0.5'], 'pip_version': '==20.2.3'}})
    def f():
        if False:
            i = 10
            return i + 15
        import pip
        return pip.__version__
    assert ray.get(f.remote()) == '20.2.3'
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))