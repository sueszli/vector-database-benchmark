import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock
import pytest
from pytest_lazyfixture import lazy_fixture
import ray
import ray.experimental.internal_kv as kv
from ray._private.ray_constants import RAY_RUNTIME_ENV_URI_PIN_EXPIRATION_S_ENV_VAR
from ray._private.test_utils import chdir, check_local_files_gced, wait_for_condition
from ray._private.utils import get_directory_size_bytes
S3_PACKAGE_URI = 's3://runtime-env-test/test_runtime_env.zip'
TEMP_URI_EXPIRATION_S = 40 if sys.platform == 'win32' else 20

@pytest.fixture(scope='class')
def working_dir_and_pymodules_disable_URI_cache():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, {'RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB': '0', 'RAY_RUNTIME_ENV_PY_MODULES_CACHE_SIZE_GB': '0', RAY_RUNTIME_ENV_URI_PIN_EXPIRATION_S_ENV_VAR: '0'}):
        print('URI caching disabled (cache size set to 0).')
        yield

@pytest.fixture(scope='class')
def URI_cache_10_MB():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, {'RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB': '0.01', 'RAY_RUNTIME_ENV_PY_MODULES_CACHE_SIZE_GB': '0.01', RAY_RUNTIME_ENV_URI_PIN_EXPIRATION_S_ENV_VAR: '0'}):
        print('URI cache size set to 0.01 GB.')
        yield

@pytest.fixture(scope='class')
def disable_temporary_uri_pinning():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, {RAY_RUNTIME_ENV_URI_PIN_EXPIRATION_S_ENV_VAR: '0'}):
        print('temporary URI pinning disabled.')
        yield

def check_internal_kv_gced():
    if False:
        while True:
            i = 10
    return len(kv._internal_kv_list('gcs://')) == 0

def get_local_file_whitelist(cluster, option):
    if False:
        while True:
            i = 10
    if sys.platform == 'win32' and option != 'py_modules':
        runtime_dir = Path(cluster.list_all_nodes()[0].get_runtime_env_dir_path()) / 'working_dir_files'
        pkg_dirs = list(Path(runtime_dir).iterdir())
        if pkg_dirs:
            return {pkg_dirs[0].name}
    return {}

class TestGC:

    @pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
    @pytest.mark.parametrize('option', ['working_dir', 'py_modules'])
    @pytest.mark.parametrize('source', [S3_PACKAGE_URI, lazy_fixture('tmp_working_dir')])
    def test_job_level_gc(self, start_cluster, working_dir_and_pymodules_disable_URI_cache, disable_temporary_uri_pinning, option: str, source: str):
        if False:
            for i in range(10):
                print('nop')
        "Tests that job-level working_dir is GC'd when the job exits."
        NUM_NODES = 3
        (cluster, address) = start_cluster
        for i in range(NUM_NODES - 1):
            cluster.add_node(num_cpus=1, runtime_env_dir_name=f'node_{i}_runtime_resources')
            print(f'Added node with runtime_env_dir_name "node_{i}_runtime_resources".')
        print(f'Added all {NUM_NODES} nodes.')
        if option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': source})
            print('Initialized ray with working_dir runtime_env.')
        elif option == 'py_modules':
            if source != S3_PACKAGE_URI:
                source = str(Path(source) / 'test_module')
            ray.init(address, runtime_env={'py_modules': [source, Path(os.path.dirname(__file__)) / 'pip_install_test-0.5-py3-none-any.whl']})
            print('Initialized ray with py_modules runtime_env.')
        if source == S3_PACKAGE_URI and option != 'py_modules':
            assert check_internal_kv_gced()
        else:
            assert not check_internal_kv_gced()
        print(f'kv check 1 passed with source "{source}" and option "{option}".')

        @ray.remote(num_cpus=1)
        class A:

            def test_import(self):
                if False:
                    return 10
                import test_module
                if option == 'py_modules':
                    import pip_install_test
                test_module.one()
        num_cpus = int(ray.available_resources()['CPU'])
        print(f'{num_cpus} cpus available.')
        actors = [A.remote() for _ in range(num_cpus)]
        print(f'Created {len(actors)} actors.')
        ray.get([a.test_import.remote() for a in actors])
        print('Got responses from all actors.')
        if source == S3_PACKAGE_URI and option != 'py_modules':
            assert check_internal_kv_gced()
        else:
            assert not check_internal_kv_gced()
        print(f'kv check 2 passed with source "{source}" and option "{option}".')
        assert not check_local_files_gced(cluster)
        print('check_local_files_gced() check passed.')
        ray.shutdown()
        print('Ray has been shut down.')
        ray.init(address=address)
        print(f'Reconnected to Ray at address "{address}".')
        wait_for_condition(check_internal_kv_gced)
        print('check_internal_kv_gced passed wait_for_condition block.')
        whitelist = get_local_file_whitelist(cluster, option)
        wait_for_condition(lambda : check_local_files_gced(cluster, whitelist=whitelist))
        print('check_local_files_gced passed wait_for_condition block.')

    @pytest.mark.parametrize('option', ['working_dir', 'py_modules'])
    def test_actor_level_gc(self, start_cluster, working_dir_and_pymodules_disable_URI_cache, disable_temporary_uri_pinning, option: str):
        if False:
            return 10
        "Tests that actor-level working_dir is GC'd when the actor exits."
        NUM_NODES = 5
        (cluster, address) = start_cluster
        for i in range(NUM_NODES - 1):
            cluster.add_node(num_cpus=1, runtime_env_dir_name=f'node_{i}_runtime_resources')
            print(f'Added node with runtime_env_dir_name "node_{i}_runtime_resources".')
        print(f'Added all {NUM_NODES} nodes.')
        ray.init(address)
        print(f'Initialized Ray at address "{address}".')

        @ray.remote(num_cpus=1)
        class A:

            def check(self):
                if False:
                    print('Hello World!')
                import test_module
                test_module.one()
        if option == 'working_dir':
            A = A.options(runtime_env={'working_dir': S3_PACKAGE_URI})
        else:
            A = A.options(runtime_env={'py_modules': [S3_PACKAGE_URI]})
        print(f'Created deployment A with option "{option}".')
        num_cpus = int(ray.available_resources()['CPU'])
        print(f'{num_cpus} cpus available.')
        actors = [A.remote() for _ in range(num_cpus)]
        print(f'Created {len(actors)} actors.')
        ray.get([a.check.remote() for a in actors])
        print('Got responses from all actors.')
        for i in range(num_cpus):
            assert not check_local_files_gced(cluster)
            print(f'check_local_files_gced assertion passed for cpu {i}.')
            ray.kill(actors[i])
            print(f'Issued ray.kill for actor {i}.')
        whitelist = get_local_file_whitelist(cluster, option)
        wait_for_condition(lambda : check_local_files_gced(cluster, whitelist))
        print('check_local_files_gced passed wait_for_condition block.')

    @pytest.mark.parametrize('option', ['working_dir', 'py_modules'])
    @pytest.mark.parametrize('source', [S3_PACKAGE_URI, lazy_fixture('tmp_working_dir')])
    def test_detached_actor_gc(self, start_cluster, working_dir_and_pymodules_disable_URI_cache, disable_temporary_uri_pinning, option: str, source: str):
        if False:
            while True:
                i = 10
        "Tests that URIs for detached actors are GC'd only when they exit."
        (cluster, address) = start_cluster
        if option == 'working_dir':
            ray.init(address, namespace='test', runtime_env={'working_dir': source})
        elif option == 'py_modules':
            if source != S3_PACKAGE_URI:
                source = str(Path(source) / 'test_module')
            ray.init(address, namespace='test', runtime_env={'py_modules': [source, Path(os.path.dirname(__file__)) / 'pip_install_test-0.5-py3-none-any.whl']})
        print(f'Initialized Ray with option "{option}".')
        if source == S3_PACKAGE_URI and option != 'py_modules':
            assert check_internal_kv_gced()
        else:
            assert not check_internal_kv_gced()
        print(f'kv check 1 passed with source "{source}" and option "{option}".')

        @ray.remote
        class A:

            def test_import(self):
                if False:
                    print('Hello World!')
                import test_module
                if option == 'py_modules':
                    import pip_install_test
                test_module.one()
        a = A.options(name='test', lifetime='detached').remote()
        print('Created detached actor with name "test".')
        ray.get(a.test_import.remote())
        print('Got response from "test" actor.')
        if source == S3_PACKAGE_URI and option != 'py_modules':
            assert check_internal_kv_gced()
        else:
            assert not check_internal_kv_gced()
        print(f'kv check 2 passed with source "{source}" and option "{option}".')
        assert not check_local_files_gced(cluster)
        print('check_local_files_gced() check passed.')
        ray.shutdown()
        print('Ray has been shut down.')
        ray.init(address, namespace='test')
        print(f'Reconnected to Ray at address "{address}" and namespace "test".')
        if source == S3_PACKAGE_URI and option != 'py_modules':
            assert check_internal_kv_gced()
        else:
            assert not check_internal_kv_gced()
        print(f'kv check 3 passed with source "{source}" and option "{option}".')
        assert not check_local_files_gced(cluster)
        print('check_local_files_gced() check passed.')
        a = ray.get_actor('test')
        print('Got "test" actor.')
        ray.get(a.test_import.remote())
        print('Got response from "test" actor.')
        ray.kill(a)
        print('Issued ray.kill() request to "test" actor.')
        wait_for_condition(check_internal_kv_gced)
        print('check_internal_kv_gced passed wait_for_condition block.')
        whitelist = get_local_file_whitelist(cluster, option)
        wait_for_condition(lambda : check_local_files_gced(cluster, whitelist=whitelist))
        print('check_local_files_gced passed wait_for_condition block.')

    def test_hit_cache_size_limit(self, start_cluster, URI_cache_10_MB, disable_temporary_uri_pinning):
        if False:
            while True:
                i = 10
        'Test eviction happens when we exceed a nonzero (10MB) cache size.'
        NUM_NODES = 3
        (cluster, address) = start_cluster
        for i in range(NUM_NODES - 1):
            cluster.add_node(num_cpus=1, runtime_env_dir_name=f'node_{i}_runtime_resources')
            print(f'Added node with runtime_env_dir_name "node_{i}_runtime_resources".')
        print(f'Added all {NUM_NODES} nodes.')
        with tempfile.TemporaryDirectory() as tmp_dir, chdir(tmp_dir):
            print('Entered tempfile context manager.')
            with open('test_file_1', 'wb') as f:
                f.write(os.urandom(8 * 1024 * 1024))
            print('Wrote random bytes to "test_file_1" file.')
            ray.init(address, runtime_env={'working_dir': tmp_dir})
            print(f'Initialized Ray at "{address}" with working_dir.')

            @ray.remote
            def f():
                if False:
                    while True:
                        i = 10
                pass
            ray.get(f.remote())
            print('Created and received response from task "f".')
            ray.shutdown()
            print('Ray has been shut down.')
            with open('test_file_2', 'wb') as f:
                f.write(os.urandom(4 * 1024 * 1024))
            print('Wrote random bytes to "test_file_2".')
            os.remove('test_file_1')
            print('Removed "test_file_1".')
            ray.init(address, runtime_env={'working_dir': tmp_dir})
            print(f'Reinitialized Ray at address "{address}" with working_dir.')
            for (idx, node) in enumerate(cluster.list_all_nodes()):
                local_dir = os.path.join(node.get_runtime_env_dir_path(), 'working_dir_files')
                print('Created local_dir path.')

                def local_dir_size_near_4mb():
                    if False:
                        for i in range(10):
                            print('nop')
                    return 3 < get_directory_size_bytes(local_dir) / 1024 ** 2 < 5
                wait_for_condition(local_dir_size_near_4mb)
                print(f'get_directory_size_bytes assertion {idx} passed.')

@pytest.fixture(scope='class')
def skip_local_gc():
    if False:
        i = 10
        return i + 15
    with mock.patch.dict(os.environ, {'RAY_RUNTIME_ENV_SKIP_LOCAL_GC': '1'}):
        print('RAY_RUNTIME_ENV_SKIP_LOCAL_GC enabled.')
        yield

@pytest.mark.skip('#23617 must be resolved for skip_local_gc to work.')
class TestSkipLocalGC:

    @pytest.mark.parametrize('source', [lazy_fixture('tmp_working_dir')])
    def test_skip_local_gc_env_var(self, skip_local_gc, start_cluster, working_dir_and_pymodules_disable_URI_cache, disable_temporary_uri_pinning, source):
        if False:
            for i in range(10):
                print('nop')
        (cluster, address) = start_cluster
        ray.init(address, namespace='test', runtime_env={'working_dir': source})

        @ray.remote
        class A:

            def test_import(self):
                if False:
                    print('Hello World!')
                import test_module
                test_module.one()
        a = A.remote()
        ray.get(a.test_import.remote())
        ray.shutdown()
        time.sleep(1)
        assert not check_local_files_gced(cluster)

@pytest.mark.parametrize('expiration_s', [0, TEMP_URI_EXPIRATION_S])
@pytest.mark.parametrize('source', [lazy_fixture('tmp_working_dir')])
def test_pin_runtime_env_uri(start_cluster, source, expiration_s, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Test that temporary GCS URI references are deleted after expiration_s.'
    monkeypatch.setenv(RAY_RUNTIME_ENV_URI_PIN_EXPIRATION_S_ENV_VAR, str(expiration_s))
    (cluster, address) = start_cluster
    start = time.time()
    ray.init(address, namespace='test', runtime_env={'working_dir': source})

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        pass
    ray.get(f.remote())
    ray.shutdown()
    ray.init(address=address)
    time_until_first_check = time.time() - start
    print('Starting Internal KV checks at time ', time_until_first_check)
    assert time_until_first_check < TEMP_URI_EXPIRATION_S, 'URI expired before we could check it. Try bumping the expiration time.'
    if expiration_s > 0:
        assert not check_internal_kv_gced()
        wait_for_condition(check_internal_kv_gced, timeout=4 * expiration_s)
        time_until_gc = time.time() - start
        assert expiration_s < time_until_gc < 4 * expiration_s
        print("Internal KV was GC'ed at time ", time_until_gc)
    else:
        wait_for_condition(check_internal_kv_gced)
        print("Internal KV was GC'ed at time ", time.time() - start)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))