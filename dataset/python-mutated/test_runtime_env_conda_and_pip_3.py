import os
import pytest
import sys
import time
from ray._private.test_utils import wait_for_condition, check_local_files_gced, generate_runtime_env_dict
import ray
from unittest import mock
if not os.environ.get('CI'):
    os.environ['RAY_RUNTIME_ENV_LOCAL_DEV_MODE'] = '1'

class TestGC:

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Needs PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('field', ['conda', 'pip'])
    @pytest.mark.parametrize('spec_format', ['file', 'python_object'])
    def test_actor_level_gc(self, runtime_env_disable_URI_cache, start_cluster, field, spec_format, tmp_path):
        if False:
            print('Hello World!')
        "Tests that actor-level working_dir is GC'd when the actor exits."
        (cluster, address) = start_cluster
        ray.init(address)
        runtime_env = generate_runtime_env_dict(field, spec_format, tmp_path)

        @ray.remote
        class A:

            def test_import(self):
                if False:
                    return 10
                import pip_install_test
                return True
        NUM_ACTORS = 5
        actors = [A.options(runtime_env=runtime_env).remote() for _ in range(NUM_ACTORS)]
        ray.get([a.test_import.remote() for a in actors])
        for i in range(5):
            assert not check_local_files_gced(cluster)
            ray.kill(actors[i])
        wait_for_condition(lambda : check_local_files_gced(cluster), timeout=30)

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Needs PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('ray_start_cluster', [{'num_nodes': 1, '_system_config': {'num_workers_soft_limit': 0}}, {'num_nodes': 1, '_system_config': {'num_workers_soft_limit': 5}}], indirect=True)
    @pytest.mark.parametrize('field', ['conda', 'pip'])
    @pytest.mark.parametrize('spec_format', ['file', 'python_object'])
    def test_task_level_gc(self, runtime_env_disable_URI_cache, ray_start_cluster, field, spec_format, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        "Tests that task-level working_dir is GC'd when the task exits."
        cluster = ray_start_cluster
        soft_limit_zero = False
        system_config = cluster.list_all_nodes()[0]._ray_params._system_config
        if 'num_workers_soft_limit' in system_config and system_config['num_workers_soft_limit'] == 0:
            soft_limit_zero = True
        runtime_env = generate_runtime_env_dict(field, spec_format, tmp_path)

        @ray.remote
        def f():
            if False:
                for i in range(10):
                    print('nop')
            import pip_install_test
            return True

        @ray.remote
        class A:

            def test_import(self):
                if False:
                    i = 10
                    return i + 15
                import pip_install_test
                return True
        ray.get(f.options(runtime_env=runtime_env).remote())
        if soft_limit_zero:
            wait_for_condition(lambda : check_local_files_gced(cluster))
        else:
            assert not check_local_files_gced(cluster)
        actor = A.options(runtime_env=runtime_env).remote()
        ray.get(actor.test_import.remote())
        assert not check_local_files_gced(cluster)
        ray.kill(actor)
        if soft_limit_zero:
            wait_for_condition(lambda : check_local_files_gced(cluster))
        else:
            assert not check_local_files_gced(cluster)
        ray.get(f.options(runtime_env=runtime_env).remote())
        if soft_limit_zero:
            wait_for_condition(lambda : check_local_files_gced(cluster))
        else:
            assert not check_local_files_gced(cluster)

@pytest.fixture(scope='class')
def skip_local_gc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.dict(os.environ, {'RAY_RUNTIME_ENV_SKIP_LOCAL_GC': '1'}):
        print('RAY_RUNTIME_ENV_SKIP_LOCAL_GC enabled.')
        yield

class TestSkipLocalGC:

    @pytest.mark.skipif(os.environ.get('CI') and sys.platform != 'linux', reason='Requires PR wheels built in CI, so only run on linux CI machines.')
    @pytest.mark.parametrize('field', ['conda', 'pip'])
    def test_skip_local_gc_env_var(self, skip_local_gc, start_cluster, field, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        (cluster, address) = start_cluster
        runtime_env = generate_runtime_env_dict(field, 'python_object', tmp_path)
        ray.init(address, namespace='test', runtime_env=runtime_env)

        @ray.remote
        def f():
            if False:
                print('Hello World!')
            import pip_install_test
            return True
        assert ray.get(f.remote())
        ray.shutdown()
        time.sleep(10)
        assert not check_local_files_gced(cluster)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))