import os
import sys
import pytest
import ray

def test_environment_variables_task(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote
    def get_env(key):
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get(key)
    assert ray.get(get_env.options(runtime_env={'env_vars': {'a': 'b'}}).remote('a')) == 'b'

def test_environment_variables_actor(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class EnvGetter:

        def get(self, key):
            if False:
                while True:
                    i = 10
            return os.environ.get(key)
    a = EnvGetter.options(runtime_env={'env_vars': {'a': 'b', 'c': 'd'}}).remote()
    assert ray.get(a.get.remote('a')) == 'b'
    assert ray.get(a.get.remote('c')) == 'd'

def test_environment_variables_nested_task(ray_start_regular):
    if False:
        return 10

    @ray.remote
    def get_env(key):
        if False:
            print('Hello World!')
        print(os.environ)
        return os.environ.get(key)

    @ray.remote
    def get_env_wrapper(key):
        if False:
            while True:
                i = 10
        return ray.get(get_env.remote(key))
    assert ray.get(get_env_wrapper.options(runtime_env={'env_vars': {'a': 'b'}}).remote('a')) == 'b'

def test_environment_variables_multitenancy(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(runtime_env={'env_vars': {'foo1': 'bar1', 'foo2': 'bar2'}})

    @ray.remote
    def get_env(key):
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get(key)
    assert ray.get(get_env.remote('foo1')) == 'bar1'
    assert ray.get(get_env.remote('foo2')) == 'bar2'
    assert ray.get(get_env.options(runtime_env={'env_vars': {'foo1': 'baz1'}}).remote('foo1')) == 'baz1'
    assert ray.get(get_env.options(runtime_env={'env_vars': {'foo1': 'baz1'}}).remote('foo2')) == 'bar2'

def test_environment_variables_complex(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(runtime_env={'env_vars': {'a': 'job_a', 'b': 'job_b', 'z': 'job_z'}})

    @ray.remote
    def get_env(key):
        if False:
            return 10
        return os.environ.get(key)

    @ray.remote
    class NestedEnvGetter:

        def get(self, key):
            if False:
                i = 10
                return i + 15
            return os.environ.get(key)

        def get_task(self, key):
            if False:
                print('Hello World!')
            return ray.get(get_env.remote(key))

    @ray.remote
    class EnvGetter:

        def get(self, key):
            if False:
                while True:
                    i = 10
            return os.environ.get(key)

        def get_task(self, key):
            if False:
                for i in range(10):
                    print('nop')
            return ray.get(get_env.remote(key))

        def nested_get(self, key):
            if False:
                for i in range(10):
                    print('nop')
            aa = NestedEnvGetter.options(runtime_env={'env_vars': {'c': 'e', 'd': 'dd'}}).remote()
            return ray.get(aa.get.remote(key))
    a = EnvGetter.options(runtime_env={'env_vars': {'a': 'b', 'c': 'd'}}).remote()
    assert ray.get(a.get.remote('a')) == 'b'
    assert ray.get(a.get_task.remote('a')) == 'b'
    assert ray.get(a.nested_get.remote('a')) == 'b'
    assert ray.get(a.nested_get.remote('c')) == 'e'
    assert ray.get(a.nested_get.remote('d')) == 'dd'
    assert ray.get(get_env.options(runtime_env={'env_vars': {'a': 'b'}}).remote('a')) == 'b'
    assert ray.get(a.get.remote('z')) == 'job_z'
    assert ray.get(a.get_task.remote('z')) == 'job_z'
    assert ray.get(a.nested_get.remote('z')) == 'job_z'
    assert ray.get(get_env.options(runtime_env={'env_vars': {'a': 'b'}}).remote('z')) == 'job_z'

def test_environment_variables_reuse(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    "Test that new tasks don't incorrectly reuse previous environments."
    ray.init()
    env_var_name = 'TEST123'
    val1 = 'VAL1'
    val2 = 'VAL2'
    assert os.environ.get(env_var_name) is None

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        return os.environ.get(env_var_name)

    @ray.remote
    def g():
        if False:
            while True:
                i = 10
        return os.environ.get(env_var_name)
    assert ray.get(f.remote()) is None
    assert ray.get(f.options(runtime_env={'env_vars': {env_var_name: val1}}).remote()) == val1
    assert ray.get(f.remote()) is None
    assert ray.get(g.remote()) is None
    assert ray.get(f.options(runtime_env={'env_vars': {env_var_name: val2}}).remote()) == val2
    assert ray.get(g.remote()) is None
    assert ray.get(f.remote()) is None

@pytest.mark.skipif(sys.platform == 'darwin', reason='Flaky on Travis CI.')
def test_environment_variables_env_caching(shutdown_only):
    if False:
        print('Hello World!')
    'Test that workers with specified envs are cached and reused.\n\n    When a new task or actor is created with a new runtime env, a\n    new worker process is started.  If a subsequent task or actor\n    uses the same runtime env, the same worker process should be\n    used.  This function checks the pid of the worker to test this.\n    '
    ray.init()
    env_var_name = 'TEST123'
    val1 = 'VAL1'
    val2 = 'VAL2'
    assert os.environ.get(env_var_name) is None

    def task():
        if False:
            return 10
        return (os.environ.get(env_var_name), os.getpid())

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        return task()

    @ray.remote
    def g():
        if False:
            for i in range(10):
                print('nop')
        return task()

    def get_options(val):
        if False:
            return 10
        return {'runtime_env': {'env_vars': {env_var_name: val}}}
    assert ray.get(f.remote())[0] is None
    (ret_val1, pid1) = ray.get(f.options(**get_options(val1)).remote())
    assert ret_val1 == val1
    (ret_val2, pid2) = ray.get(g.options(**get_options(val2)).remote())
    assert ret_val2 == val2
    assert pid1 != pid2
    (_, pid3) = ray.get(g.remote())
    assert pid2 != pid3
    (_, pid4) = ray.get(g.options(**get_options(val2)).remote())
    assert pid4 == pid2
    (_, pid5) = ray.get(f.options(**get_options(val2)).remote())
    assert pid5 != pid1
    (_, pid6) = ray.get(f.options(**get_options(val1)).remote())
    assert pid6 == pid1
    (_, pid7) = ray.get(g.options(**get_options(val1)).remote())
    assert pid7 == pid1

def test_appendable_environ(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def get_env(key):
        if False:
            i = 10
            return i + 15
        return os.environ.get(key)
    custom_env = os.path.pathsep + '/usr/local/bin'
    remote_env = ray.get(get_env.options(runtime_env={'env_vars': {'PATH': '${PATH}' + custom_env}}).remote('PATH'))
    assert remote_env.endswith(custom_env)
    assert len(remote_env) > len(custom_env)
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))