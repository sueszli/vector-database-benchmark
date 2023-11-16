import threading
import os
import sys
import logging
import pytest
import ray

def test_setup_func_basic(shutdown_only):
    if False:
        while True:
            i = 10

    def configure_logging(level: int):
        if False:
            return 10
        logger = logging.getLogger('')
        logger.setLevel(level)
    ray.init(num_cpus=1, runtime_env={'worker_process_setup_hook': lambda : configure_logging(logging.DEBUG), 'env_vars': {'ABC': '123'}})

    @ray.remote
    def f(level):
        if False:
            print('Hello World!')
        logger = logging.getLogger('')
        assert logging.getLevelName(logger.getEffectiveLevel()) == level
        return True

    @ray.remote
    class Actor:

        def __init__(self, level):
            if False:
                for i in range(10):
                    print('nop')
            logger = logging.getLogger('')
            assert logging.getLevelName(logger.getEffectiveLevel()) == level

        def ready(self):
            if False:
                while True:
                    i = 10
            return True

        def get_env_var(self, key):
            if False:
                i = 10
                return i + 15
            return os.getenv(key)
    for _ in range(10):
        assert ray.get(f.remote('DEBUG'))
    a = Actor.remote('DEBUG')
    assert ray.get(a.__ray_ready__.remote())
    assert ray.get(a.get_env_var.remote('ABC')) == '123'

def test_setup_func_failure(shutdown_only):
    if False:
        i = 10
        return i + 15
    '\n    Verify when deserilization failed, it raises an exception.\n    '

    class CustomClass:
        """
        Custom class that can serialize but canont deserialize.
        It is used to test deserialization failure.
        """

        def __getstate__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.__dict__

        def __setstate__(self, state):
            if False:
                return 10
            raise RuntimeError('Deserialization not allowed')
    c = CustomClass()

    def setup():
        if False:
            return 10
        print(c)
    ray.init(num_cpus=1, runtime_env={'worker_process_setup_hook': setup})

    @ray.remote
    class A:
        pass
    a = A.remote()
    with pytest.raises(ray.exceptions.RayActorError) as e:
        ray.get(a.__ray_ready__.remote())
    assert 'Deserialization not allowed' in str(e.value)
    '\n    Verify when the serialization fails, ray.init fails.\n    '
    ray.shutdown()
    lock = threading.Lock()
    with pytest.raises(ray.exceptions.RuntimeEnvSetupError) as e:
        ray.init(num_cpus=0, runtime_env={'worker_process_setup_hook': lambda : print(lock)})
    assert 'Failed to export the setup function.' in str(e.value)
    '\n    Verify when the setup hook failed, it raises an exception.\n    '
    ray.shutdown()

    def setup_func():
        if False:
            i = 10
            return i + 15
        raise ValueError('Setup Failed')
    ray.init(num_cpus=1, runtime_env={'worker_process_setup_hook': setup_func})

    @ray.remote
    class A:
        pass
    a = A.remote()
    with pytest.raises(ray.exceptions.RayActorError) as e:
        ray.get(a.__ray_ready__.remote())
    assert 'Setup Failed' in str(e.value)
    assert 'Failed to execute the setup hook method.' in str(e.value)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))