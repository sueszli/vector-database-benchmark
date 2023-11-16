import sys
import os
import pytest
import ray
from ray._private.test_utils import run_string_as_driver_nonblocking

def test_simple(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=1)

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                return 10
            return 'ok'

        def pid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()
    for ns in [None, 'test']:
        a = Actor.options(name='x', namespace=ns, get_if_exists=True).remote()
        b = Actor.options(name='x', namespace=ns, get_if_exists=True).remote()
        assert ray.get(a.ping.remote()) == 'ok'
        assert ray.get(b.ping.remote()) == 'ok'
        assert ray.get(b.pid.remote()) == ray.get(a.pid.remote())
    with pytest.raises(TypeError):
        Actor.options(name=object(), get_if_exists=True).remote()
    with pytest.raises(TypeError):
        Actor.options(name='x', namespace=object(), get_if_exists=True).remote()
    with pytest.raises(ValueError):
        Actor.options(num_cpus=1, get_if_exists=True).remote()

def test_shared_actor(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=1)

    @ray.remote(name='x', namespace='test', get_if_exists=True)
    class SharedActor:

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'ok'

        def pid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()
    a = SharedActor.remote()
    b = SharedActor.remote()
    assert ray.get(a.ping.remote()) == 'ok'
    assert ray.get(b.ping.remote()) == 'ok'
    assert ray.get(b.pid.remote()) == ray.get(a.pid.remote())

def test_no_verbose_output():
    if False:
        while True:
            i = 10
    script = '\nimport ray\n\n@ray.remote\nclass Actor:\n    def ping(self):\n        return "ok"\n\n\n@ray.remote\ndef getter(name):\n    actor = Actor.options(\n        name="foo", lifetime="detached", namespace="n", get_if_exists=True).remote()\n    ray.get(actor.ping.remote())\n\n\ndef do_run(name):\n    name = "actor_" + str(name)\n    tasks = [getter.remote(name) for i in range(4)]\n    ray.get(tasks)\n    try:\n        ray.kill(ray.get_actor(name, namespace="n"))  # Cleanup\n    except:\n        pass\n\n\nfor i in range(100):\n    do_run(i)\n\nprint("DONE")\n'
    proc = run_string_as_driver_nonblocking(script)
    out_str = proc.stdout.read().decode('ascii') + proc.stderr.read().decode('ascii')
    out = []
    for line in out_str.split('\n'):
        if 'local Ray instance' not in line and 'The object store' not in line:
            out.append(line)
    valid = ''.join(out)
    assert 'DONE' in valid, out_str
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))