import os
import pytest
import unittest
import ray
from ray._private.utils import update_envs

@pytest.mark.skipif("sys.platform != 'linux'")
def test_environ_file_on_linux(ray_start_10_cpus):
    if False:
        return 10

    @ray.remote
    class Actor1:

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

        def get_env_from_proc(self):
            if False:
                i = 10
                return i + 15
            pid = os.getpid()
            env = {}
            with open('/proc/%s/environ' % pid) as fd:
                for envspec in fd.read().split('\x00'):
                    if not envspec:
                        continue
                    (varname, varval) = envspec.split('=', 1)
                    env[varname] = varval
            return env

        def get_os_environ(self):
            if False:
                i = 10
                return i + 15
            return os.environ
    a = Actor1.remote()
    actor_proc_environ = ray.get(a.get_env_from_proc.remote())
    actor_os_environ = ray.get(a.get_os_environ.remote())
    assert len(actor_proc_environ) > 0
    assert len(actor_os_environ) > 0

def test_update_envs():
    if False:
        for i in range(10):
            print('nop')
    with unittest.mock.patch.dict(os.environ):
        env_vars = {'PATH': '/test/lib/path:${PATH}', 'LD_LIBRARY_PATH': '/test/path1:${LD_LIBRARY_PATH}:./test/path2', 'DYLD_LIBRARY_PATH': '${DYLD_LIBRARY_PATH}:/test/path', 'LD_PRELOAD': ''}
        old_path = os.environ['PATH']
        os.environ['LD_LIBRARY_PATH'] = './'
        os.environ['DYLD_LIBRARY_PATH'] = '/lib64'
        os.environ['LD_PRELOAD'] = '/lib:/usr/local/lib'
        update_envs(env_vars)
        assert os.environ['PATH'] == '/test/lib/path:' + old_path
        assert os.environ['LD_LIBRARY_PATH'] == '/test/path1:./:./test/path2'
        assert os.environ['DYLD_LIBRARY_PATH'] == '/lib64:/test/path'
        assert os.environ['LD_PRELOAD'] == env_vars['LD_PRELOAD']
        os.environ['LD_LIBRARY_PATH'] = ''
        del os.environ['DYLD_LIBRARY_PATH']
        del os.environ['LD_PRELOAD']
        update_envs(env_vars)
        assert os.environ['LD_LIBRARY_PATH'] == '/test/path1::./test/path2'
        assert os.environ['DYLD_LIBRARY_PATH'] == ':/test/path'
        assert os.environ['LD_PRELOAD'] == env_vars['LD_PRELOAD']
if __name__ == '__main__':
    import pytest
    import sys
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))