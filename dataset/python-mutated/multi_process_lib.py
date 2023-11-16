"""Library for multi-process testing."""
import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test

def is_oss():
    if False:
        while True:
            i = 10
    'Returns whether the test is run under OSS.'
    return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]

def _is_enabled():
    if False:
        print('Hello World!')
    tpu_args = [arg for arg in sys.argv if arg.startswith('--tpu')]
    if is_oss() and tpu_args:
        return False
    if sys.version_info == (3, 8) and platform.system() == 'Linux':
        return False
    return sys.platform != 'win32'

class _AbslProcess:
    """A process that runs using absl.app.run."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(_AbslProcess, self).__init__(*args, **kwargs)
        self._run_impl = getattr(self, 'run')
        self.run = self._run_with_absl

    def _run_with_absl(self):
        if False:
            for i in range(10):
                print('nop')
        app.run(lambda _: self._run_impl())
if _is_enabled():

    class AbslForkServerProcess(_AbslProcess, multiprocessing.context.ForkServerProcess):
        """An absl-compatible Forkserver process.

    Note: Forkserver is not available in windows.
    """

    class AbslForkServerContext(multiprocessing.context.ForkServerContext):
        _name = 'absl_forkserver'
        Process = AbslForkServerProcess
    multiprocessing = AbslForkServerContext()
    Process = multiprocessing.Process
else:

    class Process(object):
        """A process that skips test (until windows is supported)."""

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            del args, kwargs
            raise unittest.SkipTest('TODO(b/150264776): Windows is not supported in MultiProcessRunner.')
_test_main_called = False

def _set_spawn_exe_path():
    if False:
        for i in range(10):
            print('nop')
    "Set the path to the executable for spawned processes.\n\n  This utility searches for the binary the parent process is using, and sets\n  the executable of multiprocessing's context accordingly.\n\n  Raises:\n    RuntimeError: If the binary path cannot be determined.\n  "
    if sys.argv[0].endswith('.py'):

        def guess_path(package_root):
            if False:
                print('Hello World!')
            if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
                package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
                binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
                possible_path = os.path.join(package_root_base, package_root, binary)
                logging.info('Guessed test binary path: %s', possible_path)
                if os.access(possible_path, os.X_OK):
                    return possible_path
                return None
        path = guess_path('org_tensorflow')
        if not path:
            path = guess_path('org_keras')
        if path is None:
            logging.error('Cannot determine binary path. sys.argv[0]=%s os.environ=%s', sys.argv[0], os.environ)
            raise RuntimeError('Cannot determine binary path')
        sys.argv[0] = path
    multiprocessing.get_context().set_executable(sys.argv[0])

def _if_spawn_run_and_exit():
    if False:
        for i in range(10):
            print('nop')
    'If spawned process, run requested spawn task and exit. Else a no-op.'
    is_spawned = '-c' in sys.argv[1:] and sys.argv[sys.argv.index('-c') + 1].startswith('from multiprocessing.')
    if not is_spawned:
        return
    cmd = sys.argv[sys.argv.index('-c') + 1]
    sys.argv = sys.argv[0:1]
    exec(cmd)
    sys.exit(0)

def test_main():
    if False:
        while True:
            i = 10
    'Main function to be called within `__main__` of a test file.'
    global _test_main_called
    _test_main_called = True
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if _is_enabled():
        _set_spawn_exe_path()
        _if_spawn_run_and_exit()
    test.main()

def initialized():
    if False:
        i = 10
        return i + 15
    'Returns whether the module is initialized.'
    return _test_main_called