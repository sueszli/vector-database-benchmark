import sys
import os
test_filename = sys.argv[1]
del sys.argv[1]
if test_filename == 'test_urllib2_localnet.py' and os.environ.get('APPVEYOR'):
    os.environ['GEVENT_DEBUG'] = 'TRACE'
print('Running with patch_all(): %s' % (test_filename,))
from gevent import monkey
monkey.patch_all()
from .sysinfo import PY3
from .patched_tests_setup import disable_tests_in_source
from . import support
from . import resources
from . import SkipTest
from . import util

def threading_setup():
    if False:
        for i in range(10):
            print('nop')
    if PY3:
        return (1, ())
    return (1,)

def threading_cleanup(*_args):
    if False:
        return 10
    return
support.threading_setup = threading_setup
support.threading_cleanup = threading_cleanup
import contextlib

@contextlib.contextmanager
def wait_threads_exit(timeout=None):
    if False:
        print('Hello World!')
    yield
support.wait_threads_exit = wait_threads_exit
try:
    from test import support as ts
except ImportError:
    pass
else:
    ts.threading_setup = threading_setup
    ts.threading_cleanup = threading_cleanup
    ts.wait_threads_exit = wait_threads_exit
try:
    from test.support import threading_helper
except ImportError:
    pass
else:
    threading_helper.wait_threads_exit = wait_threads_exit
    threading_helper.threading_setup = threading_setup
    threading_helper.threading_cleanup = threading_cleanup
resources.setup_resources()
if not os.path.exists(test_filename) and os.sep not in test_filename:
    for d in util.find_stdlib_tests():
        if os.path.exists(os.path.join(d, test_filename)):
            os.chdir(d)
            break
__file__ = os.path.join(os.getcwd(), test_filename)
test_name = os.path.splitext(test_filename)[0]
if sys.version_info[0] >= 3:
    module_file = open(test_filename, encoding='utf-8')
else:
    module_file = open(test_filename)
with module_file:
    module_source = module_file.read()
module_source = disable_tests_in_source(module_source, test_name)
import tempfile
(temp_handle, temp_path) = tempfile.mkstemp(prefix=test_name, suffix='.py', text=True)
os.write(temp_handle, module_source.encode('utf-8') if not isinstance(module_source, bytes) else module_source)
os.close(temp_handle)
try:
    module_code = compile(module_source, temp_path, 'exec', dont_inherit=True)
    exec(module_code, globals())
except SkipTest as e:
    print(e)
    print('Ran 0 tests in 0.0s')
    print('OK (skipped=0)')
finally:
    try:
        os.remove(temp_path)
    except OSError:
        pass