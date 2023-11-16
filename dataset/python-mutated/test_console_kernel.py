"""
Tests for the console kernel.
"""
import ast
import asyncio
import os
import os.path as osp
from textwrap import dedent
from contextlib import contextmanager
import time
from subprocess import Popen, PIPE
import sys
import inspect
import uuid
from collections import namedtuple
import pytest
from flaky import flaky
from jupyter_core import paths
from jupyter_client import BlockingKernelClient
import numpy as np
from spyder_kernels.utils.iofuncs import iofunctions
from spyder_kernels.utils.test_utils import get_kernel, get_log_text
from spyder_kernels.customize.spyderpdb import SpyderPdb
from spyder_kernels.comms.commbase import CommBase
FILES_PATH = os.path.dirname(os.path.realpath(__file__))
TIMEOUT = 15
SETUP_TIMEOUT = 60
TURTLE_ACTIVE = False
try:
    import turtle
    turtle.Screen()
    turtle.bye()
    TURTLE_ACTIVE = True
except:
    pass

@contextmanager
def setup_kernel(cmd):
    if False:
        while True:
            i = 10
    'start an embedded kernel in a subprocess, and wait for it to be ready\n\n    This function was taken from the ipykernel project.\n    We plan to remove it.\n\n    Yields\n    -------\n    client: jupyter_client.BlockingKernelClient connected to the kernel\n    '
    kernel = Popen([sys.executable, '-c', cmd], stdout=PIPE, stderr=PIPE)
    try:
        connection_file = os.path.join(paths.jupyter_runtime_dir(), 'kernel-%i.json' % kernel.pid)
        tic = time.time()
        while not os.path.exists(connection_file) and kernel.poll() is None and (time.time() < tic + SETUP_TIMEOUT):
            time.sleep(0.1)
        if kernel.poll() is not None:
            (o, e) = kernel.communicate()
            raise IOError('Kernel failed to start:\n%s' % e)
        if not os.path.exists(connection_file):
            if kernel.poll() is None:
                kernel.terminate()
            raise IOError('Connection file %r never arrived' % connection_file)
        client = BlockingKernelClient(connection_file=connection_file)
        tic = time.time()
        while True:
            try:
                client.load_connection_file()
                break
            except ValueError:
                if time.time() > tic + SETUP_TIMEOUT:
                    raise IOError('Kernel failed to write connection file')
        client.start_channels()
        client.wait_for_ready()
        try:
            yield client
        finally:
            client.stop_channels()
    finally:
        kernel.terminate()

class Comm:
    """
    Comm base class, copied from qtconsole without the qt stuff
    """

    def __init__(self, target_name, kernel_client, msg_callback=None, close_callback=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a new comm. Must call open to use.\n        '
        self.target_name = target_name
        self.kernel_client = kernel_client
        self.comm_id = uuid.uuid1().hex
        self._msg_callback = msg_callback
        self._close_callback = close_callback
        self._send_channel = self.kernel_client.shell_channel

    def _send_msg(self, msg_type, content, data, metadata, buffers):
        if False:
            i = 10
            return i + 15
        '\n        Send a message on the shell channel.\n        '
        if data is None:
            data = {}
        if content is None:
            content = {}
        content['comm_id'] = self.comm_id
        content['data'] = data
        msg = self.kernel_client.session.msg(msg_type, content, metadata=metadata)
        if buffers:
            msg['buffers'] = buffers
        return self._send_channel.send(msg)

    def open(self, data=None, metadata=None, buffers=None):
        if False:
            for i in range(10):
                print('nop')
        'Open the kernel-side version of this comm'
        return self._send_msg('comm_open', {'target_name': self.target_name}, data, metadata, buffers)

    def send(self, data=None, metadata=None, buffers=None):
        if False:
            print('Hello World!')
        'Send a message to the kernel-side version of this comm'
        return self._send_msg('comm_msg', {}, data, metadata, buffers)

    def close(self, data=None, metadata=None, buffers=None):
        if False:
            print('Hello World!')
        'Close the kernel-side version of this comm'
        return self._send_msg('comm_close', {}, data, metadata, buffers)

    def on_msg(self, callback):
        if False:
            print('Hello World!')
        'Register a callback for comm_msg\n\n        Will be called with the `data` of any comm_msg messages.\n\n        Call `on_msg(None)` to disable an existing callback.\n        '
        self._msg_callback = callback

    def on_close(self, callback):
        if False:
            while True:
                i = 10
        'Register a callback for comm_close\n\n        Will be called with the `data` of the close message.\n\n        Call `on_close(None)` to disable an existing callback.\n        '
        self._close_callback = callback

    def handle_msg(self, msg):
        if False:
            return 10
        'Handle a comm_msg message'
        if self._msg_callback:
            return self._msg_callback(msg)

    def handle_close(self, msg):
        if False:
            i = 10
            return i + 15
        'Handle a comm_close message'
        if self._close_callback:
            return self._close_callback(msg)

@pytest.fixture
def kernel(request):
    if False:
        i = 10
        return i + 15
    'Console kernel fixture'
    kernel = get_kernel()
    kernel.namespace_view_settings = {'check_all': False, 'exclude_private': True, 'exclude_uppercase': True, 'exclude_capitalized': False, 'exclude_unsupported': False, 'exclude_callables_and_modules': True, 'excluded_names': ['nan', 'inf', 'infty', 'little_endian', 'colorbar_doc', 'typecodes', '__builtins__', '__main__', '__doc__', 'NaN', 'Inf', 'Infinity', 'sctypes', 'rcParams', 'rcParamsDefault', 'sctypeNA', 'typeNA', 'False_', 'True_'], 'minmax': False, 'filter_on': True}

    def reset_kernel():
        if False:
            for i in range(10):
                print('nop')
        asyncio.run(kernel.do_execute('reset -f', True))
    request.addfinalizer(reset_kernel)
    return kernel

def test_magics(kernel):
    if False:
        print('Hello World!')
    'Check available magics in the kernel.'
    line_magics = kernel.shell.magics_manager.magics['line']
    cell_magics = kernel.shell.magics_manager.magics['cell']
    for magic in ['alias', 'alias_magic', 'autocall', 'automagic', 'autosave', 'bookmark', 'cd', 'clear', 'colors', 'config', 'connect_info', 'debug', 'dhist', 'dirs', 'doctest_mode', 'ed', 'edit', 'env', 'gui', 'hist', 'history', 'killbgscripts', 'ldir', 'less', 'load', 'load_ext', 'loadpy', 'logoff', 'logon', 'logstart', 'logstate', 'logstop', 'ls', 'lsmagic', 'macro', 'magic', 'matplotlib', 'mkdir', 'more', 'notebook', 'page', 'pastebin', 'pdb', 'pdef', 'pdoc', 'pfile', 'pinfo', 'pinfo2', 'popd', 'pprint', 'precision', 'prun', 'psearch', 'psource', 'pushd', 'pwd', 'pycat', 'pylab', 'qtconsole', 'quickref', 'recall', 'rehashx', 'reload_ext', 'rep', 'rerun', 'reset', 'reset_selective', 'rmdir', 'run', 'save', 'sc', 'set_env', 'sx', 'system', 'tb', 'time', 'timeit', 'unalias', 'unload_ext', 'who', 'who_ls', 'whos', 'xdel', 'xmode']:
        msg = "magic '%s' is not in line_magics" % magic
        assert magic in line_magics, msg
    for magic in ['!', 'HTML', 'SVG', 'bash', 'capture', 'debug', 'file', 'html', 'javascript', 'js', 'latex', 'perl', 'prun', 'pypy', 'python', 'python2', 'python3', 'ruby', 'script', 'sh', 'svg', 'sx', 'system', 'time', 'timeit', 'writefile']:
        assert magic in cell_magics

def test_get_namespace_view(kernel):
    if False:
        return 10
    '\n    Test the namespace view of the kernel.\n    '
    execute = asyncio.run(kernel.do_execute('a = 1', True))
    nsview = repr(kernel.get_namespace_view())
    assert "'a':" in nsview
    assert "'type': 'int'" in nsview or "'type': u'int'" in nsview
    assert "'size': 1" in nsview
    assert "'view': '1'" in nsview
    assert "'numpy_type': 'Unknown'" in nsview
    assert "'python_type': 'int'" in nsview

@pytest.mark.parametrize('filter_on', [True, False])
def test_get_namespace_view_filter_on(kernel, filter_on):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the namespace view of the kernel with filters on and off.\n    '
    execute = asyncio.run(kernel.do_execute('a = 1', True))
    asyncio.run(kernel.do_execute('TestFilterOff = 1', True))
    settings = kernel.namespace_view_settings
    settings['filter_on'] = filter_on
    settings['exclude_capitalized'] = True
    nsview = kernel.get_namespace_view()
    if not filter_on:
        assert 'a' in nsview
        assert 'TestFilterOff' in nsview
    else:
        assert 'TestFilterOff' not in nsview
        assert 'a' in nsview
    settings['filter_on'] = True
    settings['exclude_capitalized'] = False

def test_get_var_properties(kernel):
    if False:
        i = 10
        return i + 15
    '\n    Test the properties fo the variables in the namespace.\n    '
    asyncio.run(kernel.do_execute('a = 1', True))
    var_properties = repr(kernel.get_var_properties())
    assert "'a'" in var_properties
    assert "'is_list': False" in var_properties
    assert "'is_dict': False" in var_properties
    assert "'len': 1" in var_properties
    assert "'is_array': False" in var_properties
    assert "'is_image': False" in var_properties
    assert "'is_data_frame': False" in var_properties
    assert "'is_series': False" in var_properties
    assert "'array_shape': None" in var_properties
    assert "'array_ndim': None" in var_properties

def test_get_value(kernel):
    if False:
        return 10
    'Test getting the value of a variable.'
    name = 'a'
    asyncio.run(kernel.do_execute('a = 124', True))
    assert kernel.get_value(name) == 124

def test_set_value(kernel):
    if False:
        while True:
            i = 10
    'Test setting the value of a variable.'
    name = 'a'
    asyncio.run(kernel.do_execute('a = 0', True))
    value = 10
    kernel.set_value(name, value)
    log_text = get_log_text(kernel)
    assert "'__builtin__': <module " in log_text
    assert "'__builtins__': <module " in log_text
    assert "'_ih': ['']" in log_text
    assert "'_oh': {}" in log_text
    assert "'a': 10" in log_text

def test_remove_value(kernel):
    if False:
        i = 10
        return i + 15
    'Test the removal of a variable.'
    name = 'a'
    asyncio.run(kernel.do_execute('a = 1', True))
    var_properties = repr(kernel.get_var_properties())
    assert "'a'" in var_properties
    assert "'is_list': False" in var_properties
    assert "'is_dict': False" in var_properties
    assert "'len': 1" in var_properties
    assert "'is_array': False" in var_properties
    assert "'is_image': False" in var_properties
    assert "'is_data_frame': False" in var_properties
    assert "'is_series': False" in var_properties
    assert "'array_shape': None" in var_properties
    assert "'array_ndim': None" in var_properties
    kernel.remove_value(name)
    var_properties = repr(kernel.get_var_properties())
    assert var_properties == '{}'

def test_copy_value(kernel):
    if False:
        i = 10
        return i + 15
    'Test the copy of a variable.'
    orig_name = 'a'
    new_name = 'b'
    asyncio.run(kernel.do_execute('a = 1', True))
    var_properties = repr(kernel.get_var_properties())
    assert "'a'" in var_properties
    assert "'is_list': False" in var_properties
    assert "'is_dict': False" in var_properties
    assert "'len': 1" in var_properties
    assert "'is_array': False" in var_properties
    assert "'is_image': False" in var_properties
    assert "'is_data_frame': False" in var_properties
    assert "'is_series': False" in var_properties
    assert "'array_shape': None" in var_properties
    assert "'array_ndim': None" in var_properties
    kernel.copy_value(orig_name, new_name)
    var_properties = repr(kernel.get_var_properties())
    assert "'a'" in var_properties
    assert "'b'" in var_properties
    assert "'is_list': False" in var_properties
    assert "'is_dict': False" in var_properties
    assert "'len': 1" in var_properties
    assert "'is_array': False" in var_properties
    assert "'is_image': False" in var_properties
    assert "'is_data_frame': False" in var_properties
    assert "'is_series': False" in var_properties
    assert "'array_shape': None" in var_properties
    assert "'array_ndim': None" in var_properties

@pytest.mark.parametrize('load', [(True, 'val1 = 0', {'val1': np.array(1)}), (False, 'val1 = 0', {'val1': 0, 'val1_000': np.array(1)})])
def test_load_npz_data(kernel, load):
    if False:
        for i in range(10):
            print('nop')
    'Test loading data from npz filename.'
    namespace_file = osp.join(FILES_PATH, 'load_data.npz')
    extention = '.npz'
    (overwrite, execute, variables) = load
    asyncio.run(kernel.do_execute(execute, True))
    kernel.load_data(namespace_file, extention, overwrite=overwrite)
    for (var, value) in variables.items():
        assert value == kernel.get_value(var)

def test_load_data(kernel):
    if False:
        while True:
            i = 10
    'Test loading data from filename.'
    namespace_file = osp.join(FILES_PATH, 'load_data.spydata')
    extention = '.spydata'
    kernel.load_data(namespace_file, extention)
    var_properties = repr(kernel.get_var_properties())
    assert "'a'" in var_properties
    assert "'is_list': False" in var_properties
    assert "'is_dict': False" in var_properties
    assert "'len': 1" in var_properties
    assert "'is_array': False" in var_properties
    assert "'is_image': False" in var_properties
    assert "'is_data_frame': False" in var_properties
    assert "'is_series': False" in var_properties
    assert "'array_shape': None" in var_properties
    assert "'array_ndim': None" in var_properties

def test_save_namespace(kernel):
    if False:
        print('Hello World!')
    'Test saving the namespace into filename.'
    namespace_file = osp.join(FILES_PATH, 'save_data.spydata')
    asyncio.run(kernel.do_execute('b = 1', True))
    kernel.save_namespace(namespace_file)
    assert osp.isfile(namespace_file)
    load_func = iofunctions.load_funcs['.spydata']
    (data, error_message) = load_func(namespace_file)
    assert data == {'b': 1}
    assert not error_message
    os.remove(namespace_file)
    assert not osp.isfile(namespace_file)

def test_is_defined(kernel):
    if False:
        i = 10
        return i + 15
    'Test method to tell if object is defined.'
    obj = 'debug'
    assert kernel.is_defined(obj)

def test_get_doc(kernel):
    if False:
        while True:
            i = 10
    'Test to get object documentation dictionary.'
    objtxt = 'help'
    assert "Define the builtin 'help'" in kernel.get_doc(objtxt)['docstring'] or "Define the built-in 'help'" in kernel.get_doc(objtxt)['docstring']

def test_get_source(kernel):
    if False:
        return 10
    'Test to get object source.'
    objtxt = 'help'
    assert 'class _Helper' in kernel.get_source(objtxt)

@pytest.mark.skipif(os.name == 'nt', reason="Doesn't work on Windows")
def test_output_from_c_libraries(kernel, capsys):
    if False:
        i = 10
        return i + 15
    'Test that the wurlitzer extension is working.'
    code = "\nimport ctypes\nlibc = ctypes.CDLL(None)\nlibc.printf(('Hello from C\\n').encode('utf8'))\n"
    kernel._load_wurlitzer()
    asyncio.run(kernel.do_execute(code, True))
    captured = capsys.readouterr()
    assert captured.out == 'Hello from C\n'

@flaky(max_runs=3)
def test_cwd_in_sys_path():
    if False:
        print('Hello World!')
    '\n    Test that cwd stays as the first element in sys.path after the\n    kernel has started.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        reply = client.execute_interactive('import sys; sys_path = sys.path', user_expressions={'output': 'sys_path'}, timeout=TIMEOUT)
        user_expressions = reply['content']['user_expressions']
        str_value = user_expressions['output']['data']['text/plain']
        value = ast.literal_eval(str_value)
        assert '' in value

@flaky(max_runs=3)
def test_multiprocessing(tmpdir):
    if False:
        print('Hello World!')
    '\n    Test that multiprocessing works.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f', timeout=TIMEOUT)
        code = "\nfrom multiprocessing import Pool\n\ndef f(x):\n    return x*x\n\nif __name__ == '__main__':\n    with Pool(5) as p:\n        result = p.map(f, [1, 2, 3])\n"
        p = tmpdir.join('mp-test.py')
        p.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.inspect('result')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']

@flaky(max_runs=3)
def test_multiprocessing_2(tmpdir):
    if False:
        return 10
    '\n    Test that multiprocessing works.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f', timeout=TIMEOUT)
        code = "\nfrom multiprocessing import Pool\n\nclass myClass():\n    def __init__(self, i):\n        self.i = i + 10\n\ndef myFunc(i):\n    return myClass(i)\n\nif __name__ == '__main__':\n    with Pool(5) as p:\n        result = p.map(myFunc, [1, 2, 3])\n    result = [r.i for r in result]\n"
        p = tmpdir.join('mp-test.py')
        p.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.inspect('result')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']
        assert '[11, 12, 13]' in content['data']['text/plain']

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin' and sys.version_info[:2] == (3, 8), reason='Fails on Mac with Python 3.8')
@pytest.mark.skipif(os.environ.get('USE_CONDA') != 'true', reason="Doesn't work with pip packages")
def test_dask_multiprocessing(tmpdir):
    if False:
        return 10
    '\n    Test that dask multiprocessing works.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f')
        code = "\nfrom dask.distributed import Client\n\nif __name__=='__main__':\n    client = Client()\n    client.close()\n    x = 'hello'\n"
        p = tmpdir.join('mp-test.py')
        p.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.inspect('x')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']

@flaky(max_runs=3)
def test_runfile(tmpdir):
    if False:
        return 10
    '\n    Test that runfile uses the proper name space for execution.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f', timeout=TIMEOUT)
        code = "result = 'hello world'; error # make an error"
        d = tmpdir.join('defined-test.py')
        d.write(code)
        code = dedent("\n        try:\n            result3 = result\n        except NameError:\n            result2 = 'hello world'\n        ")
        u = tmpdir.join('undefined-test.py')
        u.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(d))), timeout=TIMEOUT)
        client.inspect('result')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']
        client.execute_interactive('%runfile {}'.format(repr(str(u))), timeout=TIMEOUT)
        client.inspect('result2')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']
        msg = client.execute_interactive('%runfile {} --current-namespace'.format(repr(str(u))), timeout=TIMEOUT)
        content = msg['content']
        client.inspect('result3')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']
        client.inspect('__file__')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert not content['found']

@flaky(max_runs=3)
@pytest.mark.skipif(sys.platform == 'darwin' and sys.version_info[:2] == (3, 8), reason='Fails on Mac with Python 3.8')
def test_np_threshold(kernel):
    if False:
        return 10
    "Test that setting Numpy threshold doesn't make the Variable Explorer slow."
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive("\nimport numpy as np;\nnp.set_printoptions(\n    threshold=np.inf,\n    suppress=True,\n    formatter={'float_kind':'{:0.2f}'.format})\n    ", timeout=TIMEOUT)
        client.execute_interactive('\nx = np.random.rand(75000,5);\na = np.array([123412341234.123412341234])\n', timeout=TIMEOUT)
        client.execute_interactive("\nt = np.get_printoptions()['threshold'];\ns = np.get_printoptions()['suppress'];\nf = np.get_printoptions()['formatter']\n", timeout=TIMEOUT)
        client.inspect('a')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'data' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']['data']['text/plain']
        assert '123412341234.12' in content
        client.inspect('t')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'data' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']['data']['text/plain']
        assert 'inf' in content
        client.inspect('s')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'data' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']['data']['text/plain']
        assert 'True' in content
        client.inspect('f')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'data' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']['data']['text/plain']
        assert "{'float_kind': <built-in method format of str object" in content

@flaky(max_runs=3)
@pytest.mark.skipif(not TURTLE_ACTIVE, reason="Doesn't work on non-interactive settings or Python without Tk")
def test_turtle_launch(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test turtle scripts running in the same kernel.'
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f', timeout=TIMEOUT)
        code = '\nimport turtle\nwn=turtle.Screen()\nwn.bgcolor("lightgreen")\ntess = turtle.Turtle() # Create tess and set some attributes\ntess.color("hotpink")\ntess.pensize(5)\n\ntess.forward(80) # Make tess draw equilateral triangle\ntess.left(120)\ntess.forward(80)\ntess.left(120)\ntess.forward(80)\ntess.left(120) # Complete the triangle\n\nturtle.bye()\n'
        p = tmpdir.join('turtle-test.py')
        p.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.inspect('tess')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']
        code = code + 'a = 10'
        p = tmpdir.join('turtle-test1.py')
        p.write(code)
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT)
        client.inspect('a')
        msg = client.get_shell_msg(timeout=TIMEOUT)
        while 'found' not in msg['content']:
            msg = client.get_shell_msg(timeout=TIMEOUT)
        content = msg['content']
        assert content['found']

@flaky(max_runs=3)
def test_matplotlib_inline(kernel):
    if False:
        print('Hello World!')
    "Test that the default backend for our kernels is 'inline'."
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        code = 'import matplotlib; backend = matplotlib.get_backend()'
        reply = client.execute_interactive(code, user_expressions={'output': 'backend'}, timeout=TIMEOUT)
        user_expressions = reply['content']['user_expressions']
        str_value = user_expressions['output']['data']['text/plain']
        value = ast.literal_eval(str_value)
        assert 'inline' in value

def test_do_complete(kernel):
    if False:
        return 10
    '\n    Check do complete works in normal and debugging mode.\n    '
    asyncio.run(kernel.do_execute('abba = 1', True))
    assert kernel.get_value('abba') == 1
    match = kernel.do_complete('ab', 2)
    assert 'abba' in match['matches']
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.completenames = lambda *ignore: ['baba']
    kernel.shell._namespace_stack = [pdb_obj]
    match = kernel.do_complete('ba', 2)
    assert 'baba' in match['matches']
    pdb_obj.curframe = None

@pytest.mark.parametrize('exclude_callables_and_modules', [True, False])
@pytest.mark.parametrize('exclude_unsupported', [True, False])
def test_callables_and_modules(kernel, exclude_callables_and_modules, exclude_unsupported):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that callables and modules are in the namespace view only\n    when the right options are passed to the kernel.\n    '
    asyncio.run(kernel.do_execute('import numpy', True))
    asyncio.run(kernel.do_execute('a = 10', True))
    asyncio.run(kernel.do_execute('def f(x): return x', True))
    settings = kernel.namespace_view_settings
    settings['exclude_callables_and_modules'] = exclude_callables_and_modules
    settings['exclude_unsupported'] = exclude_unsupported
    nsview = kernel.get_namespace_view()
    if not exclude_callables_and_modules:
        assert 'numpy' in nsview.keys()
        assert 'f' in nsview.keys()
    else:
        assert 'numpy' not in nsview.keys()
        assert 'f' not in nsview.keys()
    assert 'a' in nsview.keys()
    settings['exclude_callables_and_modules'] = True
    settings['exclude_unsupported'] = False

def test_comprehensions_with_locals_in_pdb(kernel):
    if False:
        print('Hello World!')
    '\n    Test that evaluating comprehensions with locals works in Pdb.\n\n    Also test that we use the right frame globals, in case the user\n    wants to work with them.\n\n    This is a regression test for spyder-ide/spyder#13909.\n    '
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.curframe_locals = pdb_obj.curframe.f_locals
    kernel.shell._namespace_stack = [pdb_obj]
    kernel.shell.pdb_session.default('zz = 10')
    assert kernel.get_value('zz') == 10
    kernel.shell.pdb_session.default('compr = [zz * i for i in [1, 2, 3]]')
    assert kernel.get_value('compr') == [10, 20, 30]
    kernel.shell.pdb_session.default("in_globals = 'zz' in globals()")
    assert kernel.get_value('in_globals') == False
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

def test_comprehensions_with_locals_in_pdb_2(kernel):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that evaluating comprehensions with locals works in Pdb.\n\n    This is a regression test for spyder-ide/spyder#16790.\n    '
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.curframe_locals = pdb_obj.curframe.f_locals
    kernel.shell._namespace_stack = [pdb_obj]
    kernel.shell.pdb_session.default('aa = [1, 2]')
    kernel.shell.pdb_session.default('bb = [3, 4]')
    kernel.shell.pdb_session.default('res = []')
    kernel.shell.pdb_session.default('for c0 in aa: res.append([(c0, c1) for c1 in bb])')
    assert kernel.get_value('res') == [[(1, 3), (1, 4)], [(2, 3), (2, 4)]]
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

def test_namespaces_in_pdb(kernel):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test namespaces in pdb\n    '
    get_ipython = lambda : kernel.shell
    kernel.shell.user_ns['test'] = 0
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.curframe_locals = pdb_obj.curframe.f_locals
    kernel.shell._namespace_stack = [pdb_obj]
    pdb_obj.default("globals()['test2'] = 0")
    assert pdb_obj.curframe.f_globals['test2'] == 0
    old_error = pdb_obj.error
    pdb_obj._error_occured = False

    def error_wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        print(args, kwargs)
        pdb_obj._error_occured = True
        return old_error(*args, **kwargs)
    pdb_obj.error = error_wrapper
    pdb_obj.curframe.f_globals['test3'] = 0
    pdb_obj.default('%timeit test3')
    assert not pdb_obj._error_occured
    pdb_obj.curframe_locals['test4'] = 0
    pdb_obj.default('%timeit test4')
    assert not pdb_obj._error_occured
    pdb_obj.default('%timeit test')
    assert pdb_obj._error_occured
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

def test_functions_with_locals_in_pdb(kernel):
    if False:
        i = 10
        return i + 15
    '\n    Test that functions with locals work in Pdb.\n\n    This is a regression test for spyder-ide/spyder-kernels#345\n    '
    pdb_obj = SpyderPdb()
    Frame = namedtuple('Frame', ['f_globals'])
    pdb_obj.curframe = Frame(f_globals=kernel.shell.user_ns)
    pdb_obj.curframe_locals = kernel.shell.user_ns
    kernel.shell._namespace_stack = [pdb_obj]
    kernel.shell.pdb_session.default('def fun_a(): return [i for i in range(1)]')
    kernel.shell.pdb_session.default('zz = fun_a()')
    assert kernel.get_value('zz') == [0]
    kernel.shell.pdb_session.default('a = 1')
    kernel.shell.pdb_session.default('def fun_a(): return a')
    kernel.shell.pdb_session.default('zz = fun_a()')
    assert kernel.get_value('zz') == 1
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

def test_functions_with_locals_in_pdb_2(kernel):
    if False:
        print('Hello World!')
    '\n    Test that functions with locals work in Pdb.\n\n    This is another regression test for spyder-ide/spyder-kernels#345\n    '
    baba = 1
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.curframe_locals = pdb_obj.curframe.f_locals
    kernel.shell._namespace_stack = [pdb_obj]
    kernel.shell.pdb_session.default('def fun_a(): return [i for i in range(1)]')
    kernel.shell.pdb_session.default('zz = fun_a()')
    assert kernel.get_value('zz') == [0]
    kernel.shell.pdb_session.default('a = 1')
    kernel.shell.pdb_session.default('def fun_a(): return a')
    kernel.shell.pdb_session.default('zz = fun_a()')
    assert kernel.get_value('zz') == 1
    kernel.shell.pdb_session.default('ll = locals().keys()')
    assert 'baba' in kernel.get_value('ll')
    kernel.shell.pdb_session.default('gg = globals().keys()')
    assert 'baba' not in kernel.get_value('gg')
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

def test_locals_globals_in_pdb(kernel):
    if False:
        i = 10
        return i + 15
    '\n    Test thal locals and globals work properly in Pdb.\n    '
    a = 1
    pdb_obj = SpyderPdb()
    pdb_obj.curframe = inspect.currentframe()
    pdb_obj.curframe_locals = pdb_obj.curframe.f_locals
    kernel.shell._namespace_stack = [pdb_obj]
    assert kernel.get_value('a') == 1
    kernel.shell.pdb_session.default('test = "a" in globals()')
    assert kernel.get_value('test') == False
    kernel.shell.pdb_session.default('test = "a" in locals()')
    assert kernel.get_value('test') == True
    kernel.shell.pdb_session.default('def f(): return a')
    kernel.shell.pdb_session.default('test = f()')
    assert kernel.get_value('test') == 1
    kernel.shell.pdb_session.default('a = 2')
    assert kernel.get_value('a') == 2
    kernel.shell.pdb_session.default('test = "a" in globals()')
    assert kernel.get_value('test') == False
    kernel.shell.pdb_session.default('test = "a" in locals()')
    assert kernel.get_value('test') == True
    pdb_obj.curframe = None
    pdb_obj.curframe_locals = None

@flaky(max_runs=3)
@pytest.mark.parametrize('backend', [None, 'inline', 'tk', 'qt'])
@pytest.mark.skipif(os.environ.get('USE_CONDA') != 'true', reason="Doesn't work with pip packages")
@pytest.mark.skipif(sys.version_info[:2] < (3, 9), reason="Too flaky in Python 3.7/8 and doesn't work in older versions")
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on Mac')
def test_get_interactive_backend(backend):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we correctly get the interactive backend set in the kernel.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        if backend is not None:
            client.execute_interactive('%matplotlib {}'.format(backend), timeout=TIMEOUT)
            client.execute_interactive('import time; time.sleep(.1)', timeout=TIMEOUT)
        code = 'backend = get_ipython().kernel.get_mpl_interactive_backend()'
        reply = client.execute_interactive(code, user_expressions={'output': 'backend'}, timeout=TIMEOUT)
        user_expressions = reply['content']['user_expressions']
        value = user_expressions['output']['data']['text/plain']
        value = value[1:-1]
        if backend is not None:
            assert value == backend
        else:
            assert value == 'inline'

def test_global_message(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that using `global` triggers a warning.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        client.execute_interactive('%reset -f', timeout=TIMEOUT)
        code = 'def foo1():\n    global x\n    x = 2\nx = 1\nprint(x)\n'
        p = tmpdir.join('test.py')
        p.write(code)
        global found
        found = False

        def check_found(msg):
            if False:
                i = 10
                return i + 15
            if 'text' in msg['content']:
                if 'WARNING: This file contains a global statement' in msg['content']['text']:
                    global found
                    found = True
        client.execute_interactive('%runfile {} --current-namespace'.format(repr(str(p))), timeout=TIMEOUT, output_hook=check_found)
        assert not found
        client.execute_interactive('%runfile {}'.format(repr(str(p))), timeout=TIMEOUT, output_hook=check_found)
        assert found

@flaky(max_runs=3)
def test_debug_namespace(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    Test that the kernel uses the proper namespace while debugging.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    with setup_kernel(cmd) as client:
        d = tmpdir.join('pdb-ns-test.py')
        d.write('def func():\n    bb = "hello"\n    breakpoint()\nfunc()')
        msg_id = client.execute('%runfile {}'.format(repr(str(d))))
        client.get_stdin_msg(timeout=TIMEOUT)
        client.input('bb')
        t0 = time.time()
        while True:
            assert time.time() - t0 < 5
            msg = client.get_iopub_msg(timeout=TIMEOUT)
            if msg.get('msg_type') == 'stream':
                if 'hello' in msg['content'].get('text'):
                    break
        client.get_stdin_msg(timeout=TIMEOUT)
        client.input("get_ipython().kernel.get_value('bb')")
        t0 = time.time()
        while True:
            assert time.time() - t0 < 5
            msg = client.get_iopub_msg(timeout=TIMEOUT)
            if msg.get('msg_type') == 'stream':
                if 'hello' in msg['content'].get('text'):
                    break

def test_interrupt():
    if False:
        i = 10
        return i + 15
    '\n    Test that the kernel can be interrupted by calling a comm handler.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    import pickle
    with setup_kernel(cmd) as client:
        kernel_comm = CommBase()
        comm = Comm(kernel_comm._comm_name, client)
        comm.open(data={'pickle_highest_protocol': pickle.HIGHEST_PROTOCOL})
        comm._send_channel = client.control_channel
        kernel_comm._register_comm(comm)
        client.execute_interactive('import time', timeout=TIMEOUT)
        t0 = time.time()
        msg_id = client.execute('for i in range(100): time.sleep(.1)')
        time.sleep(0.2)
        kernel_comm.remote_call().raise_interrupt_signal()
        while True:
            assert time.time() - t0 < 5
            msg = client.get_shell_msg(timeout=TIMEOUT)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            break
        assert time.time() - t0 < 5
        if os.name == 'nt':
            return
        t0 = time.time()
        msg_id = client.execute('time.sleep(10)')
        time.sleep(0.2)
        kernel_comm.remote_call().raise_interrupt_signal()
        while True:
            assert time.time() - t0 < 5
            msg = client.get_shell_msg(timeout=TIMEOUT)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            break
        assert time.time() - t0 < 5

def test_enter_debug_after_interruption():
    if False:
        i = 10
        return i + 15
    '\n    Test that we can enter the debugger after interrupting the current\n    execution.\n    '
    cmd = 'from spyder_kernels.console import start; start.main()'
    import pickle
    with setup_kernel(cmd) as client:
        kernel_comm = CommBase()
        comm = Comm(kernel_comm._comm_name, client)
        comm.open(data={'pickle_highest_protocol': pickle.HIGHEST_PROTOCOL})
        comm._send_channel = client.control_channel
        kernel_comm._register_comm(comm)
        client.execute_interactive('import time', timeout=TIMEOUT)
        t0 = time.time()
        msg_id = client.execute('for i in range(100): time.sleep(.1)')
        time.sleep(0.2)
        kernel_comm.remote_call().request_pdb_stop()
        while True:
            assert time.time() - t0 < 5
            msg = client.get_iopub_msg(timeout=TIMEOUT)
            if msg.get('msg_type') == 'stream':
                print(msg['content'].get('text'))
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            if msg.get('msg_type') == 'comm_msg':
                if msg['content'].get('data', {}).get('content', {}).get('call_name') == 'pdb_input':
                    break
                comm.handle_msg(msg)
        assert time.time() - t0 < 5

def test_non_strings_in_locals(kernel):
    if False:
        i = 10
        return i + 15
    '\n    Test that we can hande non-string entries in `locals` when bulding the\n    namespace view.\n\n    This is a regression test for issue spyder-ide/spyder#19145\n    '
    execute = asyncio.run(kernel.do_execute('locals().update({1:2})', True))
    nsview = repr(kernel.get_namespace_view())
    assert '1:' in nsview

def test_django_settings(kernel):
    if False:
        i = 10
        return i + 15
    "\n    Test that we don't generate errors when importing `django.conf.settings`.\n\n    This is a regression test for issue spyder-ide/spyder#19516\n    "
    execute = asyncio.run(kernel.do_execute('from django.conf import settings', True))
    nsview = repr(kernel.get_namespace_view())
    assert "'settings':" in nsview
if __name__ == '__main__':
    pytest.main()