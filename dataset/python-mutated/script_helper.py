import collections
import importlib
import sys
import os
import os.path
import subprocess
import py_compile
import zipfile
from importlib.util import source_from_cache
from test import support
from test.support.import_helper import make_legacy_pyc
__cached_interp_requires_environment = None

def interpreter_requires_environment():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns True if our sys.executable interpreter requires environment\n    variables in order to be able to run at all.\n\n    This is designed to be used with @unittest.skipIf() to annotate tests\n    that need to use an assert_python*() function to launch an isolated\n    mode (-I) or no environment mode (-E) sub-interpreter process.\n\n    A normal build & test does not run into this situation but it can happen\n    when trying to run the standard library test suite from an interpreter that\n    doesn't have an obvious home with Python's current home finding logic.\n\n    Setting PYTHONHOME is one way to get most of the testsuite to run in that\n    situation.  PYTHONPATH or PYTHONUSERSITE are other common environment\n    variables that might impact whether or not the interpreter can start.\n    "
    global __cached_interp_requires_environment
    if __cached_interp_requires_environment is None:
        if 'PYTHONHOME' in os.environ:
            __cached_interp_requires_environment = True
            return True
        try:
            subprocess.check_call([sys.executable, '-E', '-c', 'import sys; sys.exit(0)'])
        except subprocess.CalledProcessError:
            __cached_interp_requires_environment = True
        else:
            __cached_interp_requires_environment = False
    return __cached_interp_requires_environment

class _PythonRunResult(collections.namedtuple('_PythonRunResult', ('rc', 'out', 'err'))):
    """Helper for reporting Python subprocess run results"""

    def fail(self, cmd_line):
        if False:
            while True:
                i = 10
        'Provide helpful details about failed subcommand runs'
        maxlen = 80 * 100
        (out, err) = (self.out, self.err)
        if len(out) > maxlen:
            out = b'(... truncated stdout ...)' + out[-maxlen:]
        if len(err) > maxlen:
            err = b'(... truncated stderr ...)' + err[-maxlen:]
        out = out.decode('ascii', 'replace').rstrip()
        err = err.decode('ascii', 'replace').rstrip()
        raise AssertionError('Process return code is %d\ncommand line: %r\n\nstdout:\n---\n%s\n---\n\nstderr:\n---\n%s\n---' % (self.rc, cmd_line, out, err))

def run_python_until_end(*args, **env_vars):
    if False:
        for i in range(10):
            print('nop')
    env_required = interpreter_requires_environment()
    cwd = env_vars.pop('__cwd', None)
    if '__isolated' in env_vars:
        isolated = env_vars.pop('__isolated')
    else:
        isolated = not env_vars and (not env_required)
    cmd_line = [sys.executable, '-X', 'faulthandler']
    if isolated:
        cmd_line.append('-I')
    elif not env_vars and (not env_required):
        cmd_line.append('-E')
    if env_vars.pop('__cleanenv', None):
        env = {}
        if sys.platform == 'win32':
            env['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    else:
        env = os.environ.copy()
    if 'TERM' not in env_vars:
        env['TERM'] = ''
    env.update(env_vars)
    cmd_line.extend(args)
    proc = subprocess.Popen(cmd_line, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd)
    with proc:
        try:
            (out, err) = proc.communicate()
        finally:
            proc.kill()
            subprocess._cleanup()
    rc = proc.returncode
    return (_PythonRunResult(rc, out, err), cmd_line)

def _assert_python(expected_success, /, *args, **env_vars):
    if False:
        for i in range(10):
            print('nop')
    (res, cmd_line) = run_python_until_end(*args, **env_vars)
    if res.rc and expected_success or (not res.rc and (not expected_success)):
        res.fail(cmd_line)
    return res

def assert_python_ok(*args, **env_vars):
    if False:
        print('Hello World!')
    '\n    Assert that running the interpreter with `args` and optional environment\n    variables `env_vars` succeeds (rc == 0) and return a (return code, stdout,\n    stderr) tuple.\n\n    If the __cleanenv keyword is set, env_vars is used as a fresh environment.\n\n    Python is started in isolated mode (command line option -I),\n    except if the __isolated keyword is set to False.\n    '
    return _assert_python(True, *args, **env_vars)

def assert_python_failure(*args, **env_vars):
    if False:
        print('Hello World!')
    '\n    Assert that running the interpreter with `args` and optional environment\n    variables `env_vars` fails (rc != 0) and return a (return code, stdout,\n    stderr) tuple.\n\n    See assert_python_ok() for more options.\n    '
    return _assert_python(False, *args, **env_vars)

def spawn_python(*args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw):
    if False:
        print('Hello World!')
    'Run a Python subprocess with the given arguments.\n\n    kw is extra keyword args to pass to subprocess.Popen. Returns a Popen\n    object.\n    '
    cmd_line = [sys.executable]
    if not interpreter_requires_environment():
        cmd_line.append('-E')
    cmd_line.extend(args)
    env = kw.setdefault('env', dict(os.environ))
    env['TERM'] = 'vt100'
    return subprocess.Popen(cmd_line, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, **kw)

def kill_python(p):
    if False:
        i = 10
        return i + 15
    'Run the given Popen process until completion and return stdout.'
    p.stdin.close()
    data = p.stdout.read()
    p.stdout.close()
    p.wait()
    subprocess._cleanup()
    return data

def make_script(script_dir, script_basename, source, omit_suffix=False):
    if False:
        i = 10
        return i + 15
    script_filename = script_basename
    if not omit_suffix:
        script_filename += os.extsep + 'py'
    script_name = os.path.join(script_dir, script_filename)
    with open(script_name, 'w', encoding='utf-8') as script_file:
        script_file.write(source)
    importlib.invalidate_caches()
    return script_name

def make_zip_script(zip_dir, zip_basename, script_name, name_in_zip=None):
    if False:
        i = 10
        return i + 15
    zip_filename = zip_basename + os.extsep + 'zip'
    zip_name = os.path.join(zip_dir, zip_filename)
    with zipfile.ZipFile(zip_name, 'w') as zip_file:
        if name_in_zip is None:
            parts = script_name.split(os.sep)
            if len(parts) >= 2 and parts[-2] == '__pycache__':
                legacy_pyc = make_legacy_pyc(source_from_cache(script_name))
                name_in_zip = os.path.basename(legacy_pyc)
                script_name = legacy_pyc
            else:
                name_in_zip = os.path.basename(script_name)
        zip_file.write(script_name, name_in_zip)
    return (zip_name, os.path.join(zip_name, name_in_zip))

def make_pkg(pkg_dir, init_source=''):
    if False:
        for i in range(10):
            print('nop')
    os.mkdir(pkg_dir)
    make_script(pkg_dir, '__init__', init_source)

def make_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename, source, depth=1, compiled=False):
    if False:
        for i in range(10):
            print('nop')
    unlink = []
    init_name = make_script(zip_dir, '__init__', '')
    unlink.append(init_name)
    init_basename = os.path.basename(init_name)
    script_name = make_script(zip_dir, script_basename, source)
    unlink.append(script_name)
    if compiled:
        init_name = py_compile.compile(init_name, doraise=True)
        script_name = py_compile.compile(script_name, doraise=True)
        unlink.extend((init_name, script_name))
    pkg_names = [os.sep.join([pkg_name] * i) for i in range(1, depth + 1)]
    script_name_in_zip = os.path.join(pkg_names[-1], os.path.basename(script_name))
    zip_filename = zip_basename + os.extsep + 'zip'
    zip_name = os.path.join(zip_dir, zip_filename)
    with zipfile.ZipFile(zip_name, 'w') as zip_file:
        for name in pkg_names:
            init_name_in_zip = os.path.join(name, init_basename)
            zip_file.write(init_name, init_name_in_zip)
        zip_file.write(script_name, script_name_in_zip)
    for name in unlink:
        os.unlink(name)
    return (zip_name, os.path.join(zip_name, script_name_in_zip))

def run_test_script(script):
    if False:
        return 10
    if support.verbose:

        def title(text):
            if False:
                while True:
                    i = 10
            return f'===== {text} ======'
        name = f'script {os.path.basename(script)}'
        print()
        print(title(name), flush=True)
        args = [sys.executable, '-E', '-X', 'faulthandler', '-u', script, '-v']
        proc = subprocess.run(args)
        print(title(f'{name} completed: exit code {proc.returncode}'), flush=True)
        if proc.returncode:
            raise AssertionError(f'{name} failed')
    else:
        assert_python_ok('-u', script, '-v')