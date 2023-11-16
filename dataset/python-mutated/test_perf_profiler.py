import unittest
import subprocess
import sys
import sysconfig
import os
import pathlib
from test import support
from test.support.script_helper import make_script, assert_python_failure, assert_python_ok
from test.support.os_helper import temp_dir

def supports_trampoline_profiling():
    if False:
        for i in range(10):
            print('nop')
    perf_trampoline = sysconfig.get_config_var('PY_HAVE_PERF_TRAMPOLINE')
    if not perf_trampoline:
        return False
    return int(perf_trampoline) == 1
if not supports_trampoline_profiling():
    raise unittest.SkipTest('perf trampoline profiling not supported')

class TestPerfTrampoline(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.perf_files = set(pathlib.Path('/tmp/').glob('perf-*.map'))

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        files_to_delete = set(pathlib.Path('/tmp/').glob('perf-*.map')) - self.perf_files
        for file in files_to_delete:
            file.unlink()

    def test_trampoline_works(self):
        if False:
            return 10
        code = 'if 1:\n                def foo():\n                    pass\n\n                def bar():\n                    foo()\n\n                def baz():\n                    bar()\n\n                baz()\n                '
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, '-Xperf', script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
        self.assertNotIn('Error:', stderr)
        self.assertEqual(stdout, '')
        perf_file = pathlib.Path(f'/tmp/perf-{process.pid}.map')
        self.assertTrue(perf_file.exists())
        perf_file_contents = perf_file.read_text()
        self.assertIn(f'py::foo:{script}', perf_file_contents)
        self.assertIn(f'py::bar:{script}', perf_file_contents)
        self.assertIn(f'py::baz:{script}', perf_file_contents)

    def test_trampoline_works_with_forks(self):
        if False:
            i = 10
            return i + 15
        code = 'if 1:\n                import os, sys\n\n                def foo_fork():\n                    pass\n\n                def bar_fork():\n                    foo_fork()\n\n                def baz_fork():\n                    bar_fork()\n\n                def foo():\n                    pid = os.fork()\n                    if pid == 0:\n                        print(os.getpid())\n                        baz_fork()\n                    else:\n                        _, status = os.waitpid(-1, 0)\n                        sys.exit(status)\n\n                def bar():\n                    foo()\n\n                def baz():\n                    bar()\n\n                baz()\n                '
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, '-Xperf', script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
        self.assertEqual(process.returncode, 0)
        self.assertNotIn('Error:', stderr)
        child_pid = int(stdout.strip())
        perf_file = pathlib.Path(f'/tmp/perf-{process.pid}.map')
        perf_child_file = pathlib.Path(f'/tmp/perf-{child_pid}.map')
        self.assertTrue(perf_file.exists())
        self.assertTrue(perf_child_file.exists())
        perf_file_contents = perf_file.read_text()
        self.assertIn(f'py::foo:{script}', perf_file_contents)
        self.assertIn(f'py::bar:{script}', perf_file_contents)
        self.assertIn(f'py::baz:{script}', perf_file_contents)
        child_perf_file_contents = perf_child_file.read_text()
        self.assertIn(f'py::foo_fork:{script}', child_perf_file_contents)
        self.assertIn(f'py::bar_fork:{script}', child_perf_file_contents)
        self.assertIn(f'py::baz_fork:{script}', child_perf_file_contents)

    def test_sys_api(self):
        if False:
            print('Hello World!')
        code = 'if 1:\n                import sys\n\n                def foo():\n                    pass\n\n                def spam():\n                    pass\n\n                def bar():\n                    sys.deactivate_stack_trampoline()\n                    foo()\n                    sys.activate_stack_trampoline("perf")\n                    spam()\n\n                def baz():\n                    bar()\n\n                sys.activate_stack_trampoline("perf")\n                baz()\n                '
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
        self.assertNotIn('Error:', stderr)
        self.assertEqual(stdout, '')
        perf_file = pathlib.Path(f'/tmp/perf-{process.pid}.map')
        self.assertTrue(perf_file.exists())
        perf_file_contents = perf_file.read_text()
        self.assertNotIn(f'py::foo:{script}', perf_file_contents)
        self.assertIn(f'py::spam:{script}', perf_file_contents)
        self.assertIn(f'py::bar:{script}', perf_file_contents)
        self.assertIn(f'py::baz:{script}', perf_file_contents)

    def test_sys_api_with_existing_trampoline(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'if 1:\n                import sys\n                sys.activate_stack_trampoline("perf")\n                sys.activate_stack_trampoline("perf")\n                '
        assert_python_ok('-c', code)

    def test_sys_api_with_invalid_trampoline(self):
        if False:
            print('Hello World!')
        code = 'if 1:\n                import sys\n                sys.activate_stack_trampoline("invalid")\n                '
        (rc, out, err) = assert_python_failure('-c', code)
        self.assertIn('invalid backend: invalid', err.decode())

    def test_sys_api_get_status(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'if 1:\n                import sys\n                sys.activate_stack_trampoline("perf")\n                assert sys.is_stack_trampoline_active() is True\n                sys.deactivate_stack_trampoline()\n                assert sys.is_stack_trampoline_active() is False\n                '
        assert_python_ok('-c', code)

def is_unwinding_reliable():
    if False:
        print('Hello World!')
    cflags = sysconfig.get_config_var('PY_CORE_CFLAGS')
    if not cflags:
        return False
    return 'no-omit-frame-pointer' in cflags

def perf_command_works():
    if False:
        i = 10
        return i + 15
    try:
        cmd = ['perf', '--help']
        stdout = subprocess.check_output(cmd, universal_newlines=True)
    except (subprocess.SubprocessError, OSError):
        return False
    if 'perf.data' not in stdout:
        return False
    with temp_dir() as script_dir:
        try:
            output_file = script_dir + '/perf_output.perf'
            cmd = ('perf', 'record', '-g', '--call-graph=fp', '-o', output_file, '--', sys.executable, '-c', 'print("hello")')
            stdout = subprocess.check_output(cmd, cwd=script_dir, universal_newlines=True, stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, OSError):
            return False
        if 'hello' not in stdout:
            return False
    return True

def run_perf(cwd, *args, **env_vars):
    if False:
        print('Hello World!')
    if env_vars:
        env = os.environ.copy()
        env.update(env_vars)
    else:
        env = None
    output_file = cwd + '/perf_output.perf'
    base_cmd = ('perf', 'record', '-g', '--call-graph=fp', '-o', output_file, '--')
    proc = subprocess.run(base_cmd + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if proc.returncode:
        print(proc.stderr)
        raise ValueError(f'Perf failed with return code {proc.returncode}')
    base_cmd = ('perf', 'script')
    proc = subprocess.run(('perf', 'script', '-i', output_file), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=True)
    return (proc.stdout.decode('utf-8', 'replace'), proc.stderr.decode('utf-8', 'replace'))

@unittest.skipUnless(perf_command_works(), "perf command doesn't work")
@unittest.skipUnless(is_unwinding_reliable(), 'Unwinding is unreliable')
@support.skip_if_sanitizer(address=True, memory=True, ub=True)
class TestPerfProfiler(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.perf_files = set(pathlib.Path('/tmp/').glob('perf-*.map'))

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        files_to_delete = set(pathlib.Path('/tmp/').glob('perf-*.map')) - self.perf_files
        for file in files_to_delete:
            file.unlink()

    def test_python_calls_appear_in_the_stack_if_perf_activated(self):
        if False:
            for i in range(10):
                print('nop')
        with temp_dir() as script_dir:
            code = 'if 1:\n                def foo(n):\n                    x = 0\n                    for i in range(n):\n                        x += i\n\n                def bar(n):\n                    foo(n)\n\n                def baz(n):\n                    bar(n)\n\n                baz(10000000)\n                '
            script = make_script(script_dir, 'perftest', code)
            (stdout, stderr) = run_perf(script_dir, sys.executable, '-Xperf', script)
            self.assertNotIn('Error:', stderr)
            self.assertIn(f'py::foo:{script}', stdout)
            self.assertIn(f'py::bar:{script}', stdout)
            self.assertIn(f'py::baz:{script}', stdout)

    def test_python_calls_do_not_appear_in_the_stack_if_perf_activated(self):
        if False:
            print('Hello World!')
        with temp_dir() as script_dir:
            code = 'if 1:\n                def foo(n):\n                    x = 0\n                    for i in range(n):\n                        x += i\n\n                def bar(n):\n                    foo(n)\n\n                def baz(n):\n                    bar(n)\n\n                baz(10000000)\n                '
            script = make_script(script_dir, 'perftest', code)
            (stdout, stderr) = run_perf(script_dir, sys.executable, script)
            self.assertNotIn('Error:', stderr)
            self.assertNotIn(f'py::foo:{script}', stdout)
            self.assertNotIn(f'py::bar:{script}', stdout)
            self.assertNotIn(f'py::baz:{script}', stdout)

    def test_pre_fork_compile(self):
        if False:
            i = 10
            return i + 15
        code = 'if 1:\n                import sys\n                import os\n                import sysconfig\n                from _testinternalcapi import (\n                    compile_perf_trampoline_entry,\n                    perf_trampoline_set_persist_after_fork,\n                )\n\n                def foo_fork():\n                    pass\n\n                def bar_fork():\n                    foo_fork()\n\n                def foo():\n                    pass\n\n                def bar():\n                    foo()\n\n                def compile_trampolines_for_all_functions():\n                    perf_trampoline_set_persist_after_fork(1)\n                    for _, obj in globals().items():\n                        if callable(obj) and hasattr(obj, \'__code__\'):\n                            compile_perf_trampoline_entry(obj.__code__)\n\n                if __name__ == "__main__":\n                    compile_trampolines_for_all_functions()\n                    pid = os.fork()\n                    if pid == 0:\n                        print(os.getpid())\n                        bar_fork()\n                    else:\n                        bar()\n                '
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, '-Xperf', script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
        self.assertEqual(process.returncode, 0)
        self.assertNotIn('Error:', stderr)
        child_pid = int(stdout.strip())
        perf_file = pathlib.Path(f'/tmp/perf-{process.pid}.map')
        perf_child_file = pathlib.Path(f'/tmp/perf-{child_pid}.map')
        self.assertTrue(perf_file.exists())
        self.assertTrue(perf_child_file.exists())
        perf_file_contents = perf_file.read_text()
        self.assertIn(f'py::foo:{script}', perf_file_contents)
        self.assertIn(f'py::bar:{script}', perf_file_contents)
        self.assertIn(f'py::foo_fork:{script}', perf_file_contents)
        self.assertIn(f'py::bar_fork:{script}', perf_file_contents)
        child_perf_file_contents = perf_child_file.read_text()
        self.assertIn(f'py::foo_fork:{script}', child_perf_file_contents)
        self.assertIn(f'py::bar_fork:{script}', child_perf_file_contents)
        perf_file_lines = perf_file_contents.split('\n')
        for line in perf_file_lines:
            if f'py::foo_fork:{script}' in line or f'py::bar_fork:{script}' in line:
                self.assertIn(line, child_perf_file_contents)
if __name__ == '__main__':
    unittest.main()