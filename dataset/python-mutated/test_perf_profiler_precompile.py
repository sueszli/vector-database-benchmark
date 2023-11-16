import pathlib
import subprocess
import sys
import sysconfig
import unittest
from test.support.os_helper import temp_dir
from test.support.script_helper import assert_python_ok, make_script
try:
    from cinder import _is_compile_perf_trampoline_pre_fork_enabled
except:
    raise unittest.SkipTest('pre-fork perf-trampoline compilation is not enabled')

def supports_trampoline_profiling():
    if False:
        while True:
            i = 10
    perf_trampoline = sysconfig.get_config_var('PY_HAVE_PERF_TRAMPOLINE')
    if not perf_trampoline:
        return False
    return int(perf_trampoline) == 1
if not supports_trampoline_profiling():
    raise unittest.SkipTest('perf trampoline profiling not supported')
if not _is_compile_perf_trampoline_pre_fork_enabled():
    raise unittest.SkipTest('pre-fork perf-trampoline compilation is not enabled')

class TestPerfTrampolinePreCompile(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.perf_files = set(pathlib.Path('/tmp/').glob('perf-*.map'))

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        super().tearDown()
        files_to_delete = set(pathlib.Path('/tmp/').glob('perf-*.map')) - self.perf_files
        for file in files_to_delete:
            file.unlink()

    def test_trampoline_works(self):
        if False:
            while True:
                i = 10
        code = 'if 1:\n                import sys\n                import os\n                import sysconfig\n                from cinder import _compile_perf_trampoline_pre_fork\n\n                def foo_fork():\n                    pass\n                def bar_fork():\n                    foo_fork()\n                def baz_fork():\n                    bar_fork()\n\n\n                def foo():\n                    pass\n                def bar():\n                    foo()\n                def baz():\n                    bar()\n\n                if __name__ == "__main__":\n                    _compile_perf_trampoline_pre_fork()\n                    pid = os.fork()\n                    if pid == 0:\n                        print(os.getpid())\n                        baz_fork()\n                    else:\n                        baz()\n                '
        (rc, out, err) = assert_python_ok('-c', code)
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, '-X', 'perf-trampoline-prefork-compilation', '-X', 'perf', script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
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
        self.assertIn(f'py::foo_fork:{script}', perf_file_contents)
        self.assertIn(f'py::bar_fork:{script}', perf_file_contents)
        self.assertIn(f'py::baz_fork:{script}', perf_file_contents)
        child_perf_file_contents = perf_child_file.read_text()
        self.assertIn(f'py::foo_fork:{script}', child_perf_file_contents)
        self.assertIn(f'py::bar_fork:{script}', child_perf_file_contents)
        self.assertIn(f'py::baz_fork:{script}', child_perf_file_contents)
        perf_file_lines = perf_file_contents.split('\n')
        for line in perf_file_lines:
            if f'py::foo_fork:{script}' in line or f'py::bar_fork:{script}' in line or f'py::baz_fork:{script}' in line:
                self.assertIn(line, child_perf_file_contents)

    def test_trampoline_works_with_gced_functions(self):
        if False:
            return 10
        code = 'if 1:\n                import os\n                import gc\n                from cinder import _compile_perf_trampoline_pre_fork\n\n                def baz_fork():\n                    pass\n\n                def baz():\n                    pass\n\n                if __name__ == "__main__":\n\n                    def tmp_fn():\n                        pass\n\n                    # ensure this is registered with the JIT\n                    tmp_fn()\n\n                    # ensure it\'s GC\'d\n                    del tmp_fn\n                    gc.collect()\n\n                    _compile_perf_trampoline_pre_fork()\n                    pid = os.fork()\n                    if pid == 0:\n                        print(os.getpid())\n                        baz_fork()\n                    else:\n                        baz()\n                '
        (rc, out, err) = assert_python_ok('-c', code)
        with temp_dir() as script_dir:
            script = make_script(script_dir, 'perftest', code)
            with subprocess.Popen([sys.executable, '-X', 'perf-trampoline-prefork-compilation', '-X', 'perf', script], universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
                (stdout, stderr) = process.communicate()
                self.assertNotIn('Error:', stderr)
                self.assertEqual(process.returncode, 0)