from __future__ import annotations
import os
import sys
from ast import literal_eval
from textwrap import dedent
from virtualenv.activation import PythonActivator
from virtualenv.info import IS_WIN

def test_python(raise_on_non_source_class, activation_tester):
    if False:
        while True:
            i = 10

    class Python(raise_on_non_source_class):

        def __init__(self, session) -> None:
            if False:
                print('Hello World!')
            super().__init__(PythonActivator, session, sys.executable, activate_script='activate_this.py', extension='py', non_source_fail_message="You must use exec(open(this_file).read(), {'__file__': this_file})")
            self.unix_line_ending = not IS_WIN

        def env(self, tmp_path):
            if False:
                i = 10
                return i + 15
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            for key in ('VIRTUAL_ENV', 'PYTHONPATH'):
                env.pop(str(key), None)
            env['PATH'] = os.pathsep.join([str(tmp_path), str(tmp_path / 'other')])
            return env

        @staticmethod
        def _get_test_lines(activate_script):
            if False:
                return 10
            raw = f"""\n            import os\n            import sys\n            import platform\n\n            def print_r(value):\n                print(repr(value))\n\n            print_r(os.environ.get("VIRTUAL_ENV"))\n            print_r(os.environ.get("VIRTUAL_ENV_PROMPT"))\n            print_r(os.environ.get("PATH").split(os.pathsep))\n            print_r(sys.path)\n\n            file_at = {str(activate_script)!r}\n            # CPython 2 requires non-ascii path open to be unicode\n            with open(file_at, "r", encoding='utf-8') as file_handler:\n                content = file_handler.read()\n            exec(content, {{"__file__": file_at}})\n\n            print_r(os.environ.get("VIRTUAL_ENV"))\n            print_r(os.environ.get("VIRTUAL_ENV_PROMPT"))\n            print_r(os.environ.get("PATH").split(os.pathsep))\n            print_r(sys.path)\n\n            import pydoc_test\n            print_r(pydoc_test.__file__)\n            """
            return dedent(raw).splitlines()

        def assert_output(self, out, raw, tmp_path):
            if False:
                while True:
                    i = 10
            out = [literal_eval(i) for i in out]
            assert out[0] is None
            assert out[1] is None
            prev_path = out[2]
            prev_sys_path = out[3]
            assert out[4] == str(self._creator.dest)
            assert out[5] == str(self._creator.env_name)
            new_path = out[6]
            assert [str(self._creator.bin_dir), *prev_path] == new_path
            new_sys_path = out[7]
            new_lib_paths = {str(i) for i in self._creator.libs}
            assert prev_sys_path == new_sys_path[len(new_lib_paths):]
            assert new_lib_paths == set(new_sys_path[:len(new_lib_paths)])
            dest = self.norm_path(self._creator.purelib / 'pydoc_test.py')
            found = self.norm_path(out[8])
            assert found.startswith(dest)

        def non_source_activate(self, activate_script):
            if False:
                while True:
                    i = 10
            act = str(activate_script)
            return [*self._invoke_script, '-c', f'exec(open({act!r}).read())']
    activation_tester(Python)