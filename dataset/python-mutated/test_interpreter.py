import pytest
from .support import PyScriptTest
pytest.skip(reason="NEXT: pyscript API changed doesn't expose pyscript to window anymore", allow_module_level=True)

class TestInterpreterAccess(PyScriptTest):
    """Test accessing Python objects from JS via pyscript.interpreter"""

    def test_interpreter_python_access(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n                x = 1\n                def py_func():\n                    return 2\n            </script>\n            ')
        self.run_js("\n            const x = await pyscript.interpreter.globals.get('x');\n            const py_func = await pyscript.interpreter.globals.get('py_func');\n            const py_func_res = await py_func();\n            console.log(`x is ${x}`);\n            console.log(`py_func() returns ${py_func_res}`);\n            ")
        assert self.console.log.lines[-2:] == ['x is 1', 'py_func() returns 2']

    def test_interpreter_script_execution(self):
        if False:
            while True:
                i = 10
        'Test running Python code from js via pyscript.interpreter'
        self.pyscript_run('')
        self.run_js('\n            const interface = pyscript.interpreter._remote.interface;\n            await interface.runPython(\'print("Interpreter Ran This")\');\n            ')
        expected_message = 'Interpreter Ran This'
        assert self.console.log.lines[-1] == expected_message
        py_terminal = self.page.wait_for_selector('py-terminal')
        assert py_terminal.text_content() == expected_message

    def test_backward_compatibility_runtime_script_execution(self):
        if False:
            while True:
                i = 10
        'Test running Python code from js via pyscript.runtime'
        self.pyscript_run('')
        self.run_js('\n            const interface = pyscript.runtime._remote.interpreter;\n            await interface.runPython(\'print("Interpreter Ran This")\');\n            ')
        expected_message = 'Interpreter Ran This'
        assert self.console.log.lines[-1] == expected_message
        py_terminal = self.page.wait_for_selector('py-terminal')
        assert py_terminal.text_content() == expected_message

    def test_backward_compatibility_runtime_python_access(self):
        if False:
            while True:
                i = 10
        'Test accessing Python objects from JS via pyscript.runtime'
        self.pyscript_run('\n            <script type="py">\n                x = 1\n                def py_func():\n                    return 2\n            </script>\n            ')
        self.run_js("\n            const x = await pyscript.interpreter.globals.get('x');\n            const py_func = await pyscript.interpreter.globals.get('py_func');\n            const py_func_res = await py_func();\n            console.log(`x is ${x}`);\n            console.log(`py_func() returns ${py_func_res}`);\n            ")
        assert self.console.log.lines[-2:] == ['x is 1', 'py_func() returns 2']