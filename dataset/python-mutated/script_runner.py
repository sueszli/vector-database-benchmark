import ast
import sys
import tokenize
import types
from inspect import CO_COROUTINE
from gradio.wasm_utils import app_id_context

class modified_sys_path:
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, script_path: str):
        if False:
            return 10
        self._script_path = script_path
        self._added_path = False

    def __enter__(self):
        if False:
            return 10
        if self._script_path not in sys.path:
            sys.path.insert(0, self._script_path)
            self._added_path = True

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        if self._added_path:
            try:
                sys.path.remove(self._script_path)
            except ValueError:
                pass
        return False

def _new_module(name: str) -> types.ModuleType:
    if False:
        while True:
            i = 10
    'Create a new module with the given name.'
    return types.ModuleType(name)

async def _run_script(app_id: str, script_path: str) -> None:
    with tokenize.open(script_path) as f:
        filebody = f.read()
    await _run_code(app_id, filebody, script_path)

async def _run_code(app_id: str, filebody: str, script_path: str='<string>') -> None:
    bytecode = compile(filebody, script_path, mode='exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=1, optimize=-1)
    module = _new_module('__main__')
    sys.modules['__main__'] = module
    module.__dict__['__file__'] = script_path
    with modified_sys_path(script_path), app_id_context(app_id):
        if bytecode.co_flags & CO_COROUTINE:
            await eval(bytecode, module.__dict__)
        else:
            exec(bytecode, module.__dict__)