import dagster._check as check
from dagster._builtins import Bool, Float, Int, String
from .dagster_type import Any as RuntimeAny, List
from .python_dict import PythonDict
from .python_set import PythonSet
from .python_tuple import PythonTuple
SUPPORTED_RUNTIME_BUILTINS = {int: Int, float: Float, bool: Bool, str: String, list: List(RuntimeAny), tuple: PythonTuple, set: PythonSet, dict: PythonDict}

def is_supported_runtime_python_builtin(ttype):
    if False:
        i = 10
        return i + 15
    return ttype in SUPPORTED_RUNTIME_BUILTINS

def remap_python_builtin_for_runtime(ttype):
    if False:
        i = 10
        return i + 15
    'This function remaps a python type to a Dagster type, or passes it through if it cannot be\n    remapped.\n    '
    from dagster._core.types.dagster_type import resolve_dagster_type
    check.param_invariant(is_supported_runtime_python_builtin(ttype), 'ttype')
    return resolve_dagster_type(SUPPORTED_RUNTIME_BUILTINS[ttype])