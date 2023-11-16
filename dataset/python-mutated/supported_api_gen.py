"""
Generate 'Supported pandas APIs' documentation file
"""
import warnings
from enum import Enum, unique
from inspect import getmembers, isclass, isfunction, signature
from typing import Any, Callable, Dict, List, NamedTuple, Set, TextIO, Tuple
import pyspark.pandas as ps
import pyspark.pandas.groupby as psg
import pyspark.pandas.window as psw
import pandas as pd
import pandas.core.groupby as pdg
import pandas.core.window as pdw
from pyspark.loose_version import LooseVersion
from pyspark.pandas.exceptions import PandasNotImplementedError
MAX_MISSING_PARAMS_SIZE = 5
COMMON_PARAMETER_SET = {'kwargs', 'args', 'cls'}
MODULE_GROUP_MATCH = [(pd, ps), (pdw, psw), (pdg, psg)]
RST_HEADER = "\n=====================\nSupported pandas API\n=====================\n\n.. currentmodule:: pyspark.pandas\n\nThe following table shows the pandas APIs that implemented or non-implemented from pandas API on\nSpark. Some pandas API do not implement full parameters, so the third column shows missing\nparameters for each API.\n\n* 'Y' in the second column means it's implemented including its whole parameter.\n* 'N' means it's not implemented yet.\n* 'P' means it's partially implemented with the missing of some parameters.\n\nAll API in the list below computes the data with distributed execution except the ones that require\nthe local execution by design. For example, `DataFrame.to_numpy() <https://spark.apache.org/docs/\nlatest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.to_numpy.html>`__\nrequires to collect the data to the driver side.\n\nIf there is non-implemented pandas API or parameter you want, you can create an `Apache Spark\nJIRA <https://issues.apache.org/jira/projects/SPARK/summary>`__ to request or to contribute by\nyour own.\n\nThe API list is updated based on the `latest pandas official API reference\n<https://pandas.pydata.org/docs/reference/index.html#>`__.\n\n"

@unique
class Implemented(Enum):
    IMPLEMENTED = 'Y'
    NOT_IMPLEMENTED = 'N'
    PARTIALLY_IMPLEMENTED = 'P'

class SupportedStatus(NamedTuple):
    """
    Defines a supported status for specific pandas API
    """
    implemented: str
    missing: str

def generate_supported_api(output_rst_file_path: str) -> None:
    if False:
        print('Hello World!')
    '\n    Generate supported APIs status dictionary.\n\n    Parameters\n    ----------\n    output_rst_file_path : str\n        The path to the document file in RST format.\n\n    Write supported APIs documentation.\n    '
    pandas_latest_version = '2.1.3'
    if LooseVersion(pd.__version__) != LooseVersion(pandas_latest_version):
        msg = 'Warning: Latest version of pandas (%s) is required to generate the documentation; however, your version was %s' % (pandas_latest_version, pd.__version__)
        warnings.warn(msg, UserWarning)
        raise ImportError(msg)
    all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]] = {}
    for (pd_module_group, ps_module_group) in MODULE_GROUP_MATCH:
        pd_modules = _get_pd_modules(pd_module_group)
        _update_all_supported_status(all_supported_status, pd_modules, pd_module_group, ps_module_group)
    _write_rst(output_rst_file_path, all_supported_status)

def _create_supported_by_module(module_name: str, pd_module_group: Any, ps_module_group: Any) -> Dict[str, SupportedStatus]:
    if False:
        while True:
            i = 10
    '\n    Retrieves supported status of pandas module\n\n    Parameters\n    ----------\n    module_name : str\n        Class name that exists in the path of the module.\n    pd_module_group : Any\n        Specific path of importable pandas module.\n    ps_module_group: Any\n        Specific path of importable pyspark.pandas module.\n    '
    pd_module = getattr(pd_module_group, module_name) if module_name else pd_module_group
    try:
        ps_module = getattr(ps_module_group, module_name) if module_name else ps_module_group
    except (AttributeError, PandasNotImplementedError):
        return {}
    pd_funcs = dict([m for m in getmembers(pd_module, isfunction) if not m[0].startswith('_') and m[0] in pd_module.__dict__])
    if not pd_funcs:
        return {}
    ps_funcs = dict([m for m in getmembers(ps_module, isfunction) if not m[0].startswith('_') and m[0] in ps_module.__dict__])
    return _organize_by_implementation_status(module_name, pd_funcs, ps_funcs, pd_module_group, ps_module_group)

def _organize_by_implementation_status(module_name: str, pd_funcs: Dict[str, Callable], ps_funcs: Dict[str, Callable], pd_module_group: Any, ps_module_group: Any) -> Dict[str, SupportedStatus]:
    if False:
        return 10
    '\n    Check the implementation status and parameters of both modules.\n\n    Parameters\n    ----------\n    module_name : str\n        Class name that exists in the path of the module.\n    pd_funcs: Dict[str, Callable]\n        function name and function object mapping of pandas module.\n    ps_funcs: Dict[str, Callable]\n        function name and function object mapping of pyspark.pandas module.\n    pd_module_group : Any\n        Specific path of importable pandas module.\n    ps_module_group: Any\n        Specific path of importable pyspark.pandas module.\n    '
    pd_dict = {}
    for (pd_func_name, pd_func) in pd_funcs.items():
        ps_func = ps_funcs.get(pd_func_name)
        if ps_func:
            missing_set = set(signature(pd_func).parameters) - set(signature(ps_func).parameters) - COMMON_PARAMETER_SET
            if missing_set:
                pd_dict[pd_func_name] = SupportedStatus(implemented=Implemented.PARTIALLY_IMPLEMENTED.value, missing=_transform_missing(module_name, pd_func_name, missing_set, pd_module_group.__name__, ps_module_group.__name__))
            else:
                pd_dict[pd_func_name] = SupportedStatus(implemented=Implemented.IMPLEMENTED.value, missing='')
        else:
            pd_dict[pd_func_name] = SupportedStatus(implemented=Implemented.NOT_IMPLEMENTED.value, missing='')
    return pd_dict

def _transform_missing(module_name: str, pd_func_name: str, missing_set: Set[str], pd_module_path: str, ps_module_path: str) -> str:
    if False:
        print('Hello World!')
    '\n    Transform missing parameters into table information string.\n\n    Parameters\n    ----------\n    module_name : str\n        Class name that exists in the path of the module.\n    pd_func_name : str\n        Name of pandas API.\n    missing_set : Set[str]\n        A set of parameters not yet implemented.\n    pd_module_path : str\n        Path string of pandas module.\n    ps_module_path : str\n        Path string of pyspark.pandas module.\n\n    Examples\n    --------\n    >>> _transform_missing("DataFrame", "add", {"axis", "fill_value", "level"},\n    ...                     "pandas.DataFrame", "pyspark.pandas.DataFrame")\n    \'``axis`` , ``fill_value`` , ``level``\'\n    '
    missing_str = ' , '.join(('``%s``' % x for x in sorted(missing_set)[:MAX_MISSING_PARAMS_SIZE]))
    if len(missing_set) > MAX_MISSING_PARAMS_SIZE:
        module_dot_func = '%s.%s' % (module_name, pd_func_name) if module_name else pd_func_name
        additional_str = ' and more. See the ' + '`%s.%s ' % (pd_module_path, module_dot_func) + '<https://pandas.pydata.org/docs/reference/api/' + '%s.%s.html>`__ and ' % (pd_module_path, module_dot_func) + '`%s.%s ' % (ps_module_path, module_dot_func) + '<https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/' + '%s.%s.html>`__ for detail.' % (ps_module_path, module_dot_func)
        missing_str += additional_str
    return missing_str

def _get_pd_modules(pd_module_group: Any) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns sorted pandas member list from pandas module path.\n\n    Parameters\n    ----------\n    pd_module_group : Any\n        Specific path of importable pandas module.\n    '
    return sorted([m[0] for m in getmembers(pd_module_group, isclass) if not m[0].startswith('_')])

def _update_all_supported_status(all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]], pd_modules: List[str], pd_module_group: Any, ps_module_group: Any) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Updates supported status across multiple module paths.\n\n    Parameters\n    ----------\n    all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]]\n        Data that stores the supported status across multiple module paths.\n    pd_modules: List[str]\n        Name list of pandas modules.\n    pd_module_group : Any\n        Specific path of importable pandas module.\n    ps_module_group: Any\n        Specific path of importable pyspark.pandas module.\n    '
    pd_modules += ['']
    for module_name in pd_modules:
        supported_status = _create_supported_by_module(module_name, pd_module_group, ps_module_group)
        if supported_status:
            all_supported_status[module_name, ps_module_group.__name__] = supported_status

def _write_table(module_name: str, module_path: str, supported_status: Dict[str, SupportedStatus], w_fd: TextIO) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Write table by using Sphinx list-table directive.\n    '
    lines = []
    if module_name:
        lines.append(module_name)
    else:
        lines.append('General Function')
    lines.append(' API\n')
    lines.append('-' * 100)
    lines.append('\n')
    lines.append('.. currentmodule:: %s' % module_path)
    if module_name:
        lines.append('.%s\n' % module_name)
    else:
        lines.append('\n')
    lines.append('\n')
    lines.append('.. list-table::\n')
    lines.append('    :header-rows: 1\n')
    lines.append('\n')
    lines.append('    * - API\n')
    lines.append('      - Implemented\n')
    lines.append('      - Missing parameters\n')
    for (func_str, status) in supported_status.items():
        func_str = _escape_func_str(func_str)
        if status.implemented == Implemented.NOT_IMPLEMENTED.value:
            lines.append('    * - %s\n' % func_str)
        else:
            lines.append('    * - :func:`%s`\n' % func_str)
        lines.append('      - %s\n' % status.implemented)
        lines.append('      - \n') if not status.missing else lines.append('      - %s\n' % status.missing)
    w_fd.writelines(lines)

def _escape_func_str(func_str: str) -> str:
    if False:
        return 10
    '\n    Transforms which affecting rst data format.\n    '
    if func_str.endswith('_'):
        return func_str[:-1] + '\\_'
    else:
        return func_str

def _write_rst(output_rst_file_path: str, all_supported_status: Dict[Tuple[str, str], Dict[str, SupportedStatus]]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Writes the documentation to the target file path.\n    '
    with open(output_rst_file_path, 'w') as w_fd:
        w_fd.write(RST_HEADER)
        for (module_info, supported_status) in all_supported_status.items():
            (module, module_path) = module_info
            if supported_status:
                _write_table(module, module_path, supported_status, w_fd)
                w_fd.write('\n')

def _test() -> None:
    if False:
        while True:
            i = 10
    import doctest
    import sys
    import pyspark.pandas.supported_api_gen
    globs = pyspark.pandas.supported_api_gen.__dict__.copy()
    (failure_count, test_count) = doctest.testmod(pyspark.pandas.supported_api_gen, globs=globs)
    if failure_count:
        sys.exit(-1)
if __name__ == '__main__':
    _test()