"""
Validate docstrings using pydocstyle and numpydoc.

Example usage:
python scripts/doc_checker.py asv_bench/benchmarks/utils.py modin/pandas
"""
import argparse
import ast
import functools
import inspect
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types
from typing import List
from numpydoc.docscrape import NumpyDocString
from numpydoc.validate import Docstring
for mod_name in ('cudf', 'cupy'):
    try:
        __import__(mod_name)
    except ImportError:
        sys.modules[mod_name] = types.ModuleType(mod_name, f'fake {mod_name} for checking docstrings')
if not hasattr(sys.modules['cudf'], 'DataFrame'):
    sys.modules['cudf'].DataFrame = type('DataFrame', (object,), {})
if not hasattr(sys.modules['cupy'], 'ndarray'):
    sys.modules['cupy'].ndarray = type('ndarray', (object,), {})
logging.basicConfig(stream=sys.stdout, format='%(levelname)s:%(message)s', level=logging.INFO)
MODIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODIN_PATH)
NUMPYDOC_BASE_ERROR_CODES = {*('GL01', 'GL02', 'GL03', 'GL05', 'GL06', 'GL07', 'GL08', 'GL09', 'GL10'), *('SS02', 'SS03', 'SS04', 'SS05', 'PR01', 'PR02', 'PR03', 'PR04', 'PR05'), *('PR08', 'PR09', 'PR10', 'RT01', 'RT04', 'RT05', 'SA02', 'SA03')}
MODIN_ERROR_CODES = {'MD01': "'{parameter}' description should be '[type], default: [value]', found: '{found}'", 'MD02': "Spelling error in line: {line}, found: '{word}', reference: '{reference}'", 'MD03': "Section contents is over-indented (in section '{section}')"}

def get_optional_args(doc: Docstring) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get optional parameters for the object for which the docstring is checked.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n\n    Returns\n    -------\n    dict\n        Dict with default argument names and its values.\n    '
    obj = doc.obj
    if not callable(obj) or inspect.isclass(obj):
        return {}
    signature = inspect.signature(obj)
    return {k: v.default for (k, v) in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def check_optional_args(doc: Docstring) -> list:
    if False:
        return 10
    '\n    Check type description of optional arguments.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n\n    Returns\n    -------\n    list\n        List of tuples with Modin error code and its description.\n    '
    if not doc.doc_parameters:
        return []
    optional_args = get_optional_args(doc)
    if not optional_args:
        return []
    errors = []
    for parameter in optional_args:
        if parameter not in doc.doc_parameters:
            continue
        type_line = doc.doc_parameters[parameter][0]
        has_default = 'default: ' in type_line
        has_optional = 'optional' in type_line
        if not has_default ^ has_optional:
            errors.append(('MD01', MODIN_ERROR_CODES['MD01'].format(parameter=parameter, found=type_line)))
    return errors

def check_spelling_words(doc: Docstring) -> list:
    if False:
        print('Hello World!')
    '\n    Check spelling of chosen words in doc.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n\n    Returns\n    -------\n    list\n        List of tuples with Modin error code and its description.\n\n    Notes\n    -----\n    Any special words enclosed in apostrophes(") are treated as python string\n    constants and are not checked for spelling.\n    '
    if not doc.raw_doc:
        return []
    components = set(['Modin', 'pandas', 'NumPy', 'Ray', 'Dask'] + ['PyArrow', 'HDK', 'XGBoost', 'Plasma'])
    check_words = '|'.join((x.lower() for x in components))
    pattern = '\n    (?:                     # non-capturing group\n        [^-\\\\\\w\\/]          # any symbol except: \'-\', \'\\\', \'/\' and any from [a-zA-Z0-9_]\n        | ^                 # or line start\n    )\n    ({check_words})         # words to check, example - "modin|pandas|numpy"\n    (?:                     # non-capturing group\n        [^-"\\.\\/\\w\\\\]       # any symbol except: \'-\', \'"\', \'.\', \'\\\', \'/\' and any from [a-zA-Z0-9_]\n        | \\.\\s              # or \'.\' and any whitespace\n        | \\.$               # or \'.\' and line end\n        | $                 # or line end\n    )\n    '.format(check_words=check_words)
    results = [set(re.findall(pattern, line, re.I | re.VERBOSE)) - components for line in doc.raw_doc.splitlines()]
    docstring_start_line = None
    for (idx, line) in enumerate(inspect.getsourcelines(doc.code_obj)[0]):
        if '"""' in line or "'''" in line:
            docstring_start_line = doc.source_file_def_line + idx
            break
    errors = []
    for (line_idx, words_in_line) in enumerate(results):
        for word in words_in_line:
            reference = [x for x in components if x.lower() == word.lower()][0]
            errors.append(('MD02', MODIN_ERROR_CODES['MD02'].format(line=docstring_start_line + line_idx, word=word, reference=reference)))
    return errors

def check_docstring_indention(doc: Docstring) -> list:
    if False:
        print('Hello World!')
    '\n    Check indention of docstring since numpydoc reports weird results.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n\n    Returns\n    -------\n    list\n        List of tuples with Modin error code and its description.\n    '
    from modin.utils import _get_indent
    numpy_docstring = NumpyDocString(doc.clean_doc)
    numpy_docstring._doc.reset()
    numpy_docstring._parse_summary()
    sections = list(numpy_docstring._read_sections())
    errors = []
    for section in sections:
        description = '\n'.join(section[1])
        if _get_indent(description) != 0:
            errors.append(('MD03', MODIN_ERROR_CODES['MD03'].format(section=section[0])))
    return errors

def validate_modin_error(doc: Docstring, results: dict) -> list:
    if False:
        while True:
            i = 10
    '\n    Validate custom Modin errors.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n    results : dict\n        Dictionary that numpydoc.validate.validate return.\n\n    Returns\n    -------\n    dict\n        Updated dict with Modin custom errors.\n    '
    errors = check_optional_args(doc)
    errors += check_spelling_words(doc)
    errors += check_docstring_indention(doc)
    results['errors'].extend(errors)
    return results

def skip_check_if_noqa(doc: Docstring, err_code: str, noqa_checks: list) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "\n    Skip the check that matches `err_code` if `err_code` found in noqa string.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n    err_code : str\n        Error code found by numpydoc.\n    noqa_checks : list\n        Found noqa checks.\n\n    Returns\n    -------\n    bool\n        Return True if 'noqa' found.\n    "
    if noqa_checks == ['all']:
        return True
    if err_code == 'GL08':
        name = doc.name.split('.')[-1]
        if name == '__init__':
            return True
    return err_code in noqa_checks

def get_noqa_checks(doc: Docstring) -> list:
    if False:
        i = 10
        return i + 15
    '\n    Get codes after `# noqa`.\n\n    Parameters\n    ----------\n    doc : numpydoc.validate.Docstring\n        Docstring handler.\n\n    Returns\n    -------\n    list\n        List with codes.\n\n    Notes\n    -----\n    If noqa doesn\'t have any codes - returns ["all"].\n    '
    source = doc.method_source
    if not source:
        return []
    noqa_str = ''
    if not inspect.ismodule(doc.obj):
        for line in source.split('\n'):
            if ')' in line and ':' in line.split(')', 1)[1]:
                noqa_str = line
                break
    else:
        if not doc.raw_doc:
            return []
        lines = source.split('\n')
        for (idx, line) in enumerate(lines):
            if '"""' in line or "'''" in line:
                noqa_str = lines[idx - 1]
                break
    if '# noqa:' in noqa_str:
        noqa_checks = noqa_str.split('# noqa:', 1)[1].split(',')
    elif '# noqa' in noqa_str:
        noqa_checks = ['all']
    else:
        noqa_checks = []
    return [check.strip() for check in noqa_checks]

def validate_object(import_path: str) -> list:
    if False:
        return 10
    '\n    Check docstrings of an entity that can be imported.\n\n    Parameters\n    ----------\n    import_path : str\n        Python-like import path.\n\n    Returns\n    -------\n    errors : list\n        List with string representations of errors.\n    '
    from numpydoc.validate import validate
    errors = []
    doc = Docstring(import_path)
    if getattr(doc.obj, '__doc_inherited__', False) or (isinstance(doc.obj, property) and getattr(doc.obj.fget, '__doc_inherited__', False)):
        return errors
    results = validate(import_path)
    results = validate_modin_error(doc, results)
    noqa_checks = get_noqa_checks(doc)
    for (err_code, err_desc) in results['errors']:
        if err_code not in NUMPYDOC_BASE_ERROR_CODES and err_code not in MODIN_ERROR_CODES or skip_check_if_noqa(doc, err_code, noqa_checks):
            continue
        errors.append(':'.join([import_path, str(results['file_line']), err_code, err_desc]))
    return errors

def numpydoc_validate(path: pathlib.Path) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform numpydoc checks.\n\n    Parameters\n    ----------\n    path : pathlib.Path\n        Filename or directory path for check.\n\n    Returns\n    -------\n    is_successfull : bool\n        Return True if all checks are successful.\n    '
    is_successfull = True
    if path.is_file():
        walker = ((str(path.parent), [], [path.name]),)
    else:
        walker = os.walk(path)
    for (root, _, files) in walker:
        if '__pycache__' in root:
            continue
        for _file in files:
            if not _file.endswith('.py'):
                continue
            current_path = os.path.join(root, _file)
            module_name = current_path.replace('/', '.').replace('\\', '.')
            module_name = os.path.splitext(module_name)[0]
            with open(current_path) as fd:
                file_contents = fd.read()
            module = ast.parse(file_contents)

            def is_public_func(node):
                if False:
                    i = 10
                    return i + 15
                return isinstance(node, ast.FunctionDef) and (not node.name.startswith('__') or node.name.endswith('__'))
            functions = [node for node in module.body if is_public_func(node)]
            classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
            methods = [f'{module_name}.{_class.name}.{node.name}' for _class in classes for node in _class.body if is_public_func(node)]
            to_validate = [module_name] + [f'{module_name}.{x.name}' for x in functions + classes] + methods
            results = list(map(validate_object, to_validate))
            is_successfull_file = not any(results)
            if not is_successfull_file:
                logging.info(f'NUMPYDOC OUTPUT FOR {current_path}')
            [logging.error(error) for errors in results for error in errors]
            is_successfull &= is_successfull_file
    return is_successfull

def pydocstyle_validate(path: pathlib.Path, add_ignore: List[str], use_numpydoc: bool) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Perform pydocstyle checks.\n\n    Parameters\n    ----------\n    path : pathlib.Path\n        Filename or directory path for check.\n    add_ignore : List[int]\n        `pydocstyle` error codes which are not verified.\n    use_numpydoc : bool\n        Disable duplicate `pydocstyle` checks if `numpydoc` is in use.\n\n    Returns\n    -------\n    bool\n        Return True if all pydocstyle checks are successful.\n    '
    pydocstyle = 'pydocstyle'
    if not shutil.which(pydocstyle):
        raise ValueError(f'{pydocstyle} not found in PATH')
    if use_numpydoc:
        add_ignore.extend(['D100', 'D101', 'D102', 'D103', 'D104', 'D105'])
    result = subprocess.run([pydocstyle, '--convention', 'numpy', '--add-ignore', ','.join(add_ignore), str(path)], text=True, capture_output=True)
    if result.returncode:
        logging.info(f'PYDOCSTYLE OUTPUT FOR {path}')
        logging.error(result.stdout)
        logging.error(result.stderr)
    return True if result.returncode == 0 else False

def monkeypatching():
    if False:
        for i in range(10):
            print('nop')
    'Monkeypatch not installed modules and decorators which change __doc__ attribute.'
    from unittest.mock import Mock
    import pandas.util
    import ray
    import modin.utils

    def monkeypatch(*args, **kwargs):
        if False:
            while True:
                i = 10
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        return lambda cls_or_func: cls_or_func
    ray.remote = monkeypatch
    pandas.util.cache_readonly = property
    sys.modules['pyarrow.gandiva'] = Mock()
    sys.modules['sqlalchemy'] = Mock()
    modin.utils.instancer = functools.wraps(modin.utils.instancer)(lambda cls: cls)

    def load_obj(name, old_load_obj=Docstring._load_obj):
        if False:
            for i in range(10):
                print('nop')
        obj = old_load_obj(name)
        if isinstance(obj, property):
            obj = obj.fget
        return obj
    Docstring._load_obj = staticmethod(load_obj)
    sys.modules['pyhdk'] = Mock()
    sys.modules['pyhdk'].__version__ = '999'
    sys.modules['pyhdk.hdk'] = Mock()
    sys.modules['pyhdk._sql'] = Mock()
    sys.getdlopenflags = Mock()
    sys.setdlopenflags = Mock()
    xgboost_mock = Mock()

    class Booster:
        ...
    xgboost_mock.Booster = Booster
    sys.modules['xgboost'] = xgboost_mock

def validate(paths: List[pathlib.Path], add_ignore: List[str], use_numpydoc: bool) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Perform pydocstyle and numpydoc checks.\n\n    Parameters\n    ----------\n    paths : List[pathlib.Path]\n        Filenames of directories for check.\n    add_ignore : List[str]\n        `pydocstyle` error codes which are not verified.\n    use_numpydoc : bool\n        Determine if numpydoc checks are needed.\n\n    Returns\n    -------\n    is_successfull : bool\n        Return True if all checks are successful.\n    '
    is_successfull = True
    for path in paths:
        if not pydocstyle_validate(path, add_ignore, use_numpydoc):
            is_successfull = False
        if use_numpydoc:
            if not numpydoc_validate(path):
                is_successfull = False
    return is_successfull

def check_args(args: argparse.Namespace):
    if False:
        return 10
    '\n    Check the obtained values for correctness.\n\n    Parameters\n    ----------\n    args : argparse.Namespace\n        Parser arguments.\n\n    Raises\n    ------\n    ValueError\n        Occurs in case of non-existent files or directories.\n    '
    for path in args.paths:
        if not path.exists():
            raise ValueError(f'{path} does not exist')
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(MODIN_PATH):
            raise ValueError('it is unsupported to use this script on files from another ' + f"repository; script' repo '{MODIN_PATH}', " + f"input path '{abs_path}'")

def get_args() -> argparse.Namespace:
    if False:
        print('Hello World!')
    '\n    Get args from cli with validation.\n\n    Returns\n    -------\n    argparse.Namespace\n    '
    parser = argparse.ArgumentParser(description='Check docstrings by using pydocstyle and numpydoc')
    parser.add_argument('paths', nargs='+', type=pathlib.Path, help='Filenames or directories; in case of direstories perform recursive check')
    parser.add_argument('--add-ignore', nargs='*', default=[], help='Pydocstyle error codes; for example: D100,D100,D102')
    parser.add_argument('--disable-numpydoc', default=False, action='store_true', help='Determine if numpydoc checks are not needed')
    args = parser.parse_args()
    check_args(args)
    return args
if __name__ == '__main__':
    args = get_args()
    monkeypatching()
    if not validate(args.paths, args.add_ignore, not args.disable_numpydoc):
        logging.error('INVALID DOCUMENTATION FOUND')
        exit(1)
    logging.info('SUCCESSFUL CHECK')