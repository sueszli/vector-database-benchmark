import inspect
import os
import sys
import traceback
import types
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union
from lightning.app.utilities.exceptions import MisconfigurationException
if TYPE_CHECKING:
    from lightning.app.core import LightningApp, LightningFlow, LightningWork
    from lightning.app.plugin.plugin import LightningPlugin
from lightning.app.utilities.app_helpers import Logger, _mock_missing_imports
logger = Logger(__name__)

def _prettifiy_exception(filepath: str):
    if False:
        i = 10
        return i + 15
    'Pretty print the exception that occurred when loading the app.'
    (exp, val, tb) = sys.exc_info()
    listing = traceback.format_exception(exp, val, tb)
    del listing[1]
    listing = [f'Found an exception when loading your application from {filepath}. Please, resolve it to run your app.\n\n'] + listing
    logger.error(''.join(listing))
    sys.exit(1)

def _load_objects_from_file(filepath: str, target_type: Type, raise_exception: bool=False, mock_imports: bool=False, env_vars: Dict[str, str]={}) -> Tuple[List[Any], types.ModuleType]:
    if False:
        print('Hello World!')
    'Load all of the top-level objects of the given type from a file.\n\n    Args:\n        filepath: The file to load from.\n        target_type: The type of object to load.\n        raise_exception: If ``True`` exceptions will be raised, otherwise exceptions will trigger system exit.\n        mock_imports: If ``True`` imports of missing packages will be replaced with a mock. This can allow the object to\n            be loaded without installing dependencies.\n\n    '
    with _patch_sys_path(os.path.dirname(os.path.abspath(filepath))):
        code = _create_code(filepath)
        with _create_fake_main_module(filepath) as module:
            try:
                with _add_to_env(env_vars), _patch_sys_argv():
                    if mock_imports:
                        with _mock_missing_imports():
                            exec(code, module.__dict__)
                    else:
                        exec(code, module.__dict__)
            except Exception as ex:
                if raise_exception:
                    raise ex
                _prettifiy_exception(filepath)
    return ([v for v in module.__dict__.values() if isinstance(v, target_type)], module)

def _load_plugin_from_file(filepath: str) -> 'LightningPlugin':
    if False:
        while True:
            i = 10
    from lightning.app.plugin.plugin import LightningPlugin
    (plugins, _) = _load_objects_from_file(filepath, LightningPlugin, raise_exception=True, mock_imports=False)
    if len(plugins) > 1:
        raise RuntimeError(f'There should not be multiple plugins instantiated within the file. Found {plugins}')
    if len(plugins) == 1:
        return plugins[0]
    raise RuntimeError(f'The provided file {filepath} does not contain a Plugin.')

def load_app_from_file(filepath: str, raise_exception: bool=False, mock_imports: bool=False, env_vars: Dict[str, str]={}) -> 'LightningApp':
    if False:
        while True:
            i = 10
    'Load a LightningApp from a file.\n\n    Arguments:\n        filepath:  The path to the file containing the LightningApp.\n        raise_exception: If True, raise an exception if the app cannot be loaded.\n\n    '
    from lightning.app.core.app import LightningApp
    (apps, main_module) = _load_objects_from_file(filepath, LightningApp, raise_exception=raise_exception, mock_imports=mock_imports, env_vars=env_vars)
    sys.path.append(os.path.dirname(os.path.abspath(filepath)))
    sys.modules['__main__'] = main_module
    if len(apps) > 1:
        raise MisconfigurationException(f'There should not be multiple apps instantiated within a file. Found {apps}')
    if len(apps) == 1:
        return apps[0]
    raise MisconfigurationException(f'The provided file {filepath} does not contain a LightningApp. Instantiate your app at the module level like so: `app = LightningApp(flow, ...)`')

def _new_module(name):
    if False:
        i = 10
        return i + 15
    'Create a new module with the given name.'
    return types.ModuleType(name)

def open_python_file(filename):
    if False:
        i = 10
        return i + 15
    "Open a read-only Python file taking proper care of its encoding.\n\n    In Python 3, we would like all files to be opened with utf-8 encoding. However, some author like to specify PEP263\n    headers in their source files with their own encodings. In that case, we should respect the author's encoding.\n\n    "
    import tokenize
    if hasattr(tokenize, 'open'):
        return tokenize.open(filename)
    return open(filename, encoding='utf-8')

def _create_code(script_path: str):
    if False:
        print('Hello World!')
    with open_python_file(script_path) as f:
        filebody = f.read()
    return compile(filebody, script_path, mode='exec', flags=0, dont_inherit=1, optimize=-1)

@contextmanager
def _create_fake_main_module(script_path):
    if False:
        while True:
            i = 10
    module = _new_module('__main__')
    old_main_module = sys.modules['__main__']
    sys.modules['__main__'] = module
    module.__dict__['__file__'] = os.path.abspath(script_path)
    try:
        yield module
    finally:
        sys.modules['__main__'] = old_main_module

@contextmanager
def _patch_sys_path(append):
    if False:
        print('Hello World!')
    'A context manager that appends the given value to the path once entered.\n\n    Args:\n        append: The value to append to the path.\n\n    '
    if append in sys.path:
        yield
        return
    sys.path.append(append)
    try:
        yield
    finally:
        sys.path.remove(append)

@contextmanager
def _add_to_env(envs: Dict[str, str]):
    if False:
        for i in range(10):
            print('nop')
    'This function adds the given environment variables to the current environment.'
    original_envs = dict(os.environ)
    os.environ.update(envs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_envs)

@contextmanager
def _patch_sys_argv():
    if False:
        i = 10
        return i + 15
    'This function modifies the ``sys.argv`` by extracting the arguments after ``--app_args`` and removed everything\n    else before executing the user app script.\n\n    The command: ``lightning run app app.py --without-server --app_args --use_gpu --env ...`` will be converted into\n    ``app.py --use_gpu``\n\n    '
    from lightning.app.cli.lightning_cli import run_app
    original_argv = copy(sys.argv)
    if sys.argv[:3] == ['lightning', 'run', 'app']:
        sys.argv = sys.argv[3:]
    if '--app_args' not in sys.argv:
        new_argv = sys.argv[:1]
    else:
        options = [p.opts[0] for p in run_app.params[1:] if p.opts[0] != '--app_args']
        argv_slice = sys.argv
        first_index = argv_slice.index('--app_args') + 1
        matches = [argv_slice.index(opt) for opt in options if opt in argv_slice and argv_slice.index(opt) >= first_index]
        last_index = len(argv_slice) if not matches else min(matches)
        new_argv = [argv_slice[0]] + argv_slice[first_index:last_index]
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original_argv

def component_to_metadata(obj: Union['LightningWork', 'LightningFlow']) -> Dict:
    if False:
        while True:
            i = 10
    from lightning.app.core import LightningWork
    extras = {}
    if isinstance(obj, LightningWork):
        extras = {'local_build_config': obj.local_build_config.to_dict(), 'cloud_build_config': obj.cloud_build_config.to_dict(), 'cloud_compute': obj.cloud_compute.to_dict()}
    return dict(affiliation=obj.name.split('.'), cls_name=obj.__class__.__name__, module=obj.__module__, docstring=inspect.getdoc(obj.__init__), **extras)

def extract_metadata_from_app(app: 'LightningApp') -> List:
    if False:
        return 10
    metadata = {flow.name: component_to_metadata(flow) for flow in app.flows}
    metadata.update({work.name: component_to_metadata(work) for work in app.works})
    return [metadata[key] for key in sorted(metadata.keys())]