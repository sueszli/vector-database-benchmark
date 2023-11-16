import logging
import fnmatch
import os
import sys
import types
from typing import Optional, Set
LOGGER = logging.getLogger(__name__)

def file_is_in_folder_glob(filepath, folderpath_glob) -> bool:
    if False:
        i = 10
        return i + 15
    'Test whether a file is in some folder with globbing support.\n\n    Parameters\n    ----------\n    filepath : str\n        A file path.\n    folderpath_glob: str\n        A path to a folder that may include globbing.\n\n    '
    if not folderpath_glob.endswith('*'):
        if folderpath_glob.endswith('/'):
            folderpath_glob += '*'
        else:
            folderpath_glob += '/*'
    file_dir = os.path.dirname(filepath) + '/'
    return fnmatch.fnmatch(file_dir, folderpath_glob)

def get_directory_size(directory: str) -> int:
    if False:
        print('Hello World!')
    'Return the size of a directory in bytes.'
    total_size = 0
    for (dirpath, _, filenames) in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def file_in_pythonpath(filepath) -> bool:
    if False:
        while True:
            i = 10
    'Test whether a filepath is in the same folder of a path specified in the PYTHONPATH env variable.\n\n\n    Parameters\n    ----------\n    filepath : str\n        An absolute file path.\n\n    Returns\n    -------\n    boolean\n        True if contained in PYTHONPATH, False otherwise. False if PYTHONPATH is not defined or empty.\n\n    '
    pythonpath = os.environ.get('PYTHONPATH', '')
    if len(pythonpath) == 0:
        return False
    absolute_paths = [os.path.abspath(path) for path in pythonpath.split(os.pathsep)]
    return any((file_is_in_folder_glob(os.path.normpath(filepath), path) for path in absolute_paths))

def get_module_paths(module: types.ModuleType) -> Set[str]:
    if False:
        print('Hello World!')
    paths_extractors = [lambda m: [m.__file__], lambda m: [m.__spec__.origin], lambda m: [p for p in m.__path__._path]]
    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            pass
        except Exception as e:
            LOGGER.warning(f'Examining the path of {module.__name__} raised: {e}')
        all_paths.update([os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)])
    return all_paths

def _is_valid_path(path: Optional[str]) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))

def unload_local_modules(target_dir_path: str='.'):
    if False:
        while True:
            i = 10
    " Unload all modules that are in the target directory or in a subdirectory of it.\n    It is necessary to unload modules before re-executing a script that imports the modules,\n    so that the new version of the modules is loaded.\n    The module unloading feature is extracted from Streamlit's LocalSourcesWatcher (https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py)\n    and packaged as a standalone function.\n    "
    target_dir_path = os.path.abspath(target_dir_path)
    loaded_modules = {}
    module_paths = {name: get_module_paths(module) for (name, module) in dict(sys.modules).items()}
    for (name, paths) in module_paths.items():
        for path in paths:
            if file_is_in_folder_glob(path, target_dir_path) or file_in_pythonpath(path):
                loaded_modules[path] = name
    for module_name in loaded_modules.values():
        if module_name is not None and module_name in sys.modules:
            del sys.modules[module_name]