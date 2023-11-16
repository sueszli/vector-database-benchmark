"""Resource management library."""
import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
try:
    from rules_python.python.runfiles import runfiles
except ImportError:
    runfiles = None

@tf_export(v1=['resource_loader.load_resource'])
def load_resource(path):
    if False:
        print('Hello World!')
    "Load the resource at given path, where path is relative to tensorflow/.\n\n  Args:\n    path: a string resource path relative to tensorflow/.\n\n  Returns:\n    The contents of that resource.\n\n  Raises:\n    IOError: If the path is not found, or the resource can't be opened.\n  "
    with open(get_path_to_datafile(path), 'rb') as f:
        return f.read()

@tf_export(v1=['resource_loader.get_data_files_path'])
def get_data_files_path():
    if False:
        i = 10
        return i + 15
    'Get a direct path to the data files colocated with the script.\n\n  Returns:\n    The directory where files specified in data attribute of py_test\n    and py_binary are stored.\n  '
    return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))

@tf_export(v1=['resource_loader.get_root_dir_with_all_resources'])
def get_root_dir_with_all_resources():
    if False:
        for i in range(10):
            print('nop')
    'Get a root directory containing all the data attributes in the build rule.\n\n  Returns:\n    The path to the specified file present in the data attribute of py_test\n    or py_binary. Falls back to returning the same as get_data_files_path if it\n    fails to detect a bazel runfiles directory.\n  '
    script_dir = get_data_files_path()
    directories = [script_dir]
    data_files_dir = ''
    while True:
        candidate_dir = directories[-1]
        current_directory = _os.path.basename(candidate_dir)
        if '.runfiles' in current_directory:
            if len(directories) > 1:
                data_files_dir = directories[-2]
            break
        else:
            new_candidate_dir = _os.path.dirname(candidate_dir)
            if new_candidate_dir == candidate_dir:
                break
            else:
                directories.append(new_candidate_dir)
    return data_files_dir or script_dir

@tf_export(v1=['resource_loader.get_path_to_datafile'])
def get_path_to_datafile(path):
    if False:
        i = 10
        return i + 15
    "Get the path to the specified file in the data dependencies.\n\n  The path is relative to tensorflow/\n\n  Args:\n    path: a string resource path relative to tensorflow/\n\n  Returns:\n    The path to the specified file present in the data attribute of py_test\n    or py_binary.\n\n  Raises:\n    IOError: If the path is not found, or the resource can't be opened.\n  "
    if runfiles:
        r = runfiles.Create()
        new_fpath = r.Rlocation(_os.path.abspath(_os.path.join('tensorflow', path)))
        if new_fpath is not None and _os.path.exists(new_fpath):
            return new_fpath
    old_filepath = _os.path.join(_os.path.dirname(_inspect.getfile(_sys._getframe(1))), path)
    return old_filepath

@tf_export(v1=['resource_loader.readahead_file_path'])
def readahead_file_path(path, readahead='128M'):
    if False:
        print('Hello World!')
    'Readahead files not implemented; simply returns given path.'
    return path