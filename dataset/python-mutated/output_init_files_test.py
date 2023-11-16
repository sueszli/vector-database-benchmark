"""Tests for api_init_files.bzl and api_init_files_v1.bzl."""
import argparse
import importlib
import sys
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator

def _traverse_packages(packages):
    if False:
        while True:
            i = 10
    for package in packages:
        importlib.import_module(package)

def _get_module_from_symbol(symbol):
    if False:
        i = 10
        return i + 15
    if '.' not in symbol:
        return ''
    return '.'.join(symbol.split('.')[:-1])

def _get_modules(package, attr_name, constants_attr_name):
    if False:
        i = 10
        return i + 15
    'Get list of TF API modules.\n\n  Args:\n    package: We only look at modules that contain package in the name.\n    attr_name: Attribute set on TF symbols that contains API names.\n    constants_attr_name: Attribute set on TF modules that contains\n      API constant names.\n\n  Returns:\n    Set of TensorFlow API modules.\n  '
    modules = set()
    for module in list(sys.modules.values()):
        if not module or not hasattr(module, '__name__') or package not in module.__name__:
            continue
        for module_contents_name in dir(module):
            attr = getattr(module, module_contents_name)
            (_, attr) = tf_decorator.unwrap(attr)
            if module_contents_name == constants_attr_name:
                for (exports, _) in attr:
                    modules.update([_get_module_from_symbol(export) for export in exports])
                continue
            if hasattr(attr, '__dict__') and attr_name in attr.__dict__:
                modules.update([_get_module_from_symbol(export) for export in getattr(attr, attr_name)])
    return modules

def _get_files_set(path, start_tag, end_tag):
    if False:
        for i in range(10):
            print('nop')
    'Get set of file paths from the given file.\n\n  Args:\n    path: Path to file. File at `path` is expected to contain a list of paths\n      where entire list starts with `start_tag` and ends with `end_tag`. List\n      must be comma-separated and each path entry must be surrounded by double\n      quotes.\n    start_tag: String that indicates start of path list.\n    end_tag: String that indicates end of path list.\n\n  Returns:\n    List of string paths.\n  '
    with open(path, 'r') as f:
        contents = f.read()
        start = contents.find(start_tag) + len(start_tag) + 1
        end = contents.find(end_tag)
        contents = contents[start:end]
        file_paths = [file_path.strip().strip('"') for file_path in contents.split(',')]
        return set((file_path for file_path in file_paths if file_path))

def _module_to_paths(module):
    if False:
        i = 10
        return i + 15
    "Get all API __init__.py file paths for the given module.\n\n  Args:\n    module: Module to get file paths for.\n\n  Returns:\n    List of paths for the given module. For e.g. module foo.bar\n    requires 'foo/__init__.py' and 'foo/bar/__init__.py'.\n  "
    submodules = []
    module_segments = module.split('.')
    for i in range(len(module_segments)):
        submodules.append('.'.join(module_segments[:i + 1]))
    paths = []
    for submodule in submodules:
        if not submodule:
            paths.append('__init__.py')
            continue
        paths.append('%s/__init__.py' % submodule.replace('.', '/'))
    return paths

class OutputInitFilesTest(test.TestCase):
    """Test that verifies files that list paths for TensorFlow API."""

    def _validate_paths_for_modules(self, actual_paths, expected_paths, file_to_update_on_error):
        if False:
            print('Hello World!')
        'Validates that actual_paths match expected_paths.\n\n    Args:\n      actual_paths: */__init__.py file paths listed in file_to_update_on_error.\n      expected_paths: */__init__.py file paths that we need to create for\n        TensorFlow API.\n      file_to_update_on_error: File that contains list of */__init__.py files.\n        We include it in error message printed if the file list needs to be\n        updated.\n    '
        self.assertTrue(actual_paths)
        self.assertTrue(expected_paths)
        missing_paths = expected_paths - actual_paths
        extra_paths = actual_paths - expected_paths
        missing_paths = ["'%s'" % path for path in missing_paths]
        extra_paths = ["'%s'" % path for path in extra_paths]
        self.assertFalse(missing_paths, 'Please add %s to %s.' % (',\n'.join(sorted(missing_paths)), file_to_update_on_error))
        self.assertFalse(extra_paths, 'Redundant paths, please remove %s in %s.' % (',\n'.join(sorted(extra_paths)), file_to_update_on_error))

    def test_V2_init_files(self):
        if False:
            print('Hello World!')
        modules = _get_modules('tensorflow', '_tf_api_names', '_tf_api_constants')
        file_path = resource_loader.get_path_to_datafile('api_init_files.bzl')
        paths = _get_files_set(file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
        module_paths = set((f for module in modules for f in _module_to_paths(module)))
        self._validate_paths_for_modules(paths, module_paths, file_to_update_on_error=file_path)

    def test_V1_init_files(self):
        if False:
            return 10
        modules = _get_modules('tensorflow', '_tf_api_names_v1', '_tf_api_constants_v1')
        file_path = resource_loader.get_path_to_datafile('api_init_files_v1.bzl')
        paths = _get_files_set(file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
        module_paths = set((f for module in modules for f in _module_to_paths(module)))
        self._validate_paths_for_modules(paths, module_paths, file_to_update_on_error=file_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', type=str, default='', help='Comma separated list of packages to traverse.')
    (FLAGS, unparsed) = parser.parse_known_args()
    _traverse_packages(FLAGS.packages.split(','))
    sys.argv = [sys.argv[0]] + unparsed
    test.main()