"""Function for loading TensorFlow plugins."""
import errno
import hashlib
import importlib
import os
import platform
import sys
from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export('load_op_library')
def load_op_library(library_filename):
    if False:
        for i in range(10):
            print('nop')
    'Loads a TensorFlow plugin, containing custom ops and kernels.\n\n  Pass "library_filename" to a platform-specific mechanism for dynamically\n  loading a library. The rules for determining the exact location of the\n  library are platform-specific and are not documented here. When the\n  library is loaded, ops and kernels registered in the library via the\n  `REGISTER_*` macros are made available in the TensorFlow process. Note\n  that ops with the same name as an existing op are rejected and not\n  registered with the process.\n\n  Args:\n    library_filename: Path to the plugin.\n      Relative or absolute filesystem path to a dynamic library file.\n\n  Returns:\n    A python module containing the Python wrappers for Ops defined in\n    the plugin.\n\n  Raises:\n    RuntimeError: when unable to load the library or get the python wrappers.\n  '
    lib_handle = py_tf.TF_LoadLibrary(library_filename)
    try:
        wrappers = _pywrap_python_op_gen.GetPythonWrappers(py_tf.TF_GetOpList(lib_handle))
    finally:
        py_tf.TF_DeleteLibraryHandle(lib_handle)
    module_name = hashlib.sha1(wrappers).hexdigest()
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_spec = importlib.machinery.ModuleSpec(module_name, None)
    module = importlib.util.module_from_spec(module_spec)
    exec(wrappers, module.__dict__)
    setattr(module, '_IS_TENSORFLOW_PLUGIN', True)
    sys.modules[module_name] = module
    return module

@deprecation.deprecated(date=None, instructions='Use `tf.load_library` instead.')
@tf_export(v1=['load_file_system_library'])
def load_file_system_library(library_filename):
    if False:
        print('Hello World!')
    'Loads a TensorFlow plugin, containing file system implementation.\n\n  Pass `library_filename` to a platform-specific mechanism for dynamically\n  loading a library. The rules for determining the exact location of the\n  library are platform-specific and are not documented here.\n\n  Args:\n    library_filename: Path to the plugin.\n      Relative or absolute filesystem path to a dynamic library file.\n\n  Returns:\n    None.\n\n  Raises:\n    RuntimeError: when unable to load the library.\n  '
    py_tf.TF_LoadLibrary(library_filename)

def _is_shared_object(filename):
    if False:
        i = 10
        return i + 15
    'Check the file to see if it is a shared object, only using extension.'
    if platform.system() == 'Linux':
        if filename.endswith('.so'):
            return True
        else:
            index = filename.rfind('.so.')
            if index == -1:
                return False
            else:
                return filename[index + 4].isdecimal()
    elif platform.system() == 'Darwin':
        return filename.endswith('.dylib')
    elif platform.system() == 'Windows':
        return filename.endswith('.dll')
    else:
        return False

@tf_export('load_library')
def load_library(library_location):
    if False:
        while True:
            i = 10
    'Loads a TensorFlow plugin.\n\n  "library_location" can be a path to a specific shared object, or a folder.\n  If it is a folder, all shared objects that are named "libtfkernel*" will be\n  loaded. When the library is loaded, kernels registered in the library via the\n  `REGISTER_*` macros are made available in the TensorFlow process.\n\n  Args:\n    library_location: Path to the plugin or the folder of plugins.\n      Relative or absolute filesystem path to a dynamic library file or folder.\n\n  Returns:\n    None\n\n  Raises:\n    OSError: When the file to be loaded is not found.\n    RuntimeError: when unable to load the library.\n  '
    if os.path.exists(library_location):
        if os.path.isdir(library_location):
            directory_contents = os.listdir(library_location)
            kernel_libraries = [os.path.join(library_location, f) for f in directory_contents if _is_shared_object(f)]
        else:
            kernel_libraries = [library_location]
        for lib in kernel_libraries:
            py_tf.TF_LoadLibrary(lib)
    else:
        raise OSError(errno.ENOENT, 'The file or folder to load kernel libraries from does not exist.', library_location)

def load_pluggable_device_library(library_location):
    if False:
        return 10
    'Loads a TensorFlow PluggableDevice plugin.\n\n  "library_location" can be a path to a specific shared object, or a folder.\n  If it is a folder, all shared objects will be loaded. when the library is\n  loaded, devices/kernels registered in the library via StreamExecutor C API\n  and Kernel/Op Registration C API are made available in TensorFlow process.\n\n  Args:\n    library_location: Path to the plugin or folder of plugins. Relative or\n      absolute filesystem path to a dynamic library file or folder.\n\n  Raises:\n    OSError: When the file to be loaded is not found.\n    RuntimeError: when unable to load the library.\n  '
    if os.path.exists(library_location):
        if os.path.isdir(library_location):
            directory_contents = os.listdir(library_location)
            pluggable_device_libraries = [os.path.join(library_location, f) for f in directory_contents if _is_shared_object(f)]
        else:
            pluggable_device_libraries = [library_location]
        for lib in pluggable_device_libraries:
            py_tf.TF_LoadPluggableDeviceLibrary(lib)
        context.context().reinitialize_physical_devices()
    else:
        raise OSError(errno.ENOENT, 'The file or folder to load pluggable device libraries from does not exist.', library_location)

@tf_export('experimental.register_filesystem_plugin')
def register_filesystem_plugin(plugin_location):
    if False:
        i = 10
        return i + 15
    'Loads a TensorFlow FileSystem plugin.\n\n  Args:\n    plugin_location: Path to the plugin. Relative or absolute filesystem plugin\n      path to a dynamic library file.\n\n  Returns:\n    None\n\n  Raises:\n    OSError: When the file to be loaded is not found.\n    RuntimeError: when unable to load the library.\n  '
    if os.path.exists(plugin_location):
        py_tf.TF_RegisterFilesystemPlugin(plugin_location)
    else:
        raise OSError(errno.ENOENT, 'The file to load file system plugin from does not exist.', plugin_location)