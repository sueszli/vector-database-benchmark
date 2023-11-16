"""Bring in all of the public TensorFlow interface into this module."""
import distutils as _distutils
import importlib
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys
import typing as _typing
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.platform import tf_logging as _logging
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader
if 'dev' in __version__:
    _logging.warning('\n\n  TensorFlow\'s `tf-nightly` package will soon be updated to TensorFlow 2.0.\n\n  Please upgrade your code to TensorFlow 2.0:\n    * https://www.tensorflow.org/guide/migrate\n\n  Or install the latest stable TensorFlow 1.X release:\n    * `pip install -U "tensorflow==1.*"`\n\n  Otherwise your code may be broken by the change.\n\n  ')
_API_MODULE = _sys.modules[__name__].bitwise
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
    __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
    __path__.append(_tf_api_dir)
_current_module.compat.v2
if _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == 'true' or _os.getenv('TF_USE_MODULAR_FILESYSTEM', '0') == '1':
    import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem
_estimator_module = 'tensorflow_estimator.python.estimator.api._v1.estimator'
estimator = _LazyLoader('estimator', globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
    _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, 'estimator', estimator)
_tf_uses_legacy_keras = _os.environ.get('TF_USE_LEGACY_KERAS', None) in ('true', 'True', '1')
setattr(_current_module, 'keras', _KerasLazyLoader(globals(), mode='v1'))
_module_dir = _module_util.get_parent_dir_for_name('keras._tf_keras.keras')
_current_module.__path__ = [_module_dir] + _current_module.__path__
if _tf_uses_legacy_keras:
    _module_dir = _module_util.get_parent_dir_for_name('tf_keras.api._v1.keras')
else:
    _module_dir = _module_util.get_parent_dir_for_name('keras.api._v1.keras')
_current_module.__path__ = [_module_dir] + _current_module.__path__
_CONTRIB_WARNING = '\nThe TensorFlow contrib module will not be included in TensorFlow 2.0.\nFor more information, please see:\n  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md\n  * https://github.com/tensorflow/addons\n  * https://github.com/tensorflow/io (for I/O related ops)\nIf you depend on functionality not listed there, please file an issue.\n'
contrib = _LazyLoader('contrib', globals(), 'tensorflow.contrib', _CONTRIB_WARNING)
if '__all__' in vars():
    vars()['__all__'].append('contrib')
from tensorflow.python.platform import flags
_current_module.app.flags = flags
setattr(_current_module, 'flags', flags)
_major_api_version = 1
_current_module.layers = _KerasLazyLoader(globals(), submodule='__internal__.legacy.layers', name='layers', mode='v1')
if _tf_uses_legacy_keras:
    _module_dir = _module_util.get_parent_dir_for_name('tf_keras.api._v1.keras.__internal__.legacy.layers')
else:
    _module_dir = _module_util.get_parent_dir_for_name('keras.api._v1.keras.__internal__.legacy.layers')
_current_module.__path__ = [_module_dir] + _current_module.__path__
_current_module.nn.rnn_cell = _KerasLazyLoader(globals(), submodule='__internal__.legacy.rnn_cell', name='rnn_cell', mode='v1')
if _tf_uses_legacy_keras:
    _module_dir = _module_util.get_parent_dir_for_name('tf_keras.api._v1.keras.__internal__.legacy.rnn_cell')
else:
    _module_dir = _module_util.get_parent_dir_for_name('keras.api._v1.keras.__internal__.legacy.rnn_cell')
_current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__
del importlib
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi
_site_packages_dirs = []
_site_packages_dirs += [] if _site.USER_SITE is None else [_site.USER_SITE]
_site_packages_dirs += [p for p in _sys.path if 'site-packages' in p]
if 'getsitepackages' in dir(_site):
    _site_packages_dirs += _site.getsitepackages()
if 'sysconfig' in dir(_distutils):
    _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]
_site_packages_dirs = list(set(_site_packages_dirs))
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
    if False:
        for i in range(10):
            print('nop')
    return any((_current_file_location.startswith(dir_) for dir_ in _site_packages_dirs))
if _running_from_pip_package():
    _tf_dir = _os.path.dirname(_current_file_location)
    _kernel_dir = _os.path.join(_tf_dir, 'core', 'kernels')
    if _os.path.exists(_kernel_dir):
        _ll.load_library(_kernel_dir)
    for _s in _site_packages_dirs:
        _plugin_dir = _os.path.join(_s, 'tensorflow-plugins')
        if _os.path.exists(_plugin_dir):
            _ll.load_library(_plugin_dir)
            _ll.load_pluggable_device_library(_plugin_dir)
if _os.getenv('TF_PLUGGABLE_DEVICE_LIBRARY_PATH', ''):
    _ll.load_pluggable_device_library(_os.getenv('TF_PLUGGABLE_DEVICE_LIBRARY_PATH'))
if _typing.TYPE_CHECKING:
    from tensorflow_estimator.python.estimator.api._v1 import estimator as estimator
try:
    del python
except NameError:
    pass
try:
    del core
except NameError:
    pass
try:
    del compiler
except NameError:
    pass