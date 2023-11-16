"""Provides wrapper for TensorFlow modules."""
import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
FastModuleType = fast_module_type.get_fast_module_type_class()
_PER_MODULE_WARNING_LIMIT = 1
compat_v1_usage_gauge = monitoring.BoolGauge('/tensorflow/api/compat/v1', 'compat.v1 usage')

def get_rename_v2(name):
    if False:
        for i in range(10):
            print('nop')
    if name not in all_renames_v2.symbol_renames:
        return None
    return all_renames_v2.symbol_renames[name]

def _call_location():
    if False:
        return 10
    'Extracts the caller filename and line number as a string.\n\n  Returns:\n    A string describing the caller source location.\n  '
    frame = tf_inspect.currentframe()
    assert frame.f_back.f_code.co_name == '_tfmw_add_deprecation_warning', 'This function should be called directly from _tfmw_add_deprecation_warning, as the caller is identified heuristically by chopping off the top stack frames.'
    for _ in range(3):
        parent = frame.f_back
        if parent is None:
            break
        frame = parent
    return '{}:{}'.format(frame.f_code.co_filename, frame.f_lineno)

def contains_deprecation_decorator(decorators):
    if False:
        for i in range(10):
            print('nop')
    return any((d.decorator_name == 'deprecated' for d in decorators))

def has_deprecation_decorator(symbol):
    if False:
        return 10
    'Checks if given object has a deprecation decorator.\n\n  We check if deprecation decorator is in decorators as well as\n  whether symbol is a class whose __init__ method has a deprecation\n  decorator.\n  Args:\n    symbol: Python object.\n\n  Returns:\n    True if symbol has deprecation decorator.\n  '
    (decorators, symbol) = tf_decorator.unwrap(symbol)
    if contains_deprecation_decorator(decorators):
        return True
    if tf_inspect.isfunction(symbol):
        return False
    if not tf_inspect.isclass(symbol):
        return False
    if not hasattr(symbol, '__init__'):
        return False
    (init_decorators, _) = tf_decorator.unwrap(symbol.__init__)
    return contains_deprecation_decorator(init_decorators)

class TFModuleWrapper(FastModuleType):
    """Wrapper for TF modules to support deprecation messages and lazyloading."""
    compat_v1_usage_recorded = False

    def __init__(self, wrapped, module_name, public_apis=None, deprecation=True, has_lite=False):
        if False:
            print('Hello World!')
        super(TFModuleWrapper, self).__init__(wrapped.__name__)
        FastModuleType.set_getattr_callback(self, TFModuleWrapper._getattr)
        FastModuleType.set_getattribute_callback(self, TFModuleWrapper._getattribute)
        self.__dict__.update(wrapped.__dict__)
        self._tfmw_wrapped_module = wrapped
        self._tfmw_module_name = module_name
        self._tfmw_public_apis = public_apis
        self._tfmw_print_deprecation_warnings = deprecation
        self._tfmw_has_lite = has_lite
        self._tfmw_is_compat_v1 = wrapped.__name__.endswith('.compat.v1')
        if self._tfmw_public_apis:
            self._tfmw_wrapped_module.__all__ = list(self._tfmw_public_apis.keys())
            self.__all__ = list(self._tfmw_public_apis.keys())
        elif hasattr(self._tfmw_wrapped_module, '__all__'):
            self.__all__ = self._tfmw_wrapped_module.__all__
        else:
            self._tfmw_wrapped_module.__all__ = [attr for attr in dir(self._tfmw_wrapped_module) if not attr.startswith('_')]
            self.__all__ = self._tfmw_wrapped_module.__all__
        self._tfmw_deprecated_checked = set()
        self._tfmw_warning_count = 0

    def _tfmw_add_deprecation_warning(self, name, attr):
        if False:
            return 10
        'Print deprecation warning for attr with given name if necessary.'
        if self._tfmw_warning_count < _PER_MODULE_WARNING_LIMIT and name not in self._tfmw_deprecated_checked:
            self._tfmw_deprecated_checked.add(name)
            if self._tfmw_module_name:
                full_name = 'tf.%s.%s' % (self._tfmw_module_name, name)
            else:
                full_name = 'tf.%s' % name
            rename = get_rename_v2(full_name)
            if rename and (not has_deprecation_decorator(attr)):
                call_location = _call_location()
                if not call_location.startswith('<'):
                    logging.warning('From %s: The name %s is deprecated. Please use %s instead.\n', _call_location(), full_name, rename)
                    self._tfmw_warning_count += 1
                    return True
        return False

    def _tfmw_import_module(self, name):
        if False:
            print('Hello World!')
        'Lazily loading the modules.'
        if self._tfmw_is_compat_v1 and name != 'app' and (not TFModuleWrapper.compat_v1_usage_recorded):
            TFModuleWrapper.compat_v1_usage_recorded = True
            compat_v1_usage_gauge.get_cell().set(True)
        symbol_loc_info = self._tfmw_public_apis[name]
        if symbol_loc_info[0]:
            module = importlib.import_module(symbol_loc_info[0])
            attr = getattr(module, symbol_loc_info[1])
        else:
            attr = importlib.import_module(symbol_loc_info[1])
        setattr(self._tfmw_wrapped_module, name, attr)
        self.__dict__[name] = attr
        self._fastdict_insert(name, attr)
        return attr

    def _getattribute(self, name):
        if False:
            i = 10
            return i + 15
        'Imports and caches pre-defined API.\n\n    Warns if necessary.\n\n    This method is a replacement for __getattribute__(). It will be added into\n    the extended python module as a callback to reduce API overhead.\n    '
        func__fastdict_insert = object.__getattribute__(self, '_fastdict_insert')
        if name == 'lite':
            if self._tfmw_has_lite:
                attr = self._tfmw_import_module(name)
                setattr(self._tfmw_wrapped_module, 'lite', attr)
                func__fastdict_insert(name, attr)
                return attr
        attr = object.__getattribute__(self, name)
        if name.startswith('__') or name.startswith('_tfmw_') or name.startswith('_fastdict_'):
            func__fastdict_insert(name, attr)
            return attr
        if not (self._tfmw_print_deprecation_warnings and self._tfmw_add_deprecation_warning(name, attr)):
            func__fastdict_insert(name, attr)
        return attr

    def _getattr(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Imports and caches pre-defined API.\n\n    Warns if necessary.\n\n    This method is a replacement for __getattr__(). It will be added into the\n    extended python module as a callback to reduce API overhead. Instead of\n    relying on implicit AttributeError handling, this added callback function\n    will\n    be called explicitly from the extended C API if the default attribute lookup\n    fails.\n    '
        try:
            attr = getattr(self._tfmw_wrapped_module, name)
        except AttributeError:
            if not self._tfmw_public_apis:
                raise
            if name not in self._tfmw_public_apis:
                raise
            attr = self._tfmw_import_module(name)
        if self._tfmw_print_deprecation_warnings:
            self._tfmw_add_deprecation_warning(name, attr)
        return attr

    def __setattr__(self, arg, val):
        if False:
            for i in range(10):
                print('nop')
        if not arg.startswith('_tfmw_'):
            setattr(self._tfmw_wrapped_module, arg, val)
            self.__dict__[arg] = val
            if arg not in self.__all__ and arg != '__all__':
                self.__all__.append(arg)
            if self._fastdict_key_in(arg):
                self._fastdict_insert(arg, val)
        super(TFModuleWrapper, self).__setattr__(arg, val)

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._tfmw_public_apis:
            return list(set(self._tfmw_public_apis.keys()).union(set([attr for attr in dir(self._tfmw_wrapped_module) if not attr.startswith('_')])))
        else:
            return dir(self._tfmw_wrapped_module)

    def __delattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name.startswith('_tfmw_'):
            super(TFModuleWrapper, self).__delattr__(name)
        else:
            delattr(self._tfmw_wrapped_module, name)
            self.__dict__.pop(name)
            if name in self.__all__:
                self.__all__.remove(name)
            self._fastdict_pop(name)

    def __repr__(self):
        if False:
            return 10
        return self._tfmw_wrapped_module.__repr__()

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (importlib.import_module, (self.__name__,))