"""Utilities for exporting TensorFlow symbols to the API.

Exporting a function or a class:

To export a function or a class use tf_export decorator. For e.g.:
```python
@tf_export('foo', 'bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
tf_export explicitly. For e.g.:
```python
foo = get_foo(...)
tf_export('foo', 'bar.foo')(foo)
```


Exporting a constant
```python
foo = 1
tf_export('consts.foo').export_constant(__name__, 'foo')
```
"""
from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
ESTIMATOR_API_NAME = 'estimator'
KERAS_API_NAME = 'keras'
TENSORFLOW_API_NAME = 'tensorflow'
SUBPACKAGE_NAMESPACES = [ESTIMATOR_API_NAME]

class _Attributes(NamedTuple):
    names: str
    constants: str
API_ATTRS = {TENSORFLOW_API_NAME: _Attributes('_tf_api_names', '_tf_api_constants'), ESTIMATOR_API_NAME: _Attributes('_estimator_api_names', '_estimator_api_constants'), KERAS_API_NAME: _Attributes('_keras_api_names', '_keras_api_constants')}
API_ATTRS_V1 = {TENSORFLOW_API_NAME: _Attributes('_tf_api_names_v1', '_tf_api_constants_v1'), ESTIMATOR_API_NAME: _Attributes('_estimator_api_names_v1', '_estimator_api_constants_v1'), KERAS_API_NAME: _Attributes('_keras_api_names_v1', '_keras_api_constants_v1')}

class InvalidSymbolNameError(Exception):
    """Raised when trying to export symbol as an invalid or unallowed name."""
_NAME_TO_SYMBOL_MAPPING: dict[str, Any] = dict()

def get_symbol_from_name(name: str) -> Optional[Any]:
    if False:
        while True:
            i = 10
    return _NAME_TO_SYMBOL_MAPPING.get(name)

def get_canonical_name_for_symbol(symbol: Any, api_name: str=TENSORFLOW_API_NAME, add_prefix_to_v1_names: bool=False) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    "Get canonical name for the API symbol.\n\n  Example:\n  ```python\n  from tensorflow.python.util import tf_export\n  cls = tf_export.get_symbol_from_name('keras.optimizers.Adam')\n\n  # Gives `<class 'keras.optimizer_v2.adam.Adam'>`\n  print(cls)\n\n  # Gives `keras.optimizers.Adam`\n  print(tf_export.get_canonical_name_for_symbol(cls, api_name='keras'))\n  ```\n\n  Args:\n    symbol: API function or class.\n    api_name: API name (tensorflow or estimator).\n    add_prefix_to_v1_names: Specifies whether a name available only in V1 should\n      be prefixed with compat.v1.\n\n  Returns:\n    Canonical name for the API symbol (for e.g. initializers.zeros) if\n    canonical name could be determined. Otherwise, returns None.\n  "
    if not hasattr(symbol, '__dict__'):
        return None
    api_names_attr = API_ATTRS[api_name].names
    (_, undecorated_symbol) = tf_decorator.unwrap(symbol)
    if api_names_attr not in undecorated_symbol.__dict__:
        return None
    api_names = getattr(undecorated_symbol, api_names_attr)
    deprecated_api_names = undecorated_symbol.__dict__.get('_tf_deprecated_api_names', [])
    canonical_name = get_canonical_name(api_names, deprecated_api_names)
    if canonical_name:
        return canonical_name
    api_names_attr = API_ATTRS_V1[api_name].names
    api_names = getattr(undecorated_symbol, api_names_attr)
    v1_canonical_name = get_canonical_name(api_names, deprecated_api_names)
    if add_prefix_to_v1_names:
        return 'compat.v1.%s' % v1_canonical_name
    return v1_canonical_name

def get_canonical_name(api_names: Sequence[str], deprecated_api_names: Sequence[str]) -> Optional[str]:
    if False:
        return 10
    'Get preferred endpoint name.\n\n  Args:\n    api_names: API names iterable.\n    deprecated_api_names: Deprecated API names iterable.\n\n  Returns:\n    Returns one of the following in decreasing preference:\n    - first non-deprecated endpoint\n    - first endpoint\n    - None\n  '
    non_deprecated_name = next((name for name in api_names if name not in deprecated_api_names), None)
    if non_deprecated_name:
        return non_deprecated_name
    if api_names:
        return api_names[0]
    return None

def get_v1_names(symbol: Any) -> Sequence[str]:
    if False:
        i = 10
        return i + 15
    'Get a list of TF 1.* names for this symbol.\n\n  Args:\n    symbol: symbol to get API names for.\n\n  Returns:\n    List of all API names for this symbol including TensorFlow and\n    Estimator names.\n  '
    names_v1 = []
    tensorflow_api_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].names
    estimator_api_attr_v1 = API_ATTRS_V1[ESTIMATOR_API_NAME].names
    keras_api_attr_v1 = API_ATTRS_V1[KERAS_API_NAME].names
    if not hasattr(symbol, '__dict__'):
        return names_v1
    if tensorflow_api_attr_v1 in symbol.__dict__:
        names_v1.extend(getattr(symbol, tensorflow_api_attr_v1))
    if estimator_api_attr_v1 in symbol.__dict__:
        names_v1.extend(getattr(symbol, estimator_api_attr_v1))
    if keras_api_attr_v1 in symbol.__dict__:
        names_v1.extend(getattr(symbol, keras_api_attr_v1))
    return names_v1

def get_v2_names(symbol: Any) -> Sequence[str]:
    if False:
        while True:
            i = 10
    'Get a list of TF 2.0 names for this symbol.\n\n  Args:\n    symbol: symbol to get API names for.\n\n  Returns:\n    List of all API names for this symbol including TensorFlow and\n    Estimator names.\n  '
    names_v2 = []
    tensorflow_api_attr = API_ATTRS[TENSORFLOW_API_NAME].names
    estimator_api_attr = API_ATTRS[ESTIMATOR_API_NAME].names
    keras_api_attr = API_ATTRS[KERAS_API_NAME].names
    if not hasattr(symbol, '__dict__'):
        return names_v2
    if tensorflow_api_attr in symbol.__dict__:
        names_v2.extend(getattr(symbol, tensorflow_api_attr))
    if estimator_api_attr in symbol.__dict__:
        names_v2.extend(getattr(symbol, estimator_api_attr))
    if keras_api_attr in symbol.__dict__:
        names_v2.extend(getattr(symbol, keras_api_attr))
    return names_v2

def get_v1_constants(module: Any) -> Sequence[str]:
    if False:
        print('Hello World!')
    'Get a list of TF 1.* constants in this module.\n\n  Args:\n    module: TensorFlow module.\n\n  Returns:\n    List of all API constants under the given module including TensorFlow and\n    Estimator constants.\n  '
    constants_v1 = []
    tensorflow_constants_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].constants
    estimator_constants_attr_v1 = API_ATTRS_V1[ESTIMATOR_API_NAME].constants
    if hasattr(module, tensorflow_constants_attr_v1):
        constants_v1.extend(getattr(module, tensorflow_constants_attr_v1))
    if hasattr(module, estimator_constants_attr_v1):
        constants_v1.extend(getattr(module, estimator_constants_attr_v1))
    return constants_v1

def get_v2_constants(module: Any) -> Sequence[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get a list of TF 2.0 constants in this module.\n\n  Args:\n    module: TensorFlow module.\n\n  Returns:\n    List of all API constants under the given module including TensorFlow and\n    Estimator constants.\n  '
    constants_v2 = []
    tensorflow_constants_attr = API_ATTRS[TENSORFLOW_API_NAME].constants
    estimator_constants_attr = API_ATTRS[ESTIMATOR_API_NAME].constants
    if hasattr(module, tensorflow_constants_attr):
        constants_v2.extend(getattr(module, tensorflow_constants_attr))
    if hasattr(module, estimator_constants_attr):
        constants_v2.extend(getattr(module, estimator_constants_attr))
    return constants_v2
T = TypeVar('T')

class api_export(object):
    """Provides ways to export symbols to the TensorFlow API."""
    _names: Sequence[str]
    _names_v1: Sequence[str]
    _api_name: str

    def __init__(self, *args: str, api_name: str=TENSORFLOW_API_NAME, v1: Optional[Sequence[str]]=None, allow_multiple_exports: bool=True):
        if False:
            i = 10
            return i + 15
        'Export under the names *args (first one is considered canonical).\n\n    Args:\n      *args: API names in dot delimited format.\n      api_name: Name of the API you want to generate (e.g. `tensorflow` or\n        `estimator`). Default is `tensorflow`.\n      v1: Names for the TensorFlow V1 API. If not set, we will use V2 API names\n        both for TensorFlow V1 and V2 APIs.\n      allow_multiple_exports: Deprecated.\n    '
        self._names = args
        self._names_v1 = v1 if v1 is not None else args
        self._api_name = api_name
        self._validate_symbol_names()

    def _validate_symbol_names(self) -> None:
        if False:
            return 10
        'Validate you are exporting symbols under an allowed package.\n\n    We need to ensure things exported by tf_export, estimator_export, etc.\n    export symbols under disjoint top-level package names.\n\n    For TensorFlow, we check that it does not export anything under subpackage\n    names used by components (estimator, keras, etc.).\n\n    For each component, we check that it exports everything under its own\n    subpackage.\n\n    Raises:\n      InvalidSymbolNameError: If you try to export symbol under disallowed name.\n    '
        all_symbol_names = set(self._names) | set(self._names_v1)
        if self._api_name == TENSORFLOW_API_NAME:
            for subpackage in SUBPACKAGE_NAMESPACES:
                if any((n.startswith(subpackage) for n in all_symbol_names)):
                    raise InvalidSymbolNameError('@tf_export is not allowed to export symbols under %s.*' % subpackage)
        elif not all((n.startswith(self._api_name) for n in all_symbol_names)):
            raise InvalidSymbolNameError('Can only export symbols under package name of component. e.g. tensorflow_estimator must export all symbols under tf.estimator')

    def __call__(self, func: T) -> T:
        if False:
            return 10
        'Calls this decorator.\n\n    Args:\n      func: decorated symbol (function or class).\n\n    Returns:\n      The input function with _tf_api_names attribute set.\n    '
        api_names_attr = API_ATTRS[self._api_name].names
        api_names_attr_v1 = API_ATTRS_V1[self._api_name].names
        (_, undecorated_func) = tf_decorator.unwrap(func)
        self.set_attr(undecorated_func, api_names_attr, self._names)
        self.set_attr(undecorated_func, api_names_attr_v1, self._names_v1)
        for name in self._names:
            _NAME_TO_SYMBOL_MAPPING[name] = func
        for name_v1 in self._names_v1:
            _NAME_TO_SYMBOL_MAPPING['compat.v1.%s' % name_v1] = func
        return func

    def set_attr(self, func: Any, api_names_attr: str, names: Sequence[str]) -> None:
        if False:
            return 10
        setattr(func, api_names_attr, names)

    def export_constant(self, module_name: str, name: str) -> None:
        if False:
            while True:
                i = 10
        'Store export information for constants/string literals.\n\n    Export information is stored in the module where constants/string literals\n    are defined.\n\n    e.g.\n    ```python\n    foo = 1\n    bar = 2\n    tf_export("consts.foo").export_constant(__name__, \'foo\')\n    tf_export("consts.bar").export_constant(__name__, \'bar\')\n    ```\n\n    Args:\n      module_name: (string) Name of the module to store constant at.\n      name: (string) Current constant name.\n    '
        module = sys.modules[module_name]
        api_constants_attr = API_ATTRS[self._api_name].constants
        api_constants_attr_v1 = API_ATTRS_V1[self._api_name].constants
        if not hasattr(module, api_constants_attr):
            setattr(module, api_constants_attr, [])
        getattr(module, api_constants_attr).append((self._names, name))
        if not hasattr(module, api_constants_attr_v1):
            setattr(module, api_constants_attr_v1, [])
        getattr(module, api_constants_attr_v1).append((self._names_v1, name))

def kwarg_only(f: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'A wrapper that throws away all non-kwarg arguments.'
    f_argspec = tf_inspect.getfullargspec(f)

    def wrapper(*args, **kwargs):
        if False:
            return 10
        if args:
            raise TypeError('{f} only takes keyword args (possible keys: {kwargs}). Please pass these args as kwargs instead.'.format(f=f.__name__, kwargs=f_argspec.args))
        return f(**kwargs)
    return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)

class ExportType(Protocol):

    def __call__(self, *v2: str, v1: Optional[Sequence[str]]=None, allow_multiple_exports: bool=True) -> api_export:
        if False:
            for i in range(10):
                print('nop')
        ...
tf_export: ExportType = functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
keras_export: ExportType = functools.partial(api_export, api_name=KERAS_API_NAME)