import copy
from itertools import chain
import json
import logging
import os
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from os import PathLike
from typing import Any, Dict, List, Union, Optional, TypeVar, Iterable, Set
try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: str, **_kwargs) -> str:
        if False:
            while True:
                i = 10
        logger.warning(f'error loading _jsonnet (this is expected on Windows), treating {filename} as plain json')
        with open(filename, 'r') as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs) -> str:
        if False:
            while True:
                i = 10
        logger.warning('error loading _jsonnet (this is expected on Windows), treating snippet as plain json')
        return expr
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
logger = logging.getLogger(__name__)

def infer_and_cast(value: Any):
    if False:
        return 10
    "\n    In some cases we'll be feeding params dicts to functions we don't own;\n    for example, PyTorch optimizers. In that case we can't use `pop_int`\n    or similar to force casts (which means you can't specify `int` parameters\n    using environment variables). This function takes something that looks JSON-like\n    and recursively casts things that look like (bool, int, float) to (bool, int, float).\n    "
    if isinstance(value, (int, float, bool)):
        return value
    elif isinstance(value, list):
        return [infer_and_cast(item) for item in value]
    elif isinstance(value, dict):
        return {key: infer_and_cast(item) for (key, item) in value.items()}
    elif isinstance(value, str):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                return value
    else:
        raise ValueError(f'cannot infer type of {value}')

def _is_encodable(value: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    We need to filter out environment variables that can\'t\n    be unicode-encoded to avoid a "surrogates not allowed"\n    error in jsonnet.\n    '
    return value == '' or value.encode('utf-8', 'ignore') != b''

def _environment_variables() -> Dict[str, str]:
    if False:
        return 10
    '\n    Wraps `os.environ` to filter out non-encodable values.\n    '
    return {key: value for (key, value) in os.environ.items() if _is_encodable(value)}
T = TypeVar('T', dict, list)

def with_overrides(original: T, overrides_dict: Dict[str, Any], prefix: str='') -> T:
    if False:
        print('Hello World!')
    merged: T
    keys: Union[Iterable[str], Iterable[int]]
    if isinstance(original, list):
        merged = [None] * len(original)
        keys = range(len(original))
    elif isinstance(original, dict):
        merged = {}
        keys = chain(original.keys(), (k for k in overrides_dict if '.' not in k and k not in original))
    elif prefix:
        raise ValueError(f"overrides for '{prefix[:-1]}.*' expected list or dict in original, found {type(original)} instead")
    else:
        raise ValueError(f'expected list or dict, found {type(original)} instead')
    used_override_keys: Set[str] = set()
    for key in keys:
        if str(key) in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[str(key)])
            used_override_keys.add(str(key))
        else:
            overrides_subdict = {}
            for o_key in overrides_dict:
                if o_key.startswith(f'{key}.'):
                    overrides_subdict[o_key[len(f'{key}.'):]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = with_overrides(original[key], overrides_subdict, prefix=prefix + f'{key}.')
            else:
                merged[key] = copy.deepcopy(original[key])
    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f'overrides dict contains unused keys: {unused_override_keys}')
    return merged

def parse_overrides(serialized_overrides: str, ext_vars: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    if serialized_overrides:
        ext_vars = {**_environment_variables(), **(ext_vars or {})}
        return json.loads(evaluate_snippet('', serialized_overrides, ext_vars=ext_vars))
    else:
        return {}

def _is_dict_free(obj: Any) -> bool:
    if False:
        print('Hello World!')
    "\n    Returns False if obj is a dict, or if it's a list with an element that _has_dict.\n    "
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all((_is_dict_free(item) for item in obj))
    else:
        return True

class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.

    There are currently two benefits of a `Params` object over a plain dictionary for parameter
    passing:

    1. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    2. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON file, because
       those may not specify what default values were used, whereas this will log them.

    !!! Consumption
        The convention for using a `Params` object in AllenNLP is that you will consume the parameters
        as you read them, so that there are none left when you've read everything you expect.  This
        lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
        that the parameter dictionary is empty.  You should do this when you're done handling
        parameters, by calling `Params.assert_empty`.
    """
    DEFAULT = object()

    def __init__(self, params: Dict[str, Any], history: str='') -> None:
        if False:
            print('Hello World!')
        self.params = _replace_none(params)
        self.history = history

    def pop(self, key: str, default: Any=DEFAULT, keep_as_dict: bool=False) -> Any:
        if False:
            return 10
        '\n        Performs the functionality associated with dict.pop(key), along with checking for\n        returned dictionaries, replacing them with Param objects with an updated history\n        (unless keep_as_dict is True, in which case we leave them as dictionaries).\n\n        If `key` is not present in the dictionary, and no default was specified, we raise a\n        `ConfigurationError`, instead of the typical `KeyError`.\n        '
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                msg = f'key "{key}" is required'
                if self.history:
                    msg += f' at location "{self.history}"'
                raise ConfigurationError(msg)
        else:
            value = self.params.pop(key, default)
        if keep_as_dict or _is_dict_free(value):
            logger.info(f'{self.history}{key} = {value}')
            return value
        else:
            return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any=DEFAULT) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        '\n        Performs a pop and coerces to an int.\n        '
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any=DEFAULT) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a pop and coerces to a float.\n        '
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any=DEFAULT) -> Optional[bool]:
        if False:
            print('Hello World!')
        '\n        Performs a pop and coerces to a bool.\n        '
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise ValueError('Cannot convert variable to bool: ' + value)

    def get(self, key: str, default: Any=DEFAULT):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs the functionality associated with dict.get(key) but also checks for returned\n        dicts and returns a Params object in their place with an updated history.\n        '
        default = None if default is self.DEFAULT else default
        value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any], default_to_first_choice: bool=False, allow_class_names: bool=True) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the value of `key` in the `params` dictionary, ensuring that the value is one of\n        the given choices. Note that this `pops` the key from params, modifying the dictionary,\n        consistent with how parameters are processed in this codebase.\n\n        # Parameters\n\n        key: `str`\n\n            Key to get the value from in the param dictionary\n\n        choices: `List[Any]`\n\n            A list of valid options for values corresponding to `key`.  For example, if you\'re\n            specifying the type of encoder to use for some part of your model, the choices might be\n            the list of encoder classes we know about and can instantiate.  If the value we find in\n            the param dictionary is not in `choices`, we raise a `ConfigurationError`, because\n            the user specified an invalid value in their parameter file.\n\n        default_to_first_choice: `bool`, optional (default = `False`)\n\n            If this is `True`, we allow the `key` to not be present in the parameter\n            dictionary.  If the key is not present, we will use the return as the value the first\n            choice in the `choices` list.  If this is `False`, we raise a\n            `ConfigurationError`, because specifying the `key` is required (e.g., you `have` to\n            specify your model class when running an experiment, but you can feel free to use\n            default settings for encoders if you want).\n\n        allow_class_names: `bool`, optional (default = `True`)\n\n            If this is `True`, then we allow unknown choices that look like fully-qualified class names.\n            This is to allow e.g. specifying a model type as my_library.my_model.MyModel\n            and importing it on the fly. Our check for "looks like" is extremely lenient\n            and consists of checking that the value contains a \'.\'.\n        '
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and '.' in value
        if value not in choices and (not ok_because_class_name):
            key_str = self.history + key
            message = f'{value} not in acceptable choices for {key_str}: {choices}. You should either use the --include-package flag to make sure the correct module is loaded, or use a fully qualified class name in your config file like {{"model": "my_module.models.MyModel"}} to have it imported automatically.'
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: bool=False, infer_type_and_cast: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Sometimes we need to just represent the parameters as a dict, for instance when we pass\n        them to PyTorch code.\n\n        # Parameters\n\n        quiet: `bool`, optional (default = `False`)\n\n            Whether to log the parameters before returning them as a dict.\n\n        infer_type_and_cast: `bool`, optional (default = `False`)\n\n            If True, we infer types and cast (e.g. things that look like floats to floats).\n        '
        if infer_type_and_cast:
            params_as_dict = infer_and_cast(self.params)
        else:
            params_as_dict = self.params
        if quiet:
            return params_as_dict

        def log_recursively(parameters, history):
            if False:
                print('Hello World!')
            for (key, value) in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + '.'
                    log_recursively(value, new_local_history)
                else:
                    logger.info(f'{history}{key} = {value}')
        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Returns the parameters of a flat dictionary from keys to values.\n        Nested structure is collapsed with periods.\n        '
        flat_params = {}

        def recurse(parameters, path):
            if False:
                for i in range(10):
                    print('nop')
            for (key, value) in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value
        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> 'Params':
        if False:
            i = 10
            return i + 15
        '\n        Uses `copy.deepcopy()` to create a duplicate (but fully distinct)\n        copy of these Params.\n        '
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str):
        if False:
            print('Hello World!')
        '\n        Raises a `ConfigurationError` if `self.params` is not empty.  We take `class_name` as\n        an argument so that the error message gives some idea of where an error happened, if there\n        was one.  `class_name` should be the name of the `calling` class, the one that got extra\n        parameters (if there are any).\n        '
        if self.params:
            raise ConfigurationError('Extra parameters passed to {}: {}'.format(class_name, self.params))

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError(str(key))

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.params[key] = value

    def __delitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self.params[key]

    def __iter__(self):
        if False:
            return 10
        return iter(self.params)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if False:
            while True:
                i = 10
        if isinstance(value, dict):
            new_history = self.history + new_history + '.'
            return Params(value, history=new_history)
        if isinstance(value, list):
            value = [self._check_is_dict(f'{new_history}.{i}', v) for (i, v) in enumerate(value)]
        return value

    @classmethod
    def from_file(cls, params_file: Union[str, PathLike], params_overrides: Union[str, Dict[str, Any]]='', ext_vars: dict=None) -> 'Params':
        if False:
            for i in range(10):
                print('nop')
        '\n        Load a `Params` object from a configuration file.\n\n        # Parameters\n\n        params_file: `str`\n\n            The path to the configuration file to load.\n\n        params_overrides: `Union[str, Dict[str, Any]]`, optional (default = `""`)\n\n            A dict of overrides that can be applied to final object.\n            e.g. `{"model.embedding_dim": 10}` will change the value of "embedding_dim"\n            within the "model" object of the config to 10. If you wanted to override the entire\n            "model" object of the config, you could do `{"model": {"type": "other_type", ...}}`.\n\n        ext_vars: `dict`, optional\n\n            Our config files are Jsonnet, which allows specifying external variables\n            for later substitution. Typically we substitute these using environment\n            variables; however, you can also specify them here, in which case they\n            take priority over environment variables.\n            e.g. {"HOME_DIR": "/Users/allennlp/home"}\n        '
        if ext_vars is None:
            ext_vars = {}
        params_file = cached_path(params_file)
        ext_vars = {**_environment_variables(), **ext_vars}
        file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))
        if isinstance(params_overrides, dict):
            params_overrides = json.dumps(params_overrides)
        overrides_dict = parse_overrides(params_overrides, ext_vars=ext_vars)
        if overrides_dict:
            param_dict = with_overrides(file_dict, overrides_dict)
        else:
            param_dict = file_dict
        return cls(param_dict)

    def to_file(self, params_file: str, preference_orders: List[List[str]]=None) -> None:
        if False:
            while True:
                i = 10
        with open(params_file, 'w') as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: List[List[str]]=None) -> OrderedDict:
        if False:
            print('Hello World!')
        '\n        Returns Ordered Dict of Params from list of partial order preferences.\n\n        # Parameters\n\n        preference_orders: `List[List[str]]`, optional\n\n            `preference_orders` is list of partial preference orders. ["A", "B", "C"] means\n            "A" > "B" > "C". For multiple preference_orders first will be considered first.\n            Keys not found, will have last but alphabetical preference. Default Preferences:\n            `[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",\n            "test_data_path", "trainer", "vocabulary"], ["type"]]`\n        '
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(['dataset_reader', 'iterator', 'model', 'train_data_path', 'validation_data_path', 'test_data_path', 'trainer', 'vocabulary'])
            preference_orders.append(['type'])

        def order_func(key):
            if False:
                while True:
                    i = 10
            order_tuple = [order.index(key) if key in order else len(order) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            if False:
                print('Hello World!')
            result = OrderedDict()
            for (key, val) in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
            return result
        return order_dict(params_dict, order_func)

    def get_hash(self) -> str:
        if False:
            print('Hello World!')
        "\n        Returns a hash code representing the current state of this `Params` object.  We don't\n        want to implement `__hash__` because that has deeper python implications (and this is a\n        mutable object), but this will give you a representation of the current state.\n        We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the\n        latter is reset on each new program invocation, as discussed here:\n        https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.\n        "
        dumped = json.dumps(self.params, sort_keys=True)
        hashed = zlib.adler32(dumped.encode())
        return str(hashed)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.history}Params({self.params})'

def pop_choice(params: Dict[str, Any], key: str, choices: List[Any], default_to_first_choice: bool=False, history: str='?.', allow_class_names: bool=True) -> Any:
    if False:
        i = 10
        return i + 15
    '\n    Performs the same function as `Params.pop_choice`, but is required in order to deal with\n    places that the Params object is not welcome, such as inside Keras layers.  See the docstring\n    of that method for more detail on how this function works.\n\n    This method adds a `history` parameter, in the off-chance that you know it, so that we can\n    reproduce `Params.pop_choice` exactly.  We default to using "?." if you don\'t know the\n    history, so you\'ll have to fix that in the log if you want to actually recover the logged\n    parameters.\n    '
    value = Params(params, history).pop_choice(key, choices, default_to_first_choice, allow_class_names=allow_class_names)
    return value

def _replace_none(params: Any) -> Any:
    if False:
        i = 10
        return i + 15
    if params == 'None':
        return None
    elif isinstance(params, dict):
        for (key, value) in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params

def remove_keys_from_params(params: Params, keys: List[str]=['pretrained_file', 'initializer']):
    if False:
        i = 10
        return i + 15
    if isinstance(params, Params):
        param_keys = params.keys()
        for key in keys:
            if key in param_keys:
                del params[key]
        for value in params.values():
            if isinstance(value, Params):
                remove_keys_from_params(value, keys)