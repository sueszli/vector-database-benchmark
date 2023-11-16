from copy import deepcopy
from functools import partial
import importlib
import json
import os
import re
import yaml
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils import force_list, merge_dicts

@DeveloperAPI
def from_config(cls, config=None, **kwargs):
    if False:
        return 10
    'Uses the given config to create an object.\n\n    If `config` is a dict, an optional "type" key can be used as a\n    "constructor hint" to specify a certain class of the object.\n    If `config` is not a dict, `config`\'s value is used directly as this\n    "constructor hint".\n\n    The rest of `config` (if it\'s a dict) will be used as kwargs for the\n    constructor. Additional keys in **kwargs will always have precedence\n    (overwrite keys in `config` (if a dict)).\n    Also, if the config-dict or **kwargs contains the special key "_args",\n    it will be popped from the dict and used as *args list to be passed\n    separately to the constructor.\n\n    The following constructor hints are valid:\n    - None: Use `cls` as constructor.\n    - An already instantiated object: Will be returned as is; no\n        constructor call.\n    - A string or an object that is a key in `cls`\'s `__type_registry__`\n        dict: The value in `__type_registry__` for that key will be used\n        as the constructor.\n    - A python callable: Use that very callable as constructor.\n    - A string: Either a json/yaml filename or the name of a python\n        module+class (e.g. "ray.rllib. [...] .[some class name]")\n\n    Args:\n        cls: The class to build an instance for (from `config`).\n        config (Optional[dict, str]): The config dict or type-string or\n            filename.\n\n    Keyword Args:\n        kwargs: Optional possibility to pass the constructor arguments in\n            here and use `config` as the type-only info. Then we can call\n            this like: from_config([type]?, [**kwargs for constructor])\n            If `config` is already a dict, then `kwargs` will be merged\n            with `config` (overwriting keys in `config`) after "type" has\n            been popped out of `config`.\n            If a constructor of a Configurable needs *args, the special\n            key `_args` can be passed inside `kwargs` with a list value\n            (e.g. kwargs={"_args": [arg1, arg2, arg3]}).\n\n    Returns:\n        any: The object generated from the config.\n    '
    if config is None and isinstance(cls, (dict, str)):
        config = cls
        cls = None
    elif isinstance(cls, type) and isinstance(config, cls):
        return config
    try:
        config = deepcopy(config)
    except Exception:
        pass
    if isinstance(config, dict):
        type_ = config.pop('type', None)
        if type_ is None and isinstance(cls, str):
            type_ = cls
        ctor_kwargs = config
        ctor_kwargs.update(kwargs)
    else:
        type_ = config
        if type_ is None and 'type' in kwargs:
            type_ = kwargs.pop('type')
        ctor_kwargs = kwargs
    ctor_args = force_list(ctor_kwargs.pop('_args', []))
    if type_ is None:
        if cls is not None and hasattr(cls, '__default_constructor__') and (cls.__default_constructor__ is not None) and (ctor_args == []) and (not hasattr(cls.__bases__[0], '__default_constructor__') or cls.__bases__[0].__default_constructor__ is None or cls.__bases__[0].__default_constructor__ is not cls.__default_constructor__):
            constructor = cls.__default_constructor__
            if isinstance(constructor, partial):
                kwargs = merge_dicts(ctor_kwargs, constructor.keywords)
                constructor = partial(constructor.func, **kwargs)
                ctor_kwargs = {}
        else:
            constructor = cls
    else:
        constructor = _lookup_type(cls, type_)
        if constructor is not None:
            pass
        elif type_ is False or type_ is None:
            return type_
        elif callable(type_):
            constructor = type_
        elif isinstance(type_, str):
            if re.search('\\.(yaml|yml|json)$', type_):
                return from_file(cls, type_, *ctor_args, **ctor_kwargs)
            obj = yaml.safe_load(type_)
            if isinstance(obj, dict):
                return from_config(cls, obj)
            try:
                obj = from_config(cls, json.loads(type_))
            except json.JSONDecodeError:
                pass
            else:
                return obj
            if type_.find('.') != -1:
                (module_name, function_name) = type_.rsplit('.', 1)
                try:
                    module = importlib.import_module(module_name)
                    constructor = getattr(module, function_name)
                except (ModuleNotFoundError, ImportError, AttributeError):
                    pass
            if constructor is None:
                if isinstance(cls, str):
                    raise ValueError(f'Full classpath specifier ({type_}) must be a valid full [module].[class] string! E.g.: `my.cool.module.MyCoolClass`.')
                try:
                    module = importlib.import_module(cls.__module__)
                    constructor = getattr(module, type_)
                except (ModuleNotFoundError, ImportError, AttributeError):
                    try:
                        package_name = importlib.import_module(cls.__module__).__package__
                        module = __import__(package_name, fromlist=[type_])
                        constructor = getattr(module, type_)
                    except (ModuleNotFoundError, ImportError, AttributeError):
                        pass
            if constructor is None:
                raise ValueError(f"String specifier ({type_}) must be a valid filename, a [module].[class], a class within '{cls.__module__}', or a key into {cls.__name__}.__type_registry__!")
    if not constructor:
        raise TypeError("Invalid type '{}'. Cannot create `from_config`.".format(type_))
    try:
        object_ = constructor(*ctor_args, **ctor_kwargs)
    except TypeError as e:
        if re.match("Can't instantiate abstract class", e.args[0]):
            return None
        raise e
    if type(constructor).__name__ != 'function':
        assert isinstance(object_, constructor.func if isinstance(constructor, partial) else constructor)
    return object_

@DeveloperAPI
def from_file(cls, filename, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create object from config saved in filename. Expects json or yaml file.\n\n    Args:\n        filename: File containing the config (json or yaml).\n\n    Returns:\n        any: The object generated from the file.\n    '
    path = os.path.join(os.getcwd(), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("File '{}' not found!".format(filename))
    with open(path, 'rt') as fp:
        if path.endswith('.yaml') or path.endswith('.yml'):
            config = yaml.safe_load(fp)
        else:
            config = json.load(fp)
    config['_args'] = args
    return from_config(cls, config=config, **kwargs)

def _lookup_type(cls, type_):
    if False:
        while True:
            i = 10
    if cls is not None and hasattr(cls, '__type_registry__') and isinstance(cls.__type_registry__, dict) and (type_ in cls.__type_registry__ or (isinstance(type_, str) and re.sub('[\\W_]', '', type_.lower()) in cls.__type_registry__)):
        available_class_for_type = cls.__type_registry__.get(type_)
        if available_class_for_type is None:
            available_class_for_type = cls.__type_registry__[re.sub('[\\W_]', '', type_.lower())]
        return available_class_for_type
    return None

class _NotProvided:
    """Singleton class to provide a "not provided" value for AlgorithmConfig signatures.

    Using the only instance of this class indicates that the user does NOT wish to
    change the value of some property.

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        config = AlgorithmConfig()
        # Print out the default learning rate.
        print(config.lr)

    .. testoutput::

        0.001

    .. testcode::
        :skipif: True

        # Print out the default `preprocessor_pref`.
        print(config.preprocessor_pref)

    .. testoutput::

        "deepmind"

    .. testcode::
        :skipif: True

        # Will only set the `preprocessor_pref` property (to None) and leave
        # all other properties at their default values.
        config.training(preprocessor_pref=None)
        config.preprocessor_pref is None

    .. testoutput::

        True

    .. testcode::
        :skipif: True

        # Still the same value (didn't touch it in the call to `.training()`.
        print(config.lr)

    .. testoutput::

        0.001
    """

    class __NotProvided:
        pass
    instance = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        if _NotProvided.instance is None:
            _NotProvided.instance = _NotProvided.__NotProvided()
NotProvided = _NotProvided()