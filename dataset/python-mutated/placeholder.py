from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
ID_HASH_LENGTH = 8

def create_resolvers_map():
    if False:
        return 10
    return defaultdict(list)

def _id_hash(path_tuple):
    if False:
        i = 10
        return i + 15
    'Compute a hash for the specific placeholder based on its path.'
    return hashlib.sha1(str(path_tuple).encode('utf-8')).hexdigest()[:ID_HASH_LENGTH]

class _FunctionResolver:
    """Replaced value for function typed objects."""
    TOKEN = '__fn_ph'

    def __init__(self, hash, fn):
        if False:
            while True:
                i = 10
        self.hash = hash
        self._fn = fn

    def resolve(self, config: Dict):
        if False:
            i = 10
            return i + 15
        'Some functions take a resolved spec dict as input.\n\n        Note: Function placeholders are independently sampled during\n        resolution. Therefore their random states are not restored.\n        '
        return self._fn.sample(config=config)

    def get_placeholder(self) -> str:
        if False:
            i = 10
            return i + 15
        return (self.TOKEN, self.hash)

class _RefResolver:
    """Replaced value for all other non-primitive objects."""
    TOKEN = '__ref_ph'

    def __init__(self, hash, value):
        if False:
            i = 10
            return i + 15
        self.hash = hash
        self._value = value

    def resolve(self):
        if False:
            print('Hello World!')
        return self._value

    def get_placeholder(self) -> str:
        if False:
            while True:
                i = 10
        return (self.TOKEN, self.hash)

def _is_primitive(x):
    if False:
        while True:
            i = 10
    'Returns True if x is a primitive type.\n\n    Primitive types are int, float, str, bool, and None.\n    '
    return isinstance(x, (int, float, str, bool)) or x is None

@DeveloperAPI
def inject_placeholders(config: Any, resolvers: defaultdict, id_prefix: Tuple=(), path_prefix: Tuple=()) -> Dict:
    if False:
        print('Hello World!')
    'Replaces reference objects contained by a config dict with placeholders.\n\n    Given a config dict, this function replaces all reference objects contained\n    by this dict with placeholder strings. It recursively expands nested dicts\n    and lists, and properly handles Tune native search objects such as Categorical\n    and Function.\n    This makes sure the config dict only contains primitive typed values, which\n    can then be handled by different search algorithms.\n\n    A few details about id_prefix and path_prefix. Consider the following config,\n    where "param1" is a simple grid search of 3 tuples.\n\n    config = {\n        "param1": tune.grid_search([\n            (Cat, None, None),\n            (None, Dog, None),\n            (None, None, Fish),\n        ]),\n    }\n\n    We will replace the 3 objects contained with placeholders. And after trial\n    expansion, the config may look like this:\n\n    config = {\n        "param1": (None, (placeholder, hash), None)\n    }\n\n    Now you need 2 pieces of information to resolve the placeholder. One is the\n    path of ("param1", 1), which tells you that the first element of the tuple\n    under "param1" key is a placeholder that needs to be resolved.\n    The other is the mapping from the placeholder to the actual object. In this\n    case hash -> Dog.\n\n    id and path prefixes serve exactly this purpose here. The difference between\n    these two is that id_prefix is the location of the value in the pre-injected\n    config tree. So if a value is the second option in a grid_search, it gets an\n    id part of 1. Injected placeholders all get unique id prefixes. path prefix\n    identifies a placeholder in the expanded config tree. So for example, all\n    options of a single grid_search will get the same path prefix. This is how\n    we know which location has a placeholder to be resolved in the post-expansion\n    tree.\n\n    Args:\n        config: The config dict to replace references in.\n        resolvers: A dict from path to replaced objects.\n        id_prefix: The prefix to prepend to id every single placeholders.\n        path_prefix: The prefix to prepend to every path identifying\n            potential locations of placeholders in an expanded tree.\n\n    Returns:\n        The config with all references replaced.\n    '
    if isinstance(config, dict) and 'grid_search' in config and (len(config) == 1):
        config['grid_search'] = [inject_placeholders(choice, resolvers, id_prefix + (i,), path_prefix) for (i, choice) in enumerate(config['grid_search'])]
        return config
    elif isinstance(config, dict):
        return {k: inject_placeholders(v, resolvers, id_prefix + (k,), path_prefix + (k,)) for (k, v) in config.items()}
    elif isinstance(config, list):
        return [inject_placeholders(elem, resolvers, id_prefix + (i,), path_prefix + (i,)) for (i, elem) in enumerate(config)]
    elif isinstance(config, tuple):
        return tuple((inject_placeholders(elem, resolvers, id_prefix + (i,), path_prefix + (i,)) for (i, elem) in enumerate(config)))
    elif _is_primitive(config):
        return config
    elif isinstance(config, Categorical):
        config.categories = [inject_placeholders(choice, resolvers, id_prefix + (i,), path_prefix) for (i, choice) in enumerate(config.categories)]
        return config
    elif isinstance(config, Function):
        id_hash = _id_hash(id_prefix)
        v = _FunctionResolver(id_hash, config)
        resolvers[path_prefix].append(v)
        return v.get_placeholder()
    elif not isinstance(config, Domain):
        id_hash = _id_hash(id_prefix)
        v = _RefResolver(id_hash, config)
        resolvers[path_prefix].append(v)
        return v.get_placeholder()
    else:
        return config

def _get_placeholder(config: Any, prefix: Tuple, path: Tuple):
    if False:
        return 10
    if not path:
        return (prefix, config)
    key = path[0]
    if isinstance(config, tuple):
        if config[0] in (_FunctionResolver.TOKEN, _RefResolver.TOKEN):
            return (prefix, config)
        elif key < len(config):
            return _get_placeholder(config[key], prefix=prefix + (path[0],), path=path[1:])
    elif isinstance(config, dict) and key in config or (isinstance(config, list) and key < len(config)):
        return _get_placeholder(config[key], prefix=prefix + (path[0],), path=path[1:])
    return (None, None)

@DeveloperAPI
def resolve_placeholders(config: Any, replaced: defaultdict):
    if False:
        print('Hello World!')
    'Replaces placeholders contained by a config dict with the original values.\n\n    Args:\n        config: The config to replace placeholders in.\n        replaced: A dict from path to replaced objects.\n    '

    def __resolve(resolver_type, args):
        if False:
            for i in range(10):
                print('nop')
        for (path, resolvers) in replaced.items():
            assert resolvers
            if not isinstance(resolvers[0], resolver_type):
                continue
            (prefix, ph) = _get_placeholder(config, (), path)
            if not ph:
                continue
            for resolver in resolvers:
                if resolver.hash != ph[1]:
                    continue
                assign_value(config, prefix, resolver.resolve(*args))
    __resolve(_RefResolver, args=())
    __resolve(_FunctionResolver, args=(config,))