import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
logger = logging.getLogger(__name__)

@DeveloperAPI
def generate_variants(unresolved_spec: Dict, constant_grid_search: bool=False, random_state: 'RandomState'=None) -> Generator[Tuple[Dict, Dict], None, None]:
    if False:
        print('Hello World!')
    'Generates variants from a spec (dict) with unresolved values.\n\n    There are two types of unresolved values:\n\n        Grid search: These define a grid search over values. For example, the\n        following grid search values in a spec will produce six distinct\n        variants in combination:\n\n            "activation": grid_search(["relu", "tanh"])\n            "learning_rate": grid_search([1e-3, 1e-4, 1e-5])\n\n        Lambda functions: These are evaluated to produce a concrete value, and\n        can express dependencies or conditional distributions between values.\n        They can also be used to express random search (e.g., by calling\n        into the `random` or `np` module).\n\n            "cpu": lambda spec: spec.config.num_workers\n            "batch_size": lambda spec: random.uniform(1, 1000)\n\n    Finally, to support defining specs in plain JSON / YAML, grid search\n    and lambda functions can also be defined alternatively as follows:\n\n        "activation": {"grid_search": ["relu", "tanh"]}\n        "cpu": {"eval": "spec.config.num_workers"}\n\n    Use `format_vars` to format the returned dict of hyperparameters.\n\n    Yields:\n        (Dict of resolved variables, Spec object)\n    '
    for (resolved_vars, spec) in _generate_variants_internal(unresolved_spec, constant_grid_search=constant_grid_search, random_state=random_state):
        assert not _unresolved_values(spec)
        yield (resolved_vars, spec)

@PublicAPI(stability='beta')
def grid_search(values: Iterable) -> Dict[str, Iterable]:
    if False:
        i = 10
        return i + 15
    'Specify a grid of values to search over.\n\n    Values specified in a grid search are guaranteed to be sampled.\n\n    If multiple grid search variables are defined, they are combined with the\n    combinatorial product. This means every possible combination of values will\n    be sampled.\n\n    Example:\n\n        >>> from ray import tune\n        >>> param_space={\n        ...   "x": tune.grid_search([10, 20]),\n        ...   "y": tune.grid_search(["a", "b", "c"])\n        ... }\n\n    This will create a grid of 6 samples:\n    ``{"x": 10, "y": "a"}``, ``{"x": 10, "y": "b"}``, etc.\n\n    When specifying ``num_samples`` in the\n    :class:`TuneConfig <ray.tune.tune_config.TuneConfig>`, this will specify\n    the number of random samples per grid search combination.\n\n    For instance, in the example above, if ``num_samples=4``,\n    a total of 24 trials will be started -\n    4 trials for each of the 6 grid search combinations.\n\n    Args:\n        values: An iterable whose parameters will be used for creating a trial grid.\n\n    '
    return {'grid_search': values}
_STANDARD_IMPORTS = {'random': random, 'np': numpy}
_MAX_RESOLUTION_PASSES = 20

def _resolve_nested_dict(nested_dict: Dict) -> Dict[Tuple, Any]:
    if False:
        while True:
            i = 10
    'Flattens a nested dict by joining keys into tuple of paths.\n\n    Can then be passed into `format_vars`.\n    '
    res = {}
    for (k, v) in nested_dict.items():
        if isinstance(v, dict):
            for (k_, v_) in _resolve_nested_dict(v).items():
                res[(k,) + k_] = v_
        else:
            res[k,] = v
    return res

@DeveloperAPI
def format_vars(resolved_vars: Dict) -> str:
    if False:
        return 10
    'Format variables to be used as experiment tags.\n\n    Experiment tags are used in directory names, so this method makes sure\n    the resulting tags can be legally used in directory names on all systems.\n\n    The input to this function is a dict of the form\n    ``{("nested", "config", "path"): "value"}``. The output will be a comma\n    separated string of the form ``last_key=value``, so in this example\n    ``path=value``.\n\n    Note that the sanitizing implies that empty strings are possible return\n    values. This is expected and acceptable, as it is not a common case and\n    the resulting directory names will still be valid.\n\n    Args:\n        resolved_vars: Dictionary mapping from config path tuples to a value.\n\n    Returns:\n        Comma-separated key=value string.\n    '
    vars = resolved_vars.copy()
    for v in ['run', 'env', 'resources_per_trial']:
        vars.pop(v, None)
    return ','.join((f'{_clean_value(k[-1])}={_clean_value(v)}' for (k, v) in sorted(vars.items())))

def _flatten_resolved_vars(resolved_vars: Dict) -> Dict:
    if False:
        while True:
            i = 10
    'Formats the resolved variable dict into a mapping of (str -> value).'
    flattened_resolved_vars_dict = {}
    for (pieces, value) in resolved_vars.items():
        if pieces[0] == 'config':
            pieces = pieces[1:]
        pieces = [str(piece) for piece in pieces]
        flattened_resolved_vars_dict['/'.join(pieces)] = value
    return flattened_resolved_vars_dict

def _clean_value(value: Any) -> str:
    if False:
        return 10
    'Format floats and replace invalid string characters with ``_``.'
    if isinstance(value, float):
        return f'{value:.4f}'
    else:
        invalid_alphabet = '[^a-zA-Z0-9_-]+'
        return re.sub(invalid_alphabet, '_', str(value)).strip('_')

@DeveloperAPI
def parse_spec_vars(spec: Dict) -> Tuple[List[Tuple[Tuple, Any]], List[Tuple[Tuple, Any]], List[Tuple[Tuple, Any]]]:
    if False:
        return 10
    (resolved, unresolved) = _split_resolved_unresolved_values(spec)
    resolved_vars = list(resolved.items())
    if not unresolved:
        return (resolved_vars, [], [])
    grid_vars = []
    domain_vars = []
    for (path, value) in unresolved.items():
        if value.is_grid():
            grid_vars.append((path, value))
        else:
            domain_vars.append((path, value))
    grid_vars.sort()
    return (resolved_vars, domain_vars, grid_vars)

def _count_spec_samples(spec: Dict, num_samples=1) -> int:
    if False:
        i = 10
        return i + 15
    'Count samples for a specific spec'
    (_, domain_vars, grid_vars) = parse_spec_vars(spec)
    grid_count = 1
    for (path, domain) in grid_vars:
        grid_count *= len(domain.categories)
    return num_samples * grid_count

def _count_variants(spec: Dict, presets: Optional[List[Dict]]=None) -> int:
    if False:
        print('Hello World!')

    def deep_update(d, u):
        if False:
            return 10
        for (k, v) in u.items():
            if isinstance(v, Mapping):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    total_samples = 0
    total_num_samples = spec.get('num_samples', 1)
    for preset in presets:
        preset_spec = copy.deepcopy(spec)
        deep_update(preset_spec['config'], preset)
        total_samples += _count_spec_samples(preset_spec, 1)
        total_num_samples -= 1
    if total_num_samples > 0:
        total_samples += _count_spec_samples(spec, total_num_samples)
    return total_samples

def _generate_variants_internal(spec: Dict, constant_grid_search: bool=False, random_state: 'RandomState'=None) -> Tuple[Dict, Dict]:
    if False:
        while True:
            i = 10
    spec = copy.deepcopy(spec)
    (_, domain_vars, grid_vars) = parse_spec_vars(spec)
    if not domain_vars and (not grid_vars):
        yield ({}, spec)
        return
    to_resolve = domain_vars
    all_resolved = True
    if constant_grid_search:
        (all_resolved, resolved_vars) = _resolve_domain_vars(spec, domain_vars, allow_fail=True, random_state=random_state)
        if not all_resolved:
            to_resolve = [(r, d) for (r, d) in to_resolve if r not in resolved_vars]
    grid_search = _grid_search_generator(spec, grid_vars)
    for resolved_spec in grid_search:
        if not constant_grid_search or not all_resolved:
            (_, resolved_vars) = _resolve_domain_vars(resolved_spec, to_resolve, random_state=random_state)
        for (resolved, spec) in _generate_variants_internal(resolved_spec, constant_grid_search=constant_grid_search, random_state=random_state):
            for (path, value) in grid_vars:
                resolved_vars[path] = _get_value(spec, path)
            for (k, v) in resolved.items():
                if k in resolved_vars and v != resolved_vars[k] and _is_resolved(resolved_vars[k]):
                    raise ValueError('The variable `{}` could not be unambiguously resolved to a single value. Consider simplifying your configuration.'.format(k))
                resolved_vars[k] = v
            yield (resolved_vars, spec)

def _get_preset_variants(spec: Dict, config: Dict, constant_grid_search: bool=False, random_state: 'RandomState'=None):
    if False:
        for i in range(10):
            print('nop')
    'Get variants according to a spec, initialized with a config.\n\n    Variables from the spec are overwritten by the variables in the config.\n    Thus, we may end up with less sampled parameters.\n\n    This function also checks if values used to overwrite search space\n    parameters are valid, and logs a warning if not.\n    '
    spec = copy.deepcopy(spec)
    (resolved, _, _) = parse_spec_vars(config)
    for (path, val) in resolved:
        try:
            domain = _get_value(spec['config'], path)
            if isinstance(domain, dict):
                if 'grid_search' in domain:
                    domain = Categorical(domain['grid_search'])
                else:
                    domain = None
        except IndexError as exc:
            raise ValueError(f"Pre-set config key `{'/'.join(path)}` does not correspond to a valid key in the search space definition. Please add this path to the `param_space` variable passed to `tune.Tuner()`.") from exc
        if domain:
            if isinstance(domain, Domain):
                if not domain.is_valid(val):
                    logger.warning(f"Pre-set value `{val}` is not within valid values of parameter `{'/'.join(path)}`: {domain.domain_str}")
            elif domain != val:
                logger.warning(f"Pre-set value `{val}` is not equal to the value of parameter `{'/'.join(path)}`: {domain}")
        assign_value(spec['config'], path, val)
    return _generate_variants_internal(spec, constant_grid_search=constant_grid_search, random_state=random_state)

@DeveloperAPI
def assign_value(spec: Dict, path: Tuple, value: Any):
    if False:
        return 10
    'Assigns a value to a nested dictionary.\n\n    Handles the special case of tuples, in which case the tuples\n    will be re-constructed to accomodate the updated value.\n    '
    parent_spec = None
    parent_key = None
    for k in path[:-1]:
        parent_spec = spec
        parent_key = k
        spec = spec[k]
    key = path[-1]
    if not isinstance(spec, tuple):
        spec[key] = value
    else:
        if parent_spec is None:
            raise ValueError('Cannot assign value to a tuple.')
        assert isinstance(key, int), 'Tuple key must be an int.'
        parent_spec[parent_key] = spec[:key] + (value,) + spec[key + 1:]

def _get_value(spec: Dict, path: Tuple) -> Any:
    if False:
        print('Hello World!')
    for k in path:
        spec = spec[k]
    return spec

def _resolve_domain_vars(spec: Dict, domain_vars: List[Tuple[Tuple, Domain]], allow_fail: bool=False, random_state: 'RandomState'=None) -> Tuple[bool, Dict]:
    if False:
        print('Hello World!')
    resolved = {}
    error = True
    num_passes = 0
    while error and num_passes < _MAX_RESOLUTION_PASSES:
        num_passes += 1
        error = False
        for (path, domain) in domain_vars:
            if path in resolved:
                continue
            try:
                value = domain.sample(_UnresolvedAccessGuard(spec), random_state=random_state)
            except RecursiveDependencyError as e:
                error = e
            except Exception:
                raise ValueError('Failed to evaluate expression: {}: {}'.format(path, domain))
            else:
                assign_value(spec, path, value)
                resolved[path] = value
    if error:
        if not allow_fail:
            raise error
        else:
            return (False, resolved)
    return (True, resolved)

def _grid_search_generator(unresolved_spec: Dict, grid_vars: List) -> Generator[Dict, None, None]:
    if False:
        i = 10
        return i + 15
    value_indices = [0] * len(grid_vars)

    def increment(i):
        if False:
            print('Hello World!')
        value_indices[i] += 1
        if value_indices[i] >= len(grid_vars[i][1]):
            value_indices[i] = 0
            if i + 1 < len(value_indices):
                return increment(i + 1)
            else:
                return True
        return False
    if not grid_vars:
        yield unresolved_spec
        return
    while value_indices[-1] < len(grid_vars[-1][1]):
        spec = copy.deepcopy(unresolved_spec)
        for (i, (path, values)) in enumerate(grid_vars):
            assign_value(spec, path, values[value_indices[i]])
        yield spec
        if grid_vars:
            done = increment(0)
            if done:
                break

def _is_resolved(v) -> bool:
    if False:
        print('Hello World!')
    (resolved, _) = _try_resolve(v)
    return resolved

def _try_resolve(v) -> Tuple[bool, Any]:
    if False:
        return 10
    if isinstance(v, Domain):
        return (False, v)
    elif isinstance(v, dict) and len(v) == 1 and ('eval' in v):
        return (False, Function(lambda spec: eval(v['eval'], _STANDARD_IMPORTS, {'spec': spec})))
    elif isinstance(v, dict) and len(v) == 1 and ('grid_search' in v):
        grid_values = v['grid_search']
        return (False, Categorical(grid_values).grid())
    return (True, v)

def _split_resolved_unresolved_values(spec: Dict) -> Tuple[Dict[Tuple, Any], Dict[Tuple, Any]]:
    if False:
        i = 10
        return i + 15
    resolved_vars = {}
    unresolved_vars = {}
    for (k, v) in spec.items():
        (resolved, v) = _try_resolve(v)
        if not resolved:
            unresolved_vars[k,] = v
        elif isinstance(v, dict):
            (_resolved_children, _unresolved_children) = _split_resolved_unresolved_values(v)
            for (path, value) in _resolved_children.items():
                resolved_vars[(k,) + path] = value
            for (path, value) in _unresolved_children.items():
                unresolved_vars[(k,) + path] = value
        elif isinstance(v, (list, tuple)):
            for (i, elem) in enumerate(v):
                (_resolved_children, _unresolved_children) = _split_resolved_unresolved_values({i: elem})
                for (path, value) in _resolved_children.items():
                    resolved_vars[(k,) + path] = value
                for (path, value) in _unresolved_children.items():
                    unresolved_vars[(k,) + path] = value
        else:
            resolved_vars[k,] = v
    return (resolved_vars, unresolved_vars)

def _unresolved_values(spec: Dict) -> Dict[Tuple, Any]:
    if False:
        print('Hello World!')
    return _split_resolved_unresolved_values(spec)[1]

def _has_unresolved_values(spec: Dict) -> bool:
    if False:
        print('Hello World!')
    return True if _unresolved_values(spec) else False

class _UnresolvedAccessGuard(dict):

    def __init__(self, *args, **kwds):
        if False:
            return 10
        super(_UnresolvedAccessGuard, self).__init__(*args, **kwds)
        self.__dict__ = self

    def __getattribute__(self, item):
        if False:
            while True:
                i = 10
        value = dict.__getattribute__(self, item)
        if not _is_resolved(value):
            raise RecursiveDependencyError('`{}` recursively depends on {}'.format(item, value))
        elif isinstance(value, dict):
            return _UnresolvedAccessGuard(value)
        else:
            return value

@DeveloperAPI
class RecursiveDependencyError(Exception):

    def __init__(self, msg: str):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self, msg)