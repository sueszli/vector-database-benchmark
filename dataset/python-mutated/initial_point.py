import functools
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Set, Union
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.variable import TensorVariable
from pymc.logprob.transforms import RVTransform
from pymc.pytensorf import compile_pymc, find_rng_nodes, replace_rng_nodes, reseed_rngs
from pymc.util import get_transformed_name, get_untransformed_name, is_transformed_name
StartDict = Dict[Union[Variable, str], Union[np.ndarray, Variable, str]]
PointType = Dict[str, np.ndarray]

def convert_str_to_rv_dict(model, start: StartDict) -> Dict[TensorVariable, Optional[Union[np.ndarray, Variable, str]]]:
    if False:
        i = 10
        return i + 15
    'Helper function for converting a user-provided start dict with str keys of (transformed) variable names\n    to a dict mapping the RV tensors to untransformed initvals.\n    TODO: Deprecate this functionality and only accept TensorVariables as keys\n    '
    initvals = {}
    for (key, initval) in start.items():
        if isinstance(key, str):
            if is_transformed_name(key):
                rv = model[get_untransformed_name(key)]
                initvals[rv] = model.rvs_to_transforms[rv].backward(initval, *rv.owner.inputs)
            else:
                initvals[model[key]] = initval
        else:
            initvals[key] = initval
    return initvals

def make_initial_point_fns_per_chain(*, model, overrides: Optional[Union[StartDict, Sequence[Optional[StartDict]]]], jitter_rvs: Optional[Set[TensorVariable]]=None, chains: int) -> List[Callable]:
    if False:
        for i in range(10):
            print('nop')
    'Create an initial point function for each chain, as defined by initvals\n\n    If a single initval dictionary is passed, the function is replicated for each\n    chain, otherwise a unique function is compiled for each entry in the dictionary.\n\n    Parameters\n    ----------\n    overrides : optional, list or dict\n        Initial value strategy overrides that should take precedence over the defaults from the model.\n        A sequence of None or dicts will be treated as chain-wise strategies and must have the same length as `seeds`.\n    jitter_rvs : set, optional\n        Random variable tensors for which U(-1, 1) jitter shall be applied.\n        (To the transformed space if applicable.)\n\n    Raises\n    ------\n    ValueError\n        If the number of entries in initvals is different than the number of chains\n\n    '
    if isinstance(overrides, dict) or overrides is None:
        ipfns = [make_initial_point_fn(model=model, overrides=overrides, jitter_rvs=jitter_rvs, return_transformed=True)] * chains
    elif len(overrides) == chains:
        ipfns = [make_initial_point_fn(model=model, jitter_rvs=jitter_rvs, overrides=chain_overrides, return_transformed=True) for chain_overrides in overrides]
    else:
        raise ValueError(f'Number of initval dicts ({len(overrides)}) does not match the number of chains ({chains}).')
    return ipfns

def make_initial_point_fn(*, model, overrides: Optional[StartDict]=None, jitter_rvs: Optional[Set[TensorVariable]]=None, default_strategy: str='moment', return_transformed: bool=True) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    'Create seeded function that computes initial values for all free model variables.\n\n    Parameters\n    ----------\n    jitter_rvs : set\n        The set (or list or tuple) of random variables for which a U(-1, +1) jitter should be\n        added to the initial value. Only available for variables that have a transform or real-valued support.\n    default_strategy : str\n        Which of { "moment", "prior" } to prefer if the initval setting for an RV is None.\n    overrides : dict\n        Initial value (strategies) to use instead of what\'s specified in `Model.initial_values`.\n    return_transformed : bool\n        If `True` the returned variables will correspond to transformed initial values.\n    '
    sdict_overrides = convert_str_to_rv_dict(model, overrides or {})
    initval_strats = {**model.rvs_to_initial_values, **sdict_overrides}
    initial_values = make_initial_point_expression(free_rvs=model.free_RVs, rvs_to_transforms=model.rvs_to_transforms, initval_strategies=initval_strats, jitter_rvs=jitter_rvs, default_strategy=default_strategy, return_transformed=return_transformed)
    initial_values = replace_rng_nodes(initial_values)
    func = compile_pymc(inputs=[], outputs=initial_values, mode=pytensor.compile.mode.FAST_COMPILE)
    varnames = []
    for var in model.free_RVs:
        transform = model.rvs_to_transforms[var]
        if transform is not None and return_transformed:
            name = get_transformed_name(var.name, transform)
        else:
            name = var.name
        varnames.append(name)

    def make_seeded_function(func):
        if False:
            print('Hello World!')
        rngs = find_rng_nodes(func.maker.fgraph.outputs)

        @functools.wraps(func)
        def inner(seed, *args, **kwargs):
            if False:
                return 10
            reseed_rngs(rngs, seed)
            values = func(*args, **kwargs)
            return dict(zip(varnames, values))
        return inner
    return make_seeded_function(func)

def make_initial_point_expression(*, free_rvs: Sequence[TensorVariable], rvs_to_transforms: Dict[TensorVariable, RVTransform], initval_strategies: Dict[TensorVariable, Optional[Union[np.ndarray, Variable, str]]], jitter_rvs: Set[TensorVariable]=None, default_strategy: str='moment', return_transformed: bool=False) -> List[TensorVariable]:
    if False:
        print('Hello World!')
    'Creates the tensor variables that need to be evaluated to obtain an initial point.\n\n    Parameters\n    ----------\n    free_rvs : list\n        Tensors of free random variables in the model.\n    rvs_to_values : dict\n        Mapping of free random variable tensors to value variable tensors.\n    initval_strategies : dict\n        Mapping of free random variable tensors to initial value strategies.\n        For example the `Model.initial_values` dictionary.\n    jitter_rvs : set\n        The set (or list or tuple) of random variables for which a U(-1, +1) jitter should be\n        added to the initial value. Only available for variables that have a transform or real-valued support.\n    default_strategy : str\n        Which of { "moment", "prior" } to prefer if the initval strategy setting for an RV is None.\n    return_transformed : bool\n        Switches between returning the tensors for untransformed or transformed initial points.\n\n    Returns\n    -------\n    initial_points : list of TensorVariable\n        PyTensor expressions for initial values of the free random variables.\n    '
    from pymc.distributions.distribution import moment
    if jitter_rvs is None:
        jitter_rvs = set()
    initial_values = []
    initial_values_transformed = []
    for variable in free_rvs:
        strategy = initval_strategies.get(variable, None)
        if strategy is None:
            strategy = default_strategy
        if isinstance(strategy, str):
            if strategy == 'moment':
                try:
                    value = moment(variable)
                except NotImplementedError:
                    warnings.warn(f'Moment not defined for variable {variable} of type {variable.owner.op.__class__.__name__}, defaulting to a draw from the prior. This can lead to difficulties during tuning. You can manually define an initval or implement a moment dispatched function for this distribution.', UserWarning)
                    value = variable
            elif strategy == 'prior':
                value = variable
            else:
                raise ValueError(f'Invalid string strategy: {strategy}. It must be one of ["moment", "prior"]')
        else:
            value = pt.as_tensor(strategy, dtype=variable.dtype).astype(variable.dtype)
        transform = rvs_to_transforms.get(variable, None)
        if transform is not None:
            value = transform.forward(value, *variable.owner.inputs)
        if variable in jitter_rvs:
            jitter = pt.random.uniform(-1, 1, size=value.shape)
            jitter.name = f'{variable.name}_jitter'
            value = value + jitter
        value = value.astype(variable.dtype)
        initial_values_transformed.append(value)
        if transform is not None:
            value = transform.backward(value, *variable.owner.inputs)
        initial_values.append(value)
    all_outputs: List[TensorVariable] = []
    all_outputs.extend(free_rvs)
    all_outputs.extend(initial_values)
    all_outputs.extend(initial_values_transformed)
    copy_graph = FunctionGraph(outputs=all_outputs, clone=True)
    n_variables = len(free_rvs)
    free_rvs_clone = copy_graph.outputs[:n_variables]
    initial_values_clone = copy_graph.outputs[n_variables:-n_variables]
    initial_values_transformed_clone = copy_graph.outputs[-n_variables:]
    graph = FunctionGraph(outputs=free_rvs_clone, clone=False)
    replacements = reversed(list(zip(free_rvs_clone, initial_values_clone)))
    graph.replace_all(replacements, import_missing=True)
    if not return_transformed:
        return graph.outputs
    return initial_values_transformed_clone