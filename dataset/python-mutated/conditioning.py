import warnings
from typing import Any, List, Mapping, Optional, Sequence, Union
from pytensor import Variable
from pytensor.graph import ancestors
from pytensor.graph.basic import walk
from pytensor.graph.op import HasInnerGraph
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pymc import Model
from pymc.logprob.transforms import RVTransform
from pymc.model.fgraph import ModelDeterministic, ModelFreeRV, extract_dims, fgraph_from_model, model_deterministic, model_free_rv, model_from_fgraph, model_named, model_observed_rv
from pymc.model.transform.basic import ModelVariable, parse_vars, prune_vars_detached_from_observed
from pymc.pytensorf import _replace_vars_in_graphs, toposort_replace
from pymc.util import get_transformed_name, get_untransformed_name

def observe(model: Model, vars_to_observations: Mapping[Union['str', TensorVariable], Any]) -> Model:
    if False:
        return 10
    'Convert free RVs or Deterministics to observed RVs.\n\n    Parameters\n    ----------\n    model: PyMC Model\n    vars_to_observations: Dict of variable or name to TensorLike\n        Dictionary that maps model variables (or names) to observed values.\n        Observed values must have a shape and data type that is compatible\n        with the original model variable.\n\n    Returns\n    -------\n    new_model: PyMC model\n        A distinct PyMC model with the relevant variables observed.\n        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        import pymc as pm\n\n        with pm.Model() as m:\n            x = pm.Normal("x")\n            y = pm.Normal("y", x)\n            z = pm.Normal("z", y)\n\n        m_new = pm.observe(m, {y: 0.5})\n\n    Deterministic variables can also be observed.\n    This relies on PyMC ability to infer the logp of the underlying expression\n\n    .. code-block:: python\n\n        import pymc as pm\n\n        with pm.Model() as m:\n            x = pm.Normal("x")\n            y = pm.Normal.dist(x, shape=(5,))\n            y_censored = pm.Deterministic("y_censored", pm.math.clip(y, -1, 1))\n\n        new_m = pm.observe(m, {y_censored: [0.9, 0.5, 0.3, 1, 1]})\n\n\n    '
    vars_to_observations = {model[var] if isinstance(var, str) else var: obs for (var, obs) in vars_to_observations.items()}
    valid_model_vars = set(model.free_RVs + model.deterministics)
    if any((var not in valid_model_vars for var in vars_to_observations)):
        raise ValueError(f'At least one var is not a free variable or deterministic in the model')
    (fgraph, memo) = fgraph_from_model(model)
    replacements = {}
    for (var, obs) in vars_to_observations.items():
        model_var = memo[var]
        assert isinstance(model_var.owner.op, (ModelFreeRV, ModelDeterministic))
        assert model_var in fgraph.variables
        var = model_var.owner.inputs[0]
        var.name = model_var.name
        dims = extract_dims(model_var)
        model_obs_rv = model_observed_rv(var, var.type.filter_variable(obs), *dims)
        replacements[model_var] = model_obs_rv
    toposort_replace(fgraph, tuple(replacements.items()))
    return model_from_fgraph(fgraph)

def replace_vars_in_graphs(graphs: Sequence[TensorVariable], replacements) -> List[TensorVariable]:
    if False:
        while True:
            i = 10

    def replacement_fn(var, inner_replacements):
        if False:
            return 10
        if var in replacements:
            inner_replacements[var] = replacements[var]
        for inp in var.owner.inputs:
            if inp.owner is None and inp in replacements:
                inner_replacements[inp] = replacements[inp]
        return [var]
    (replaced_graphs, _) = _replace_vars_in_graphs(graphs=graphs, replacement_fn=replacement_fn)
    return replaced_graphs

def rvs_in_graph(vars: Sequence[Variable]) -> bool:
    if False:
        print('Hello World!')
    'Check if there are any rvs in the graph of vars'
    from pymc.distributions.distribution import SymbolicRandomVariable

    def expand(r):
        if False:
            for i in range(10):
                print('nop')
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))
            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs
            return inputs
    return any((node for node in walk(vars, expand, False) if node.owner and isinstance(node.owner.op, (RandomVariable, SymbolicRandomVariable))))

def do(model: Model, vars_to_interventions: Mapping[Union['str', TensorVariable], Any], prune_vars=False) -> Model:
    if False:
        print('Hello World!')
    'Replace model variables by intervention variables.\n\n    Intervention variables will either show up as `Data` or `Deterministics` in the new model,\n    depending on whether they depend on other RandomVariables or not.\n\n    Parameters\n    ----------\n    model: PyMC Model\n    vars_to_interventions: Dict of variable or name to TensorLike\n        Dictionary that maps model variables (or names) to intervention expressions.\n        Intervention expressions must have a shape and data type that is compatible\n        with the original model variable.\n    prune_vars: bool, defaults to False\n        Whether to prune model variables that are not connected to any observed variables,\n        after the interventions.\n\n    Returns\n    -------\n    new_model: PyMC model\n        A distinct PyMC model with the relevant variables replaced by the intervention expressions.\n        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.\n\n    Examples\n    --------\n    .. code-block:: python\n\n        import pymc as pm\n\n        with pm.Model() as m:\n            x = pm.Normal("x", 0, 1)\n            y = pm.Normal("y", x, 1)\n            z = pm.Normal("z", y + x, 1)\n\n        # Dummy posterior, same as calling `pm.sample`\n        idata_m = az.from_dict({rv.name: [pm.draw(rv, draws=500)] for rv in [x, y, z]})\n\n        # Replace `y` by a constant `100.0`\n        with pm.do(m, {y: 100.0}) as m_do:\n            idata_do = pm.sample_posterior_predictive(idata_m, var_names="z")\n\n    '
    do_mapping = {}
    for (var, obs) in vars_to_interventions.items():
        if isinstance(var, str):
            var = model[var]
        try:
            do_mapping[var] = var.type.filter_variable(obs)
        except TypeError as err:
            raise TypeError('Incompatible replacement type. Make sure the shape and datatype of the interventions match the original variables') from err
    if any((var not in model.named_vars.values() for var in do_mapping)):
        raise ValueError(f'At least one var is not a named variable in the model')
    (fgraph, memo) = fgraph_from_model(model, inlined_views=True)
    ir_interventions = replace_vars_in_graphs(list(do_mapping.values()), replacements=memo)
    replacements = {}
    for (var, intervention) in zip(do_mapping, ir_interventions):
        model_var = memo[var]
        assert model_var in fgraph.variables
        if model_var in ancestors([intervention]):
            intervention.name = f'do_{model_var.name}'
            warnings.warn(f'Intervention expression references the variable that is being intervened: {model_var.name}. Intervention will be given the name: {intervention.name}')
        else:
            intervention.name = model_var.name
        dims = extract_dims(model_var)
        if rvs_in_graph([intervention]):
            new_var = model_deterministic(intervention.copy(name=intervention.name), *dims)
        else:
            new_var = model_named(intervention, *dims)
        replacements[model_var] = new_var
    toposort_replace(fgraph, tuple(replacements.items()))
    model = model_from_fgraph(fgraph)
    if prune_vars:
        return prune_vars_detached_from_observed(model)
    return model

def change_value_transforms(model: Model, vars_to_transforms: Mapping[ModelVariable, Union[RVTransform, None]]) -> Model:
    if False:
        i = 10
        return i + 15
    'Change the value variables transforms in the model\n\n    Parameters\n    ----------\n    model : Model\n    vars_to_transforms : Dict\n        Dictionary that maps RVs to new transforms to be applied to the respective value variables\n\n    Returns\n    -------\n    new_model : Model\n        Model with the updated transformed value variables\n\n    Examples\n    --------\n    Extract untransformed space Hessian after finding transformed space MAP\n\n    .. code-block:: python\n\n        import pymc as pm\n        from pymc.distributions.transforms import logodds\n        from pymc.model.transform.conditioning import change_value_transforms\n\n        with pm.Model() as base_m:\n            p = pm.Uniform("p", 0, 1, transform=None)\n            w = pm.Binomial("w", n=9, p=p, observed=6)\n\n        with change_value_transforms(base_m, {"p": logodds}) as transformed_p:\n            mean_q = pm.find_MAP()\n\n        with change_value_transforms(transformed_p, {"p": None}) as untransformed_p:\n            new_p = untransformed_p[\'p\']\n            std_q = ((1 / pm.find_hessian(mean_q, vars=[new_p])) ** 0.5)[0]\n\n        print(f"  Mean, Standard deviation\\np {mean_q[\'p\']:.2}, {std_q[0]:.2}")\n        #   Mean, Standard deviation\n        # p 0.67, 0.16\n\n    '
    vars_to_transforms = {parse_vars(model, var)[0]: transform for (var, transform) in vars_to_transforms.items()}
    if set(vars_to_transforms.keys()) - set(model.free_RVs):
        raise ValueError(f'All keys must be free variables in the model: {model.free_RVs}')
    (fgraph, memo) = fgraph_from_model(model)
    vars_to_transforms = {memo[var]: transform for (var, transform) in vars_to_transforms.items()}
    replacements = {}
    for node in fgraph.apply_nodes:
        if not isinstance(node.op, ModelFreeRV):
            continue
        [dummy_rv] = node.outputs
        if dummy_rv not in vars_to_transforms:
            continue
        transform = vars_to_transforms[dummy_rv]
        (rv, value, *dims) = node.inputs
        new_value = rv.type()
        try:
            untransformed_name = get_untransformed_name(value.name)
        except ValueError:
            untransformed_name = value.name
        if transform:
            new_name = get_transformed_name(untransformed_name, transform)
        else:
            new_name = untransformed_name
        new_value.name = new_name
        new_dummy_rv = model_free_rv(rv, new_value, transform, *dims)
        replacements[dummy_rv] = new_dummy_rv
    toposort_replace(fgraph, tuple(replacements.items()))
    return model_from_fgraph(fgraph)

def remove_value_transforms(model: Model, vars: Optional[Sequence[ModelVariable]]=None) -> Model:
    if False:
        while True:
            i = 10
    'Remove the value variables transforms in the model\n\n    Parameters\n    ----------\n    model : Model\n    vars : Model variables, optional\n        Model variables for which to remove transforms. Defaults to all transformed variables\n\n    Returns\n    -------\n    new_model : Model\n        Model with the removed transformed value variables\n\n    Examples\n    --------\n    Extract untransformed space Hessian after finding transformed space MAP\n\n    .. code-block:: python\n\n        import pymc as pm\n        from pymc.model.transform.conditioning import remove_value_transforms\n\n        with pm.Model() as transformed_m:\n            p = pm.Uniform("p", 0, 1)\n            w = pm.Binomial("w", n=9, p=p, observed=6)\n            mean_q = pm.find_MAP()\n\n        with remove_value_transforms(transformed_m) as untransformed_m:\n            new_p = untransformed_m["p"]\n            std_q = ((1 / pm.find_hessian(mean_q, vars=[new_p])) ** 0.5)[0]\n            print(f"  Mean, Standard deviation\\np {mean_q[\'p\']:.2}, {std_q[0]:.2}")\n\n        #   Mean, Standard deviation\n        # p 0.67, 0.16\n\n    '
    if vars is None:
        vars = model.free_RVs
    return change_value_transforms(model, {var: None for var in vars})
__all__ = ('change_value_transforms', 'do', 'observe', 'remove_value_transforms')