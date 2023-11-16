from copy import copy
from typing import Dict, Optional, Tuple
import pytensor
from pytensor import Variable, shared
from pytensor.compile import SharedVariable
from pytensor.graph import Apply, FunctionGraph, Op, node_rewriter
from pytensor.graph.rewriting.basic import out2in
from pytensor.scalar import Identity
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.sharedvar import ScalarSharedVariable
from pymc.logprob.transforms import RVTransform
from pymc.model.core import Model
from pymc.pytensorf import StringType, find_rng_nodes, toposort_replace

class ModelVar(Op):
    """A dummy Op that describes the purpose of a Model variable and contains
    meta-information as additional inputs (value and dims).
    """

    def make_node(self, rv, *dims):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        return Apply(self, [rv, *dims], [rv.type(name=rv.name)])

    def _parse_dims(self, rv, *dims):
        if False:
            for i in range(10):
                print('nop')
        if dims:
            dims = [pytensor.as_symbolic(dim) for dim in dims]
            assert all((isinstance(dim.type, StringType) for dim in dims))
            assert len(dims) == rv.type.ndim
        return dims

    def infer_shape(self, fgraph, node, inputs_shape):
        if False:
            while True:
                i = 10
        return [inputs_shape[0]]

    def do_constant_folding(self, fgraph, node):
        if False:
            return 10
        return False

    def perform(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise RuntimeError('ModelVars should never be in a final graph!')

class ModelValuedVar(ModelVar):
    __props__ = ('transform',)

    def __init__(self, transform: Optional[RVTransform]=None):
        if False:
            i = 10
            return i + 15
        if transform is not None and (not isinstance(transform, RVTransform)):
            raise TypeError(f'transform must be None or RVTransform type, got {type(transform)}')
        self.transform = transform
        super().__init__()

    def make_node(self, rv, value, *dims):
        if False:
            return 10
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        if value is not None:
            assert isinstance(value, Variable)
            assert rv.type.dtype == value.type.dtype
            return Apply(self, [rv, value, *dims], [rv.type(name=rv.name)])

class ModelFreeRV(ModelValuedVar):
    pass

class ModelObservedRV(ModelValuedVar):
    pass

class ModelPotential(ModelVar):
    pass

class ModelDeterministic(ModelVar):
    pass

class ModelNamed(ModelVar):
    pass

def model_free_rv(rv, value, transform, *dims):
    if False:
        for i in range(10):
            print('nop')
    return ModelFreeRV(transform=transform)(rv, value, *dims)
model_observed_rv = ModelObservedRV()
model_potential = ModelPotential()
model_deterministic = ModelDeterministic()
model_named = ModelNamed()

@node_rewriter([Elemwise])
def local_remove_identity(fgraph, node):
    if False:
        print('Hello World!')
    if isinstance(node.op.scalar_op, Identity):
        return [node.inputs[0]]
remove_identity_rewrite = out2in(local_remove_identity)

def fgraph_from_model(model: Model, inlined_views=False) -> Tuple[FunctionGraph, Dict[Variable, Variable]]:
    if False:
        while True:
            i = 10
    'Convert Model to FunctionGraph.\n\n    See: model_from_fgraph\n\n    Parameters\n    ----------\n    model: PyMC model\n    inlined_views: bool, default False\n        Whether "view" variables (Deterministics and Data) should be inlined among RVs in the fgraph,\n        or show up as separate branches.\n\n    Returns\n    -------\n    fgraph: FunctionGraph\n        FunctionGraph that includes a copy of model variables, wrapped in dummy `ModelVar` Ops.\n        It should be possible to reconstruct a valid PyMC model using `model_from_fgraph`.\n\n    memo: Dict\n        A dictionary mapping original model variables to the equivalent nodes in the fgraph.\n    '
    if any((v is not None for v in model.rvs_to_initial_values.values())):
        raise NotImplementedError('Cannot convert models with non-default initial_values')
    if model.parent is not None:
        raise ValueError('Nested sub-models cannot be converted to fgraph. Convert the parent model instead')
    rvs_to_values = model.rvs_to_values
    rvs = list(rvs_to_values.keys())
    free_rvs = model.free_RVs
    observed_rvs = model.observed_RVs
    potentials = model.potentials
    named_vars = model.named_vars.values()
    old_deterministics = model.deterministics
    deterministics = [det if inlined_views else det.copy(det.name) for det in old_deterministics]
    old_value_vars = list(rvs_to_values.values())
    unnamed_value_vars = [val for val in old_value_vars if val not in named_vars]
    named_value_vars = [val if inlined_views else val.copy(val.name) for val in old_value_vars if val in named_vars]
    value_vars = old_value_vars.copy()
    if inlined_views:
        for named_val in named_value_vars:
            idx = value_vars.index(named_val)
            value_vars[idx] = named_val
    accounted_for = set(free_rvs + observed_rvs + potentials + old_deterministics + old_value_vars)
    other_named_vars = [var if inlined_views else var.copy(var.name) for var in named_vars if var not in accounted_for]
    model_vars = rvs + potentials + deterministics + other_named_vars + named_value_vars + unnamed_value_vars
    memo = {}
    shared_vars_to_copy = find_rng_nodes(model_vars)
    shared_vars_to_copy += [v for v in model.dim_lengths.values() if isinstance(v, SharedVariable)]
    shared_vars_to_copy += [v for v in model.named_vars.values() if isinstance(v, SharedVariable)]
    for var in shared_vars_to_copy:
        if isinstance(var, ScalarSharedVariable):
            new_var = shared(var.get_value(borrow=False).item())
        else:
            new_var = shared(var.get_value(borrow=False))
        assert new_var.type == var.type
        new_var.name = var.name
        new_var.tag = copy(var.tag)
        memo[var] = new_var
    fgraph = FunctionGraph(outputs=model_vars, clone=True, memo=memo, copy_orphans=True, copy_inputs=True)
    fgraph._coords = model._coords.copy()
    fgraph._dim_lengths = {k: memo.get(v, v) for (k, v) in model._dim_lengths.items()}
    rvs_to_transforms = model.rvs_to_transforms
    named_vars_to_dims = model.named_vars_to_dims
    free_rvs_to_transforms = {memo[k]: tr for (k, tr) in rvs_to_transforms.items()}
    free_rvs_to_values = {memo[k]: memo[v] for (k, v) in zip(rvs, value_vars) if k in free_rvs}
    observed_rvs_to_values = {memo[k]: memo[v] for (k, v) in zip(rvs, value_vars) if k in observed_rvs}
    potentials = [memo[k] for k in potentials]
    deterministics = [memo[k] for k in deterministics]
    named_vars = [memo[k] for k in other_named_vars + named_value_vars]
    vars = fgraph.outputs
    new_vars = []
    for var in vars:
        dims = named_vars_to_dims.get(var.name, ())
        if var in free_rvs_to_values:
            new_var = model_free_rv(var, free_rvs_to_values[var], free_rvs_to_transforms[var], *dims)
        elif var in observed_rvs_to_values:
            new_var = model_observed_rv(var, observed_rvs_to_values[var], *dims)
        elif var in potentials:
            new_var = model_potential(var, *dims)
        elif var in deterministics:
            new_var = model_deterministic(var, *dims)
        elif var in named_vars:
            new_var = model_named(var, *dims)
        else:
            new_var = var
        new_vars.append(new_var)
    replacements = tuple(zip(vars, new_vars))
    toposort_replace(fgraph, replacements, reverse=True)
    inverse_memo = {v: k for (k, v) in memo.items()}
    for (var, model_var) in replacements:
        if not inlined_views and (model_var.owner and isinstance(model_var.owner.op, (ModelDeterministic, ModelNamed))):
            var = var.owner.inputs[0]
        original_var = inverse_memo[var]
        memo[original_var] = model_var
    first_idx_to_remove = len(fgraph.outputs) - len(unnamed_value_vars)
    for _ in unnamed_value_vars:
        fgraph.remove_output(first_idx_to_remove)
    remove_identity_rewrite.apply(fgraph)
    return (fgraph, memo)

def model_from_fgraph(fgraph: FunctionGraph) -> Model:
    if False:
        print('Hello World!')
    'Convert FunctionGraph to PyMC model.\n\n    This requires nodes to be properly tagged with `ModelVar` dummy Ops.\n\n    See: fgraph_from_model\n    '

    def first_non_model_var(var):
        if False:
            return 10
        if var.owner and isinstance(var.owner.op, ModelVar):
            new_var = var.owner.inputs[0]
            return first_non_model_var(new_var)
        else:
            return var
    model = Model()
    if model.parent is not None:
        raise RuntimeError('model_to_fgraph cannot be called inside a PyMC model context')
    model._coords = getattr(fgraph, '_coords', {})
    model._dim_lengths = getattr(fgraph, '_dim_lengths', {})
    fgraph = fgraph.clone()
    model_dummy_vars = [model_node.outputs[0] for model_node in fgraph.toposort() if isinstance(model_node.op, ModelVar)]
    model_dummy_vars_to_vars = {dummy_var: first_non_model_var(dummy_var.owner.inputs[0]) for dummy_var in model_dummy_vars}
    toposort_replace(fgraph, tuple(model_dummy_vars_to_vars.items()))
    for model_var in model_dummy_vars:
        if isinstance(model_var.owner.op, ModelFreeRV):
            (var, value, *dims) = model_var.owner.inputs
            transform = model_var.owner.op.transform
            model.free_RVs.append(var)
            model.create_value_var(var, transform=None, value_var=value)
            model.rvs_to_transforms[var] = transform
            model.set_initval(var, initval=None)
        elif isinstance(model_var.owner.op, ModelObservedRV):
            (var, value, *dims) = model_var.owner.inputs
            model.observed_RVs.append(var)
            model.create_value_var(var, transform=None, value_var=value)
        elif isinstance(model_var.owner.op, ModelPotential):
            (var, *dims) = model_var.owner.inputs
            model.potentials.append(var)
        elif isinstance(model_var.owner.op, ModelDeterministic):
            (var, *dims) = model_var.owner.inputs
            if var in model.basic_RVs:
                var = var.copy()
            model.deterministics.append(var)
        elif isinstance(model_var.owner.op, ModelNamed):
            (var, *dims) = model_var.owner.inputs
        else:
            raise TypeError(f'Unexpected ModelVar type {type(model_var)}')
        var.name = model_var.name
        dims = [dim.data for dim in dims] if dims else None
        model.add_named_variable(var, dims=dims)
    return model

def clone_model(model: Model) -> Model:
    if False:
        i = 10
        return i + 15
    'Clone a PyMC model.\n\n    Recreates a PyMC model with clones of the original variables.\n    Shared variables will point to the same container but be otherwise different objects.\n    Constants are not cloned.\n\n\n    Examples\n    --------\n    .. code-block:: python\n\n        import pymc as pm\n        from pymc.model.fgraph import clone_model\n\n        with pm.Model() as m:\n            p = pm.Beta("p", 1, 1)\n            x = pm.Bernoulli("x", p=p, shape=(3,))\n\n        with clone_model(m) as clone_m:\n            # Access cloned variables by name\n            clone_x = clone_m["x"]\n\n            # z will be part of clone_m but not m\n            z = pm.Deterministic("z", clone_x + 1)\n\n    '
    return model_from_fgraph(fgraph_from_model(model)[0])

def extract_dims(var) -> Tuple:
    if False:
        print('Hello World!')
    dims = ()
    node = var.owner
    if node and isinstance(node.op, ModelVar):
        if isinstance(node.op, ModelValuedVar):
            dims = node.inputs[2:]
        else:
            dims = node.inputs[1:]
    return dims
__all__ = ('fgraph_from_model', 'model_from_fgraph', 'clone_model')