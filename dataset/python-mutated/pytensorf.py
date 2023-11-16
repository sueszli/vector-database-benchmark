import warnings
from typing import Callable, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import scipy.sparse as sps
from pytensor import scalar
from pytensor.compile import Function, Mode, get_mode
from pytensor.gradient import grad
from pytensor.graph import Type, node_rewriter, rewrite_graph
from pytensor.graph.basic import Apply, Constant, Variable, clone_get_equiv, graph_inputs, walk
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.scalar.basic import Cast
from pytensor.scan.op import Scan
from pytensor.tensor.basic import _as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.var import RandomGeneratorSharedVariable, RandomStateSharedVariable
from pytensor.tensor.rewriting.basic import topo_constant_folding
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from pytensor.tensor.variable import TensorConstant, TensorVariable
from pymc.exceptions import NotConstantValueError
from pymc.logprob.transforms import RVTransform
from pymc.logprob.utils import CheckParameterValue
from pymc.util import makeiter
from pymc.vartypes import continuous_types, isgenerator, typefilter
PotentialShapeType = Union[int, np.ndarray, Sequence[Union[int, Variable]], TensorVariable]
__all__ = ['gradient', 'hessian', 'hessian_diag', 'inputvars', 'cont_inputs', 'floatX', 'intX', 'smartfloatX', 'jacobian', 'CallableTensor', 'join_nonshared_inputs', 'make_shared_replacements', 'generator', 'convert_observed_data', 'compile_pymc']

def convert_observed_data(data):
    if False:
        for i in range(10):
            print('nop')
    'Convert user provided dataset to accepted formats.'
    if hasattr(data, 'to_numpy') and hasattr(data, 'isnull'):
        vals = data.to_numpy()
        null_data = data.isnull()
        if hasattr(null_data, 'to_numpy'):
            mask = null_data.to_numpy()
        else:
            mask = null_data
        if mask.any():
            ret = np.ma.MaskedArray(vals, mask)
        else:
            ret = vals
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.ma.MaskedArray):
            if not data.mask.any():
                ret = data.filled()
            else:
                ret = data
        else:
            mask = np.isnan(data)
            if np.any(mask):
                ret = np.ma.MaskedArray(data, mask)
            else:
                ret = data
    elif isinstance(data, Variable):
        ret = data
    elif sps.issparse(data):
        ret = data
    elif isgenerator(data):
        ret = generator(data)
    else:
        ret = np.asarray(data)
    if hasattr(data, 'dtype'):
        if 'int' in str(data.dtype):
            return intX(ret)
        else:
            return floatX(ret)
    else:
        return floatX(ret)

@_as_tensor_variable.register(pd.Series)
@_as_tensor_variable.register(pd.DataFrame)
def dataframe_to_tensor_variable(df: pd.DataFrame, *args, **kwargs) -> TensorVariable:
    if False:
        for i in range(10):
            print('nop')
    return pt.as_tensor_variable(df.to_numpy(), *args, **kwargs)

def extract_obs_data(x: TensorVariable) -> np.ndarray:
    if False:
        return 10
    'Extract data from observed symbolic variables.\n\n    Raises\n    ------\n    TypeError\n\n    '
    if isinstance(x, Constant):
        return x.data
    if isinstance(x, SharedVariable):
        return x.get_value()
    if x.owner and isinstance(x.owner.op, Elemwise) and isinstance(x.owner.op.scalar_op, Cast):
        array_data = extract_obs_data(x.owner.inputs[0])
        return array_data.astype(x.type.dtype)
    if x.owner and isinstance(x.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)):
        array_data = extract_obs_data(x.owner.inputs[0])
        mask_idx = tuple((extract_obs_data(i) for i in x.owner.inputs[2:]))
        mask = np.zeros_like(array_data)
        mask[mask_idx] = 1
        return np.ma.MaskedArray(array_data, mask)
    raise TypeError(f'Data cannot be extracted from {x}')

def walk_model(graphs: Iterable[TensorVariable], stop_at_vars: Optional[Set[TensorVariable]]=None, expand_fn: Callable[[TensorVariable], Iterable[TensorVariable]]=lambda var: []) -> Generator[TensorVariable, None, None]:
    if False:
        while True:
            i = 10
    'Walk model graphs and yield their nodes.\n\n    Parameters\n    ----------\n    graphs\n        The graphs to walk.\n    stop_at_vars\n        A list of variables at which the walk will terminate.\n    expand_fn\n        A function that returns the next variable(s) to be traversed.\n    '
    if stop_at_vars is None:
        stop_at_vars = set()

    def expand(var):
        if False:
            return 10
        new_vars = expand_fn(var)
        if var.owner and var not in stop_at_vars:
            new_vars.extend(reversed(var.owner.inputs))
        return new_vars
    yield from walk(graphs, expand, bfs=False)

def _replace_vars_in_graphs(graphs: Iterable[TensorVariable], replacement_fn: Callable[[TensorVariable], Dict[TensorVariable, TensorVariable]], **kwargs) -> Tuple[List[TensorVariable], Dict[TensorVariable, TensorVariable]]:
    if False:
        print('Hello World!')
    'Replace variables in graphs.\n\n    This will *not* recompute test values.\n\n    Parameters\n    ----------\n    graphs\n        The graphs in which random variables are to be replaced.\n    replacement_fn\n        A callable called on each graph output that populates a replacement dictionary and returns\n        nodes that should be investigated further.\n\n    Returns\n    -------\n    Tuple containing the transformed graphs and a ``dict`` of the replacements\n    that were made.\n    '
    replacements = {}

    def expand_replace(var):
        if False:
            for i in range(10):
                print('nop')
        new_nodes = []
        if var.owner:
            new_nodes.extend(replacement_fn(var, replacements))
        return new_nodes
    for var in walk_model(graphs, expand_fn=expand_replace, **kwargs):
        pass
    if replacements:
        inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
        equiv = {k: k for k in replacements.keys()}
        equiv = clone_get_equiv(inputs, graphs, False, False, equiv)
        fg = FunctionGraph([equiv[i] for i in inputs], [equiv[o] for o in graphs], clone=False)
        toposort = fg.toposort()
        sorted_replacements = sorted(tuple(replacements.items()), key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner is not None else -1, reverse=True)
        fg.replace_all(sorted_replacements, import_missing=True)
        graphs = list(fg.outputs)
    return (graphs, replacements)

def rvs_to_value_vars(graphs: Iterable[Variable], apply_transforms: bool=True, **kwargs) -> List[Variable]:
    if False:
        i = 10
        return i + 15
    "Clone and replace random variables in graphs with their value variables.\n\n    This will *not* recompute test values in the resulting graphs.\n\n    Parameters\n    ----------\n    graphs\n        The graphs in which to perform the replacements.\n    apply_transforms\n        If ``True``, apply each value variable's transform.\n    "
    warnings.warn('rvs_to_value_vars is deprecated. Use model.replace_rvs_by_values instead', FutureWarning)

    def populate_replacements(random_var: TensorVariable, replacements: Dict[TensorVariable, TensorVariable]) -> List[TensorVariable]:
        if False:
            print('Hello World!')
        value_var = getattr(random_var.tag, 'observations', getattr(random_var.tag, 'value_var', None))
        if value_var is None:
            return []
        transform = getattr(value_var.tag, 'transform', None)
        if transform is not None and apply_transforms:
            value_var = transform.backward(value_var, *random_var.owner.inputs)
        replacements[random_var] = value_var
        return [value_var]
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    equiv = clone_get_equiv(inputs, graphs, False, False, {})
    graphs = [equiv[n] for n in graphs]
    (graphs, _) = _replace_vars_in_graphs(graphs, replacement_fn=populate_replacements, **kwargs)
    return graphs

def replace_rvs_by_values(graphs: Sequence[TensorVariable], *, rvs_to_values: Dict[TensorVariable, TensorVariable], rvs_to_transforms: Optional[Dict[TensorVariable, RVTransform]]=None, **kwargs) -> List[TensorVariable]:
    if False:
        i = 10
        return i + 15
    'Clone and replace random variables in graphs with their value variables.\n\n    This will *not* recompute test values in the resulting graphs.\n\n    Parameters\n    ----------\n    graphs\n        The graphs in which to perform the replacements.\n    rvs_to_values\n        Mapping between the original graph RVs and respective value variables\n    rvs_to_transforms, optional\n        Mapping between the original graph RVs and respective value transforms\n    '
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    equiv = clone_get_equiv(inputs, graphs, False, False, {})
    graphs = [equiv[n] for n in graphs]
    equiv_rvs_to_values = {}
    equiv_rvs_to_transforms = {}
    for (rv, value) in rvs_to_values.items():
        equiv_rv = equiv.get(rv, rv)
        equiv_rvs_to_values[equiv_rv] = equiv.get(value, value)
        if rvs_to_transforms is not None:
            equiv_rvs_to_transforms[equiv_rv] = rvs_to_transforms[rv]

    def poulate_replacements(rv, replacements):
        if False:
            return 10
        value = equiv_rvs_to_values.get(rv, None)
        if value is None:
            return []
        if rvs_to_transforms is not None:
            transform = equiv_rvs_to_transforms.get(rv, None)
            if transform is not None:
                value = transform.backward(value, *rv.owner.inputs)
                value = rv.type.filter_variable(value, allow_convert=True)
                value.name = rv.name
        replacements[rv] = value
        return [value]
    (graphs, _) = _replace_vars_in_graphs(graphs, replacement_fn=poulate_replacements, **kwargs)
    return graphs

def inputvars(a):
    if False:
        while True:
            i = 10
    '\n    Get the inputs into PyTensor variables\n\n    Parameters\n    ----------\n        a: PyTensor variable\n\n    Returns\n    -------\n        r: list of tensor variables that are inputs\n    '
    return [v for v in graph_inputs(makeiter(a)) if isinstance(v, TensorVariable) and (not isinstance(v, TensorConstant))]

def cont_inputs(a):
    if False:
        return 10
    '\n    Get the continuous inputs into PyTensor variables\n\n    Parameters\n    ----------\n        a: PyTensor variable\n\n    Returns\n    -------\n        r: list of tensor variables that are continuous inputs\n    '
    return typefilter(inputvars(a), continuous_types)

def floatX(X):
    if False:
        return 10
    '\n    Convert an PyTensor tensor or numpy array to pytensor.config.floatX type.\n    '
    try:
        return X.astype(pytensor.config.floatX)
    except AttributeError:
        return np.asarray(X, dtype=pytensor.config.floatX)
_conversion_map = {'float64': 'int32', 'float32': 'int16', 'float16': 'int8', 'float8': 'int8'}

def intX(X):
    if False:
        i = 10
        return i + 15
    '\n    Convert a pytensor tensor or numpy array to pytensor.tensor.int32 type.\n    '
    intX = _conversion_map[pytensor.config.floatX]
    try:
        return X.astype(intX)
    except AttributeError:
        return np.asarray(X, dtype=intX)

def smartfloatX(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts numpy float values to floatX and leaves values of other types unchanged.\n    '
    if str(x.dtype).startswith('float'):
        x = floatX(x)
    return x
'\nPyTensor derivative functions\n'

def gradient1(f, v):
    if False:
        i = 10
        return i + 15
    'flat gradient of f wrt v'
    return pt.flatten(grad(f, v, disconnected_inputs='warn'))
empty_gradient = pt.zeros(0, dtype='float32')

def gradient(f, vars=None):
    if False:
        for i in range(10):
            print('nop')
    if vars is None:
        vars = cont_inputs(f)
    if vars:
        return pt.concatenate([gradient1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient

def jacobian1(f, v):
    if False:
        for i in range(10):
            print('nop')
    'jacobian of f wrt v'
    f = pt.flatten(f)
    idx = pt.arange(f.shape[0], dtype='int32')

    def grad_i(i):
        if False:
            while True:
                i = 10
        return gradient1(f[i], v)
    return pytensor.map(grad_i, idx)[0]

def jacobian(f, vars=None):
    if False:
        print('Hello World!')
    if vars is None:
        vars = cont_inputs(f)
    if vars:
        return pt.concatenate([jacobian1(f, v) for v in vars], axis=1)
    else:
        return empty_gradient

def jacobian_diag(f, x):
    if False:
        for i in range(10):
            print('nop')
    idx = pt.arange(f.shape[0], dtype='int32')

    def grad_ii(i, f, x):
        if False:
            print('Hello World!')
        return grad(f[i], x)[i]
    return pytensor.scan(grad_ii, sequences=[idx], n_steps=f.shape[0], non_sequences=[f, x], name='jacobian_diag')[0]

@pytensor.config.change_flags(compute_test_value='ignore')
def hessian(f, vars=None):
    if False:
        for i in range(10):
            print('nop')
    return -jacobian(gradient(f, vars), vars)

@pytensor.config.change_flags(compute_test_value='ignore')
def hessian_diag1(f, v):
    if False:
        return 10
    g = gradient1(f, v)
    idx = pt.arange(g.shape[0], dtype='int32')

    def hess_ii(i):
        if False:
            for i in range(10):
                print('nop')
        return gradient1(g[i], v)[i]
    return pytensor.map(hess_ii, idx)[0]

@pytensor.config.change_flags(compute_test_value='ignore')
def hessian_diag(f, vars=None):
    if False:
        return 10
    if vars is None:
        vars = cont_inputs(f)
    if vars:
        return -pt.concatenate([hessian_diag1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient

class IdentityOp(scalar.UnaryScalarOp):

    @staticmethod
    def st_impl(x):
        if False:
            while True:
                i = 10
        return x

    def impl(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x

    def grad(self, inp, grads):
        if False:
            for i in range(10):
                print('nop')
        return grads

    def c_code(self, node, name, inp, out, sub):
        if False:
            for i in range(10):
                print('nop')
        return f'{out[0]} = {inp[0]};'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(self, type(other))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(type(self))
scalar_identity = IdentityOp(scalar.upgrade_to_float, name='scalar_identity')
identity = Elemwise(scalar_identity, name='identity')

def make_shared_replacements(point, vars, model):
    if False:
        i = 10
        return i + 15
    '\n    Makes shared replacements for all *other* variables than the ones passed.\n\n    This way functions can be called many times without setting unchanging variables. Allows us\n    to use func.trust_input by removing the need for DictToArrayBijection and kwargs.\n\n    Parameters\n    ----------\n    point: dictionary mapping variable names to sample values\n    vars: list of variables not to make shared\n    model: model\n\n    Returns\n    -------\n    Dict of variable -> new shared variable\n    '
    othervars = set(model.value_vars) - set(vars)
    return {var: pytensor.shared(point[var.name], var.name + '_shared', shape=var.type.shape) for var in othervars}

def join_nonshared_inputs(point: Dict[str, np.ndarray], outputs: List[TensorVariable], inputs: List[TensorVariable], shared_inputs: Optional[Dict[TensorVariable, TensorSharedVariable]]=None, make_inputs_shared: bool=False) -> Tuple[List[TensorVariable], TensorVariable]:
    if False:
        while True:
            i = 10
    '\n    Create new outputs and input TensorVariables where the non-shared inputs are joined\n    in a single raveled vector input.\n\n    Parameters\n    ----------\n    point : dict of {str : array_like}\n        Dictionary that maps each input variable name to a numerical variable. The values\n        are used to extract the shape of each input variable to establish a correct\n        mapping between joined and original inputs. The shape of each variable is\n        assumed to be fixed.\n    outputs : list of TensorVariable\n        List of output TensorVariables whose non-shared inputs will be replaced\n        by a joined vector input.\n    inputs : list of TensorVariable\n        List of input TensorVariables which will be replaced by a joined vector input.\n    shared_inputs : dict of {TensorVariable : TensorSharedVariable}, optional\n        Dict of TensorVariable and their associated TensorSharedVariable in\n        subgraph replacement.\n    make_inputs_shared : bool, default False\n        Whether to make the joined vector input a shared variable.\n\n    Returns\n    -------\n    new_outputs : list of TensorVariable\n        List of new outputs `outputs` TensorVariables that depend on `joined_inputs` and new shared variables as inputs.\n    joined_inputs : TensorVariable\n        Joined input vector TensorVariable for the `new_outputs`\n\n    Examples\n    --------\n    Join the inputs of a simple PyTensor graph.\n\n    .. code-block:: python\n\n        import pytensor.tensor as pt\n        import numpy as np\n\n        from pymc.pytensorf import join_nonshared_inputs\n\n        # Original non-shared inputs\n        x = pt.scalar("x")\n        y = pt.vector("y")\n        # Original output\n        out = x + y\n        print(out.eval({x: np.array(1), y: np.array([1, 2, 3])})) # [2, 3, 4]\n\n        # New output and inputs\n        [new_out], joined_inputs = join_nonshared_inputs(\n            point={ # Only shapes matter\n                "x": np.zeros(()),\n                "y": np.zeros(3),\n            },\n            outputs=[out],\n            inputs=[x, y],\n        )\n        print(new_out.eval({\n            joined_inputs: np.array([1, 1, 2, 3]),\n        })) # [2, 3, 4]\n\n    Join the input value variables of a model logp.\n\n    .. code-block:: python\n\n        import pymc as pm\n\n        with pm.Model() as model:\n            mu_pop = pm.Normal("mu_pop")\n            sigma_pop = pm.HalfNormal("sigma_pop")\n            mu = pm.Normal("mu", mu_pop, sigma_pop, shape=(3, ))\n\n            y = pm.Normal("y", mu, 1.0, observed=[0, 1, 2])\n\n        print(model.compile_logp()({\n            "mu_pop": 0,\n            "sigma_pop_log__": 1,\n            "mu": [0, 1, 2],\n        })) # -12.691227342634292\n\n        initial_point = model.initial_point()\n        inputs = model.value_vars\n\n        [logp], joined_inputs = join_nonshared_inputs(\n            point=initial_point,\n            outputs=[model.logp()],\n            inputs=inputs,\n        )\n\n        print(logp.eval({\n            joined_inputs: [0, 1, 0, 1, 2],\n        })) # -12.691227342634292\n\n    Same as above but with the `mu_pop` value variable being shared.\n\n    .. code-block:: python\n\n        from pytensor import shared\n\n        mu_pop_input, *other_inputs = inputs\n        shared_mu_pop_input = shared(0.0)\n\n        [logp], other_joined_inputs = join_nonshared_inputs(\n            point=initial_point,\n            outputs=[model.logp()],\n            inputs=other_inputs,\n            shared_inputs={\n                mu_pop_input: shared_mu_pop_input\n            },\n        )\n\n        print(logp.eval({\n            other_joined_inputs: [1, 0, 1, 2],\n        })) # -12.691227342634292\n    '
    if not inputs:
        raise ValueError('Empty list of input variables.')
    raveled_inputs = pt.concatenate([var.ravel() for var in inputs])
    if not make_inputs_shared:
        tensor_type = raveled_inputs.type
        joined_inputs = tensor_type('joined_inputs')
    else:
        joined_values = np.concatenate([point[var.name].ravel() for var in inputs])
        joined_inputs = pytensor.shared(joined_values, 'joined_inputs')
    if pytensor.config.compute_test_value != 'off':
        joined_inputs.tag.test_value = raveled_inputs.tag.test_value
    replace: Dict[TensorVariable, TensorVariable] = {}
    last_idx = 0
    for var in inputs:
        shape = point[var.name].shape
        arr_len = np.prod(shape, dtype=int)
        replace[var] = joined_inputs[last_idx:last_idx + arr_len].reshape(shape).astype(var.dtype)
        last_idx += arr_len
    if shared_inputs is not None:
        replace.update(shared_inputs)
    new_outputs = [pytensor.clone_replace(output, replace, rebuild_strict=False) for output in outputs]
    return (new_outputs, joined_inputs)

class PointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments."""

    def __init__(self, f):
        if False:
            print('Hello World!')
        self.f = f

    def __call__(self, state):
        if False:
            return 10
        return self.f(**state)

class CallableTensor:
    """Turns a symbolic variable with one input into a function that returns symbolic arguments
    with the one variable replaced with the input.
    """

    def __init__(self, tensor):
        if False:
            print('Hello World!')
        self.tensor = tensor

    def __call__(self, input):
        if False:
            return 10
        'Replaces the single input of symbolic variable to be the passed argument.\n\n        Parameters\n        ----------\n        input: TensorVariable\n        '
        (oldinput,) = inputvars(self.tensor)
        return pytensor.clone_replace(self.tensor, {oldinput: input}, rebuild_strict=False)

class GeneratorOp(Op):
    """
    Generator Op is designed for storing python generators inside pytensor graph.

    __call__ creates TensorVariable
        It has 2 new methods
        - var.set_gen(gen): sets new generator
        - var.set_default(value): sets new default value (None erases default value)

    If generator is exhausted, variable will produce default value if it is not None,
    else raises `StopIteration` exception that can be caught on runtime.

    Parameters
    ----------
    gen: generator that implements __next__ (py3) or next (py2) method
        and yields np.arrays with same types
    default: np.array with the same type as generator produces
    """
    __props__ = ('generator',)

    def __init__(self, gen, default=None):
        if False:
            while True:
                i = 10
        from pymc.data import GeneratorAdapter
        super().__init__()
        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        self.generator = gen
        self.set_default(default)

    def make_node(self, *inputs):
        if False:
            while True:
                i = 10
        gen_var = self.generator.make_variable(self)
        return Apply(self, [], [gen_var])

    def perform(self, node, inputs, output_storage, params=None):
        if False:
            i = 10
            return i + 15
        if self.default is not None:
            output_storage[0][0] = next(self.generator, self.default)
        else:
            output_storage[0][0] = next(self.generator)

    def do_constant_folding(self, fgraph, node):
        if False:
            for i in range(10):
                print('nop')
        return False
    __call__ = pytensor.config.change_flags(compute_test_value='off')(Op.__call__)

    def set_gen(self, gen):
        if False:
            print('Hello World!')
        from pymc.data import GeneratorAdapter
        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        if not gen.tensortype == self.generator.tensortype:
            raise ValueError('New generator should yield the same type')
        self.generator = gen

    def set_default(self, value):
        if False:
            print('Hello World!')
        if value is None:
            self.default = None
        else:
            value = np.asarray(value, self.generator.tensortype.dtype)
            t1 = (False,) * value.ndim
            t2 = self.generator.tensortype.broadcastable
            if not t1 == t2:
                raise ValueError('Default value should have the same type as generator')
            self.default = value

def generator(gen, default=None):
    if False:
        print('Hello World!')
    '\n    Generator variable with possibility to set default value and new generator.\n    If generator is exhausted variable will produce default value if it is not None,\n    else raises `StopIteration` exception that can be caught on runtime.\n\n    Parameters\n    ----------\n    gen: generator that implements __next__ (py3) or next (py2) method\n        and yields np.arrays with same types\n    default: np.array with the same type as generator produces\n\n    Returns\n    -------\n    TensorVariable\n        It has 2 new methods\n        - var.set_gen(gen): sets new generator\n        - var.set_default(value): sets new default value (None erases default value)\n    '
    return GeneratorOp(gen, default)()

def floatX_array(x):
    if False:
        print('Hello World!')
    return floatX(np.array(x))

def ix_(*args):
    if False:
        return 10
    '\n    PyTensor np.ix_ analog\n\n    See numpy.lib.index_tricks.ix_ for reference\n    '
    out = []
    nd = len(args)
    for (k, new) in enumerate(args):
        if new is None:
            out.append(slice(None))
        new = pt.as_tensor(new)
        if new.ndim != 1:
            raise ValueError('Cross index must be 1 dimensional')
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)

def largest_common_dtype(tensors):
    if False:
        print('Hello World!')
    dtypes = {str(t.dtype) if hasattr(t, 'dtype') else smartfloatX(np.asarray(t)).dtype for t in tensors}
    return np.stack([np.ones((), dtype=dtype) for dtype in dtypes]).dtype

@node_rewriter(tracks=[CheckParameterValue])
def local_remove_check_parameter(fgraph, node):
    if False:
        while True:
            i = 10
    'Rewrite that removes CheckParameterValue\n\n    This is used when compile_rv_inplace\n    '
    if isinstance(node.op, CheckParameterValue):
        return [node.inputs[0]]

@node_rewriter(tracks=[CheckParameterValue])
def local_check_parameter_to_ninf_switch(fgraph, node):
    if False:
        print('Hello World!')
    if not node.op.can_be_replaced_by_ninf:
        return None
    (logp_expr, *logp_conds) = node.inputs
    if len(logp_conds) > 1:
        logp_cond = pt.all(logp_conds)
    else:
        (logp_cond,) = logp_conds
    out = pt.switch(logp_cond, logp_expr, -np.inf)
    out.name = node.op.msg
    if out.dtype != node.outputs[0].dtype:
        out = pt.cast(out, node.outputs[0].dtype)
    return [out]
pytensor.compile.optdb['canonicalize'].register('local_remove_check_parameter', local_remove_check_parameter, use_db_name_as_tag=False)
pytensor.compile.optdb['canonicalize'].register('local_check_parameter_to_ninf_switch', local_check_parameter_to_ninf_switch, use_db_name_as_tag=False)

def find_rng_nodes(variables: Iterable[Variable]) -> List[Union[RandomStateSharedVariable, RandomGeneratorSharedVariable]]:
    if False:
        return 10
    'Return RNG variables in a graph'
    return [node for node in graph_inputs(variables) if isinstance(node, (RandomStateSharedVariable, RandomGeneratorSharedVariable))]

def replace_rng_nodes(outputs: Sequence[TensorVariable]) -> Sequence[TensorVariable]:
    if False:
        for i in range(10):
            print('nop')
    'Replace any RNG nodes upstream of outputs by new RNGs of the same type\n\n    This can be used when combining a pre-existing graph with a cloned one, to ensure\n    RNGs are unique across the two graphs.\n    '
    rng_nodes = find_rng_nodes(outputs)
    if not rng_nodes:
        return outputs
    graph = FunctionGraph(outputs=outputs, clone=False)
    new_rng_nodes: List[Union[np.random.RandomState, np.random.Generator]] = []
    for rng_node in rng_nodes:
        rng_cls: type
        if isinstance(rng_node, pt.random.var.RandomStateSharedVariable):
            rng_cls = np.random.RandomState
        else:
            rng_cls = np.random.Generator
        new_rng_nodes.append(pytensor.shared(rng_cls(np.random.PCG64())))
    graph.replace_all(zip(rng_nodes, new_rng_nodes), import_missing=True)
    return graph.outputs
SeedSequenceSeed = Optional[Union[int, Sequence[int], np.ndarray, np.random.SeedSequence]]

def reseed_rngs(rngs: Sequence[SharedVariable], seed: SeedSequenceSeed) -> None:
    if False:
        print('Hello World!')
    'Create a new set of RandomState/Generator for each rng based on a seed'
    bit_generators = [np.random.PCG64(sub_seed) for sub_seed in np.random.SeedSequence(seed).spawn(len(rngs))]
    for (rng, bit_generator) in zip(rngs, bit_generators):
        new_rng: Union[np.random.RandomState, np.random.Generator]
        if isinstance(rng, pt.random.var.RandomStateSharedVariable):
            new_rng = np.random.RandomState(bit_generator)
        else:
            new_rng = np.random.Generator(bit_generator)
        rng.set_value(new_rng, borrow=True)

def collect_default_updates(outputs: Sequence[Variable], *, inputs: Optional[Sequence[Variable]]=None, must_be_shared: bool=True) -> Dict[Variable, Variable]:
    if False:
        return 10
    'Collect default update expression for shared-variable RNGs used by RVs between inputs and outputs.\n\n    Parameters\n    ----------\n    outputs: list of PyTensor variables\n        List of variables in which graphs default updates will be collected.\n    inputs: list of PyTensor variables, optional\n        Input nodes above which default updates should not be collected.\n        When not provided, search will include top level inputs (roots).\n    must_be_shared: bool, default True\n        Used internally by PyMC. Whether updates should be collected for non-shared\n        RNG input variables. This is used to collect update expressions for inner graphs.\n\n    Examples\n    --------\n    .. code:: python\n        import pymc as pm\n        from pytensor.scan import scan\n        from pymc.pytensorf import collect_default_updates\n\n        def scan_step(xtm1):\n            x = xtm1 + pm.Normal.dist()\n            x_update = collect_default_updates([x])\n            return x, x_update\n\n        x0 = pm.Normal.dist()\n\n        xs, updates = scan(\n            fn=scan_step,\n            outputs_info=[x0],\n            n_steps=10,\n        )\n\n        # PyMC makes use of the updates to seed xs properly.\n        # Without updates, it would raise an error.\n        xs_draws = pm.draw(xs, draws=10)\n\n    '
    from pymc.distributions.distribution import SymbolicRandomVariable

    def find_default_update(clients, rng: Variable) -> Union[None, Variable]:
        if False:
            for i in range(10):
                print('nop')
        rng_clients = clients.get(rng, None)
        if not rng_clients:
            return rng
        if len(rng_clients) > 1:
            warnings.warn(f'RNG Variable {rng} has multiple clients. This is likely an inconsistent random graph.', UserWarning)
            return None
        [client, _] = rng_clients[0]
        if client == 'output':
            return rng
        if isinstance(client.op, RandomVariable):
            next_rng = client.outputs[0]
        elif isinstance(client.op, SymbolicRandomVariable):
            next_rng = client.op.update(client).get(rng)
            if next_rng is None:
                raise ValueError(f'No update found for at least one RNG used in SymbolicRandomVariable Op {client.op}')
        elif isinstance(client.op, Scan):
            rng_idx = client.inputs.index(rng)
            io_map = client.op.get_oinp_iinp_iout_oout_mappings()['outer_out_from_outer_inp']
            out_idx = io_map.get(rng_idx, -1)
            if out_idx != -1:
                next_rng = client.outputs[out_idx]
            else:
                raise ValueError(f'No update found for at least one RNG used in Scan Op {client.op}.\nYou can use `pytensorf.collect_default_updates` inside the Scan function to return updates automatically.')
        else:
            return None
        return find_default_update(clients, next_rng)
    if inputs is None:
        inputs = []
    outputs = makeiter(outputs)
    fg = FunctionGraph(outputs=outputs, clone=False)
    clients = fg.clients
    rng_updates = {}
    for input_rng in (inp for inp in graph_inputs(outputs, blockers=inputs) if (not must_be_shared or isinstance(inp, SharedVariable)) and isinstance(inp.type, RandomType)):
        default_update = find_default_update(clients, input_rng)
        if getattr(input_rng, 'default_update', None):
            rng_updates[input_rng] = input_rng.default_update
        elif default_update is not None:
            rng_updates[input_rng] = default_update
    return rng_updates

def compile_pymc(inputs, outputs, random_seed: SeedSequenceSeed=None, mode=None, **kwargs) -> Function:
    if False:
        return 10
    'Use ``pytensor.function`` with specialized pymc rewrites always enabled.\n\n    This function also ensures shared RandomState/Generator used by RandomVariables\n    in the graph are updated across calls, to ensure independent draws.\n\n    Parameters\n    ----------\n    inputs: list of TensorVariables, optional\n        Inputs of the compiled PyTensor function\n    outputs: list of TensorVariables, optional\n        Outputs of the compiled PyTensor function\n    random_seed: int, array-like of int or SeedSequence, optional\n        Seed used to override any RandomState/Generator shared variables in the graph.\n        If not specified, the value of original shared variables will still be overwritten.\n    mode: optional\n        PyTensor mode used to compile the function\n\n    Included rewrites\n    -----------------\n    random_make_inplace\n        Ensures that compiled functions containing random variables will produce new\n        samples on each call.\n    local_check_parameter_to_ninf_switch\n        Replaces CheckParameterValue assertions is logp expressions with Switches\n        that return -inf in place of the assert.\n\n    Optional rewrites\n    -----------------\n    local_remove_check_parameter\n        Replaces CheckParameterValue assertions is logp expressions. This is used\n        as an alteranative to the default local_check_parameter_to_ninf_switch whenenver\n        this function is called within a model context and the model `check_bounds` flag\n        is set to False.\n    '
    rng_updates = collect_default_updates(inputs=inputs, outputs=outputs)
    if rng_updates:
        reseed_rngs(rng_updates.keys(), random_seed)
    try:
        from pymc.model import modelcontext
        model = modelcontext(None)
        check_bounds = model.check_bounds
    except TypeError:
        check_bounds = True
    check_parameter_opt = 'local_check_parameter_to_ninf_switch' if check_bounds else 'local_remove_check_parameter'
    mode = get_mode(mode)
    opt_qry = mode.provided_optimizer.including('random_make_inplace', check_parameter_opt)
    mode = Mode(linker=mode.linker, optimizer=opt_qry)
    pytensor_function = pytensor.function(inputs, outputs, updates={**rng_updates, **kwargs.pop('updates', {})}, mode=mode, **kwargs)
    return pytensor_function

def constant_fold(xs: Sequence[TensorVariable], raise_not_constant: bool=True) -> Tuple[np.ndarray, ...]:
    if False:
        while True:
            i = 10
    'Use constant folding to get constant values of a graph.\n\n    Parameters\n    ----------\n    xs: Sequence of TensorVariable\n        The variables that are to be constant folded\n    raise_not_constant: bool, default True\n        Raises NotConstantValueError if any of the variables cannot be constant folded.\n        This should only be disabled with care, as the graphs are cloned before\n        attempting constant folding, and any old non-shared inputs will not work with\n        the returned outputs\n    '
    fg = FunctionGraph(outputs=xs, features=[ShapeFeature()], clone=True)
    folded_xs = rewrite_graph(fg, custom_rewrite=topo_constant_folding).outputs
    if raise_not_constant and (not all((isinstance(folded_x, Constant) for folded_x in folded_xs))):
        raise NotConstantValueError
    return tuple((folded_x.data if isinstance(folded_x, Constant) else folded_x for folded_x in folded_xs))

def rewrite_pregrad(graph):
    if False:
        i = 10
        return i + 15
    'Apply simplifying or stabilizing rewrites to graph that are safe to use\n    pre-grad.\n    '
    return rewrite_graph(graph, include=('canonicalize', 'stabilize'))

class StringType(Type[str]):

    def clone(self, **kwargs):
        if False:
            while True:
                i = 10
        return type(self)()

    def filter(self, x, strict=False, allow_downcast=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, str):
            return x
        else:
            raise TypeError('Expected a string!')

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'string'

    @staticmethod
    def may_share_memory(a, b):
        if False:
            print('Hello World!')
        return isinstance(a, str) and a is b
stringtype = StringType()

class StringConstant(Constant):
    pass

@pytensor._as_symbolic.register(str)
def as_symbolic_string(x, **kwargs):
    if False:
        while True:
            i = 10
    return StringConstant(stringtype, x)

def toposort_replace(fgraph: FunctionGraph, replacements: Sequence[Tuple[Variable, Variable]], reverse: bool=False) -> None:
    if False:
        return 10
    'Replace multiple variables in topological order.'
    toposort = fgraph.toposort()
    sorted_replacements = sorted(replacements, key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner else -1, reverse=reverse)
    fgraph.replace_all(sorted_replacements, import_missing=True)