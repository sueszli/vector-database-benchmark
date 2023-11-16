"""
Variational inference is a great approach for doing really complex,
often intractable Bayesian inference in approximate form. Common methods
(e.g. ADVI) lack from complexity so that approximate posterior does not
reveal the true nature of underlying problem. In some applications it can
yield unreliable decisions.

Recently on NIPS 2017 `OPVI  <https://arxiv.org/abs/1610.09033/>`_ framework
was presented. It generalizes variational inference so that the problem is
build with blocks. The first and essential block is Model itself. Second is
Approximation, in some cases :math:`log Q(D)` is not really needed. Necessity
depends on the third and fourth part of that black box, Operator and
Test Function respectively.

Operator is like an approach we use, it constructs loss from given Model,
Approximation and Test Function. The last one is not needed if we minimize
KL Divergence from Q to posterior. As a drawback we need to compute :math:`loq Q(D)`.
Sometimes approximation family is intractable and :math:`loq Q(D)` is not available,
here comes LS(Langevin Stein) Operator with a set of test functions.

Test Function has more unintuitive meaning. It is usually used with LS operator
and represents all we want from our approximate distribution. For any given vector
based function of :math:`z` LS operator yields zero mean function under posterior.
:math:`loq Q(D)` is no more needed. That opens a door to rich approximation
families as neural networks.

References
----------
-   Rajesh Ranganath, Jaan Altosaar, Dustin Tran, David M. Blei
    Operator Variational Inference
    https://arxiv.org/abs/1610.09033 (2016)
"""
from __future__ import annotations
import collections
import itertools
import warnings
from typing import Any, overload
import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.tensor.shape import unbroadcast
import pymc as pm
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.pytensorf import SeedSequenceSeed, compile_pymc, find_rng_nodes, identity, reseed_rngs
from pymc.util import RandomState, WithMemoization, _get_seeds_per_chain, locally_cachedmethod, makeiter
from pymc.variational.minibatch_rv import MinibatchRandomVariable, get_scaling
from pymc.variational.updates import adagrad_window
from pymc.vartypes import discrete_types
__all__ = ['ObjectiveFunction', 'Operator', 'TestFunction', 'Group', 'Approximation']

class VariationalInferenceError(Exception):
    """Exception for VI specific cases"""

class NotImplementedInference(VariationalInferenceError, NotImplementedError):
    """Marking non functional parts of code"""

class ExplicitInferenceError(VariationalInferenceError, TypeError):
    """Exception for bad explicit inference"""

class AEVBInferenceError(VariationalInferenceError, TypeError):
    """Exception for bad aevb inference"""

class ParametrizationError(VariationalInferenceError, ValueError):
    """Error raised in case of bad parametrization"""

class GroupError(VariationalInferenceError, TypeError):
    """Error related to VI groups"""

def _known_scan_ignored_inputs(terms):
    if False:
        for i in range(10):
            print('nop')
    from pymc.data import MinibatchIndexRV
    from pymc.distributions.simulator import SimulatorRV
    return [n.owner.inputs[0] for n in pytensor.graph.ancestors(terms) if n.owner is not None and isinstance(n.owner.op, (MinibatchIndexRV, SimulatorRV))]

def append_name(name):
    if False:
        i = 10
        return i + 15

    def wrap(f):
        if False:
            return 10
        if name is None:
            return f

        def inner(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            res = f(*args, **kwargs)
            res.name = name
            return res
        return inner
    return wrap

def node_property(f):
    if False:
        while True:
            i = 10
    'A shortcut for wrapping method to accessible tensor'
    if isinstance(f, str):

        def wrapper(fn):
            if False:
                while True:
                    i = 10
            ff = append_name(f)(fn)
            f_ = pytensor.config.change_flags(compute_test_value='off')(ff)
            return property(locally_cachedmethod(f_))
        return wrapper
    else:
        f_ = pytensor.config.change_flags(compute_test_value='off')(f)
        return property(locally_cachedmethod(f_))

@pytensor.config.change_flags(compute_test_value='ignore')
def try_to_set_test_value(node_in, node_out, s):
    if False:
        print('Hello World!')
    _s = s
    if s is None:
        s = 1
    s = pytensor.compile.view_op(pt.as_tensor(s))
    if not isinstance(node_in, (list, tuple)):
        node_in = [node_in]
    if not isinstance(node_out, (list, tuple)):
        node_out = [node_out]
    for (i, o) in zip(node_in, node_out):
        if hasattr(i.tag, 'test_value'):
            if not hasattr(s.tag, 'test_value'):
                continue
            else:
                tv = i.tag.test_value[None, ...]
                tv = np.repeat(tv, s.tag.test_value, 0)
                if _s is None:
                    tv = tv[0]
                o.tag.test_value = tv

class ObjectiveUpdates(pytensor.OrderedUpdates):
    """OrderedUpdates extension for storing loss"""
    loss = None

def _warn_not_used(smth, where):
    if False:
        i = 10
        return i + 15
    warnings.warn(f'`{smth}` is not used for {where} and ignored')

class ObjectiveFunction:
    """Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op : :class:`Operator`
        OPVI Functional operator
    tf : :class:`TestFunction`
        OPVI TestFunction
    """

    def __init__(self, op: Operator, tf: TestFunction):
        if False:
            return 10
        self.op = op
        self.tf = tf
    obj_params = property(lambda self: self.op.approx.params)
    test_params = property(lambda self: self.tf.params)
    approx = property(lambda self: self.op.approx)

    def updates(self, obj_n_mc=None, tf_n_mc=None, obj_optimizer=adagrad_window, test_optimizer=adagrad_window, more_obj_params=None, more_tf_params=None, more_updates=None, more_replacements=None, total_grad_norm_constraint=None):
        if False:
            print('Hello World!')
        'Calculate gradients for objective function, test function and then\n        constructs updates for optimization step\n\n        Parameters\n        ----------\n        obj_n_mc : int\n            Number of monte carlo samples used for approximation of objective gradients\n        tf_n_mc : int\n            Number of monte carlo samples used for approximation of test function gradients\n        obj_optimizer : function (loss, params) -> updates\n            Optimizer that is used for objective params\n        test_optimizer : function (loss, params) -> updates\n            Optimizer that is used for test function params\n        more_obj_params : list\n            Add custom params for objective optimizer\n        more_tf_params : list\n            Add custom params for test function optimizer\n        more_updates : dict\n            Add custom updates to resulting updates\n        more_replacements : dict\n            Apply custom replacements before calculating gradients\n        total_grad_norm_constraint : float\n            Bounds gradient norm, prevents exploding gradient problem\n\n        Returns\n        -------\n        :class:`ObjectiveUpdates`\n        '
        if more_updates is None:
            more_updates = dict()
        resulting_updates = ObjectiveUpdates()
        if self.test_params:
            self.add_test_updates(resulting_updates, tf_n_mc=tf_n_mc, test_optimizer=test_optimizer, more_tf_params=more_tf_params, more_replacements=more_replacements, total_grad_norm_constraint=total_grad_norm_constraint)
        else:
            if tf_n_mc is not None:
                _warn_not_used('tf_n_mc', self.op)
            if more_tf_params:
                _warn_not_used('more_tf_params', self.op)
        self.add_obj_updates(resulting_updates, obj_n_mc=obj_n_mc, obj_optimizer=obj_optimizer, more_obj_params=more_obj_params, more_replacements=more_replacements, total_grad_norm_constraint=total_grad_norm_constraint)
        resulting_updates.update(more_updates)
        return resulting_updates

    def add_test_updates(self, updates, tf_n_mc=None, test_optimizer=adagrad_window, more_tf_params=None, more_replacements=None, total_grad_norm_constraint=None):
        if False:
            print('Hello World!')
        if more_tf_params is None:
            more_tf_params = []
        if more_replacements is None:
            more_replacements = dict()
        tf_target = self(tf_n_mc, more_tf_params=more_tf_params, more_replacements=more_replacements)
        grads = pm.updates.get_or_compute_grads(tf_target, self.obj_params + more_tf_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(test_optimizer(grads, self.test_params + more_tf_params))

    def add_obj_updates(self, updates, obj_n_mc=None, obj_optimizer=adagrad_window, more_obj_params=None, more_replacements=None, total_grad_norm_constraint=None):
        if False:
            for i in range(10):
                print('nop')
        if more_obj_params is None:
            more_obj_params = []
        if more_replacements is None:
            more_replacements = dict()
        obj_target = self(obj_n_mc, more_obj_params=more_obj_params, more_replacements=more_replacements)
        grads = pm.updates.get_or_compute_grads(obj_target, self.obj_params + more_obj_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(obj_optimizer(grads, self.obj_params + more_obj_params))
        if self.op.returns_loss:
            updates.loss = obj_target

    @pytensor.config.change_flags(compute_test_value='off')
    def step_function(self, obj_n_mc=None, tf_n_mc=None, obj_optimizer=adagrad_window, test_optimizer=adagrad_window, more_obj_params=None, more_tf_params=None, more_updates=None, more_replacements=None, total_grad_norm_constraint=None, score=False, fn_kwargs=None):
        if False:
            print('Hello World!')
        "Step function that should be called on each optimization step.\n\n        Generally it solves the following problem:\n\n        .. math::\n\n                \\mathbf{\\lambda^{\\*}} = \\inf_{\\lambda} \\sup_{\\theta} t(\\mathbb{E}_{\\lambda}[(O^{p,q}f_{\\theta})(z)])\n\n        Parameters\n        ----------\n        obj_n_mc: `int`\n            Number of monte carlo samples used for approximation of objective gradients\n        tf_n_mc: `int`\n            Number of monte carlo samples used for approximation of test function gradients\n        obj_optimizer: function (grads, params) -> updates\n            Optimizer that is used for objective params\n        test_optimizer: function (grads, params) -> updates\n            Optimizer that is used for test function params\n        more_obj_params: `list`\n            Add custom params for objective optimizer\n        more_tf_params: `list`\n            Add custom params for test function optimizer\n        more_updates: `dict`\n            Add custom updates to resulting updates\n        total_grad_norm_constraint: `float`\n            Bounds gradient norm, prevents exploding gradient problem\n        score: `bool`\n            calculate loss on each step? Defaults to False for speed\n        fn_kwargs: `dict`\n            Add kwargs to pytensor.function (e.g. `{'profile': True}`)\n        more_replacements: `dict`\n            Apply custom replacements before calculating gradients\n\n        Returns\n        -------\n        `pytensor.function`\n        "
        if fn_kwargs is None:
            fn_kwargs = {}
        if score and (not self.op.returns_loss):
            raise NotImplementedError('%s does not have loss' % self.op)
        updates = self.updates(obj_n_mc=obj_n_mc, tf_n_mc=tf_n_mc, obj_optimizer=obj_optimizer, test_optimizer=test_optimizer, more_obj_params=more_obj_params, more_tf_params=more_tf_params, more_updates=more_updates, more_replacements=more_replacements, total_grad_norm_constraint=total_grad_norm_constraint)
        seed = self.approx.rng.randint(2 ** 30, dtype=np.int64)
        if score:
            step_fn = compile_pymc([], updates.loss, updates=updates, random_seed=seed, **fn_kwargs)
        else:
            step_fn = compile_pymc([], [], updates=updates, random_seed=seed, **fn_kwargs)
        return step_fn

    @pytensor.config.change_flags(compute_test_value='off')
    def score_function(self, sc_n_mc=None, more_replacements=None, fn_kwargs=None):
        if False:
            return 10
        'Compile scoring function that operates which takes no inputs and returns Loss\n\n        Parameters\n        ----------\n        sc_n_mc: `int`\n            number of scoring MC samples\n        more_replacements:\n            Apply custom replacements before compiling a function\n        fn_kwargs: `dict`\n            arbitrary kwargs passed to `pytensor.function`\n\n        Returns\n        -------\n        pytensor.function\n        '
        if fn_kwargs is None:
            fn_kwargs = {}
        if not self.op.returns_loss:
            raise NotImplementedError('%s does not have loss' % self.op)
        if more_replacements is None:
            more_replacements = {}
        loss = self(sc_n_mc, more_replacements=more_replacements)
        seed = self.approx.rng.randint(2 ** 30, dtype=np.int64)
        return compile_pymc([], loss, random_seed=seed, **fn_kwargs)

    @pytensor.config.change_flags(compute_test_value='off')
    def __call__(self, nmc, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'more_tf_params' in kwargs:
            m = -1.0
        else:
            m = 1.0
        a = self.op.apply(self.tf)
        a = self.approx.set_size_and_deterministic(a, nmc, 0, kwargs.get('more_replacements'))
        return m * self.op.T(a)

class Operator:
    """**Base class for Operator**

    Parameters
    ----------
    approx: :class:`Approximation`
        an approximation instance

    Notes
    -----
    For implementing custom operator it is needed to define :func:`Operator.apply` method
    """
    has_test_function = False
    returns_loss = True
    require_logq = True
    objective_class = ObjectiveFunction
    supports_aevb = property(lambda self: not self.approx.any_histograms)
    T = identity

    def __init__(self, approx):
        if False:
            while True:
                i = 10
        self.approx = approx
        if self.require_logq and (not approx.has_logq):
            raise ExplicitInferenceError('%s requires logq, but %s does not implement itplease change inference method' % (self, approx))
    inputs = property(lambda self: self.approx.inputs)
    logp = property(lambda self: self.approx.logp)
    varlogp = property(lambda self: self.approx.varlogp)
    datalogp = property(lambda self: self.approx.datalogp)
    logq = property(lambda self: self.approx.logq)
    logp_norm = property(lambda self: self.approx.logp_norm)
    varlogp_norm = property(lambda self: self.approx.varlogp_norm)
    datalogp_norm = property(lambda self: self.approx.datalogp_norm)
    logq_norm = property(lambda self: self.approx.logq_norm)
    model = property(lambda self: self.approx.model)

    def apply(self, f):
        if False:
            i = 10
            return i + 15
        'Operator itself\n\n        .. math::\n\n            (O^{p,q}f_{\\theta})(z)\n\n        Parameters\n        ----------\n        f: :class:`TestFunction` or None\n            function that takes `z = self.input` and returns\n            same dimensional output\n\n        Returns\n        -------\n        TensorVariable\n            symbolically applied operator\n        '
        raise NotImplementedError

    def __call__(self, f=None):
        if False:
            print('Hello World!')
        if self.has_test_function:
            if f is None:
                raise ParametrizationError('Operator %s requires TestFunction' % self)
            elif not isinstance(f, TestFunction):
                f = TestFunction.from_function(f)
        else:
            if f is not None:
                warnings.warn('TestFunction for %s is redundant and removed' % self, stacklevel=3)
            else:
                pass
            f = TestFunction()
        f.setup(self.approx)
        return self.objective_class(self, f)

    def __str__(self):
        if False:
            print('Hello World!')
        return '%(op)s[%(ap)s]' % dict(op=self.__class__.__name__, ap=self.approx.__class__.__name__)

def collect_shared_to_list(params):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for getting a list from\n    usable representation of parameters\n\n    Parameters\n    ----------\n    params: {dict|None}\n\n    Returns\n    -------\n    List\n    '
    if isinstance(params, dict):
        return list((t[1] for t in sorted(params.items(), key=lambda t: t[0]) if isinstance(t[1], pytensor.compile.SharedVariable)))
    elif params is None:
        return []
    else:
        raise TypeError('Unknown type %s for %r, need dict or None')

class TestFunction:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._inited = False
        self.shared_params = None

    @property
    def params(self):
        if False:
            return 10
        return collect_shared_to_list(self.shared_params)

    def __call__(self, z):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def setup(self, approx):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def from_function(cls, f):
        if False:
            print('Hello World!')
        if not callable(f):
            raise ParametrizationError('Need callable, got %r' % f)
        obj = TestFunction()
        obj.__call__ = f
        return obj

class Group(WithMemoization):
    """**Base class for grouping variables in VI**

    Grouped Approximation is used for modelling mutual dependencies
    for a specified group of variables. Base for local and global group.

    Parameters
    ----------
    group: list
        List of PyMC variables or None indicating that group takes all the rest variables
    vfam: str
        String that marks the corresponding variational family for the group.
        Cannot be passed both with `params`
    params: dict
        Dict with variational family parameters, full description can be found below.
        Cannot be passed both with `vfam`
    random_seed: int
        Random seed for underlying random generator
    model :
        PyMC Model
    options: dict
        Special options for the group
    kwargs: Other kwargs for the group

    Notes
    -----
    Group instance/class has some important constants:

    -   **has_logq**
        Tells that distribution is defined explicitly

    These constants help providing the correct inference method for given parametrization

    Examples
    --------
    **Basic Initialization**

    :class:`Group` is a factory class. You do not need to call every ApproximationGroup explicitly.
    Passing the correct `vfam` (Variational FAMily) argument you'll tell what
    parametrization is desired for the group. This helps not to overload code with lots of classes.

    .. code:: python

        >>> group = Group([latent1, latent2], vfam='mean_field')

    The other way to select approximation is to provide `params` dictionary that has some
    predefined well shaped parameters. Keys of the dict serve as an identifier for variational family and help
    to autoselect the correct group class. To identify what approximation to use, params dict should
    have the full set of needed parameters. As there are 2 ways to instantiate the :class:`Group`
    passing both `vfam` and `params` is prohibited. Partial parametrization is prohibited by design to
    avoid corner cases and possible problems.

    .. code:: python

        >>> group = Group([latent3], params=dict(mu=my_mu, rho=my_rho))

    Important to note that in case you pass custom params they will not be autocollected by optimizer, you'll
    have to provide them with `more_obj_params` keyword.

    **Supported dict keys:**

    -   `{'mu', 'rho'}`: :class:`MeanFieldGroup`

    -   `{'mu', 'L_tril'}`: :class:`FullRankGroup`

    -   `{'histogram'}`: :class:`EmpiricalGroup`

    **Delayed Initialization**

    When you have a lot of latent variables it is impractical to do it all manually.
    To make life much simpler, You can pass `None` instead of list of variables. That case
    you'll not create shared parameters until you pass all collected groups to
    Approximation object that collects all the groups together and checks that every group is
    correctly initialized. For those groups which have group equal to `None` it will collect all
    the rest variables not covered by other groups and perform delayed init.

    .. code:: python

        >>> group_1 = Group([latent1], vfam='fr')  # latent1 has full rank approximation
        >>> group_other = Group(None, vfam='mf')  # other variables have mean field Q
        >>> approx = Approximation([group_1, group_other])

    **Summing Up**

    When you have created all the groups they need to pass all the groups to :class:`Approximation`.
    It does not accept any other parameter rather than `groups`

    .. code:: python

        >>> approx = Approximation(my_groups)

    See Also
    --------
    :class:`Approximation`

    References
    ----------
    -   Kingma, D. P., & Welling, M. (2014).
        `Auto-Encoding Variational Bayes. stat, 1050, 1. <https://arxiv.org/abs/1312.6114>`_
    """
    shared_params = None
    symbolic_initial = None
    replacements = None
    input = None
    has_logq = True
    initial_dist_name = 'normal'
    initial_dist_map = 0.0
    __param_spec__: dict = dict()
    short_name = ''
    alias_names: frozenset[str] = frozenset()
    __param_registry: dict[frozenset, Any] = dict()
    __name_registry: dict[str, Any] = dict()

    @classmethod
    def register(cls, sbcls):
        if False:
            i = 10
            return i + 15
        assert frozenset(sbcls.__param_spec__) not in cls.__param_registry, 'Duplicate __param_spec__'
        cls.__param_registry[frozenset(sbcls.__param_spec__)] = sbcls
        assert sbcls.short_name not in cls.__name_registry, 'Duplicate short_name'
        cls.__name_registry[sbcls.short_name] = sbcls
        for alias in sbcls.alias_names:
            assert alias not in cls.__name_registry, 'Duplicate alias_name'
            cls.__name_registry[alias] = sbcls
        return sbcls

    @classmethod
    def group_for_params(cls, params):
        if False:
            for i in range(10):
                print('nop')
        if frozenset(params) not in cls.__param_registry:
            raise KeyError('No such group for the following params: {!r}, only the following are supported\n\n{}'.format(params, cls.__param_registry))
        return cls.__param_registry[frozenset(params)]

    @classmethod
    def group_for_short_name(cls, name):
        if False:
            while True:
                i = 10
        if name.lower() not in cls.__name_registry:
            raise KeyError('No such group: {!r}, only the following are supported\n\n{}'.format(name, cls.__name_registry))
        return cls.__name_registry[name.lower()]

    def __new__(cls, group=None, vfam=None, params=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if cls is Group:
            if vfam is not None and params is not None:
                raise TypeError('Cannot call Group with both `vfam` and `params` provided')
            elif vfam is not None:
                return super().__new__(cls.group_for_short_name(vfam))
            elif params is not None:
                return super().__new__(cls.group_for_params(params))
            else:
                raise TypeError('Need to call Group with either `vfam` or `params` provided')
        else:
            return super().__new__(cls)

    def __init__(self, group, vfam=None, params=None, random_seed=None, model=None, options=None, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(vfam, str):
            vfam = vfam.lower()
        if options is None:
            options = dict()
        self.options = options
        self._vfam = vfam
        self.rng = np.random.RandomState(random_seed)
        model = modelcontext(model)
        self.model = model
        self.group = group
        self.user_params = params
        self._user_params = None
        self.replacements = collections.OrderedDict()
        self.ordering = collections.OrderedDict()
        self._kwargs = kwargs
        if self.group is not None:
            self.__init_group__(self.group)

    def _prepare_start(self, start=None):
        if False:
            while True:
                i = 10
        ipfn = make_initial_point_fn(model=self.model, overrides=start, jitter_rvs={}, return_transformed=True)
        start = ipfn(self.rng.randint(2 ** 30, dtype=np.int64))
        group_vars = {self.model.rvs_to_values[v].name for v in self.group}
        start = {k: v for (k, v) in start.items() if k in group_vars}
        start = DictToArrayBijection.map(start).data
        return start

    @classmethod
    def get_param_spec_for(cls, **kwargs):
        if False:
            print('Hello World!')
        res = dict()
        for (name, fshape) in cls.__param_spec__.items():
            res[name] = tuple((eval(s, kwargs) for s in fshape))
        return res

    def _check_user_params(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - checks user params, allocates them if they are correct, returns True.\n        If they are not present, returns False\n\n        Parameters\n        ----------\n        kwargs: special kwargs needed sometimes\n\n        Returns\n        -------\n        bool indicating whether to allocate new shared params\n        '
        user_params = self.user_params
        if user_params is None:
            return False
        if not isinstance(user_params, dict):
            raise TypeError('params should be a dict')
        givens = set(user_params.keys())
        needed = set(self.__param_spec__)
        if givens != needed:
            raise ParametrizationError('Passed parameters do not have a needed set of keys, they should be equal, got {givens}, needed {needed}'.format(givens=givens, needed=needed))
        self._user_params = dict()
        spec = self.get_param_spec_for(d=self.ddim, **kwargs.pop('spec_kw', {}))
        for (name, param) in self.user_params.items():
            shape = spec[name]
            self._user_params[name] = pt.as_tensor(param).reshape(shape)
        return True

    def _initial_type(self, name):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - initial type with given name. The correct type depends on `self.batched`\n\n        Parameters\n        ----------\n        name: str\n            name for tensor\n        Returns\n        -------\n        tensor\n        '
        return pt.matrix(name)

    def _input_type(self, name):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - input type with given name. The correct type depends on `self.batched`\n\n        Parameters\n        ----------\n        name: str\n            name for tensor\n        Returns\n        -------\n        tensor\n        '
        return pt.vector(name)

    @pytensor.config.change_flags(compute_test_value='off')
    def __init_group__(self, group):
        if False:
            print('Hello World!')
        if not group:
            raise GroupError('Got empty group')
        if self.group is None:
            self.group = group
        self.symbolic_initial = self._initial_type(self.__class__.__name__ + '_symbolic_initial_tensor')
        self.input = self._input_type(self.__class__.__name__ + '_symbolic_input')
        model_initial_point = self.model.initial_point(0)
        start_idx = 0
        for var in self.group:
            if var.type.numpy_dtype.name in discrete_types:
                raise ParametrizationError(f'Discrete variables are not supported by VI: {var}')
            value_var = self.model.rvs_to_values[var]
            test_var = model_initial_point[value_var.name]
            shape = test_var.shape
            size = test_var.size
            dtype = test_var.dtype
            vr = self.input[..., start_idx:start_idx + size].reshape(shape).astype(dtype)
            vr.name = value_var.name + '_vi_replacement'
            self.replacements[value_var] = vr
            self.ordering[value_var.name] = (value_var.name, slice(start_idx, start_idx + size), shape, dtype)
            start_idx += size

    def _finalize_init(self):
        if False:
            while True:
                i = 10
        '*Dev* - clean up after init'
        del self._kwargs

    @property
    def params_dict(self):
        if False:
            for i in range(10):
                print('nop')
        if self._user_params is not None:
            return self._user_params
        else:
            return self.shared_params

    @property
    def params(self):
        if False:
            print('Hello World!')
        if self.user_params is not None:
            return collect_shared_to_list(self.user_params)
        else:
            return collect_shared_to_list(self.shared_params)

    def _new_initial_shape(self, size, dim, more_replacements=None):
        if False:
            print('Hello World!')
        '*Dev* - correctly proceeds sampling with variable batch size\n\n        Parameters\n        ----------\n        size: scalar\n            sample size\n        dim: scalar\n            latent fixed dim\n        more_replacements: dict\n            replacements for latent batch shape\n\n        Returns\n        -------\n        shape vector\n        '
        return pt.stack([size, dim])

    @node_property
    def ndim(self):
        if False:
            while True:
                i = 10
        return self.ddim

    @property
    def ddim(self):
        if False:
            i = 10
            return i + 15
        return sum((s.stop - s.start for (_, s, _, _) in self.ordering.values()))

    def _new_initial(self, size, deterministic, more_replacements=None):
        if False:
            while True:
                i = 10
        "*Dev* - allocates new initial random generator\n\n        Parameters\n        ----------\n        size: scalar\n            sample size\n        deterministic: bool or scalar\n            whether to sample in deterministic manner\n        more_replacements: dict\n            more replacements passed to shape\n\n        Notes\n        -----\n        Suppose you have a AEVB setup that:\n\n            -   input `X` is purely symbolic, and `X.shape[0]` is needed to `initial` second dim\n            -   to perform inference, `X` is replaced with data tensor, however, since `X.shape[0]` in `initial`\n                remains symbolic and can't be replaced, you get `MissingInputError`\n            -   as a solution, here we perform a manual replacement for the second dim in `initial`.\n\n        Returns\n        -------\n        tensor\n        "
        if size is None:
            size = 1
        if not isinstance(deterministic, Variable):
            deterministic = np.int8(deterministic)
        (dim, dist_name, dist_map) = (self.ddim, self.initial_dist_name, self.initial_dist_map)
        dtype = self.symbolic_initial.dtype
        dim = pt.as_tensor(dim)
        size = pt.as_tensor(size)
        shape = self._new_initial_shape(size, dim, more_replacements)
        if not isinstance(deterministic, Variable):
            if deterministic:
                return pt.ones(shape, dtype) * dist_map
            else:
                return getattr(pt.random, dist_name)(size=shape)
        else:
            sample = getattr(pt.random, dist_name)(size=shape)
            initial = pt.switch(deterministic, pt.ones(shape, dtype) * dist_map, sample)
            return initial

    @node_property
    def symbolic_random(self):
        if False:
            while True:
                i = 10
        '*Dev* - abstract node that takes `self.symbolic_initial` and creates\n        approximate posterior that is parametrized with `self.params_dict`.\n\n        Implementation should take in account `self.batched`. If `self.batched` is `True`, then\n        `self.symbolic_initial` is 3d tensor, else 2d\n\n        Returns\n        -------\n        tensor\n        '
        raise NotImplementedError

    @overload
    def set_size_and_deterministic(self, node: Variable, s, d: bool, more_replacements: dict | None=None) -> Variable:
        if False:
            return 10
        ...

    @overload
    def set_size_and_deterministic(self, node: list[Variable], s, d: bool, more_replacements: dict | None=None) -> list[Variable]:
        if False:
            return 10
        ...

    @pytensor.config.change_flags(compute_test_value='off')
    def set_size_and_deterministic(self, node: Variable | list[Variable], s, d: bool, more_replacements: dict | None=None) -> Variable | list[Variable]:
        if False:
            print('Hello World!')
        '*Dev* - after node is sampled via :func:`symbolic_sample_over_posterior` or\n        :func:`symbolic_single_sample` new random generator can be allocated and applied to node\n\n        Parameters\n        ----------\n        node\n            PyTensor node(s) with symbolically applied VI replacements\n        s: scalar\n            desired number of samples\n        d: bool or int\n            whether sampling is done deterministically\n        more_replacements: dict\n            more replacements to apply\n\n        Returns\n        -------\n        :class:`Variable` or list with applied replacements, ready to use\n        '
        flat2rand = self.make_size_and_deterministic_replacements(s, d, more_replacements)
        node_out = graph_replace(node, flat2rand, strict=False)
        assert not set(makeiter(self.input)) & set(pytensor.graph.graph_inputs(makeiter(node_out)))
        try_to_set_test_value(node, node_out, s)
        assert self.symbolic_random not in set(pytensor.graph.graph_inputs(makeiter(node_out)))
        return node_out

    def to_flat_input(self, node):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - replace vars with flattened view stored in `self.inputs`'
        return graph_replace(node, self.replacements, strict=False)

    def symbolic_sample_over_posterior(self, node):
        if False:
            while True:
                i = 10
        '*Dev* - performs sampling of node applying independent samples from posterior each time.\n        Note that it is done symbolically and this node needs :func:`set_size_and_deterministic` call\n        '
        node = self.to_flat_input(node)
        random = self.symbolic_random.astype(self.symbolic_initial.dtype)
        random = pt.specify_shape(random, self.symbolic_initial.type.shape)

        def sample(post, *_):
            if False:
                print('Hello World!')
            return graph_replace(node, {self.input: post}, strict=False)
        (nodes, _) = pytensor.scan(sample, random, non_sequences=_known_scan_ignored_inputs(makeiter(random)))
        assert self.input not in set(pytensor.graph.graph_inputs(makeiter(nodes)))
        return nodes

    def symbolic_single_sample(self, node):
        if False:
            i = 10
            return i + 15
        '*Dev* - performs sampling of node applying single sample from posterior.\n        Note that it is done symbolically and this node needs\n        :func:`set_size_and_deterministic` call with `size=1`\n        '
        node = self.to_flat_input(node)
        random = self.symbolic_random.astype(self.symbolic_initial.dtype)
        return graph_replace(node, {self.input: random[0]}, strict=False)

    def make_size_and_deterministic_replacements(self, s, d, more_replacements=None):
        if False:
            while True:
                i = 10
        '*Dev* - creates correct replacements for initial depending on\n        sample size and deterministic flag\n\n        Parameters\n        ----------\n        s: scalar\n            sample size\n        d: bool or scalar\n            whether sampling is done deterministically\n        more_replacements: dict\n            replacements for shape and initial\n\n        Returns\n        -------\n        dict with replacements for initial\n        '
        initial = self._new_initial(s, d, more_replacements)
        initial = pt.specify_shape(initial, self.symbolic_initial.type.shape)
        if initial.type.broadcastable != self.symbolic_initial.type.broadcastable:
            unbroadcast_axes = (i for (i, b) in enumerate(self.symbolic_initial.type.broadcastable) if not b)
            initial = unbroadcast(initial, *unbroadcast_axes)
        if more_replacements:
            initial = graph_replace(initial, more_replacements, strict=False)
        return {self.symbolic_initial: initial}

    @node_property
    def symbolic_normalizing_constant(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - normalizing constant for `self.logq`, scales it to `minibatch_size` instead of `total_size`'
        t = self.to_flat_input(pt.max([get_scaling(v.owner.inputs[1:], v.shape) for v in self.group if isinstance(v.owner.op, MinibatchRandomVariable)] + [1.0]))
        t = self.symbolic_single_sample(t)
        return pm.floatX(t)

    @node_property
    def symbolic_logq_not_scaled(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - symbolically computed logq for `self.symbolic_random`\n        computations can be more efficient since all is known beforehand including\n        `self.symbolic_random`\n        '
        raise NotImplementedError

    @node_property
    def symbolic_logq(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - correctly scaled `self.symbolic_logq_not_scaled`'
        return self.symbolic_logq_not_scaled

    @node_property
    def logq(self):
        if False:
            return 10
        '*Dev* - Monte Carlo estimate for group `logQ`'
        return self.symbolic_logq.mean(0)

    @node_property
    def logq_norm(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - Monte Carlo estimate for group `logQ` normalized'
        return self.logq / self.symbolic_normalizing_constant

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.group is None:
            shp = 'undefined'
        else:
            shp = str(self.ddim)
        return f'{self.__class__.__name__}[{shp}]'

    @node_property
    def std(self) -> pt.TensorVariable:
        if False:
            for i in range(10):
                print('nop')
        'Standard deviation of the latent variables as an unstructured 1-dimensional tensor variable'
        raise NotImplementedError()

    @node_property
    def cov(self) -> pt.TensorVariable:
        if False:
            print('Hello World!')
        'Covariance between the latent variables as an unstructured 2-dimensional tensor variable'
        raise NotImplementedError()

    @node_property
    def mean(self) -> pt.TensorVariable:
        if False:
            for i in range(10):
                print('nop')
        'Mean of the latent variables as an unstructured 1-dimensional tensor variable'
        raise NotImplementedError()

    def var_to_data(self, shared: pt.TensorVariable) -> xarray.Dataset:
        if False:
            print('Hello World!')
        'Takes a flat 1-dimensional tensor variable and maps it to an xarray data set based on the information in\n        `self.ordering`.\n        '
        shared_nda = shared.eval()
        result = dict()
        for (name, s, shape, dtype) in self.ordering.values():
            dims = self.model.named_vars_to_dims.get(name, None)
            if dims is not None:
                coords = {d: np.array(self.model.coords[d]) for d in dims}
            else:
                coords = None
            values = shared_nda[s].reshape(shape).astype(dtype)
            result[name] = xarray.DataArray(values, coords=coords, dims=dims, name=name)
        return xarray.Dataset(result)

    @property
    def mean_data(self) -> xarray.Dataset:
        if False:
            i = 10
            return i + 15
        'Mean of the latent variables as an xarray Dataset'
        return self.var_to_data(self.mean)

    @property
    def std_data(self) -> xarray.Dataset:
        if False:
            while True:
                i = 10
        'Standard deviation of the latent variables as an xarray Dataset'
        return self.var_to_data(self.std)
group_for_params = Group.group_for_params
group_for_short_name = Group.group_for_short_name

class Approximation(WithMemoization):
    """**Wrapper for grouped approximations**

    Wraps list of groups, creates an Approximation instance that collects
    sampled variables from all the groups, also collects logQ needed for
    explicit Variational Inference.

    Parameters
    ----------
    groups: list[Group]
        List of :class:`Group` instances. They should have all model variables
    model: Model

    Notes
    -----
    Some shortcuts for single group approximations are available:

        -   :class:`MeanField`
        -   :class:`FullRank`
        -   :class:`Empirical`

    See Also
    --------
    :class:`Group`
    """

    def __init__(self, groups, model=None):
        if False:
            return 10
        self._scale_cost_to_minibatch = pytensor.shared(np.int8(1))
        model = modelcontext(model)
        if not model.free_RVs:
            raise TypeError('Model does not have an free RVs')
        self.groups = list()
        seen = set()
        rest = None
        for g in groups:
            if g.group is None:
                if rest is not None:
                    raise GroupError('More than one group is specified for the rest variables')
                else:
                    rest = g
            else:
                if set(g.group) & seen:
                    raise GroupError('Found duplicates in groups')
                seen.update(g.group)
                self.groups.append(g)
        unseen_free_RVs = [var for var in model.free_RVs if var not in seen]
        if unseen_free_RVs:
            if rest is None:
                raise GroupError('No approximation is specified for the rest variables')
            else:
                rest.__init_group__(unseen_free_RVs)
                self.groups.append(rest)
        self.model = model

    @property
    def has_logq(self):
        if False:
            print('Hello World!')
        return all(self.collect('has_logq'))

    def collect(self, item):
        if False:
            print('Hello World!')
        return [getattr(g, item) for g in self.groups]
    inputs = property(lambda self: self.collect('input'))
    symbolic_randoms = property(lambda self: self.collect('symbolic_random'))

    @property
    def scale_cost_to_minibatch(self):
        if False:
            return 10
        '*Dev* - Property to control scaling cost to minibatch'
        return bool(self._scale_cost_to_minibatch.get_value())

    @scale_cost_to_minibatch.setter
    def scale_cost_to_minibatch(self, value):
        if False:
            while True:
                i = 10
        self._scale_cost_to_minibatch.set_value(np.int8(bool(value)))

    @node_property
    def symbolic_normalizing_constant(self):
        if False:
            while True:
                i = 10
        '*Dev* - normalizing constant for `self.logq`, scales it to `minibatch_size` instead of `total_size`.\n        Here the effect is controlled by `self.scale_cost_to_minibatch`\n        '
        t = pt.max(self.collect('symbolic_normalizing_constant') + [get_scaling(obs.owner.inputs[1:], obs.shape) for obs in self.model.observed_RVs if isinstance(obs.owner.op, MinibatchRandomVariable)])
        t = pt.switch(self._scale_cost_to_minibatch, t, pt.constant(1, dtype=t.dtype))
        return pm.floatX(t)

    @node_property
    def symbolic_logq(self):
        if False:
            return 10
        '*Dev* - collects `symbolic_logq` for all groups'
        return pt.add(*self.collect('symbolic_logq'))

    @node_property
    def logq(self):
        if False:
            while True:
                i = 10
        '*Dev* - collects `logQ` for all groups'
        return pt.add(*self.collect('logq'))

    @node_property
    def logq_norm(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - collects `logQ` for all groups and normalizes it'
        return self.logq / self.symbolic_normalizing_constant

    @node_property
    def _sized_symbolic_varlogp_and_datalogp(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - computes sampled prior term from model via `pytensor.scan`'
        (varlogp_s, datalogp_s) = self.symbolic_sample_over_posterior([self.model.varlogp, self.model.datalogp])
        return (varlogp_s, datalogp_s)

    @node_property
    def sized_symbolic_varlogp(self):
        if False:
            while True:
                i = 10
        '*Dev* - computes sampled prior term from model via `pytensor.scan`'
        return self._sized_symbolic_varlogp_and_datalogp[0]

    @node_property
    def sized_symbolic_datalogp(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - computes sampled data term from model via `pytensor.scan`'
        return self._sized_symbolic_varlogp_and_datalogp[1]

    @node_property
    def sized_symbolic_logp(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - computes sampled logP from model via `pytensor.scan`'
        return self.sized_symbolic_varlogp + self.sized_symbolic_datalogp

    @node_property
    def logp(self):
        if False:
            return 10
        '*Dev* - computes :math:`E_{q}(logP)` from model via `pytensor.scan` that can be optimized later'
        return self.varlogp + self.datalogp

    @node_property
    def varlogp(self):
        if False:
            return 10
        '*Dev* - computes :math:`E_{q}(prior term)` from model via `pytensor.scan` that can be optimized later'
        return self.sized_symbolic_varlogp.mean(0)

    @node_property
    def datalogp(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - computes :math:`E_{q}(data term)` from model via `pytensor.scan` that can be optimized later'
        return self.sized_symbolic_datalogp.mean(0)

    @node_property
    def _single_symbolic_varlogp_and_datalogp(self):
        if False:
            print('Hello World!')
        '*Dev* - computes sampled prior term from model via `pytensor.scan`'
        (varlogp, datalogp) = self.symbolic_single_sample([self.model.varlogp, self.model.datalogp])
        return (varlogp, datalogp)

    @node_property
    def single_symbolic_varlogp(self):
        if False:
            i = 10
            return i + 15
        '*Dev* - for single MC sample estimate of :math:`E_{q}(prior term)` `pytensor.scan`\n        is not needed and code can be optimized'
        return self._single_symbolic_varlogp_and_datalogp[0]

    @node_property
    def single_symbolic_datalogp(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - for single MC sample estimate of :math:`E_{q}(data term)` `pytensor.scan`\n        is not needed and code can be optimized'
        return self._single_symbolic_varlogp_and_datalogp[1]

    @node_property
    def single_symbolic_logp(self):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - for single MC sample estimate of :math:`E_{q}(logP)` `pytensor.scan`\n        is not needed and code can be optimized'
        return self.single_symbolic_datalogp + self.single_symbolic_varlogp

    @node_property
    def logp_norm(self):
        if False:
            print('Hello World!')
        '*Dev* - normalized :math:`E_{q}(logP)`'
        return self.logp / self.symbolic_normalizing_constant

    @node_property
    def varlogp_norm(self):
        if False:
            print('Hello World!')
        '*Dev* - normalized :math:`E_{q}(prior term)`'
        return self.varlogp / self.symbolic_normalizing_constant

    @node_property
    def datalogp_norm(self):
        if False:
            while True:
                i = 10
        '*Dev* - normalized :math:`E_{q}(data term)`'
        return self.datalogp / self.symbolic_normalizing_constant

    @property
    def replacements(self):
        if False:
            while True:
                i = 10
        '*Dev* - all replacements from groups to replace PyMC random variables with approximation'
        return collections.OrderedDict(itertools.chain.from_iterable((g.replacements.items() for g in self.groups)))

    def make_size_and_deterministic_replacements(self, s, d, more_replacements=None):
        if False:
            while True:
                i = 10
        '*Dev* - creates correct replacements for initial depending on\n        sample size and deterministic flag\n\n        Parameters\n        ----------\n        s: scalar\n            sample size\n        d: bool\n            whether sampling is done deterministically\n        more_replacements: dict\n            replacements for shape and initial\n\n        Returns\n        -------\n        dict with replacements for initial\n        '
        if more_replacements is None:
            more_replacements = {}
        flat2rand = collections.OrderedDict()
        for g in self.groups:
            flat2rand.update(g.make_size_and_deterministic_replacements(s, d, more_replacements))
        flat2rand.update(more_replacements)
        return flat2rand

    @pytensor.config.change_flags(compute_test_value='off')
    def set_size_and_deterministic(self, node, s, d, more_replacements=None):
        if False:
            for i in range(10):
                print('nop')
        '*Dev* - after node is sampled via :func:`symbolic_sample_over_posterior` or\n        :func:`symbolic_single_sample` new random generator can be allocated and applied to node\n\n        Parameters\n        ----------\n        node: :class:`Variable`\n            PyTensor node with symbolically applied VI replacements\n        s: scalar\n            desired number of samples\n        d: bool or int\n            whether sampling is done deterministically\n        more_replacements: dict\n            more replacements to apply\n\n        Returns\n        -------\n        :class:`Variable` with applied replacements, ready to use\n        '
        _node = node
        optimizations = self.get_optimization_replacements(s, d)
        flat2rand = self.make_size_and_deterministic_replacements(s, d, more_replacements)
        node = graph_replace(node, optimizations, strict=False)
        node = graph_replace(node, flat2rand, strict=False)
        assert not set(self.symbolic_randoms) & set(pytensor.graph.graph_inputs(makeiter(node)))
        try_to_set_test_value(_node, node, s)
        return node

    def to_flat_input(self, node, more_replacements=None):
        if False:
            while True:
                i = 10
        '*Dev* - replace vars with flattened view stored in `self.inputs`'
        more_replacements = more_replacements or {}
        node = graph_replace(node, more_replacements, strict=False)
        return graph_replace(node, self.replacements, strict=False)

    def symbolic_sample_over_posterior(self, node, more_replacements=None):
        if False:
            print('Hello World!')
        '*Dev* - performs sampling of node applying independent samples from posterior each time.\n        Note that it is done symbolically and this node needs :func:`set_size_and_deterministic` call\n        '
        node = self.to_flat_input(node)

        def sample(*post):
            if False:
                return 10
            return graph_replace(node, dict(zip(self.inputs, post)), strict=False)
        (nodes, _) = pytensor.scan(sample, self.symbolic_randoms, non_sequences=_known_scan_ignored_inputs(makeiter(node)))
        assert not set(self.inputs) & set(pytensor.graph.graph_inputs(makeiter(nodes)))
        return nodes

    def symbolic_single_sample(self, node, more_replacements=None):
        if False:
            print('Hello World!')
        '*Dev* - performs sampling of node applying single sample from posterior.\n        Note that it is done symbolically and this node needs\n        :func:`set_size_and_deterministic` call with `size=1`\n        '
        node = self.to_flat_input(node, more_replacements=more_replacements)
        post = [v[0] for v in self.symbolic_randoms]
        inp = self.inputs
        return graph_replace(node, dict(zip(inp, post)), strict=False)

    def get_optimization_replacements(self, s, d):
        if False:
            return 10
        '*Dev* - optimizations for logP. If sample size is static and equal to 1:\n        then `pytensor.scan` MC estimate is replaced with single sample without call to `pytensor.scan`.\n        '
        repl = collections.OrderedDict()
        if isinstance(s, int) and s == 1 or s is None:
            repl[self.varlogp] = self.single_symbolic_varlogp
            repl[self.datalogp] = self.single_symbolic_datalogp
        return repl

    @pytensor.config.change_flags(compute_test_value='off')
    def sample_node(self, node, size=None, deterministic=False, more_replacements=None):
        if False:
            while True:
                i = 10
        'Samples given node or nodes over shared posterior\n\n        Parameters\n        ----------\n        node: PyTensor Variables (or PyTensor expressions)\n        size: None or scalar\n            number of samples\n        more_replacements: `dict`\n            add custom replacements to graph, e.g. change input source\n        deterministic: bool\n            whether to use zeros as initial distribution\n            if True - zero initial point will produce constant latent variables\n\n        Returns\n        -------\n        sampled node(s) with replacements\n        '
        node_in = node
        if more_replacements:
            node = graph_replace(node, more_replacements, strict=False)
        if not isinstance(node, (list, tuple)):
            node = [node]
        node = self.model.replace_rvs_by_values(node)
        if not isinstance(node_in, (list, tuple)):
            node = node[0]
        if size is None:
            node_out = self.symbolic_single_sample(node)
        else:
            node_out = self.symbolic_sample_over_posterior(node)
        node_out = self.set_size_and_deterministic(node_out, size, deterministic)
        try_to_set_test_value(node_in, node_out, size)
        return node_out

    def rslice(self, name):
        if False:
            i = 10
            return i + 15
        '*Dev* - vectorized sampling for named random variable without call to `pytensor.scan`.\n        This node still needs :func:`set_size_and_deterministic` to be evaluated\n        '

        def vars_names(vs):
            if False:
                i = 10
                return i + 15
            return {self.model.rvs_to_values[v].name for v in vs}
        for (vars_, random, ordering) in zip(self.collect('group'), self.symbolic_randoms, self.collect('ordering')):
            if name in vars_names(vars_):
                (name_, slc, shape, dtype) = ordering[name]
                found = random[..., slc].reshape((random.shape[0],) + shape).astype(dtype)
                found.name = name + '_vi_random_slice'
                break
        else:
            raise KeyError('%r not found' % name)
        return found

    @node_property
    def sample_dict_fn(self):
        if False:
            return 10
        s = pt.iscalar()
        names = [self.model.rvs_to_values[v].name for v in self.model.free_RVs]
        sampled = [self.rslice(name) for name in names]
        sampled = self.set_size_and_deterministic(sampled, s, 0)
        sample_fn = compile_pymc([s], sampled)
        rng_nodes = find_rng_nodes(sampled)

        def inner(draws=100, *, random_seed: SeedSequenceSeed=None):
            if False:
                i = 10
                return i + 15
            if random_seed is not None:
                reseed_rngs(rng_nodes, random_seed)
            _samples = sample_fn(draws)
            return {v_: s_ for (v_, s_) in zip(names, _samples)}
        return inner

    def sample(self, draws=500, *, random_seed: RandomState=None, return_inferencedata=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Draw samples from variational posterior.\n\n        Parameters\n        ----------\n        draws : int\n            Number of random samples.\n        random_seed : int, RandomState or Generator, optional\n            Seed for the random number generator.\n        return_inferencedata : bool\n            Return trace in Arviz format.\n\n        Returns\n        -------\n        trace: :class:`pymc.backends.base.MultiTrace`\n            Samples drawn from variational posterior.\n        '
        kwargs['log_likelihood'] = False
        if random_seed is not None:
            (random_seed,) = _get_seeds_per_chain(random_seed, 1)
        samples: dict = self.sample_dict_fn(draws, random_seed=random_seed)
        points = ({name: records[i] for (name, records) in samples.items()} for i in range(draws))
        trace = NDArray(model=self.model, test_point={name: records[0] for (name, records) in samples.items()})
        try:
            trace.setup(draws=draws, chain=0)
            for point in points:
                trace.record(point)
        finally:
            trace.close()
        multi_trace = MultiTrace([trace])
        if not return_inferencedata:
            return multi_trace
        else:
            return pm.to_inference_data(multi_trace, model=self.model, **kwargs)

    @property
    def ndim(self):
        if False:
            for i in range(10):
                print('nop')
        return sum(self.collect('ndim'))

    @property
    def ddim(self):
        if False:
            i = 10
            return i + 15
        return sum(self.collect('ddim'))

    @node_property
    def symbolic_random(self):
        if False:
            while True:
                i = 10
        return pt.concatenate(self.collect('symbolic_random'), axis=-1)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if len(self.groups) < 5:
            return 'Approximation{' + ' & '.join(map(str, self.groups)) + '}'
        else:
            forprint = self.groups[:2] + ['...'] + self.groups[-2:]
            return 'Approximation{' + ' & '.join(map(str, forprint)) + '}'

    @property
    def all_histograms(self):
        if False:
            i = 10
            return i + 15
        return all((isinstance(g, pm.approximations.EmpiricalGroup) for g in self.groups))

    @property
    def any_histograms(self):
        if False:
            print('Hello World!')
        return any((isinstance(g, pm.approximations.EmpiricalGroup) for g in self.groups))

    @node_property
    def joint_histogram(self):
        if False:
            while True:
                i = 10
        if not self.all_histograms:
            raise VariationalInferenceError('%s does not consist of all Empirical approximations')
        return pt.concatenate(self.collect('histogram'), axis=-1)

    @property
    def params(self):
        if False:
            i = 10
            return i + 15
        return sum(self.collect('params'), [])