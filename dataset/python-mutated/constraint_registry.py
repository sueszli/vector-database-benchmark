"""
PyTorch provides two global :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects. These objects both
input constraints and return transforms, but they have different guarantees on
bijectivity.

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is guaranteed to have
   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is not guaranteed to
   implement ``.log_abs_det_jacobian()``.

The ``transform_to()`` registry is useful for performing unconstrained
optimization on constrained parameters of probability distributions, which are
indicated by each distribution's ``.arg_constraints`` dict. These transforms often
overparameterize a space in order to avoid rotation; they are thus more
suitable for coordinate-wise optimization algorithms like Adam::

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints['scale'])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, where
samples from a probability distribution with constrained ``.support`` are
propagated in an unconstrained space, and algorithms are typically rotation
invariant.::

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    An example where ``transform_to`` and ``biject_to`` differ is
    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply
    exponentiates and normalizes its inputs; this is a cheap and mostly
    coordinate-wise operation appropriate for algorithms like SVI. In
    contrast, ``biject_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.StickBreakingTransform` that
    bijects its input down to a one-fewer-dimensional space; this a more
    expensive less numerically stable transform but is needed for algorithms
    like HMC.

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method either as a
function on singleton constraints::

    transform_to.register(my_constraint, my_transform)

or as a decorator on parameterized constraints::

    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)

You can create your own registry by creating a new :class:`ConstraintRegistry`
object.
"""
import numbers
from torch.distributions import constraints, transforms
__all__ = ['ConstraintRegistry', 'biject_to', 'transform_to']

class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a :class:`~torch.distributions.constraints.Constraint`\n        subclass in this registry. Usage::\n\n            @my_registry.register(MyConstraintClass)\n            def construct_transform(constraint):\n                assert isinstance(constraint, MyConstraint)\n                return MyTransform(constraint.arg_constraints)\n\n        Args:\n            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):\n                A subclass of :class:`~torch.distributions.constraints.Constraint`, or\n                a singleton object of the desired class.\n            factory (Callable): A callable that inputs a constraint object and returns\n                a  :class:`~torch.distributions.transforms.Transform` object.\n        '
        if factory is None:
            return lambda factory: self.register(constraint, factory)
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)
        if not isinstance(constraint, type) or not issubclass(constraint, constraints.Constraint):
            raise TypeError(f'Expected constraint to be either a Constraint subclass or instance, but got {constraint}')
        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        if False:
            while True:
                i = 10
        "\n        Looks up a transform to constrained space, given a constraint object.\n        Usage::\n\n            constraint = Normal.arg_constraints['scale']\n            scale = transform_to(constraint)(torch.zeros(1))  # constrained\n            u = transform_to(constraint).inv(scale)           # unconstrained\n\n        Args:\n            constraint (:class:`~torch.distributions.constraints.Constraint`):\n                A constraint object.\n\n        Returns:\n            A :class:`~torch.distributions.transforms.Transform` object.\n\n        Raises:\n            `NotImplementedError` if no transform has been registered.\n        "
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(f'Cannot transform {type(constraint).__name__} constraints') from None
        return factory(constraint)
biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()

@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    if False:
        print('Hello World!')
    return transforms.identity_transform

@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    if False:
        for i in range(10):
            print('nop')
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(base_transform, constraint.reinterpreted_batch_ndims)

@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    if False:
        for i in range(10):
            print('nop')
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(base_transform, constraint.reinterpreted_batch_ndims)

@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    if False:
        for i in range(10):
            print('nop')
    return transforms.ExpTransform()

@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    if False:
        return 10
    return transforms.ComposeTransform([transforms.ExpTransform(), transforms.AffineTransform(constraint.lower_bound, 1)])

@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    if False:
        while True:
            i = 10
    return transforms.ComposeTransform([transforms.ExpTransform(), transforms.AffineTransform(constraint.upper_bound, -1)])

@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint):
    if False:
        for i in range(10):
            print('nop')
    lower_is_0 = isinstance(constraint.lower_bound, numbers.Number) and constraint.lower_bound == 0
    upper_is_1 = isinstance(constraint.upper_bound, numbers.Number) and constraint.upper_bound == 1
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()
    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform([transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)])

@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint):
    if False:
        i = 10
        return i + 15
    return transforms.StickBreakingTransform()

@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    if False:
        for i in range(10):
            print('nop')
    return transforms.SoftmaxTransform()

@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    if False:
        i = 10
        return i + 15
    return transforms.LowerCholeskyTransform()

@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    if False:
        print('Hello World!')
    return transforms.PositiveDefiniteTransform()

@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    if False:
        return 10
    return transforms.CorrCholeskyTransform()

@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    if False:
        print('Hello World!')
    return transforms.CatTransform([biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths)

@transform_to.register(constraints.cat)
def _transform_to_cat(constraint):
    if False:
        i = 10
        return i + 15
    return transforms.CatTransform([transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths)

@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    if False:
        for i in range(10):
            print('nop')
    return transforms.StackTransform([biject_to(c) for c in constraint.cseq], constraint.dim)

@transform_to.register(constraints.stack)
def _transform_to_stack(constraint):
    if False:
        print('Hello World!')
    return transforms.StackTransform([transform_to(c) for c in constraint.cseq], constraint.dim)