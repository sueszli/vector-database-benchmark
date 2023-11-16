from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import SigmoidTransform, TanhTransform
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from .. import constraints
from ..conditional import ConditionalTransformModule
from ..torch_transform import TransformModule
from ..util import copy_docs_from
from .basic import ELUTransform, LeakyReLUTransform
eps = 1e-08

@copy_docs_from(TransformModule)
class NeuralAutoregressive(TransformModule):
    """
    An implementation of the deep Neural Autoregressive Flow (NAF) bijective
    transform of the "IAF flavour" that can be used for sampling and scoring samples
    drawn from it (but not arbitrary ones).

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> arn = AutoRegressiveNN(10, [40], param_dims=[16]*3)
    >>> transform = NeuralAutoregressive(arn, hidden_units=16)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse operation is not implemented. This would require numerical
    inversion, e.g., using a root finding method - a possibility for a future
    implementation.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a tuple of three real-valued tensors, whose last dimension is the
        input dimension, and whose penultimate dimension is equal to hidden_units.
    :type autoregressive_nn: nn.Module
    :param hidden_units: the number of hidden units to use in the NAF transformation
        (see Eq (8) in reference)
    :type hidden_units: int
    :param activation: Activation function to use. One of 'ELU', 'LeakyReLU',
        'sigmoid', or 'tanh'.
    :type activation: string

    Reference:

    [1] Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville. Neural
    Autoregressive Flows. [arXiv:1804.00779]


    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    eps = 1e-08
    autoregressive = True

    def __init__(self, autoregressive_nn, hidden_units=16, activation='sigmoid'):
        if False:
            return 10
        super().__init__(cache_size=1)
        name_to_mixin = {'ELU': ELUTransform, 'LeakyReLU': LeakyReLUTransform, 'sigmoid': SigmoidTransform, 'tanh': TanhTransform}
        if activation not in name_to_mixin:
            raise ValueError('Invalid activation function "{}"'.format(activation))
        self.T = name_to_mixin[activation]()
        self.arn = autoregressive_nn
        self.hidden_units = hidden_units
        self.logsoftmax = nn.LogSoftmax(dim=-2)
        self._cached_log_df_inv_dx = None
        self._cached_A = None
        self._cached_W_pre = None
        self._cached_C = None
        self._cached_T_C = None

    def _call(self, x):
        if False:
            return 10
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        (A, W_pre, b) = self.arn(x)
        T = self.T
        A = F.softplus(A)
        C = A * x.unsqueeze(-2) + b
        W = F.softmax(W_pre, dim=-2)
        T_C = T(C)
        D = (W * T_C).sum(dim=-2)
        y = T.inv(D)
        self._cached_log_df_inv_dx = T.inv.log_abs_det_jacobian(D, y)
        self._cached_A = A
        self._cached_W_pre = W_pre
        self._cached_C = C
        self._cached_T_C = T_C
        return y

    def log_abs_det_jacobian(self, x, y):
        if False:
            return 10
        '\n        Calculates the elementwise determinant of the log Jacobian\n        '
        A = self._cached_A
        W_pre = self._cached_W_pre
        C = self._cached_C
        T_C = self._cached_T_C
        T = self.T
        log_dydD = self._cached_log_df_inv_dx
        log_dDdx = torch.logsumexp(torch.log(A + self.eps) + self.logsoftmax(W_pre) + T.log_abs_det_jacobian(C, T_C), dim=-2)
        log_det = log_dydD + log_dDdx
        return log_det.sum(-1)

@copy_docs_from(ConditionalTransformModule)
class ConditionalNeuralAutoregressive(ConditionalTransformModule):
    """
    An implementation of the deep Neural Autoregressive Flow (NAF) bijective
    transform of the "IAF flavour" conditioning on an additiona context variable
    that can be used for sampling and scoring samples drawn from it (but not
    arbitrary ones).

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> arn = ConditionalAutoRegressiveNN(input_dim, context_dim, [40],
    ... param_dims=[16]*3)
    >>> transform = ConditionalNeuralAutoregressive(arn, hidden_units=16)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    The inverse operation is not implemented. This would require numerical
    inversion, e.g., using a root finding method - a possibility for a future
    implementation.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a tuple of three real-valued tensors, whose last dimension is the
        input dimension, and whose penultimate dimension is equal to hidden_units.
    :type autoregressive_nn: nn.Module
    :param hidden_units: the number of hidden units to use in the NAF transformation
        (see Eq (8) in reference)
    :type hidden_units: int
    :param activation: Activation function to use. One of 'ELU', 'LeakyReLU',
        'sigmoid', or 'tanh'.
    :type activation: string

    Reference:

    [1] Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville. Neural
    Autoregressive Flows. [arXiv:1804.00779]


    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, autoregressive_nn, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        if False:
            print('Hello World!')
        '\n        Conditions on a context variable, returning a non-conditional transform of\n        of type :class:`~pyro.distributions.transforms.NeuralAutoregressive`.\n        '
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return NeuralAutoregressive(cond_nn, **self.kwargs)

def neural_autoregressive(input_dim, hidden_dims=None, activation='sigmoid', width=16):
    if False:
        while True:
            i = 10
    '\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.NeuralAutoregressive` object that takes\n    care of constructing an autoregressive network with the correct input/output\n    dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param hidden_dims: The desired hidden dimensions of the autoregressive network.\n        Defaults to using [3*input_dim + 1]\n    :type hidden_dims: list[int]\n    :param activation: Activation function to use. One of \'ELU\', \'LeakyReLU\',\n        \'sigmoid\', or \'tanh\'.\n    :type activation: string\n    :param width: The width of the "multilayer perceptron" in the transform (see\n        paper). Defaults to 16\n    :type width: int\n\n    '
    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=[width] * 3)
    return NeuralAutoregressive(arn, hidden_units=width, activation=activation)

def conditional_neural_autoregressive(input_dim, context_dim, hidden_dims=None, activation='sigmoid', width=16):
    if False:
        print('Hello World!')
    '\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.ConditionalNeuralAutoregressive` object\n    that takes care of constructing an autoregressive network with the correct\n    input/output dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param context_dim: Dimension of context variable\n    :type context_dim: int\n    :param hidden_dims: The desired hidden dimensions of the autoregressive network.\n        Defaults to using [3*input_dim + 1]\n    :type hidden_dims: list[int]\n    :param activation: Activation function to use. One of \'ELU\', \'LeakyReLU\',\n        \'sigmoid\', or \'tanh\'.\n    :type activation: string\n    :param width: The width of the "multilayer perceptron" in the transform (see\n        paper). Defaults to 16\n    :type width: int\n\n    '
    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims, param_dims=[width] * 3)
    return ConditionalNeuralAutoregressive(arn, hidden_units=width, activation=activation)