import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Transform, constraints
from pyro.nn import DenseNN
from ..conditional import ConditionalTransformModule
from ..torch_transform import TransformModule
from ..util import copy_docs_from

@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, params):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    def u_hat(self, u, w):
        if False:
            return 10
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        if False:
            while True:
                i = 10
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output\n        of a previous transform)\n        '
        (x0, alpha_prime, beta_prime) = self._params() if callable(self._params) else self._params
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)
        diff = x - x0
        r = diff.norm(dim=-1, keepdim=True)
        h = (alpha + r).reciprocal()
        h_prime = -h ** 2
        beta_h = beta * h
        self._cached_logDetJ = ((x0.size(-1) - 1) * torch.log1p(beta_h) + torch.log1p(beta_h + beta * h_prime * r)).sum(-1)
        return x + beta_h * diff

    def _inverse(self, y):
        if False:
            return 10
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n        Inverts y => x. As noted above, this implementation is incapable of\n        inverting arbitrary values `y`; rather it assumes `y` is the result of a\n        previously computed application of the bijector to some `x` (which was\n        cached on the forward call)\n        '
        raise KeyError("ConditionedRadial object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        if False:
            return 10
        '\n        Calculates the elementwise determinant of the log Jacobian\n        '
        (x_old, y_old) = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)
        return self._cached_logDetJ

@copy_docs_from(ConditionedRadial)
class Radial(ConditionedRadial, TransformModule):
    """
    A 'radial' bijective transform using the equation,

        :math:`\\mathbf{y} = \\mathbf{x} + \\beta h(\\alpha,r)(\\mathbf{x} - \\mathbf{x}_0)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    and the learnable parameters are :math:`\\alpha\\in\\mathbb{R}^+`,
    :math:`\\beta\\in\\mathbb{R}`, :math:`\\mathbf{x}_0\\in\\mathbb{R}^D`, for input
    dimension :math:`D`, :math:`r=||\\mathbf{x}-\\mathbf{x}_0||_2`,
    :math:`h(\\alpha,r)=1/(\\alpha+r)`. For this to be an invertible transformation,
    the condition :math:`\\beta>-\\alpha` is enforced.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Radial(10)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the radial transform can be
    scored.

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    [1] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim):
        if False:
            print('Hello World!')
        super().__init__(self._params)
        self.x0 = nn.Parameter(torch.Tensor(input_dim))
        self.alpha_prime = nn.Parameter(torch.Tensor(1))
        self.beta_prime = nn.Parameter(torch.Tensor(1))
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        if False:
            return 10
        return (self.x0, self.alpha_prime, self.beta_prime)

    def reset_parameters(self):
        if False:
            print('Hello World!')
        stdv = 1.0 / math.sqrt(self.x0.size(0))
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

@copy_docs_from(ConditionalTransformModule)
class ConditionalRadial(ConditionalTransformModule):
    """
    A conditional 'radial' bijective transform context using the equation,

        :math:`\\mathbf{y} = \\mathbf{x} + \\beta h(\\alpha,r)(\\mathbf{x} - \\mathbf{x}_0)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    and :math:`\\alpha\\in\\mathbb{R}^+`, :math:`\\beta\\in\\mathbb{R}`,
    and :math:`\\mathbf{x}_0\\in\\mathbb{R}^D`, are the output of a function, e.g. a NN,
    with input :math:`z\\in\\mathbb{R}^{M}` representing the context variable to
    condition on. The input dimension is :math:`D`,
    :math:`r=||\\mathbf{x}-\\mathbf{x}_0||_2`, and :math:`h(\\alpha,r)=1/(\\alpha+r)`.
    For this to be an invertible transformation, the condition :math:`\\beta>-\\alpha`
    is enforced.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim, 1, 1]
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims)
    >>> transform = ConditionalRadial(hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is
    left unimplemented. However, the inverse is cached when the forward operation is
    called during sampling, and so samples drawn using the radial transform can be
    scored.

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    [1] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, nn):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.nn = nn

    def _params(self, context):
        if False:
            while True:
                i = 10
        return self.nn(context)

    def condition(self, context):
        if False:
            print('Hello World!')
        params = partial(self._params, context)
        return ConditionedRadial(params)

def radial(input_dim):
    if False:
        return 10
    '\n    A helper function to create a :class:`~pyro.distributions.transforms.Radial`\n    object for consistency with other helpers.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n\n    '
    return Radial(input_dim)

def conditional_radial(input_dim, context_dim, hidden_dims=None):
    if False:
        while True:
            i = 10
    '\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.ConditionalRadial` object that takes care\n    of constructing a dense network with the correct input/output dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param context_dim: Dimension of context variable\n    :type context_dim: int\n    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults\n        to using [input_dim * 10, input_dim * 10]\n    :type hidden_dims: list[int]\n\n\n    '
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[input_dim, 1, 1])
    return ConditionalRadial(nn)