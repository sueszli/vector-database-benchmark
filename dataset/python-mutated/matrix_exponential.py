import math
from functools import partial
import torch
import torch.nn as nn
from torch.distributions import Transform, constraints
from pyro.nn import DenseNN
from ..conditional import ConditionalTransformModule
from ..torch_transform import TransformModule
from ..util import copy_docs_from

@copy_docs_from(Transform)
class ConditionedMatrixExponential(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, weights=None, iterations=8, normalization='none', bound=None):
        if False:
            while True:
                i = 10
        super().__init__(cache_size=1)
        assert iterations > 0
        self.weights = weights
        self.iterations = iterations
        self.normalization = normalization
        self.bound = bound
        if normalization == 'weight' or normalization == 'spectral':
            raise NotImplementedError('Normalization is currently not implemented.')
        elif normalization != 'none':
            raise ValueError('Unknown normalization method: {}'.format(normalization))

    def _exp(self, x, M):
        if False:
            while True:
                i = 10
        '\n        Performs power series approximation to the vector product of x with the\n        matrix exponential of M.\n        '
        power_term = x.unsqueeze(-1)
        y = x.unsqueeze(-1)
        for idx in range(self.iterations):
            power_term = torch.matmul(M, power_term) / (idx + 1)
            y = y + power_term
        return y.squeeze(-1)

    def _trace(self, M):
        if False:
            print('Hello World!')
        '\n        Calculates the trace of a matrix and is able to do broadcasting over batch\n        dimensions, unlike `torch.trace`.\n\n        Broadcasting is necessary for the conditional version of the transform,\n        where `self.weights` may have batch dimensions corresponding the batch\n        dimensions of the context variable that was conditioned upon.\n        '
        return M.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _call(self, x):
        if False:
            i = 10
            return i + 15
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n        Invokes the bijection x => y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        M = self.weights() if callable(self.weights) else self.weights
        return self._exp(x, M)

    def _inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n        Inverts y => x.\n        '
        M = self.weights() if callable(self.weights) else self.weights
        return self._exp(y, -M)

    def log_abs_det_jacobian(self, x, y):
        if False:
            return 10
        '\n        Calculates the element-wise determinant of the log Jacobian\n        '
        M = self.weights() if callable(self.weights) else self.weights
        return self._trace(M)

@copy_docs_from(ConditionedMatrixExponential)
class MatrixExponential(ConditionedMatrixExponential, TransformModule):
    """
    A dense matrix exponential bijective transform (Hoogeboom et al., 2020) with
    equation,

        :math:`\\mathbf{y} = \\exp(M)\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`\\exp(\\cdot)` represents the matrix exponential, and the learnable
    parameters are :math:`M\\in\\mathbb{R}^D\\times\\mathbb{R}^D` for input dimension
    :math:`D`. In general, :math:`M` is not required to be invertible.

    Due to the favourable mathematical properties of the matrix exponential, the
    transform has an exact inverse and a log-determinate-Jacobian that scales in
    time-complexity as :math:`O(D)`. Both the forward and reverse operations are
    approximated with a truncated power series. For numerical stability, the
    norm of :math:`M` can be restricted with the `normalization` keyword argument.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = MatrixExponential(10)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int
    :param iterations: the number of terms to use in the truncated power series that
        approximates matrix exponentiation.
    :type iterations: int
    :param normalization: One of `['none', 'weight', 'spectral']` normalization that
        selects what type of normalization to apply to the weight matrix. `weight`
        corresponds to weight normalization (Salimans and Kingma, 2016) and
        `spectral` to spectral normalization (Miyato et al, 2018).
    :type normalization: string
    :param bound: a bound on either the weight or spectral norm, when either of
        those two types of regularization are chosen by the `normalization`
        argument. A lower value for this results in fewer required terms of the
        truncated power series to closely approximate the exact value of the matrix
        exponential.
    :type bound: float

    References:

    [1] Emiel Hoogeboom, Victor Garcia Satorras, Jakub M. Tomczak, Max Welling. The
        Convolution Exponential and Generalized Sylvester Flows. [arXiv:2006.01910]
    [2] Tim Salimans, Diederik P. Kingma. Weight Normalization: A Simple
        Reparameterization to Accelerate Training of Deep Neural Networks.
        [arXiv:1602.07868]
    [3] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. Spectral
        Normalization for Generative Adversarial Networks. ICLR 2018.

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, iterations=8, normalization='none', bound=None):
        if False:
            return 10
        super().__init__(iterations=iterations, normalization=normalization, bound=bound)
        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if False:
            while True:
                i = 10
        stdv = 1.0 / math.sqrt(self.weights.size(0))
        self.weights.data.uniform_(-stdv, stdv)

@copy_docs_from(ConditionalTransformModule)
class ConditionalMatrixExponential(ConditionalTransformModule):
    """
    A dense matrix exponential bijective transform (Hoogeboom et al., 2020) that
    conditions on an additional context variable with equation,

        :math:`\\mathbf{y} = \\exp(M)\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`\\exp(\\cdot)` represents the matrix exponential, and
    :math:`M\\in\\mathbb{R}^D\\times\\mathbb{R}^D` is the output of a neural network
    conditioning on a context variable :math:`\\mathbf{z}` for input dimension
    :math:`D`. In general, :math:`M` is not required to be invertible.

    Due to the favourable mathematical properties of the matrix exponential, the
    transform has an exact inverse and a log-determinate-Jacobian that scales in
    time-complexity as :math:`O(D)`. Both the forward and reverse operations are
    approximated with a truncated power series. For numerical stability, the
    norm of :math:`M` can be restricted with the `normalization` keyword argument.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim*input_dim]
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims)
    >>> transform = ConditionalMatrixExponential(input_dim, hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int
    :param iterations: the number of terms to use in the truncated power series that
        approximates matrix exponentiation.
    :type iterations: int
    :param normalization: One of `['none', 'weight', 'spectral']` normalization that
        selects what type of normalization to apply to the weight matrix. `weight`
        corresponds to weight normalization (Salimans and Kingma, 2016) and
        `spectral` to spectral normalization (Miyato et al, 2018).
    :type normalization: string
    :param bound: a bound on either the weight or spectral norm, when either of
        those two types of regularization are chosen by the `normalization`
        argument. A lower value for this results in fewer required terms of the
        truncated power series to closely approximate the exact value of the matrix
        exponential.
    :type bound: float

    References:

    [1] Emiel Hoogeboom, Victor Garcia Satorras, Jakub M. Tomczak, Max Welling. The
        Convolution Exponential and Generalized Sylvester Flows. [arXiv:2006.01910]
    [2] Tim Salimans, Diederik P. Kingma. Weight Normalization: A Simple
        Reparameterization to Accelerate Training of Deep Neural Networks.
        [arXiv:1602.07868]
    [3] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. Spectral
        Normalization for Generative Adversarial Networks. ICLR 2018.

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, nn, iterations=8, normalization='none', bound=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.input_dim = input_dim
        self.nn = nn
        self.iterations = iterations
        self.normalization = normalization
        self.bound = bound

    def _params(self, context):
        if False:
            for i in range(10):
                print('nop')
        return self.nn(context)

    def condition(self, context):
        if False:
            i = 10
            return i + 15
        cond_nn = partial(self.nn, context)

        def weights():
            if False:
                print('Hello World!')
            w = cond_nn()
            return w.view(w.shape[:-1] + (self.input_dim, self.input_dim))
        return ConditionedMatrixExponential(weights, iterations=self.iterations, normalization=self.normalization, bound=self.bound)

def matrix_exponential(input_dim, iterations=8, normalization='none', bound=None):
    if False:
        return 10
    "\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.MatrixExponential` object for consistency\n    with other helpers.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param iterations: the number of terms to use in the truncated power series that\n        approximates matrix exponentiation.\n    :type iterations: int\n    :param normalization: One of `['none', 'weight', 'spectral']` normalization that\n        selects what type of normalization to apply to the weight matrix. `weight`\n        corresponds to weight normalization (Salimans and Kingma, 2016) and\n        `spectral` to spectral normalization (Miyato et al, 2018).\n    :type normalization: string\n    :param bound: a bound on either the weight or spectral norm, when either of\n        those two types of regularization are chosen by the `normalization`\n        argument. A lower value for this results in fewer required terms of the\n        truncated power series to closely approximate the exact value of the matrix\n        exponential.\n    :type bound: float\n\n    "
    return MatrixExponential(input_dim, iterations=iterations, normalization=normalization, bound=bound)

def conditional_matrix_exponential(input_dim, context_dim, hidden_dims=None, iterations=8, normalization='none', bound=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.ConditionalMatrixExponential` object for\n    consistency with other helpers.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param context_dim: Dimension of context variable\n    :type context_dim: int\n    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults\n        to using [input_dim * 10, input_dim * 10]\n    :type hidden_dims: list[int]\n    :param iterations: the number of terms to use in the truncated power series that\n        approximates matrix exponentiation.\n    :type iterations: int\n    :param normalization: One of `['none', 'weight', 'spectral']` normalization that\n        selects what type of normalization to apply to the weight matrix. `weight`\n        corresponds to weight normalization (Salimans and Kingma, 2016) and\n        `spectral` to spectral normalization (Miyato et al, 2018).\n    :type normalization: string\n    :param bound: a bound on either the weight or spectral norm, when either of\n        those two types of regularization are chosen by the `normalization`\n        argument. A lower value for this results in fewer required terms of the\n        truncated power series to closely approximate the exact value of the matrix\n        exponential.\n    :type bound: float\n\n    "
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[input_dim * input_dim])
    return ConditionalMatrixExponential(input_dim, nn, iterations=iterations, normalization=normalization, bound=bound)