from functools import partial
import torch
import torch.nn as nn
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from .. import constraints
from ..conditional import ConditionalTransformModule
from ..torch_transform import TransformModule
from ..util import copy_docs_from
from .utils import clamp_preserve_gradients

@copy_docs_from(TransformModule)
class AffineAutoregressive(TransformModule):
    """
    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF), using by default Eq (10) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`\\mu_t,\\sigma_t` are calculated from an autoregressive network on
    :math:`\\mathbf{x}`, and :math:`\\sigma_t>0`.

    If the stable keyword argument is set to True then the transformation used is,

        :math:`\\mathbf{y} = \\sigma_t\\odot\\mathbf{x} + (1-\\sigma_t)\\odot\\mu_t`

    where :math:`\\sigma_t` is restricted to :math:`(0,1)`. This variant of IAF is
    claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be
    represented, presumably since the input is restricted to rescaling by a number
    on :math:`(0,1)`.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitrary value is being scored, it will calculate it manually. Note
    that this is an operation that scales as O(D) where D is the input dimension,
    and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to
    score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a real-valued mean and logit-scale as a tuple
    :type autoregressive_nn: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    References:

    [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
    Max Welling. Improving Variational Inference with Inverse Autoregressive Flow.
    [arXiv:1606.04934]

    [2] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    [3] Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. MADE: Masked
    Autoencoder for Distribution Estimation. [arXiv:1502.03509]

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1
    autoregressive = True

    def __init__(self, autoregressive_nn, log_scale_min_clip=-5.0, log_scale_max_clip=3.0, sigmoid_bias=2.0, stable=False):
        if False:
            while True:
                i = 10
        super().__init__(cache_size=1)
        self.arn = autoregressive_nn
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid_bias = sigmoid_bias
        self.stable = stable
        if stable:
            self._call = self._call_stable
            self._inverse = self._inverse_stable

    def _call(self, x):
        if False:
            while True:
                i = 10
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        (mean, log_scale) = self.arn(x)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        scale = torch.exp(log_scale)
        y = scale * x + mean
        return y

    def _inverse(self, y):
        if False:
            i = 10
            return i + 15
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n\n        Inverts y => x. Uses a previously cached inverse if available, otherwise\n        performs the inversion afresh.\n        '
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim
        for idx in perm:
            (mean, log_scale) = self.arn(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(-clamp_preserve_gradients(log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale
        x = torch.stack(x, dim=-1)
        log_scale = clamp_preserve_gradients(log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        return x

    def log_abs_det_jacobian(self, x, y):
        if False:
            while True:
                i = 10
        '\n        Calculates the elementwise determinant of the log Jacobian\n        '
        (x_old, y_old) = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        elif not self.stable:
            (_, log_scale) = self.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        else:
            (_, logit_scale) = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)

    def _call_stable(self, x):
        if False:
            return 10
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        (mean, logit_scale) = self.arn(x)
        logit_scale = logit_scale + self.sigmoid_bias
        scale = self.sigmoid(logit_scale)
        log_scale = self.logsigmoid(logit_scale)
        self._cached_log_scale = log_scale
        y = scale * x + (1 - scale) * mean
        return y

    def _inverse_stable(self, y):
        if False:
            return 10
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n\n        Inverts y => x.\n        '
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim
        for idx in perm:
            (mean, logit_scale) = self.arn(torch.stack(x, dim=-1))
            inverse_scale = 1 + torch.exp(-logit_scale[..., idx] - self.sigmoid_bias)
            x[idx] = inverse_scale * y[..., idx] + (1 - inverse_scale) * mean[..., idx]
        self._cached_log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        x = torch.stack(x, dim=-1)
        return x

@copy_docs_from(ConditionalTransformModule)
class ConditionalAffineAutoregressive(ConditionalTransformModule):
    """
    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF) that conditions on an additional context variable and uses, by default,
    Eq (10) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`\\mu_t,\\sigma_t` are calculated from an autoregressive network on
    :math:`\\mathbf{x}` and context :math:`\\mathbf{z}\\in\\mathbb{R}^M`, and
    :math:`\\sigma_t>0`.

    If the stable keyword argument is set to True then the transformation used is,

        :math:`\\mathbf{y} = \\sigma_t\\odot\\mathbf{x} + (1-\\sigma_t)\\odot\\mu_t`

    where :math:`\\sigma_t` is restricted to :math:`(0,1)`. This variant of IAF is
    claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be
    represented, presumably since the input is restricted to rescaling by a number
    on :math:`(0,1)`.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> context_dim = 4
    >>> batch_size = 3
    >>> hidden_dims = [10*input_dim, 10*input_dim]
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hypernet = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims)
    >>> transform = ConditionalAffineAutoregressive(hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitrary value is being scored, it will calculate it manually. Note
    that this is an operation that scales as O(D) where D is the input dimension,
    and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to
    score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a real-valued mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    References:

    [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
    Max Welling. Improving Variational Inference with Inverse Autoregressive Flow.
    [arXiv:1606.04934]

    [2] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    [3] Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. MADE: Masked
    Autoencoder for Distribution Estimation. [arXiv:1502.03509]

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, autoregressive_nn, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        if False:
            while True:
                i = 10
        '\n        Conditions on a context variable, returning a non-conditional transform of\n        of type :class:`~pyro.distributions.transforms.AffineAutoregressive`.\n        '
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return AffineAutoregressive(cond_nn, **self.kwargs)

def affine_autoregressive(input_dim, hidden_dims=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    A helper function to create an\n    :class:`~pyro.distributions.transforms.AffineAutoregressive` object that takes\n    care of constructing an autoregressive network with the correct input/output\n    dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param hidden_dims: The desired hidden dimensions of the autoregressive network.\n        Defaults to using [3*input_dim + 1]\n    :type hidden_dims: list[int]\n    :param log_scale_min_clip: The minimum value for clipping the log(scale) from\n        the autoregressive NN\n    :type log_scale_min_clip: float\n    :param log_scale_max_clip: The maximum value for clipping the log(scale) from\n        the autoregressive NN\n    :type log_scale_max_clip: float\n    :param sigmoid_bias: A term to add the logit of the input when using the stable\n        tranform.\n    :type sigmoid_bias: float\n    :param stable: When true, uses the alternative "stable" version of the transform\n        (see above).\n    :type stable: bool\n\n    '
    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims)
    return AffineAutoregressive(arn, **kwargs)

def conditional_affine_autoregressive(input_dim, context_dim, hidden_dims=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    A helper function to create an\n    :class:`~pyro.distributions.transforms.ConditionalAffineAutoregressive` object\n    that takes care of constructing a dense network with the correct input/output\n    dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param context_dim: Dimension of context variable\n    :type context_dim: int\n    :param hidden_dims: The desired hidden dimensions of the dense network. Defaults\n        to using [10*input_dim]\n    :type hidden_dims: list[int]\n    :param log_scale_min_clip: The minimum value for clipping the log(scale) from\n        the autoregressive NN\n    :type log_scale_min_clip: float\n    :param log_scale_max_clip: The maximum value for clipping the log(scale) from\n        the autoregressive NN\n    :type log_scale_max_clip: float\n    :param sigmoid_bias: A term to add the logit of the input when using the stable\n        tranform.\n    :type sigmoid_bias: float\n    :param stable: When true, uses the alternative "stable" version of the transform\n        (see above).\n    :type stable: bool\n\n    '
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]
    nn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims)
    return ConditionalAffineAutoregressive(nn, **kwargs)