from functools import partial
import torch
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from .. import constraints
from ..conditional import ConditionalTransformModule
from ..torch_transform import TransformModule
from ..util import copy_docs_from
from .spline import ConditionalSpline

@copy_docs_from(TransformModule)
class SplineAutoregressive(TransformModule):
    """
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020).
    Rational splines are functions that are comprised of segments that are the ratio
    of two polynomials (see :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\\theta_d}(x_d)\\ \\ \\ d=1,2,\\ldots,D`

    where :math:`\\mathbf{x}=(x_1,x_2,\\ldots,x_D)` are the inputs,
    :math:`\\mathbf{y}=(y_1,y_2,\\ldots,y_D)` are the outputs, :math:`g_{\\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\\theta_d`, and
    :math:`\\theta=(\\theta_1,\\theta_2,\\ldots,\\theta_D)` is the output of an
    autoregressive NN inputting :math:`\\mathbf{x}`.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    >>> transform = SplineAutoregressive(input_dim, hypernet, count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    autoregressive = True

    def __init__(self, input_dim, autoregressive_nn, count_bins=8, bound=3.0, order='linear'):
        if False:
            i = 10
            return i + 15
        super(SplineAutoregressive, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.spline = ConditionalSpline(autoregressive_nn, input_dim, count_bins, bound, order)

    def _call(self, x):
        if False:
            print('Hello World!')
        '\n        :param x: the input into the bijection\n        :type x: torch.Tensor\n\n        Invokes the bijection x=>y; in the prototypical context of a\n        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from\n        the base distribution (or the output of a previous transform)\n        '
        spline = self.spline.condition(x)
        y = spline(x)
        self._cache_log_detJ = spline._cache_log_detJ
        return y

    def _inverse(self, y):
        if False:
            print('Hello World!')
        '\n        :param y: the output of the bijection\n        :type y: torch.Tensor\n\n        Inverts y => x. Uses a previously cached inverse if available, otherwise\n        performs the inversion afresh.\n        '
        input_dim = y.size(-1)
        x = torch.zeros_like(y)
        for _ in range(input_dim):
            spline = self.spline.condition(x)
            x = spline._inverse(y)
        self._cache_log_detJ = spline._cache_log_detJ
        return x

    def log_abs_det_jacobian(self, x, y):
        if False:
            return 10
        '\n        Calculates the elementwise determinant of the log Jacobian\n        '
        (x_old, y_old) = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)
        return self._cache_log_detJ.sum(-1)

@copy_docs_from(ConditionalTransformModule)
class ConditionalSplineAutoregressive(ConditionalTransformModule):
    """
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020) that
    conditions on an additional context variable. Rational splines are functions
    that are comprised of segments that are the ratio of two polynomials (see
    :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\\theta_d}(x_d)\\ \\ \\ d=1,2,\\ldots,D`

    where :math:`\\mathbf{x}=(x_1,x_2,\\ldots,x_D)` are the inputs,
    :math:`\\mathbf{y}=(y_1,y_2,\\ldots,y_D)` are the outputs, :math:`g_{\\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\\theta_d`, and
    :math:`\\theta=(\\theta_1,\\theta_2,\\ldots,\\theta_D)` is the output of a
    conditional autoregressive NN inputting :math:`\\mathbf{x}` and conditioning on
    the context variable :math:`\\mathbf{z}`.

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims,
    ... param_dims=param_dims)
    >>> transform = ConditionalSplineAutoregressive(input_dim, hypernet,
    ... count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, autoregressive_nn, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.input_dim = input_dim
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        if False:
            for i in range(10):
                print('nop')
        '\n        Conditions on a context variable, returning a non-conditional transform of\n        of type :class:`~pyro.distributions.transforms.SplineAutoregressive`.\n        '
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return SplineAutoregressive(self.input_dim, cond_nn, **self.kwargs)

def spline_autoregressive(input_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    if False:
        i = 10
        return i + 15
    "\n    A helper function to create an\n    :class:`~pyro.distributions.transforms.SplineAutoregressive` object that takes\n    care of constructing an autoregressive network with the correct input/output\n    dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param hidden_dims: The desired hidden dimensions of the autoregressive network.\n        Defaults to using [3*input_dim + 1]\n    :type hidden_dims: list[int]\n    :param count_bins: The number of segments comprising the spline.\n    :type count_bins: int\n    :param bound: The quantity :math:`K` determining the bounding box,\n        :math:`[-K,K]\\times[-K,K]`, of the spline.\n    :type bound: float\n    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.\n    :type order: string\n\n    "
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    return SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)

def conditional_spline_autoregressive(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    if False:
        print('Hello World!')
    "\n    A helper function to create a\n    :class:`~pyro.distributions.transforms.ConditionalSplineAutoregressive` object\n    that takes care of constructing an autoregressive network with the correct\n    input/output dimensions.\n\n    :param input_dim: Dimension of input variable\n    :type input_dim: int\n    :param context_dim: Dimension of context variable\n    :type context_dim: int\n    :param hidden_dims: The desired hidden dimensions of the autoregressive network.\n        Defaults to using [input_dim * 10, input_dim * 10]\n    :type hidden_dims: list[int]\n    :param count_bins: The number of segments comprising the spline.\n    :type count_bins: int\n    :param bound: The quantity :math:`K` determining the bounding box,\n        :math:`[-K,K]\\times[-K,K]`, of the spline.\n    :type bound: float\n    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.\n    :type order: string\n\n    "
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims, param_dims=param_dims)
    return ConditionalSplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)