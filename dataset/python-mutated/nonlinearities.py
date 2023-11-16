"""
Non-linear activation functions for artificial neurons.
"""
import theano.tensor

def sigmoid(x):
    if False:
        return 10
    'Sigmoid activation function :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32 in [0, 1]\n        The output of the sigmoid function applied to the activation.\n    '
    return theano.tensor.nnet.sigmoid(x)

def softmax(x):
    if False:
        for i in range(10):
            print('nop')
    'Softmax activation function\n    :math:`\\varphi(\\mathbf{x})_j =\n    \\frac{e^{\\mathbf{x}_j}}{\\sum_{k=1}^K e^{\\mathbf{x}_k}}`\n    where :math:`K` is the total number of neurons in the layer. This\n    activation function gets applied row-wise.\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32 where the sum of the row is 1 and each single value is in [0, 1]\n        The output of the softmax function applied to the activation.\n    '
    return theano.tensor.nnet.softmax(x)

def tanh(x):
    if False:
        print('Hello World!')
    'Tanh activation function :math:`\\varphi(x) = \\tanh(x)`\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32 in [-1, 1]\n        The output of the tanh function applied to the activation.\n    '
    return theano.tensor.tanh(x)

class ScaledTanH(object):
    """Scaled tanh :math:`\\varphi(x) = \\tanh(\\alpha \\cdot x) \\cdot \\beta`

    This is a modified tanh function which allows to rescale both the input and
    the output of the activation.

    Scaling the input down will result in decreasing the maximum slope of the
    tanh and as a result it will be in the linear regime in a larger interval
    of the input space. Scaling the input up will increase the maximum slope
    of the tanh and thus bring it closer to a step function.

    Scaling the output changes the output interval to :math:`[-\\beta,\\beta]`.

    Parameters
    ----------
    scale_in : float32
        The scale parameter :math:`\\alpha` for the input

    scale_out : float32
        The scale parameter :math:`\\beta` for the output

    Methods
    -------
    __call__(x)
        Apply the scaled tanh function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import ScaledTanH
    >>> scaled_tanh = ScaledTanH(scale_in=0.5, scale_out=2.27)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=scaled_tanh)

    Notes
    -----
    LeCun et al. (in [1]_, Section 4.4) suggest ``scale_in=2./3`` and
    ``scale_out=1.7159``, which has :math:`\\varphi(\\pm 1) = \\pm 1`,
    maximum second derivative at 1, and an effective gain close to 1.

    By carefully matching :math:`\\alpha` and :math:`\\beta`, the nonlinearity
    can also be tuned to preserve the mean and variance of its input:

      * ``scale_in=0.5``, ``scale_out=2.4``: If the input is a random normal
        variable, the output will have zero mean and unit variance.
      * ``scale_in=1``, ``scale_out=1.6``: Same property, but with a smaller
        linear regime in input space.
      * ``scale_in=0.5``, ``scale_out=2.27``: If the input is a uniform normal
        variable, the output will have zero mean and unit variance.
      * ``scale_in=1``, ``scale_out=1.48``: Same property, but with a smaller
        linear regime in input space.

    References
    ----------
    .. [1] LeCun, Yann A., et al. (1998):
       Efficient BackProp,
       http://link.springer.com/chapter/10.1007/3-540-49430-8_2,
       http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    .. [2] Masci, Jonathan, et al. (2011):
       Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction,
       http://link.springer.com/chapter/10.1007/978-3-642-21735-7_7,
       http://people.idsia.ch/~ciresan/data/icann2011.pdf
    """

    def __init__(self, scale_in=1, scale_out=1):
        if False:
            print('Hello World!')
        self.scale_in = scale_in
        self.scale_out = scale_out

    def __call__(self, x):
        if False:
            return 10
        return theano.tensor.tanh(x * self.scale_in) * self.scale_out
ScaledTanh = ScaledTanH

def rectify(x):
    if False:
        return 10
    'Rectify activation function :math:`\\varphi(x) = \\max(0, x)`\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32\n        The output of the rectify function applied to the activation.\n    '
    return theano.tensor.nnet.relu(x)

class LeakyRectify(object):
    """Leaky rectifier :math:`\\varphi(x) = (x > 0)? x : \\alpha \\cdot x`

    The leaky rectifier was introduced in [1]_. Compared to the standard
    rectifier :func:`rectify`, it has a nonzero gradient for negative input,
    which often helps convergence.

    Parameters
    ----------
    leakiness : float
        Slope for negative input, usually between 0 and 1.
        A leakiness of 0 will lead to the standard rectifier,
        a leakiness of 1 will lead to a linear activation function,
        and any value in between will give a leaky rectifier.

    Methods
    -------
    __call__(x)
        Apply the leaky rectify function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import LeakyRectify
    >>> custom_rectify = LeakyRectify(0.1)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=custom_rectify)

    Alternatively, you can use the provided instance for leakiness=0.01:

    >>> from lasagne.nonlinearities import leaky_rectify
    >>> l2 = DenseLayer(l_in, num_units=200, nonlinearity=leaky_rectify)

    Or the one for a high leakiness of 1/3:

    >>> from lasagne.nonlinearities import very_leaky_rectify
    >>> l3 = DenseLayer(l_in, num_units=200, nonlinearity=very_leaky_rectify)

    See Also
    --------
    leaky_rectify: Instance with default leakiness of 0.01, as in [1]_.
    very_leaky_rectify: Instance with high leakiness of 1/3, as in [2]_.

    References
    ----------
    .. [1] Maas et al. (2013):
       Rectifier Nonlinearities Improve Neural Network Acoustic Models,
       http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    .. [2] Graham, Benjamin (2014):
       Spatially-sparse convolutional neural networks,
       http://arxiv.org/abs/1409.6070
    """

    def __init__(self, leakiness=0.01):
        if False:
            return 10
        self.leakiness = leakiness

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        return theano.tensor.nnet.relu(x, self.leakiness)
leaky_rectify = LeakyRectify()
leaky_rectify.__doc__ = 'leaky_rectify(x)\n\n    Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=0.01`\n    '
very_leaky_rectify = LeakyRectify(1.0 / 3)
very_leaky_rectify.__doc__ = 'very_leaky_rectify(x)\n\n     Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=1/3`\n     '

def elu(x):
    if False:
        return 10
    'Exponential Linear Unit :math:`\\varphi(x) = (x > 0) ? x : e^x - 1`\n\n    The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the\n    linear rectifier :func:`rectify`, it has a mean activation closer to zero\n    and nonzero gradient for negative input, which can help convergence.\n    Compared to the leaky rectifier :class:`LeakyRectify`, it saturates for\n    highly negative inputs.\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighed input of a neuron).\n\n    Returns\n    -------\n    float32\n        The output of the exponential linear unit for the activation.\n\n    Notes\n    -----\n    In [1]_, an additional parameter :math:`\\alpha` controls the (negative)\n    saturation value for negative inputs, but is set to 1 for all experiments.\n    It is omitted here.\n\n    References\n    ----------\n    .. [1] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):\n       Fast and Accurate Deep Network Learning by Exponential Linear Units\n       (ELUs), http://arxiv.org/abs/1511.07289\n    '
    return theano.tensor.switch(x > 0, x, theano.tensor.expm1(x))

class SELU(object):
    """
    Scaled Exponential Linear Unit
    :math:`\\varphi(x)=\\lambda \\left[(x>0) ? x : \\alpha(e^x-1)\\right]`

    The Scaled Exponential Linear Unit (SELU) was introduced in [1]_
    as an activation function that allows the construction of
    self-normalizing neural networks.

    Parameters
    ----------
    scale : float32
        The scale parameter :math:`\\lambda` for scaling all output.

    scale_neg  : float32
        The scale parameter :math:`\\alpha`
        for scaling output for nonpositive argument values.

    Methods
    -------
    __call__(x)
        Apply the SELU function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import SELU
    >>> selu = SELU(2, 3)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=selu)

    See Also
    --------
    selu: Instance with :math:`\\alpha\\approx1.6733,\\lambda\\approx1.0507`
          as used in [1]_.

    References
    ----------
    .. [1] Günter Klambauer et al. (2017):
       Self-Normalizing Neural Networks,
       https://arxiv.org/abs/1706.02515
    """

    def __init__(self, scale=1, scale_neg=1):
        if False:
            print('Hello World!')
        self.scale = scale
        self.scale_neg = scale_neg

    def __call__(self, x):
        if False:
            return 10
        return self.scale * theano.tensor.switch(x > 0.0, x, self.scale_neg * theano.tensor.expm1(x))
selu = SELU(scale=1.0507009873554805, scale_neg=1.6732632423543772)
selu.__doc__ = 'selu(x)\n\n    Instance of :class:`SELU` with :math:`\\alpha\\approx 1.6733,\n    \\lambda\\approx 1.0507`\n\n    This has a stable and attracting fixed point of :math:`\\mu=0`,\n    :math:`\\sigma=1` under the assumptions of the\n    original paper on self-normalizing neural networks.\n    '

def softplus(x):
    if False:
        print('Hello World!')
    'Softplus activation function :math:`\\varphi(x) = \\log(1 + e^x)`\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32\n        The output of the softplus function applied to the activation.\n    '
    return theano.tensor.nnet.softplus(x)

def linear(x):
    if False:
        print('Hello World!')
    'Linear activation function :math:`\\varphi(x) = x`\n\n    Parameters\n    ----------\n    x : float32\n        The activation (the summed, weighted input of a neuron).\n\n    Returns\n    -------\n    float32\n        The output of the identity applied to the activation.\n    '
    return x
identity = linear