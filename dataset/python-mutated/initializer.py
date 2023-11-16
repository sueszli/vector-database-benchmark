import numpy as np
from neon import NervanaObject
from neon.backends.backend import Tensor

class Initializer(NervanaObject):
    """
    Abstract base class from which parameter tensor initializers inherit.

    Subclasses should implement the ``fill`` method which takes as input a Tensor
    and fills the values based on the initialization scheme.
    """

    def fill(self, param):
        if False:
            while True:
                i = 10
        '\n        Initialize the provided tensor with values.\n\n        Args:\n            param (Tensor): Input Tensor.\n        '
        raise NotImplementedError()

class Constant(Initializer):
    """
    Initializes parameters as a constant.
    """

    def __init__(self, val=0.0, name='constantInit'):
        if False:
            return 10
        '\n        Class constructor.\n\n        Args:\n            val (float, optional): The value to assign to all tensor elements\n        '
        super(Constant, self).__init__(name=name)
        self.val = val

    def fill(self, param):
        if False:
            print('Hello World!')
        '\n        Fills the provided tensor.\n\n        Args:\n            param (tensor): target tensor to fill\n        '
        if isinstance(self.val, Tensor):
            assert self.val.shape == param.shape, 'Constant(Array) initializer can only fill a matching shape tensor'
        param[:] = self.val

class Array(Constant):
    """
    Initializes parameters with values specified by a provided numpy array.

    Same functionality as Constant except serialization needs to dump
    tensor values into np array

    Args:
        vals (ndarray or tensor, optional): Values to assign to the tensor elements
    """

    def get_description(self):
        if False:
            print('Hello World!')
        '\n        Returns description of the object as a dict. Transfers the\n        tensors back to a numpy array.\n        '
        desc = super(Array, self).get_description()
        if isinstance(desc['config']['val'], Tensor):
            desc['config']['val'] = desc['config']['val'].get()
        return desc

class Uniform(Initializer):
    """
    Initializes parameters with random values drawn from a uniform distribution.
    """

    def __init__(self, low=0.0, high=1.0, name='uniformInit'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Class constructor.\n\n        Args:\n            low  (float, optional): Lower bound of range.\n            high (float, optional): Upper bound of range.\n        '
        super(Uniform, self).__init__(name=name)
        (self.low, self.high) = (low, high)

    def fill(self, param):
        if False:
            return 10
        '\n        Fill the provided tensor with random values drawn from a uniform\n        distribution.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        param[:] = self.be.rng.uniform(self.low, self.high, param.shape)

class Gaussian(Initializer):
    """
    Initializes parameters with a gaussian distribution with the provided mean
    and standard deviation. Defaults to (loc=0, scale=1)
    """

    def __init__(self, loc=0.0, scale=1.0, name='gaussianInit'):
        if False:
            print('Hello World!')
        '\n        Class constructor.\n\n        Args:\n            loc   (float, optional): Mean parameter (mu). Defaults to 0.\n            scale (float, optional): Standard deviation parameter (sigma). Defaults to 1.\n            name (string, optional): Name to assign an instance of this class.\n        '
        super(Gaussian, self).__init__(name=name)
        (self.loc, self.scale) = (loc, scale)

    def fill(self, param):
        if False:
            i = 10
            return i + 15
        '\n        Fill the provided tensor with random values drawn from a gaussian\n        distribution.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        param[:] = self.be.rng.normal(self.loc, self.scale, param.shape)

class GlorotUniform(Initializer):
    """
    Initializes parameter tensors with values drawn from a uniform distribution
    ranging from :math:`-K` to :math:`K`. We define :math:`K=\\sqrt{6 / (n_{in} + n_{out})}`,
    where :math:`n_{in}` and :math:`n_{out}` are the input and output dimensions, respectively,
    of the parameter tensor. This approach normalizes the range of the initialized values
    by the tensor dimensions.

    From: "Understanding the difficulty of training deep feedforward neural networks"
    (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """

    def __init__(self, name='autouniformInit'):
        if False:
            while True:
                i = 10
        '\n        Class constructor.\n\n        Args:\n            name (string, optional): Name to assign an instance of this class\n        '
        super(GlorotUniform, self).__init__(name=name)

    def fill(self, param):
        if False:
            return 10
        '\n        Fill the provided tensor with random values drawn from the Uniform\n        distribution, using normalized bounds.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        k = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
        param[:] = self.be.rng.uniform(-k, k, param.shape)

class Xavier(Initializer):
    """
    Initializes parameter tensors with values drawn from a uniform distribution
    ranging from :math:`-K` to :math:`K` We define :math:`K=\\sqrt{3 / (n_{in})}`,
    where :math:`n_{in}` is the number of input nodes.

    Similar to Glorot except the range is normalized by the input size only.
    """

    def __init__(self, local=True, name='xavier'):
        if False:
            return 10
        '\n        Class constructor.\n\n        Args:\n            local (bool, optional): Whether the layer type is local (Convolutional) or not.\n                                      Default is True.\n            name (string, optional): Name to assign an instance of this class.\n        '
        super(Xavier, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fill the provided tensor with random values drawn from the Uniform\n        distribution, using normalized bounds.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        fan_in = param.shape[0 if self.local else 1]
        scale = np.sqrt(3.0 / fan_in)
        param[:] = self.be.rng.uniform(-scale, scale, param.shape)

class Kaiming(Initializer):
    """
    Initializes parameters with a zero-mean Gaussian distribution. The standard deviation
    is automatically set as :math:`\\sigma=\\sqrt{2 / n_{in}}`, where :math:`n_{in}` is
    the input dimension of the tensor.


    Based on the initializer described in: http://arxiv.org/pdf/1502.01852.pdf.
    """

    def __init__(self, local=True, name='Kaiming'):
        if False:
            i = 10
            return i + 15
        '\n        Class constructor.\n\n        Args:\n            local (bool, optional): Whether the layer type is local (Convolutional) or not.\n                                      Default is True.\n            name (string, optional): Name to assign an instance of this class.\n        '
        super(Kaiming, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        if False:
            while True:
                i = 10
        '\n        Fill the provided tensor with random values drawn from a gaussian\n        distribution.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        fan_in = param.shape[0 if self.local else 1]
        scale = np.sqrt(2.0 / fan_in)
        param[:] = self.be.rng.normal(0, scale, param.shape)

class IdentityInit(Initializer):
    """
    Initializes parameters with the identity matrix.
    """

    def __init__(self, local=True, name='Identity'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Class constructor.\n\n        Args:\n            local (bool, optional): Whether the layer type is local (Convolutional) or not.\n                                      Default is True.\n            name (string, optional): Name to assign an instance of this class.\n        '
        super(IdentityInit, self).__init__(name=name)
        self.local = local

    def fill(self, param):
        if False:
            while True:
                i = 10
        '\n        Fill the provided tensor with the identity matrix.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        (nin, nout) = param.shape
        w_ary = np.zeros((nin, nout), dtype=np.float32)
        w_ary[:, :nin] = np.eye(nin)
        param[:] = w_ary

class Orthonormal(Initializer):
    """
    Initializes parameters with the single value decomposition of a
    random gaussian matrix.

    Implementation taken from Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def __init__(self, scale=1.1, name='orthonormal'):
        if False:
            i = 10
            return i + 15
        '\n        Class constructor.\n\n        Args:\n            scale (float, optional): Scaling factor of values. Defaults to 1.1.\n            name (string, optional): Name to assign an instance of this class.\n        '
        super(Orthonormal, self).__init__(name=name)
        self.scale = scale

    def fill(self, param):
        if False:
            print('Hello World!')
        '\n        Fill the provided tensor using the Orthonormal method.\n\n        Args:\n            params (tensor): Tensor to fill\n        '
        a = np.random.normal(0.0, 1.0, param.shape)
        (u, _, v) = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == param.shape else v
        param[:] = self.scale * q