"""
Defines interface that any backend must implement
"""
import abc
from future.utils import with_metaclass

class Backend_ABC_Meta(abc.ABCMeta):
    """
    metaclass for the backend objects
    takes care of registering all the backend subclasses
    """

    def __init__(self, name, bases, dict_):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'backends'):
            self.backends = {}
        else:
            name = getattr(self, 'backend_name', name)
            if name not in ['Backend']:
                self.backends[name] = self
        super(Backend_ABC_Meta, self).__init__(name, bases, dict_)

class AbstractBackend(with_metaclass(Backend_ABC_Meta, object)):

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self.cleanup_backend()

    @abc.abstractmethod
    def cleanup_backend(self):
        if False:
            print('Hello World!')
        'Release any resources that have been acquired by this backend.'
        raise NotImplementedError()

    @abc.abstractmethod
    def gen_rng(self, seed=None):
        if False:
            while True:
                i = 10
        '\n        Setup the random number generator(s) and store the state\n        in self.init_rng_state.\n\n        Arguments:\n            seed (int or None): RNG seed, if the seed is None,\n                                then a seed will be randomly chosen\n\n        Returns:\n            np.random.RandomState: numpy RNG\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_get_state(self, state):
        if False:
            while True:
                i = 10
        '\n        Get the random number generator state to a specific state.\n\n        Returns a tuple since some backends have multiple RNG states\n        (e.g. on-host and on-device)\n\n        Returns:\n            tuple: array of numpy ndarray which defines the current\n                   state of the RNGs\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset the random state to the state where the Backend is first\n        initialized.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def rng_set_state(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Set the random number generator state to a specific state.\n\n        Arguments:\n            state (np.array): array which is used to define the RNG\n                              state\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def empty(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            while True:
                i = 10
        "\n        Instantiate a new instance of this backend's Tensor class, without\n        initializing element values.  This is slightly faster than\n        :py:func:`~neon.backends.Backend.array`,\n        :py:func:`~neon.backends.Backend.ones`,\n        :py:func:`~neon.backends.Backend.zeros`, but the values will be\n        random.\n\n        Arguments:\n            shape (int, list): length of each dimension of the Tensor.\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            name (str, optional): name indentifying the tensor (used in printing).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n            parallel (bool, optional): If True and using multi-GPU backend,\n                                       replicate copies of this tensor across\n                                       devices.  Defaults to False, and has no\n                                       effect on CPU, or (single) GPU backends.\n            distributed (bool, optional): If True and using multi-GPU backend,\n                                          this tensor is fragmented and\n                                          partitioned across devices.  Defaults\n                                          to False, and has no effect on CPU,\n                                          or (single) GPU backends.\n\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.array`,\n            :py:func:`~neon.backends.backend.Backend.zeros`,\n            :py:func:`~neon.backends.backend.Backend.ones`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def array(self, ary, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            while True:
                i = 10
        "\n        Instantiate a new instance of this backend's Tensor class, populating\n        elements based on ary values.\n\n        Arguments:\n            ary (array_like): input array object to construct from.  Can be\n                              built-in python scalar or list (of lists), or a\n                              numpy.ndarray\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            name (str, optional): name indentifying the tensor (used in printing).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n            parallel (bool, optional): If True and using multi-GPU backend,\n                                       replicate copies of this tensor across\n                                       devices.  Defaults to False, and has no\n                                       effect on CPU, or (single) GPU backends.\n            distributed (bool, optional): If True and using multi-GPU backend,\n                                          this tensor is fragmented and\n                                          partitioned across devices.  Defaults\n                                          to False, and has no effect on CPU,\n                                          or (single) GPU backends.\n\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.empty`,\n            :py:func:`~neon.backends.backend.Backend.zeros`,\n            :py:func:`~neon.backends.backend.Backend.ones`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def zeros(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            print('Hello World!')
        "\n        Instantiate a new instance of this backend's Tensor class, populating\n        each element with a value of 0.\n\n        Arguments:\n            shape (int, list): length of each dimension of the Tensor.\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            name (str, optional): name indentifying the tensor (used in printing).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n            parallel (bool, optional): If True and using multi-GPU backend,\n                                       replicate copies of this tensor across\n                                       devices.  Defaults to False, and has no\n                                       effect on CPU, or (single) GPU backends.\n            distributed (bool, optional): If True and using multi-GPU backend,\n                                          this tensor is fragmented and\n                                          partitioned across devices.  Defaults\n                                          to False, and has no effect on CPU,\n                                          or (single) GPU backends.\n\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.empty`,\n            :py:func:`~neon.backends.backend.Backend.ones`,\n            :py:func:`~neon.backends.backend.Backend.array`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def ones(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            print('Hello World!')
        "\n        Instantiate a new instance of this backend's Tensor class, populating\n        each element with a value of 1.\n\n        Arguments:\n            shape (int, list): length of each dimension of the Tensor.\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            name (str, optional): name indentifying the tensor (used in printing).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n            parallel (bool, optional): If True and using multi-GPU backend,\n                                       replicate copies of this tensor across\n                                       devices.  Defaults to False, and has no\n                                       effect on CPU, or (single) GPU backends.\n            distributed (bool, optional): If True and using multi-GPU backend,\n                                          this tensor is fragmented and\n                                          partitioned across devices.  Defaults\n                                          to False, and has no effect on CPU,\n                                          or (single) GPU backends.\n\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.empty`,\n            :py:func:`~neon.backends.backend.Backend.zeros`,\n            :py:func:`~neon.backends.backend.Backend.array`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def empty_like(self, other_ary, name=None, persist_values=True):
        if False:
            while True:
                i = 10
        "\n        Instantiate a new instance of this backend's Tensor class, with the\n        shape taken from other_ary.\n\n        Arguments:\n            other_ary (tensor object): Tensor to inherit the dimensions of.\n            name (str, optional): name indentifying the tensor (used in printing).\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls.\n\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.empty`,\n            :py:func:`~neon.backends.backend.Backend.ones`,\n            :py:func:`~neon.backends.backend.Backend.array`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def zeros_like(self, other_ary, name=None, persist_values=True):
        if False:
            i = 10
            return i + 15
        "\n        Instantiate a new instance of this backend's Tensor class, with the\n        shape taken from other_ary and populating each element with a value of 0.\n\n        Arguments:\n            other_ary (tensor object): Tensor to inherit the dimensions of.\n            name (str, optional): name indentifying the tensor (used in printing).\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls.\n        Returns:\n            Tensor: array object\n\n        Raises:\n            NotImplementedError: Can't be instantiated directly.\n\n        See Also:\n            :py:func:`~neon.backends.backend.Backend.empty`,\n            :py:func:`~neon.backends.backend.Backend.ones`,\n            :py:func:`~neon.backends.backend.Backend.array`\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def dot(self, a, b, out=None):
        if False:
            while True:
                i = 10
        '\n        Dot product of two Tensors.\n\n        Arguments:\n            a (Tensor): left-hand side operand.\n            b (Tensor): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n                                    Note that this object should differ from\n                                    left and right.\n\n        Returns:\n            OpTreeNode: the resulting op-tree from this operation.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform one of the following operations (* is dot product)\n        C = alpha * A * B   + beta * C\n        C = alpha * A.T * B + beta * C\n        C = alpha * A * B.T + beta * C.\n\n        relu: if true, applied before output (and prior to beta addition)\n\n        The operation will be short-circuited to: out <- alpha * left * right\n        if beta has value 0 (the default).\n\n        Arguments:\n            A (Tensor): left-hand side operand.\n            B (Tensor): right-hand side operand.\n            C (Tensor): output operand\n            alpha (float. optional): scale A*B term\n            beta (float, optional): scale C term before sum\n            relu (bool, optional): If True apply ReLu non-linearity before\n                                   output.  Defaults to False.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform one of the following operations:\n        1 For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)\n        2 For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)\n        3 For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C)\n\n        Arguments:\n            A (Tensor): left-hand input operand\n            B (Tensor): right-hand input operand\n            C (Tensor): output operand\n            alpha (float. optional): scale A*B term\n            beta (float, optional): scale C term before sum\n            relu (bool, optional): If True apply ReLu non-linearity before\n                                   output.  Defaults to False.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def make_binary_mask(self, out, keepthresh=0.5):
        if False:
            print('Hello World!')
        '\n        Create a binary mask for dropout layers.\n\n        Arguments:\n            out (Tensor): Output tensor\n            keepthresh (float, optional): fraction of ones. Defaults to 0.5\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, a, b, out=None):
        if False:
            while True:
                i = 10
        '\n        Perform element-wise addition on the operands, storing the resultant\n        values in the out Tensor. Each operand and out must have identical\n        shape or be broadcastable as such.\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def subtract(self, a, b, out=None):
        if False:
            return 10
        '\n        Perform element-wise subtraction on the operands, storing the resultant\n        values in the out Tensor. Each operand and out must have identical\n        shape or be broadcastable as such.\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def multiply(self, a, b, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform element-wise multiplication on the operands, storing the\n        resultant values in the out Tensor. Each operand and out must have\n        identical shape or be broadcastable as such.\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def divide(self, a, b, out=None):
        if False:
            return 10
        '\n        Perform element-wise division on the operands, storing the\n        resultant values in the out Tensor. Each operand and out must have\n        identical shape or be broadcastable as such.\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def true_divide(self, a, b, out=None):
        if False:
            return 10
        "\n        Here it is an alias of divide.\n        Instead of the Python traditional 'floor division', this returns a\n        true division.\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def power(self, a, b, out=None):
        if False:
            print('Hello World!')
        "\n        Perform element-wise raise of tsr values to specified power,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            b (Tensor, numeric): exponentiated value to be applied to\n                                     element.  Examples include 2 (square),\n                                     0.5 (sqaure root).\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def reciprocal(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise reciprocal of Tensor `a`, storing the result in\n        Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            power (Tensor, numeric): exponentiated value to be applied to\n                                     element.  Examples include 2 (square),\n                                     0.5 (sqaure root).\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def negative(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise negation of Tensor `a`, storing the result in\n        Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def sgn(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise indication of the sign of Tensor `a`, storing the\n        result in Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def absolute(self, a, out=None):
        if False:
            while True:
                i = 10
        "\n        Perform element-wise absolute value of Tensor `a`, storing the result in\n        Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def fabs(self, a, out=None):
        if False:
            print('Hello World!')
        "\n        Perform element-wise absolute value of Tensor `a`, storing the result\n        in Tensor out. Both Tensor's should have identical shape. Implemented as\n        an alias of absolute.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def sqrt(self, a, out=None):
        if False:
            print('Hello World!')
        "\n        Perform element-wise square-root of Tensor `a`, storing the result in\n        Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def square(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise square of Tensor `a`, storing the result in Tensor\n        out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def exp(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise exponential transformation on Tensor `a`, storing\n        the result in Tensor out. Both Tensor's should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def exp2(self, a, out=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Perform element-wise 2-based exponential transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def safelog(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise natural logarithm transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.  This log function has built in safety for underflow.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def log(self, a, out=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Perform element-wise natural logarithm transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def log2(self, a, out=None):
        if False:
            while True:
                i = 10
        "\n        Perform element-wise 2-based logarithm transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def sig(self, a, out=None):
        if False:
            print('Hello World!')
        "\n        Perform element-wise sigmoid transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def sig2(self, a, out=None):
        if False:
            print('Hello World!')
        "\n        Perform element-wise 2-based sigmoid logarithm transformation on\n        Tensor `a`, storing the result in Tensor out. Both Tensor's should\n        have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def tanh(self, a, out=None):
        if False:
            i = 10
            return i + 15
        "\n        Perform element-wise hyperbolic tangent transformation on Tensor `a`,\n        storing the result in Tensor out. Both Tensor's should have identical\n        shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def tanh2(self, a, out=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Perform element-wise 2-based hyperbolic tangent transformation on Tensor\n        `a`, storing the result in Tensor out. Both Tensor's should have\n        identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def finite(self, a, out=None):
        if False:
            return 10
        "\n        Perform element-wise test of finiteness (not infinity or not Not a\n        Number) on Tensor `a`, storing the result in Tensor out. Both Tensor's\n        should have identical shape.\n\n        Arguments:\n            a (Tensor): input to be transformed.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        "
        raise NotImplementedError()

    @abc.abstractmethod
    def equal(self, a, b, out=None):
        if False:
            while True:
                i = 10
        '\n        Performs element-wise equality testing on each element of left and\n        right, storing the result in out. Each operand is assumed to be the\n        same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def not_equal(self, a, b, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs element-wise non-equality testing on each element of left and\n        right, storing the result in out. Each operand is assumed to be the\n        same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def less(self, a, b, out=None):
        if False:
            print('Hello World!')
        '\n        Performs element-wise less than testing on each element of left and\n        right, storing the result in out. Each operand is assumed to be the\n        same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def less_equal(self, a, b, out=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs element-wise less than or equal testing on each element of\n        left and right, storing the result in out. Each operand is assumed to\n        be the same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def greater(self, a, b, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Performs element-wise greater than testing on each element of left and\n        right, storing the result in out. Each operand is assumed to be the\n        same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only theshape op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def greater_equal(self, a, b, out=None):
        if False:
            return 10
        '\n        Performs element-wise greater than or equal testing on each element of\n        left and right, storing the result in out. Each operand is assumed to\n        be the same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def maximum(self, a, b, out=None):
        if False:
            return 10
        '\n        Performs element-wise maximum value assignment based on corresponding\n        elements of left and right, storing the result in out. Each operand is\n        assumed to be the same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def minimum(self, a, b, out=None):
        if False:
            while True:
                i = 10
        '\n        Performs element-wise minimum value assignment based on corresponding\n        elements of left and right, storing the result in out. Each operand is\n        assumed to be the same shape (or broadcastable as such).\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def clip(self, a, a_min, a_max, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Performs element-wise clipping of Tensor `a`, storing the result in out.\n        The clipped value will be between [a_min, a_max].\n\n        Arguments:\n            a (Tensor, numeric): left-hand side operand.\n            b (Tensor, numeric): right-hand side operand.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def sum(self, a, axis=None, out=None, keepdims=True):
        if False:
            print('Hello World!')
        '\n        Calculates the summation of the elements along the specified axis.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the sum\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will sum over all\n                                  dimensions.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def max(self, a, axis=None, out=None, keepdims=True):
        if False:
            return 10
        '\n        Calculates the maximal element value along the specified axes.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take max over all\n                                  dimensions.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def min(self, a, axis=None, out=None, keepdims=True):
        if False:
            print('Hello World!')
        '\n        Calculates the minimal element value along the specified axes.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take min over all\n                                  dimensions.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def argmax(self, a, axis=1, out=None, keepdims=True):
        if False:
            while True:
                i = 10
        '\n        Calculates the indices of the maximal element value along the specified\n        axis.  If multiple elements contain the maximum, only the indices of\n        the first are returned.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take argmax over all\n                                  dimensions.  Defaults to 1\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def argmin(self, a, axis=1, out=None, keepdims=True):
        if False:
            while True:
                i = 10
        '\n        Calculates the indices of the minimal element value along the specified\n        axis.  If multiple elements contain the minimum, only the indices of\n        the first are returned.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take argmin over all\n                                  dimensions.  Defaults to 1\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def mean(self, a, axis=None, partial=None, out=None, keepdims=True):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the arithmetic mean of the elements along the specified\n        axes.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take mean over all\n                                  dimensions.  Defaults to None\n            partial (bool, optional): Not currently used.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def var(self, a, axis=None, partial=None, out=None, keepdims=True):
        if False:
            while True:
                i = 10
        '\n        Calculates the variance of the elements along the specified\n        axes.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take var over all\n                                  dimensions.  Defaults to None\n            partial (bool, optional): Not currently used.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def std(self, a, axis=None, partial=None, out=None, keepdims=True):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the standard deviation of the elements along the specified\n        axes.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will take std over all\n                                  dimensions.\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n            partial (bool, optional): Not currently used.\n            keepdims (bool, optional): Keep the axes being computed over in the\n                                       output (with size 1), instead of\n                                       collapsing.  Defaults to True.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def take(self, a, indices, axis, out=None):
        if False:
            print('Hello World!')
        '\n        Extract elements based on the indices along a given axis.\n\n        Arguments:\n            a (Tensor): the Tensor on which to perform the operation\n            indices (Tensor, numpy ndarray): indicies of elements to select\n            axis (int, optional): the dimension along which to compute.\n                                  If set to None, we will extract over all\n                                  dimensions (flattened first)\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def onehot(self, indices, axis, out=None):
        if False:
            print('Hello World!')
        '\n        Generate optree for converting `indices` to a onehot representation.\n\n        Arguments:\n            indices (Tensor): Elements must be of numpy integer type for gpu\n                              onehot to work.\n            axis (int): the axis along the feature length dimension\n            out (Tensor, optional): where the result will be stored. If out is\n                                    None, only the op-tree will be returned.\n\n        Returns:\n            OpTreeNode: the resulting op-tree\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def update_fc_bias(self, err, out):
        if False:
            print('Hello World!')
        '\n        Compute the updated bias gradient for a fully connected network layer.\n\n        Arguments:\n            err (Tensor): backpropagated error\n            out (Tensor): Where to store the updated gradient value.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def add_fc_bias(self, inputs, bias):
        if False:
            i = 10
            return i + 15
        '\n        Add the bias for a fully connected network layer.\n\n        Arguments:\n            inputs (Tensor): the input to update.\n            bias (Tensor): the amount to increment\n        '
        self.ng.add(inputs, bias, out=inputs)

    @abc.abstractmethod
    def conv_layer(self, dtype, N, C, K, D=1, H=1, W=1, T=1, R=1, S=1, pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1, relu=False, bsum=False):
        if False:
            while True:
                i = 10
        '\n        Create a new ConvLayer parameter object.\n        This is then passed as an argument to all the convolution operations.\n\n        Arguments:\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n\n            N (int): Number of images in mini-batch\n            C (int): Number of input feature maps\n            K (int): Number of output feature maps\n\n            D (int, optional): Depth of input image.  Defaults to 1\n            H (int, optional): Height of input image.  Defaults to 1\n            W (int, optional): Width of input image.  Defaults to 1\n\n            T (int, optional): Depth of filter kernel.  Defaults to 1\n            R (int, optional): Height of filter kernel.  Defaults to 1\n            S (int, optional): Width of filter kernel.  Defaults to 1\n\n            pad_d (int, optional): amount of zero-padding around the depth edge\n                                   Defaults to 0.\n            pad_h (int, optional): amount of zero-padding around the height edge\n                                   Defaults to 0.\n            pad_w (int, optional): amount of zero-padding around the width edge\n                                   Defaults to 0.\n\n            str_d (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_h (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_w (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n\n            relu (bool, optional): apply a relu transform to the output for\n                                   fprop or bprop.  Defaults to False\n\n            bsum (bool, optional): calculate the sum along the batchnorm axis\n                                   for fprop or bprop.  Outputs an fp32 tensor\n                                   of size Kx1.  Defaults to False.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def fprop_conv(self, layer, I, F, O, alpha=1.0, relu=False, repeat=1):
        if False:
            print('Hello World!')
        '\n        Forward propagate the inputs of a convolutional network layer to\n        produce output.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            I (Tensor): inputs\n            F (Tensor): the weights (filters)\n            O (Tensor): outputs\n            alpha (float, optional): linear scaling.  Defaults to 1.0\n            relu (bool, optional): apply ReLu before output.  Default not to.\n            repeat (int, optional): Repeat this operation the specified number\n                                    of times.  Defaults to 1.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def bprop_conv(self, layer, F, E, grad_I, alpha=1.0, repeat=1):
        if False:
            print('Hello World!')
        '\n        Backward propagate the error through a convolutional network layer.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            F (Tensor): the weights (filters)\n            E (Tensor): errors\n            grad_I (Tensor): gradient to inputs (output delta)\n            alpha (float, optional): linear scaling.  Defaults to 1.0\n            repeat (int, optional): Repeat this operation the specified number\n                                    of times.  Defaults to 1.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def update_conv(self, layer, I, E, grad_F, alpha=1.0, repeat=1):
        if False:
            i = 10
            return i + 15
        '\n        Compute the updated gradient for a convolutional network layer.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            I (Tensor): the inputs\n            E (Tensor): the errors\n            grad_F (Tensor): filter gradients (weights) to update.\n            alpha (float, optional): linear scaling.  Defaults to 1.0\n            repeat (int, optional): Repeat this operation the specified number\n                                    of times.  Defaults to 1.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def deconv_layer(self, dtype, N, C, K, P, Q, R=1, S=1, pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new Deconvolution parameter object.\n        This then is passed as an argument to all deconvolution kernels.\n\n        Arguments:\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n\n            N (int): Number of images in mini-batch\n            C (int): Number of input feature maps\n            K (int): Number of output feature maps\n\n            P (int): Height of output\n            Q (int): Width of output\n\n            R (int, optional): Height of filter kernel.  Defaults to 1\n            S (int, optional): Width of filter kernel.  Defaults to 1\n\n            pad_d (int, optional): amount of zero-padding around the depth edge\n                                   Defaults to 0.\n            pad_h (int, optional): amount of zero-padding around the height edge\n                                   Defaults to 0.\n            pad_w (int, optional): amount of zero-padding around the width edge\n                                   Defaults to 0.\n\n            str_d (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_h (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_w (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n\n        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def pool_layer(self, dtype, op, N, C, D=1, H=1, W=1, J=1, T=1, R=1, S=1, pad_j=0, pad_d=0, pad_h=0, pad_w=0, str_j=None, str_d=None, str_h=None, str_w=None):
        if False:
            while True:
                i = 10
        '\n        Create a new PoolLayer parameter object.\n        This then is passed as an argument to all pooling kernels.\n\n        Arguments:\n            op (str): "max", "avg", "l2" pooling (currently bprop only supports\n                      max, but not avg and l2)\n            N (int): Number of images in mini-batch\n\n            C (int): Number of input feature maps\n            D (int, optional): Depth of input image.  Defaults to 1\n            H (int, optional): Height of input image.  Defaults to 1\n            W (int, optional): Width of input image.  Defaults to 1\n\n            J (int, optional): Size of feature map pooling window\n                               (maxout n_pieces).  Defaults to 1\n            T (int, optional): Depth of pooling window.  Defaults to 1\n            R (int, optional): Height of pooling window.  Defaults to 1\n            S (int, optional): Width of pooling window.  Defaults to 1\n\n            pad_j (int, optional): amount of zero-padding around the fm pooling\n                                   window edge.  Defaults to 0.\n            pad_d (int, optional): amount of zero-padding around the depth edge\n                                   Defaults to 0.\n            pad_h (int, optional): amount of zero-padding around the height edge\n                                   Defaults to 0.\n            pad_w (int, optional): amount of zero-padding around the width edge\n                                   Defaults to 0.\n\n            str_j (int, optional): factor to step the filters by in the fm\n                                   pooling window direction.  Defaults to 1\n            str_d (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_h (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n            str_w (int, optional): factor to step the filters by in the depth\n                                   direction.  Defaults to 1\n\n        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def fprop_pool(self, layer, I, O):
        if False:
            print('Hello World!')
        '\n        Forward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object, different backends have\n                               different pool layers.\n            I (Tensor): Input tensor.\n            O (Tensor): output tensor.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def bprop_pool(self, layer, I, E, grad_I):
        if False:
            print('Hello World!')
        '\n        Backward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object. Different backends have\n                               different pool layers.\n            I (Tensor): Input tensor.\n            E (Tensor): Error tensor.\n            grad_I (Tensor): Gradient tensor (delta)\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def compound_bprop_lut(self, nin, inputs, error, error_t, dW, pad_idx, alpha=1.0, beta=0):
        if False:
            while True:
                i = 10
        '\n        Backward propagate lookup table layer.\n\n        Arguments:\n            nin (int): Number of input word_ids.\n            inputs (Tensor): Input tensor.\n            error (Tensor): Error tensor.\n            error_t (Tensor): Transposed error tensor.\n            dW (Tensor): Gradient tensor (delta).\n            pad_idx (int):\n            alpha (float):\n            beta (float):\n        '
        raise NotImplementedError()