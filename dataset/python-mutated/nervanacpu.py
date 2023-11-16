"""
Our CPU based backend interface and tensor data structure. Our implementation
wraps :mod:`numpy` ndarray and related operations
"""
from __future__ import division
from builtins import object, round, str, zip
import numpy as np
import sys
import logging
import time
import functools
from neon.backends.backend import Tensor, Backend, OpTreeNode, OpCollection
from neon.backends.layer_cpu import ConvLayer, DeconvLayer, PoolLayer
from neon.util.compat import xrange
_none_slice = slice(None, None, None)
logger = logging.getLogger(__name__)

class CPUTensor(Tensor):
    """
    The n-dimensional array data structure that resides in host memory,
    and is meant to be manipulated on the CPU.  wrapped `numpy.ndarray` tensor.

    Arguments:
        dtype (numpy.ndtype, optional): underlying data type of the elements.
        ary (data array, optional): optionally it can be instantiated with
                                    a data array
        persist_values (bool, optional): If set to True (the default), the
                                         values assigned to this Tensor will
                                         persist across multiple begin and end
                                         calls.  Setting to False may provide a
                                         performance increase if values do
                                         not need to be maintained across such
                                         calls

    See also:
        :class:`NervanaCPU` class
    """
    _tensor = None

    def __init__(self, backend, shape=None, dtype=np.float32, ary=None, name=None, persist_values=True, base=None):
        if False:
            i = 10
            return i + 15
        super(CPUTensor, self).__init__(backend, shape, dtype, name, persist_values)
        assert dtype in (np.float16, np.float32, np.float64, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32)
        dtype = np.dtype(dtype)
        if type(ary) != np.ndarray:
            self._tensor = np.array(ary, dtype)
        elif ary.dtype != dtype:
            self._tensor = ary.astype(dtype)
        else:
            self._tensor = ary
        while self._tensor.ndim < self._min_dims:
            self._tensor = self._tensor.reshape(self._tensor.shape + (1,))
        if shape is not None and len(shape) < self._min_dims:
            self.shape = shape + (1,) * (self._min_dims - len(shape))
        else:
            self.shape = self._tensor.shape
        shape_ = []
        size = 1
        for dim in self.shape:
            if int(dim) != dim:
                raise TypeError('shape dims must be integer values [%s]' % str(dim))
            dim = int(dim)
            shape_.append(dim)
            size *= dim
        self.shape = tuple(shape_)
        self.size = size
        self.base = base
        self.dtype = dtype
        self.is_contiguous = self._tensor.flags.c_contiguous

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string representation of this Tensor.\n\n        Returns:\n            str: the representation.\n        '
        if self._tensor.base is not None:
            base_id = id(self._tensor.base)
        else:
            base_id = id(self._tensor)
        return 'CPUTensor(base 0x%x) name:%s shape:%s dtype:%s strides:%s is_c_contiguous:%s' % (base_id, self.name, self.shape, self.dtype, self._tensor.strides, self._tensor.flags.c_contiguous)

    def __repr__(self):
        if False:
            return 10
        '\n        Returns a more unambiguous string representation of the Tensor.\n\n        Returns:\n            str: the representation.\n        '
        return self.__str__()

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the size of the leading dimension of self.\n        '
        if len(self.shape):
            return self.shape[0]
        else:
            return 0

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        "\n        Assign the specified value to a subset of elements found via slice\n        style indexing along each dimension. e.g. A[5:10, :] = 4.5.\n        Each slice consists of start_idx:stop_idx:step_size triplets.  If\n        step_size isn't specified it defaults to 1.  If start_idx isn't\n        specified it defaults to 0.  If stop_idx isn't specified it defaults\n        to the total number of elements along that dimension.  As such a slice\n        value of ':' allows one to select all elements along that dimension.\n\n        Arguments:\n            key (int, slice, tuple): indices of each dimension's slice.\n            value (numeric array, CPUTensor): values to be assigned to the\n                                              extracted element subset.  If an\n                                              array it should be the same shape\n                                              as what key indexes (or be\n                                              broadcastable as such).\n        "
        self.__getitem__(key)._assign(value)
        return self

    def __getitem__(self, key):
        if False:
            return 10
        "\n        Extract a subset view of the items via slice style indexing\n        along each dimension. e.g. A[5:10, :].  Each slice consists of\n        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it\n        defaults to 1.  If start_idx isn't specified it defaults to 0.  If\n        stop_idx isn't specified it defaults to the total number of elements\n        along that dimension.  As such a slice value of ':' allows one to\n        select all elements along that dimension. To be consistent with GPU\n        Tensors, CPU Tensors remove the axis that has size 1 unless it needs to\n        maintain 2D.\n\n        Arguments:\n            key (int, slice, tuple): indices of each dimension's slice.\n\n        Returns:\n            CPUTensor: view of self corresponding to the subset items.\n\n        "
        if not isinstance(key, tuple):
            if key == _none_slice:
                return self
            key = (key,)
        key_list = list(key)
        for (idx, k) in enumerate(key):
            if type(k) is int:
                k = self.shape[idx] + k if k < 0 else k
                key_list[idx] = slice(k, k + 1, None)
        key = tuple(key_list)
        new_shape = list(self._tensor[key].shape)
        for (idx, k) in enumerate(new_shape):
            if len(new_shape) > 2 and k is 1:
                new_shape.remove(k)
        return self.__class__(backend=self.backend, ary=self._tensor[key].reshape(new_shape), dtype=self._tensor.dtype, base=self)

    def _assign(self, value):
        if False:
            return 10
        '\n        Assign an input value to the CPU tensor. The NervanaCPU does clipping\n        for int and uint types, when overflow happens\n\n        Arguments:\n            value (CPUTensor, OpTreeNode, numeric): the value to be assigned.\n\n        '
        if isinstance(value, (CPUTensor, OpTreeNode)):
            OpTreeNode.build('assign', self, value)
        elif isinstance(value, (int, float, np.ndarray)):
            self.set(value)
        else:
            raise TypeError('Invalid type for assignment: %s' % type(value))
        return self

    def set(self, value):
        if False:
            print('Hello World!')
        '\n        Wrap the value into NervanaCPU tensor.\n\n        Arguments:\n            value: Array or single input. If it is array, check and Convert\n                   the dtype and shape. If it is single value, broadcast to\n                   the memory\n\n        Returns:\n            self\n        '
        if isinstance(value, np.ndarray):
            if value.dtype is not self.dtype:
                value = value.astype(self.dtype)
            assert value.size == self.size
            if value.ndim < self._min_dims:
                value = value.reshape(self.shape)
        self._tensor[:] = value
        return self

    def get(self):
        if False:
            while True:
                i = 10
        '\n        Return the array.\n        '
        return self._tensor.copy()

    def raw(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Access the raw buffer.\n\n        Returns:\n            pointer: A device specific pointer\n        '
        return self._tensor.ctypes.data

    def asnumpyarray(self):
        if False:
            return 10
        '\n        Deprecated.\n        Scheduled to be removed in 2.0.\n        Use get() instead.\n        '
        return self._tensor

    def take(self, indices, axis=None):
        if False:
            print('Hello World!')
        '\n        Select a subset of elements from an array across an axis.\n\n        Arguments:\n            indices (Tensor, numpy ndarray): indicies of elements to select\n            axis (int): axis across which to select the values\n\n        Returns:\n            Tensor: Tensor with selected values\n\n        '
        if type(indices) == self.__class__:
            indices = indices._tensor
        if type(indices) == np.ndarray:
            indices = indices.squeeze()
        new_shape = list(self.shape)
        new_shape[axis] = indices.size
        return self.__class__(backend=self.backend, ary=self._tensor.take(indices, axis).reshape(new_shape), dtype=self._tensor.dtype, base=self)

    def fill(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assign specified value to each element of this CPUTensor.\n\n        Arguments:\n            value (numeric): The value to be assigned to each element.\n\n        Return:\n            CPUTensor: updated view of the data.\n        '
        self._tensor.fill(value)
        return self

    def copy(self, a):
        if False:
            i = 10
            return i + 15
        '\n        Construct and return a deep copy of the Tensor passed.\n\n         Arguments:\n            a (Tensor): the object to copy\n\n        Returns:\n            Tensor: new array object with the same values as input tensor\n        '
        return self._assign(a)

    def copy_from(self, a):
        if False:
            print('Hello World!')
        '\n        Alias of copy.\n\n        Arguments:\n            a (Tensor): the object to copy\n\n        Returns:\n            Tensor: new array object with the same values as input tensor\n        '
        return self._assign(a)

    def reshape(self, *shape):
        if False:
            i = 10
            return i + 15
        '\n        Return a reshaped view.\n        '
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if shape == self.shape:
            return self
        try:
            ary = self._tensor.reshape(shape)
        except ValueError:

            def product(vec):
                if False:
                    i = 10
                    return i + 15
                return functools.reduce(lambda x, y: x * y, vec)
            raise ValueError('The total size of a reshaped tensor must be the same as its existing size. Tensor is currently shape {current_shape} and size {current_size}. Attempted to reshape to {reshape_shape} which would be size {reshape_size}.'.format(current_shape=self._tensor.shape, current_size=product(self._tensor.shape), reshape_shape=shape, reshape_size=product(shape)))
        return self.__class__(backend=self.backend, ary=ary, dtype=self._tensor.dtype, base=self)

    @property
    def T(self):
        if False:
            while True:
                i = 10
        '\n        Return a transposed view.\n\n        For 2D tensor, will do a normal transpose\n        For 3D tensor, will keep the 0 dim, swap the 1 and 2 dimensions\n\n        '
        if len(self.shape) <= 2:
            ary = self._tensor.transpose()
        else:
            ary = self._tensor.swapaxes(1, 2)
        return self.__class__(backend=self.backend, ary=ary, dtype=self._tensor.dtype, base=self)

    def transpose(self, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Return a transposed view of the data.  Alias of .T property\n        '
        if out:
            return OpTreeNode.build('assign', out, self.T)
        return self.T

    def share(self, shape, dtype=None, name=None):
        if False:
            while True:
                i = 10
        '\n        Return a view: ary, where ary.size <= self.size.\n        Allows easy sharing of temporary memory\n        This is mostly provided for compatibility, -- dtype is ignored\n        '
        size = np.prod(shape)
        if size > self.size:
            raise ValueError('total size of new array must <= size of parent')
        ary = self._tensor.ravel()[:size].reshape(shape)
        return self.__class__(backend=self.backend, ary=ary, dtype=self._tensor.dtype, base=self)

    def hist(self, tag):
        if False:
            print('Hello World!')
        '\n        Compute a histogram of the current tensor values.\n\n        Arguments:\n            tag (string): Tag to identify the current state of the tensor,\n                          useful for disambiguating multiple histograms of the\n                          same tensor at different points in time.\n\n        Returns:\n            Tensor containing the histogram data.\n\n        '
        nbins = self.backend.hist_bins
        offset = self.backend.hist_offset
        bins = np.arange(nbins + 1) + float(offset)
        bins[0] = -float('Inf')
        np_inp_log_abs = np.rint(np.log2(np.abs(self._tensor.astype(np.float32))))
        (np_hist, edges) = np.histogram(np_inp_log_abs, density=False, bins=bins)
        nc_hist = self.backend._hist_tensor(tag)._assign(np_hist)
        return nc_hist

class CustomNumpy(object):

    @staticmethod
    def argmax(x, axis=1, keepdims=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls numpy argmax with keepdims.\n        '
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmax(x, axis=axis).reshape(new_shape)

    @staticmethod
    def argmin(x, axis=1, keepdims=True):
        if False:
            while True:
                i = 10
        '\n        Calls numpy argmin with keepdims.\n        '
        new_shape = list(x.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)
        return np.argmin(x, axis=axis).reshape(new_shape)

def _assign_right_to_left(left, right):
    if False:
        while True:
            i = 10
    left[:] = right
numpy_call_dict_cpu = {'assign': _assign_right_to_left, 'neg': lambda left: -left, 'abs': lambda left: np.abs(left), 'sgn': lambda left: np.sign(left), 'sqrt': lambda left: np.sqrt(left), 'sqr': lambda left: np.square(left), 'exp': lambda left: np.exp(left), 'log': lambda left: np.log(left), 'safelog': lambda left: np.log(np.maximum(left, np.exp(-50.0))), 'exp2': lambda left: np.exp2(left), 'log2': lambda left: np.log2(left), 'sig': lambda left: 1.0 / (1.0 + np.exp(-left)), 'sig2': lambda left: 1.0 / (1.0 + np.exp2(-left)), 'tanh': lambda left: np.tanh(left), 'tanh2': lambda left: (np.exp2(2.0 * left) - 1.0) / (np.exp2(2.0 * left) + 1.0), 'transpose': lambda left: np.transpose(left), 'rint': lambda left: np.rint(left), 'add': lambda left, right: left + right, 'sub': lambda left, right: left - right, 'mul': lambda left, right: left * right, 'div': lambda left, right: left / right, 'eq': lambda left, right: left == right, 'ne': lambda left, right: left != right, 'lt': lambda left, right: left < right, 'le': lambda left, right: left <= right, 'gt': lambda left, right: left > right, 'ge': lambda left, right: left >= right, 'pow': lambda left, right: np.power(left, right), 'minimum': lambda left, right: np.minimum(left, right), 'maximum': lambda left, right: np.maximum(left, right), 'dot': lambda left, right: np.dot(left, right), 'sum': lambda op_dict, left: np.sum(left, axis=op_dict['axis'], keepdims=True), 'max': lambda op_dict, left: np.max(left, axis=op_dict['axis'], keepdims=True), 'min': lambda op_dict, left: np.min(left, axis=op_dict['axis'], keepdims=True), 'argmax': lambda op_dict, left: CustomNumpy.argmax(left, axis=op_dict['axis'], keepdims=True), 'argmin': lambda op_dict, left: CustomNumpy.argmin(left, axis=op_dict['axis'], keepdims=True)}

class NervanaCPU(Backend):
    """
    Sets up a :mod:`numpy` baseyd backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.
        tensor_cls: underlying Tensor type. For CPU backend, it will be CPU tensor

    See also:
        :class:`CPUTensor`
    """
    backend_name = 'cpu'

    def __init__(self, rng_seed=None, default_dtype=np.float32, hist_bins=64, hist_offset=-48, compat_mode=None, num_devices=None, stochastic_round=None, device_id=None, deterministic=None):
        if False:
            for i in range(10):
                print('nop')
        if default_dtype not in [np.float16, np.float32, np.float64]:
            logger.error('Default data type for nervanagpu backend must be float16, 32 or 64')
            raise ValueError
        super(NervanaCPU, self).__init__(rng_seed, default_dtype, compat_mode=compat_mode)
        try:
            if not any((x in str(np.__config__.blas_opt_info['libraries']).lower() for x in ['openblas', 'atlas', 'mkl', 'accelerate'])):
                logger.warn('No accelerated BLAS libraries found, CPU performance may suffer.  Consider installing one of openblas, Atlas, MKL, or vecLib')
        except (AttributeError, KeyError):
            if sys.platform != 'darwin':
                logger.warn('Problems inferring BLAS info, CPU performance may be suboptimal')
        self.device_type = 0
        self.device_id = 0
        self.tensor_cls = CPUTensor
        logger.info('Initialized NervanaCPU')
        (self.hist_bins, self.hist_offset) = (None, None)
        self.set_hist_buffers(hist_bins, hist_offset)
        self.use_pinned_mem = False

    def consume(self, buf_index, hostlist, devlist):
        if False:
            while True:
                i = 10
        assert 0 <= buf_index < 2, 'Can only double buffer'
        if devlist[buf_index] is None:
            devlist[buf_index] = self.empty_like(hostlist[buf_index].T, dtype=hostlist[buf_index].dtype)
        devlist[buf_index][:] = hostlist[buf_index].T

    def set_hist_buffers(self, hist_bins, hist_offset):
        if False:
            print('Hello World!')
        if hist_bins != self.hist_bins or hist_offset != self.hist_offset:
            self.hist_bins = hist_bins
            self.hist_offset = hist_offset
            self.hist_max = 4096
            self.hist_buf = self.empty((self.hist_max, hist_bins), dtype=np.int32)
            self.hist_idx = 0
            self.hist_map = dict()

    def gen_rng(self, seed=None):
        if False:
            i = 10
            return i + 15
        '\n        Generate the random number generator on host.\n\n        Arguments:\n            seed (int): random number generator seed\n\n        Returns:\n            seeded numpy RNG\n        '
        self.rng = np.random.RandomState(seed)
        self.init_rng_state = self.rng_get_state()
        return self.rng

    def rng_set_state(self, state):
        if False:
            return 10
        '\n        Set the RNG state for host RNG.\n\n        Arguments:\n            state (np.array): numpy random number state vector\n        '
        self.rng.set_state(state)

    def rng_get_state(self):
        if False:
            while True:
                i = 10
        '\n        Return the current state of the on-host RNG.\n\n        Returns:\n            np.array: the on-host RNG state vectors\n        '
        return self.rng.get_state()

    def rng_reset(self):
        if False:
            print('Hello World!')
        '\n        Reset the random state to the state where the Backend is first\n        initialized.\n        '
        self.rng_set_state(self.init_rng_state)

    def fill_normal(self, ary, mean=0, stdv=1):
        if False:
            i = 10
            return i + 15
        '\n        Fill ary with normally distributed random numbers.\n\n        Arguments:\n            ary (Tensor): Tensor to fill with random values\n            mean (float): Mean value. Default 0\n            stdv (float): standard deviation value.  Default 1\n        '
        ary[:] = np.random.standard_normal(ary.shape) * stdv + mean

    def execute(self, optree, numpy_call_dict=numpy_call_dict_cpu):
        if False:
            print('Hello World!')
        '\n        Execute the optree. Break optree into sub-optrees if necessary.\n\n        Arguments:\n            optree: (OpTreeNode): the OpTreeNode object that represents all\n                                  the operations\n        '
        if len(optree) == 3 and isinstance(optree[2], OpTreeNode) and (optree[2][0]['op'] == 'onehot'):
            assert optree[0]['op'] == 'assign'
            assert isinstance(optree[1], Tensor)
            array_output = optree[1]._tensor
            numpy_axis = optree[2][0]['axis']
            numpy_ind0 = optree[2][0]['idx']._tensor.squeeze()
            numpy_ind_len = numpy_ind0.size
            numpy_ind1 = list(range(numpy_ind_len))
            numpy_ind = np.zeros((2, numpy_ind_len), dtype=np.int32)
            numpy_ind[numpy_axis] = numpy_ind0
            numpy_ind[1 - numpy_axis] = numpy_ind1
            array_output[:] = 0
            array_output[numpy_ind.tolist()] = 1
            return array_output
        postfix_stack = optree.traverse(list())
        compute_stack = []
        for p in postfix_stack:
            if isinstance(p, dict):
                if p['op'] in OpCollection.unary_ops:
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](left))
                elif p['op'] in OpCollection.binary_ops:
                    right = compute_stack.pop()
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](left, right))
                elif p['op'] in OpCollection.reduction_ops:
                    left = compute_stack.pop()
                    compute_stack.append(numpy_call_dict[p['op']](p, left))
                elif p['op'] in OpCollection.zero_operand_ops:
                    compute_stack.append(numpy_call_dict[p['op']](None))
                else:
                    raise NotImplementedError
            elif isinstance(p, CPUTensor):
                compute_stack.append(p._tensor)
            else:
                compute_stack.append(p)
        assert len(compute_stack) == 1
        return postfix_stack[0]

    def empty(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a new instance of the CPUTensor class without initializing\n        individual element values.\n\n        Arguments:\n            shape (int, list): The size of each dimension of the Tensor.\n\n            dtype (dtype, optional): Element data type.  If not specified we\n                                     use default_dtype value\n\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n\n        Returns:\n            CPUTensor: newly created data structure reference\n        '
        dtype = self.default_dtype if dtype is None else dtype
        try:
            ary = np.zeros(shape, dtype)
        except ValueError:
            raise ValueError('Invalid shape or dtype. shape: {shape} dtype: {dtype}'.format(shape=shape, dtype=dtype))
        return self.tensor_cls(backend=self, ary=ary, dtype=dtype, name=name, persist_values=persist_values)

    def array(self, ary, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            while True:
                i = 10
        "\n        Instantiate a new instance of the CPUTensor class setting each element\n        value to what is specified in ary.\n\n        Arguments:\n            ary (numpy.ndarray): The data structure containing element values\n                                 spread across a number of dimensions.  Python\n                                 built-in types like ints and lists are\n                                 supported.\n            dtype (dtype, optional): Element data type.  If not specified we\n                                     use default_dtype value ('float32'\n                                     unless overridden).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n\n        Returns:\n            CPUTensor: newly created data structure reference\n        "
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(backend=self, ary=np.array(ary, dtype), dtype=dtype, name=name, persist_values=persist_values)

    def zeros(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Instantiate a new instance of the CPUTensor class setting each element\n        value to 0.\n\n        Arguments:\n            shape (list of ints): The size of each dimension of the Tensor.\n            dtype (dtype, optional): Element data type.  If not specified we\n                                     use default_dtype value ('float32'\n                                     unless overridden).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n\n        Returns:\n            CPUTensor: newly created data structure reference\n        "
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(backend=self, ary=np.zeros(shape, dtype), dtype=dtype, name=name, persist_values=persist_values)

    def ones(self, shape, dtype=None, name=None, persist_values=True, parallel=False, distributed=False):
        if False:
            while True:
                i = 10
        "\n        Instantiate a new instance of the CPUTensor class setting each element\n        value to 1.\n\n        Arguments:\n            shape (list of ints): The size of each dimension of the Tensor.\n            dtype (dtype, optional): Element data type.  If not specified we\n                                     use default_dtype value ('float32'\n                                     unless overridden).\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n\n        Returns:\n            CPUTensor: newly created data structure reference\n        "
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(backend=self, ary=np.ones(shape, dtype), dtype=dtype, name=name, persist_values=persist_values)

    def empty_like(self, ary, dtype=None, name=None, persist_values=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Instantiate a new instance of this backend's Tensor class, with the\n        shape taken from ary.\n\n        Arguments:\n            ary (tensor object): Tensor to inherit the dimensions of.\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n        Returns:\n            Tensor: array object\n        "
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(backend=self, ary=np.zeros(ary.shape, dtype), dtype=dtype, name=name, persist_values=persist_values)

    def zeros_like(self, ary, dtype=None, name=None, persist_values=True):
        if False:
            i = 10
            return i + 15
        "\n        Instantiate a new instance of this backend's Tensor class, with the\n        shape taken from ary and populating each element with a value of 0.\n\n        Arguments:\n            ary (tensor object): Tensor to inherit the dimensions of.\n            dtype (data-type, optional): If present, specifies the underlying\n                                         type to employ for each element.\n            persist_values (bool, optional): If set to True (the default), the\n                                             values assigned to this Tensor\n                                             will persist across multiple begin\n                                             and end calls.  Setting to False\n                                             may provide a performance increase\n                                             if values do not need to be\n                                             maintained across such calls\n        Returns:\n            Tensor: array object\n        "
        dtype = self.default_dtype if dtype is None else dtype
        return self.tensor_cls(backend=self, ary=np.zeros(ary.shape, dtype), dtype=dtype, name=name, persist_values=persist_values)

    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None):
        if False:
            while True:
                i = 10
        '\n        Doing following operations (* is dot product)\n        C = alpha * A * B   + beta * C\n        C = alpha * A.T * B + beta * C\n        C = alpha * A * B.T + beta * C.\n\n        relu: if true applied before output (and prior to beta addition)\n\n        The operation will be short-circuited to: out <- alpha * left * right\n        if beta has value 0 (the default).\n\n        Arguments:\n            A, B (CPUTensor): input operands\n            C (CPUTensor): output\n            alpha (float): scale A*B term\n            beta (float): scale C term before sum\n            relu (bool): whether to apply ReLu before output\n        '
        assert A.dtype == B.dtype == C.dtype
        assert A.shape[0] == C.shape[0]
        assert B.shape[1] == C.shape[1]
        assert A.shape[1] == B.shape[0]
        if beta == 0:
            if C._tensor.flags['C_CONTIGUOUS'] is not True:
                tmp = np.empty(C.shape, dtype=C.dtype)
                np.dot(A._tensor, B._tensor, tmp)
                C._tensor[:] = tmp.copy()
            else:
                np.dot(A._tensor, B._tensor, C._tensor)
            if relu:
                self.Relu(C._tensor, C._tensor)
        else:
            np.multiply(C._tensor, beta, C._tensor)
            tmp = np.empty(C.shape, dtype=C.dtype)
            np.dot(A._tensor, B._tensor, tmp)
            np.multiply(tmp, alpha, tmp)
            if relu:
                self.Relu(tmp, tmp)
            np.add(C._tensor, tmp, C._tensor)
        if bsum is not None:
            bsum[:] = self.sum(C, 1)
        return C

    def batched_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False):
        if False:
            i = 10
            return i + 15
        '\n        Doing following operations:\n        1 For fprop: A(K, C), B(X,C,N), C(X,K,N) --> call batched_dot(A, B, C)\n        2 For bprop: A(K, C), B(X,K,N), C(X,C,N) --> call batched_dot(A.T, B, C)\n        3 For update: A(X,K,N), B(X,C,N), C(K,C) --> call batched_dot(A, B.T, C).\n\n        Arguments:\n            A, B (CPUTensor): input operands\n            C (CPUTensor): output\n            alpha, beta, relu: see usage in dot()\n        '
        assert A.dtype == B.dtype == C.dtype
        (dima, dimb, dimc) = (0, 0, 0)
        (batch_grid, batch_loops) = (1, 1)
        if len(A.shape) == 3:
            dima = 1
        if len(B.shape) == 3:
            dimb = 1
        assert dima or dimb, 'Tensor A or B must have 3 dims to use batched_dot'
        if len(C.shape) == 3:
            dimc = 1
            batch_grid = C.shape[0]
            assert not dima or A.shape[0] == batch_grid
            assert not dimb or B.shape[0] == batch_grid
        if dima:
            batch_loops = A.shape[0]
            assert not dimb or B.shape[0] == batch_loops
        elif dimb:
            batch_loops = B.shape[0]
            assert not dima or A.shape[0] == batch_loops
        assert A.shape[0 + dima] == C.shape[0 + dimc]
        assert B.shape[1 + dimb] == C.shape[1 + dimc]
        assert A.shape[1 + dima] == B.shape[0 + dimb]
        tmp = np.zeros(C.shape)
        for i in range(batch_loops):
            if dima:
                tmp += np.dot(A._tensor[i], B._tensor[i])
            else:
                tmp[i] = np.dot(A._tensor, B._tensor[i])
        np.multiply(tmp, alpha, tmp)
        if relu:
            self.Relu(tmp, tmp)
        np.add(C._tensor * beta, tmp, C._tensor)
        return C

    def xnor_compound_dot(self, A, B, C, beta=0.0, bsum=None):
        if False:
            print('Hello World!')
        '\n        Performs XNOR GEMM\n        C = A * B\n\n        Arguments:\n            A (Tensor): left-hand side operand.\n            B (Tensor): right-hand side operand.\n            C (Tensor): output operand\n        '
        assert A.dtype == B.dtype == C.dtype
        assert A.shape[0] == C.shape[0]
        assert B.shape[1] == C.shape[1]
        assert A.shape[1] == B.shape[0]
        np.dot(A._tensor, B._tensor, C._tensor)
        if bsum is not None:
            bsum[:] = self.sum(C, 1)
        return C

    def copy_transpose(self, a, out, axes=None, repeat=1):
        if False:
            while True:
                i = 10
        '\n        Function to perform a fast copy transpose/dimshuffle operation.\n        Works just like numpy.transpose, but requires an output tensor argument.\n        '
        out._tensor[:] = np.transpose(a._tensor, axes).copy()

    def make_binary_mask(self, out, keepthresh=0.5):
        if False:
            print('Hello World!')
        '\n        Create a binary mask for dropout layers.\n\n        Arguments:\n            out (CPUTensor): Output tensor\n            keepthresh (float): fraction of ones\n        '
        out._tensor[:] = np.array(self.rng.uniform(size=out._tensor.shape) < keepthresh, dtype=out._tensor.dtype)

    def conv_layer(self, dtype, N, C, K, D=1, H=1, W=1, T=1, R=1, S=1, pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1, dil_d=1, dil_h=1, dil_w=1):
        if False:
            while True:
                i = 10
        '\n        Create a new ConvLayer parameter object.\n        This then is passed as an argument to all the convolution operations.\n\n        N: Number of images in mini-batch\n        C: Number of input feature maps\n        K: Number of output feature maps\n\n        D: Depth  of input image\n        H: Height of input image\n        W: Width  of input image\n\n        T: Depth  of filter kernel\n        R: Height of filter kernel\n        S: Width  of filter kernel\n\n        padding: amount of zero-padding around the given edge\n        strides: factor to step the filters by in a given direction\n        dilation: dilation factor for each dimension\n\n        dtype: need to know dtype to setup proper kernels and params.\n\n        bsum: calculate the sum along the batchnorm axis for fprop or bprop\n              outputs an fp32 tensor of size Kx1\n\n        '
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w)

    def fprop_conv(self, layer, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False, slope=0.0, layer_op=None):
        if False:
            i = 10
            return i + 15
        '\n        Forward propagate the inputs of a convolutional network layer to\n        produce output.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            I (CPUTensor): inputs\n            F (CPUTensor): the weights (filters)\n            O (CPUTensor): outputs\n\n        Compounding Options:\n            X: tensor to use in bprop_relu or beta\n                can be same as O for beta accumulate (this is default when None)\n                should be same shape as O\n            bias: (K,1) tensor to use for adding bias to output\n                O += bias\n            bsum: (K,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)\n                bsum = sum(O.reshape(K,-1), axis=1)\n                the sum operation is fully deterministic\n            alpha, beta:\n                O = alpha*O + beta*X\n                O = alpha*O + beta*O   (if X==O)\n            relu, slope: boolean flag to apply:\n                O = max(O, 0) + beta*min(O, 0)\n                can be combined with bias (where bias is added first)\n            brelu, slope: boolean flag to apply:\n                O *= (X > 0) + beta*(X < 0)\n                can be combined with bsum tensor to output bprop_bias\n        '
        layer.xprop_conv(I, F, O, X, bias, bsum, alpha, beta, relu, brelu, slope, layer_op=layer)

    def bprop_conv(self, layer, F, E, grad_I, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False, slope=0.0, layer_op=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Backward propagate the error through a convolutional network layer.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            F (CPUTensor): the weights (filters)\n            E (CPUTensor): errors\n            grad_I (CPUTensor): gradient to inputs (output delta)\n\n        Compounding Options:\n            X: tensor to use in bprop_relu or beta\n                can be same as grad_I for beta accumulate (this is default when None)\n                should be same shape as grad_I\n            bias: (K,1) tensor to use for adding bias to output\n                grad_I += bias\n            bsum: (K,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)\n                bsum = sum(grad_I.reshape(K,-1), axis=1)\n                the sum operation is fully deterministic\n            alpha, beta:\n                grad_I = alpha*grad_I + beta*X\n                grad_I = alpha*grad_I + beta*grad_I   (if X==grad_I)\n            relu, slope: boolean flag to apply:\n                grad_I = max(grad_I, 0) + slope*min(grad_I, 0)\n                can be combined with bias (where bias is added first)\n            brelu, slope: boolean flag to apply:\n                grad_I *= (X > 0) + slope*(X < 0)\n                can be combined with bsum tensor to output bprop_bias\n        '
        layer.xprop_conv(E, F, grad_I, X, bias, bsum, alpha, beta, relu, brelu, slope, backward=True, layer_op=layer)

    def update_conv(self, layer, I, E, U, alpha=1.0, beta=0.0, grad_bias=None, layer_op=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the updated gradient for a convolutional network layer.\n\n        Arguments:\n            layer: the conv layer as a parameter object\n            I (CPUTensor): the inputs\n            E (CPUTensor): the errors\n            U (CPUTensor): the updates\n            alpha (float): linear scaling\n            beta  (float): scaled accumulation\n        '
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == U.size
        layer.update_conv(I, E, U, alpha, beta, grad_bias=grad_bias, layer_op=layer_op)

    def deconv_layer(self, dtype, N, C, K, M, P, Q, T=1, R=1, S=1, pad_d=0, pad_h=0, pad_w=0, str_d=1, str_h=1, str_w=1, dil_d=1, dil_h=1, dil_w=1):
        if False:
            while True:
                i = 10
        '\n        Create a new DeconvLayer parameter object.\n        This then is passed as an argument to all the convolution operations.\n\n        N: Number of images in mini-batch\n        C: Number of output feature maps\n        K: Number of input feature maps\n\n        M: Depth  of input\n        P: Height of input\n        Q: Width of input\n\n        D: Depth  of output image\n        H: Height of output image\n        W: Width  of output image\n\n        T: Depth  of filter kernel\n        R: Height of filter kernel\n        S: Width  of filter kernel\n\n        padding: amount of zero-padding around the given edge\n        strides: factor to step the filters by in a given direction\n        dilation: dilation factor for each dimension\n\n        dtype: need to know dtype to setup proper kernels and params.\n        '
        return DeconvLayer(self, dtype, N, C, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w)

    def lrn_layer(self, dtype, N, C, D=1, H=1, W=1, J=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new PoolLayer parameter object.\n        This then is passed as an argument to all pooling kernels.\n\n        N: Number of images in mini-batch\n\n        C: Number of input feature maps\n        H: Height of input image\n        W: Width  of input image\n\n        J: Size of feature map pooling window (maxout n_pieces)\n\n        padding: amount of zero-padding around the given image or feature map edge\n        strides: factor to step the window by in a given direction (overlap allowed)\n\n        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.\n        '
        assert J % 2 == 1, 'Only support odd LRN window size'
        pad_c = J // 2
        op = 'lrn'
        lrn_opts = dict(T=1, R=1, S=1, pad_c=pad_c, pad_d=0, pad_h=0, pad_w=0, str_c=1, str_d=1, str_h=1, str_w=1)
        return PoolLayer(self, dtype, op, N, C, D, H, W, J, **lrn_opts)

    def fprop_lrn(self, layer, I, O, denom, alpha=None, beta=None, ascale=1, bpower=1):
        if False:
            i = 10
            return i + 15
        '\n        Forward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object, different backends have\n                               different pool layers.\n            I (Tensor): Input tensor.\n            O (Tensor): output tensor.\n            denom (Tensor): denominator tensor, stores the result of the squared pooling/contrast\n            ascale (float): scaling parameter (alpha) to multiply the pooled sum (1.25e-5 in AK)\n            bpower (float): exponential parameter (beta) to raise denominator by (0.75 in AK)\n        '
        assert layer.sizeI == I.size
        assert layer.sizeO == O.size
        (J, T, R, S) = layer.JTRS
        (C, D, H, W, N) = layer.dimI
        (K, M, P, Q, N) = layer.dimO
        (pad_c, pad_d, pad_h, pad_w) = layer.padding
        (str_c, str_d, str_h, str_w) = layer.strides
        array_I = I._tensor.reshape(layer.dimI)
        array_O = O._tensor.reshape(layer.dimO)
        array_d = denom._tensor.reshape(layer.dimO)
        for k in range(K):
            (sliceC, _) = layer.kSlice[k]
            _ascale = ascale / J
            for m in range(M):
                (sliceD, _) = layer.mSlice[m]
                for p in range(P):
                    (sliceH, _) = layer.pSlice[p]
                    for q in range(Q):
                        (sliceW, _) = layer.qSlice[q]
                        sliceI = array_I[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        array_d[k, m, p, q, :] = 1 + _ascale * np.sum(np.square(sliceI), axis=0)
        array_O[:] = array_I * np.power(array_d, -bpower)

    def bprop_lrn(self, layer, I, O, E, delta, denom, alpha=None, beta=None, ascale=1, bpower=1):
        if False:
            while True:
                i = 10
        '\n        Backward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object. Different backends have\n                               different pool layers.\n            I (Tensor): Input tensor.\n            E (Tensor): Error tensor.\n            delta (Tensor): Gradient tensor (delta)\n            denom (Tensor): denominator tensor computed during bprop\n            ascale (float): scaling parameter (alpha) to multiply the pooled sum (1.25e-5 in AK)\n            bpower (float): exponential parameter (beta) to raise denominator by (0.75 in AK)\n        '
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeI == delta.size
        (J, T, R, S) = layer.JTRS
        (C, D, H, W, N) = layer.dimI
        (K, M, P, Q, N) = layer.dimO
        (pad_c, pad_d, pad_h, pad_w) = layer.padding
        (str_c, str_d, str_h, str_w) = layer.strides
        array_I = I._tensor.reshape(layer.dimI)
        array_E = E._tensor.reshape(layer.dimO)
        array_O = O._tensor.reshape(layer.dimO)
        array_delta = delta._tensor.reshape(layer.dimI)
        array_denom = denom._tensor.reshape(layer.dimO)
        for k in range(K):
            (sliceC, _) = layer.kSlice[k]
            for m in range(M):
                (sliceD, _) = layer.mSlice[m]
                for p in range(P):
                    (sliceH, _) = layer.pSlice[p]
                    for q in range(Q):
                        (sliceW, _) = layer.qSlice[q]
                        _O = array_O[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        _E = array_E[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        _den = array_denom[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        array_delta[k, m, p, q, :] = np.sum(_O * _E / _den, axis=0)
        array_delta[:] = -2 * bpower * (ascale / float(J)) * array_delta * array_I + array_E * np.power(array_denom, -bpower)

    def pool_layer(self, dtype, op, N, C, D=1, H=1, W=1, J=1, T=1, R=1, S=1, pad_c=0, pad_d=0, pad_h=0, pad_w=0, str_c=None, str_d=None, str_h=None, str_w=None):
        if False:
            return 10
        '\n        Create a new PoolLayer parameter object.\n        This then is passed as an argument to all pooling kernels.\n\n        op: "max", "avg", "l2" pooling (currently bprop only supports max, but not avg and l2)\n        N: Number of images in mini-batch\n\n        C: Number of input feature maps\n        D: Depth  of input image\n        H: Height of input image\n        W: Width  of input image\n\n        J: Size of feature map pooling window (maxout n_pieces)\n        T: Depth  of pooling window\n        R: Height of pooling window\n        S: Width  of pooling window\n\n        padding: amount of zero-padding around the given image or feature map edge\n        strides: factor to step the window by in a given direction (overlap allowed)\n\n        Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.\n        '
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S
        return PoolLayer(self, dtype, op, N, C, D, H, W, J, T, R, S, pad_c, pad_d, pad_h, pad_w, str_c, str_d, str_h, str_w)

    def fprop_pool(self, layer, I, O, argmax=None, beta=0.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Forward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object, different backends have\n                               different pool layers.\n            I (Tensor): Input tensor.\n            O (Tensor): output tensor.\n            argmax (Tensor): tensor to store location of the maximum\n        '
        assert layer.sizeI == I.size
        assert layer.sizeO == O.size
        if layer.op == 'max':
            assert layer.sizeO == argmax.size
        op = layer.op
        (J, T, R, S) = layer.JTRS
        (C, D, H, W, N) = layer.dimI
        (K, M, P, Q, N) = layer.dimO
        (pad_c, pad_d, pad_h, pad_w) = layer.padding
        (str_c, str_d, str_h, str_w) = layer.strides
        array_I = I._tensor.reshape(layer.dimI)
        array_O = O._tensor.reshape(layer.dimO)
        if op == 'max':
            array_argmax = argmax._tensor.reshape(layer.dimO)
        for k in range(K):
            (sliceC, _) = layer.kSlice[k]
            for m in range(M):
                (sliceD, _) = layer.mSlice[m]
                for p in range(P):
                    (sliceH, _) = layer.pSlice[p]
                    for q in range(Q):
                        (sliceW, _) = layer.qSlice[q]
                        sliceI = array_I[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        if op == 'max':
                            array_argmax[k, m, p, q, :] = np.argmax(sliceI, axis=0)
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + np.max(sliceI, axis=0)
                        elif op == 'avg':
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + np.mean(sliceI, axis=0)
                        elif op == 'l2':
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + np.sqrt(np.sum(np.square(sliceI), axis=0))

    def bprop_pool(self, layer, I, O, argmax=None, alpha=1.0, beta=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Backward propagate pooling layer.\n\n        Arguments:\n            layer (PoolLayer): The pool layer object. Different backends have\n                               different pool layers.\n            I (Tensor): Input (error) tensor.\n            O (Tensor): Output (delta) tensor.\n            argmax (Tensor): tensor to store location of the maximum\n            alpha (float): linear scaling (does not work for l2 pooling)\n            beta (float): accumulation value into grad_I\n        '
        assert layer.sizeI == O.size
        assert layer.sizeO == I.size
        if layer.op == 'max':
            assert layer.sizeO == argmax.size
        op = layer.op
        (J, T, R, S) = layer.JTRS
        (C, D, H, W, N) = layer.dimI
        (K, M, P, Q, N) = layer.dimO
        (pad_c, pad_d, pad_h, pad_w) = layer.padding
        (str_c, str_d, str_h, str_w) = layer.strides
        array_E = I._tensor.reshape(layer.dimO)
        array_E[:] = array_E * alpha
        array_delta = O._tensor.reshape(layer.dimI)
        array_delta[:] = array_delta * beta
        if op == 'max':
            array_argmax = argmax._tensor.reshape(layer.dimO)
        for k in range(K):
            (sliceC, clen) = layer.kSlice[k]
            for m in range(M):
                (sliceD, dlen) = layer.mSlice[m]
                for p in range(P):
                    (sliceH, hlen) = layer.pSlice[p]
                    for q in range(Q):
                        (sliceW, wlen) = layer.qSlice[q]
                        patch_in = (sliceC, sliceD, sliceH, sliceW, slice(None))
                        patch_out = (k, m, p, q, slice(None))
                        sliceB = array_delta[patch_in].reshape((-1, N))
                        if op == 'max':
                            max_n = array_argmax[patch_out]
                            sliceB[max_n, list(range(N))] += array_E[patch_out]
                        elif op == 'avg':
                            sliceB += array_E[patch_out] * (1.0 / sliceB.shape[0])
                        else:
                            raise NotImplementedError
                        array_delta[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))

    def _roipooling_slice(self, h, stride, H, roi_offset):
        if False:
            while True:
                i = 10
        '\n        Slicing for ROIPooling along one dimension.\n        h: is the index on the pooled map (output index)\n        stride:\n        H: the max of the input map\n        roi_offset: how far hstart is from 0\n        '
        hstart = int(np.floor(float(h) * stride))
        hend = int(np.ceil(float(h + 1) * stride))
        hstart = min(max(hstart + roi_offset, 0), H)
        hend = min(max(hend + roi_offset, 0), H)
        return (slice(hstart, hend), hend - hstart)

    def roipooling_fprop(self, I, rois, O, argmax, roi_count, C, H, W, pooled_height, pooled_width, spatial_scale):
        if False:
            for i in range(10):
                print('nop')
        '\n        Function to perform fprop of ROIPooling\n\n        Arguments:\n            I (Tensor): (C, H, W, N)\n            rois (Tensor): (ROIs, 5)\n            O (Tensor): (C, pooled_height, pooled_width, roi_count)\n            argmax (Tensor): (C, pooled_height, pooled_width, roi_count)\n        '
        assert I.size == C * H * W * self.bsz, 'ROIPooling input feature map size do not match'
        assert O.size == argmax.size == C * pooled_height * pooled_width * roi_count, 'ROIPooling output shape do not match'
        assert rois.shape[1] == 5, 'ROIs should be on the row dimension'
        assert rois.shape[0] == roi_count, 'ROIs do not match with roi count'
        array_fm = I._tensor.reshape(C, H, W, self.bsz)
        array_rois = rois._tensor
        array_O = O._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_argmax = argmax._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_O[:] = 0
        array_argmax[:] = -1
        for b_id in xrange(roi_count):
            [idx, xmin, ymin, xmax, ymax] = array_rois[b_id]
            xmin = int(round(xmin * spatial_scale))
            xmax = int(round(xmax * spatial_scale))
            ymin = int(round(ymin * spatial_scale))
            ymax = int(round(ymax * spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            stride_h = float(roi_height) / float(pooled_height)
            stride_w = float(roi_width) / float(pooled_width)
            for h_out in xrange(pooled_height):
                (sliceh, lenh) = self._roipooling_slice(h_out, stride_h, H, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for w_out in xrange(pooled_width):
                    (slicew, lenw) = self._roipooling_slice(w_out, stride_w, W, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    else:
                        array_I = array_fm[:, sliceh, slicew, int(idx)].reshape(C, -1)
                        array_O[:, h_out, w_out, b_id] = np.max(array_I, axis=1)
                        max_idx_slice = np.unravel_index(np.argmax(array_I, axis=1), (lenh, lenw))
                        max_idx_slice_h = max_idx_slice[0] + sliceh.start
                        max_idx_slice_w = max_idx_slice[1] + slicew.start
                        max_idx_slice = max_idx_slice_h * W + max_idx_slice_w
                        array_argmax[:, h_out, w_out, b_id] = max_idx_slice

    def roipooling_bprop(self, I, rois, O, argmax, roi_count, C, H, W, pooled_height, pooled_width, spatial_scale):
        if False:
            return 10
        '\n        Function to perform bprop of ROIPooling.\n\n        Arguments:\n            I (Tensor): input errors (C, pooled_height, pooled_width, roi_count)\n            argmax (Tensor): max args from the fprp (C, pooled_height, pooled_width, roi_count)\n            rois (Tensor): (ROIs, 5)\n            O (Tensor): output deltas (C, H, W, N)\n        '
        assert I.size == argmax.size == C * pooled_height * pooled_width * roi_count, 'ROIPooling bprop input size do not match'
        assert O.size == C * H * W * self.bsz, 'ROIPooling bprop output size do not match'
        assert rois.shape[1] == 5, 'ROIs should be on the row dimension'
        assert rois.shape[0] == roi_count, 'ROIs do not match with roi count'
        array_E = I._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_rois = rois._tensor
        array_delta = O._tensor.reshape(C, H, W, self.bsz)
        array_argmax = argmax._tensor.reshape(C, pooled_height, pooled_width, roi_count)
        array_delta[:] = 0
        for b_id in xrange(roi_count):
            [idx, xmin, ymin, xmax, ymax] = array_rois[b_id]
            xmin = int(round(xmin * spatial_scale))
            xmax = int(round(xmax * spatial_scale))
            ymin = int(round(ymin * spatial_scale))
            ymax = int(round(ymax * spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            stride_h = float(roi_height) / float(pooled_height)
            stride_w = float(roi_width) / float(pooled_width)
            for w in range(xmin, xmax + 1):
                for h in range(ymin, ymax + 1):
                    phstart = int(np.floor(float(h - ymin) / stride_h))
                    phend = int(np.ceil(float(h - ymin + 1) / stride_h))
                    pwstart = int(np.floor(float(w - xmin) / stride_w))
                    pwend = int(np.ceil(float(w - xmin + 1) / stride_w))
                    phstart = min(max(phstart, 0), pooled_height)
                    phend = min(max(phend, 0), pooled_height)
                    pwstart = min(max(pwstart, 0), pooled_width)
                    pwend = min(max(pwend, 0), pooled_width)
                    for ph in range(phstart, phend):
                        for pw in range(pwstart, pwend):
                            max_idx_tmp = array_argmax[:, ph, pw, b_id]
                            for c in range(C):
                                if max_idx_tmp[c] == h * W + w:
                                    array_delta[c, h, w, int(idx)] += array_E[c, ph, pw, b_id]

    def nms(self, detections, threshold, normalized=False):
        if False:
            while True:
                i = 10
        '\n        Function to perform non-maximal supression.\n\n        Arguments:\n            detections (Tensor): detection boxes (box_count, 5), each row has\n                                 (x1, y1, x2, y2, score). Assume the boxes have already\n                                 been sorted based on score in descending order\n            output_mask (Tensor): pre-allocated buffer for mask output from the kernel\n            threshold (float): box overlap threshold, boxes with smaller overlaps will be kept\n            normalized (bool): whether box coordinates are normalized to image dimensions\n\n        Outputs:\n            keep_ind (list): list of indices\n        '
        if normalized is True:
            offset = 0
        else:
            offset = 1
        dets = detections.get()
        keep = np.where(dets[:, 4] != 0)[0]
        x1 = dets[keep, 0]
        y1 = dets[keep, 1]
        x2 = dets[keep, 2]
        y2 = dets[keep, 3]
        scores = dets[keep, 4]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + offset)
            h = np.maximum(0.0, yy2 - yy1 + offset)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        return keep

    def compound_fprop_bn(self, x, xsum, xvar, gmean, gvar, gamma, beta, y, eps, rho, compute_batch_sum, accumbeta=0.0, relu=False, binary=False, inference=False, outputs=None, layer=None):
        if False:
            i = 10
            return i + 15
        '\n        Function to perform batch normalization forward pass. Included\n        for API compatibility with GPU compound kernel call.\n\n        Arguments:\n            x (Tensor): Input from previous layer\n            xsum (Tensor): Precomputed batch sum over PQN dimension\n            xvar (Tensor): Buffer for variance (computed in kernel)\n            gmean (Tensor): global mean ()\n            gvar (Tensor): global variance\n            gamma (Tensor): scale parameter\n            beta (Tensor): location parameter\n            y (Tensor): normalized output\n            eps (float): constant for numerical stability\n            rho (float): exponential window averaging constant\n        '
        if inference:
            xhat = (x - gmean) / self.sqrt(gvar + eps)
            y[:] = y * accumbeta + xhat * gamma + beta
            return
        if compute_batch_sum:
            xsum[:] = self.sum(x, axis=1)
        xvar[:] = self.var(x, axis=1, binary=binary)
        xsum[:] = xsum / x.shape[1]
        gmean[:] = gmean * rho + (1.0 - rho) * xsum
        gvar[:] = gvar * rho + (1.0 - rho) * xvar
        if binary:
            xhat = self.shift(x - xsum, 1.0 / self.sqrt(xvar + eps))
            outputs = y.reshape(xhat.shape)
            outputs[:] = self.shift(xhat, gamma) + beta + accumbeta * outputs
        else:
            xhat = (x - xsum) / self.sqrt(xvar + eps)
            outputs = y.reshape(xhat.shape)
            outputs[:] = xhat * gamma + beta + accumbeta * outputs

    def compound_bprop_bn(self, delta_out, grad_gamma, grad_beta, delta_in, x, xsum, xvar, gamma, eps, binary=False, layer=None):
        if False:
            print('Hello World!')
        '\n        Function to perform batch normalization backward pass. Included\n        for API compatibility with GPU compound kernel call.\n\n        Arguments:\n            delta_out (Tensor): Delta buffer to write out to\n            grad_gamma (Tensor): Gradient w.r.t. gamma\n            grad_beta (Tensor): Gradient w.r.t. beta\n            delta_in (Tensor): Delta buffer to read from (incoming errors)\n            x (Tensor): feedforward input\n            xsum (Tensor): Batch sum over PQN dimension\n            xvar (Tensor): Batch variance\n            gamma (Tensor): scale parameter\n            eps (float): constant for numerical stability\n            binary (bool): Binary shift based computations\n        '
        if binary:
            op = self.shift
        else:

            def multiply(left, right):
                if False:
                    for i in range(10):
                        print('nop')
                return left * right
            op = multiply
        inv_v = 1.0 / self.sqrt(xvar + eps)
        xhat = op(x - xsum, inv_v)
        grad_gamma[:] = self.sum(xhat * delta_in, axis=1)
        grad_beta[:] = self.sum(delta_in, axis=1)
        xtmp = (op(xhat, grad_gamma) + grad_beta) / float(x.shape[1])
        delta_out.reshape(delta_in.shape)[:] = op(op(delta_in - xtmp, gamma), inv_v)

    def compound_bprop_lut(self, nin, inputs, error, error_t, dW, pad_idx, alpha=1.0, beta=0):
        if False:
            i = 10
            return i + 15
        '\n        Backward propagate lookup table layer.\n\n        Arguments:\n            nin (int): Number of input word_ids.\n            inputs (Tensor): Input tensor.\n            error (Tensor): Error tensor.\n            error_t (Tensor): Transposed error tensor.\n            dW (Tensor): Gradient tensor (delta).\n            pad_idx (int):\n            alpha (float):\n            beta (float):\n        '
        wrd_ids = inputs._tensor[0]
        (unqidx, inv) = np.unique(wrd_ids, return_inverse=True)
        groups = [np.where(inv == i) for i in range(len(unqidx))]
        for (wrd_id, group) in zip(unqidx, groups):
            if wrd_id != pad_idx:
                dW[wrd_id, :] = self.sum(error.take(group[0], axis=1), axis=1)
        '\n        alternative bprop\n        for (j, wrd_id) in enumerate(wrd_ids):\n            dW[:, wrd_id] = dW[:, wrd_id] + error[:, j]\n        '

    def _hist_tensor(self, tag):
        if False:
            i = 10
            return i + 15
        '\n        Create a tensor the right size for histogram data, with memory allocated\n        in the contiguous histogram buffer. Track it by tag for later reference.\n        '
        assert self.hist_idx < self.hist_max
        self.hist_map[tag] = self.hist_idx
        hist_buf = self.hist_buf[self.hist_idx]
        self.hist_idx += 1
        return hist_buf

    def dump_hist_data(self):
        if False:
            print('Hello World!')
        hist_data = self.hist_buf
        hist_map = self.hist_map
        self.hist_map = dict()
        self.hist_idx = 0
        self.hist_buf = self.empty((self.hist_max, self.hist_bins), dtype=np.int32)
        return (hist_data, hist_map)

    def Relu(self, ary, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the ReLu transformation for input array.\n\n        Arguments:\n            ary: numpy array\n            out: reference to output\n        '
        if out is not None:
            return np.maximum(ary, 0, out)
        else:
            return np.maximum(ary, 0)

    def binarize(self, ary, out, stochastic=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Binarizes input array\n\n        Arguments:\n            ary: tensor\n            out: reference to output\n            stochastic: stochastic or deterministic\n        '
        if stochastic:
            out[:] = (ary + 1) / 2.0
            self.clip(out, 0, 1, out)
            prob = self.array(np.random.uniform(0, 1, size=ary.shape))
            self.less_equal(prob, out, out)
        else:
            self.greater_equal(ary, 0, out)
        out[:] = 2 * out - 1
        return out

    def shift(self, ary, shift_ary, value=True, out=None):
        if False:
            i = 10
            return i + 15
        '\n        Shifts input array\n\n        Arguments:\n            ary: tensor\n            shift_ary: tensor of shift amount\n            out: reference to output\n        '
        if value:
            exp = self.rint(self.safelog(self.absolute(shift_ary)) / self.log(2))
            ap2 = self.multiply(self.sgn(shift_ary), self.exp2(exp))
        else:
            ap2 = self.exp2(shift_ary)
        if out is None:
            if hasattr(ary, 'shape'):
                out = self.empty_like(ary)
            else:
                out = self.empty((1, 1))
        out[:] = self.multiply(ary, ap2)
        return out

    def init_mark(self):
        if False:
            i = 10
            return i + 15
        '\n        Generate a timing mark object.\n\n        Returns:\n            timing mark (dict)\n        '
        return {'time': 0}

    def record_mark(self, marker):
        if False:
            while True:
                i = 10
        '\n        Mark the current time.\n\n        Arguments:\n            marker (time mark): timing mark generated by init_mark()\n        '
        marker['time'] = time.time()

    def synchronize_mark(self, marker):
        if False:
            for i in range(10):
                print('nop')
        '\n        Synchronize on the given marker.\n\n        Arguments:\n            marker (time mark): timing mark generated by init_mark()\n        '
        return

    def get_time(self, start, end):
        if False:
            print('Hello World!')
        '\n        Return time between start and end marks.\n\n        Arguments:\n            start (time maker): start time mark\n\n            end (time marker): end time mark\n\n        Returns:\n            time elapsed between start and end time marks in milliseconds\n        '
        return (end['time'] - start['time']) * 1000.0

    def relu_layer(self):
        if False:
            print('Hello World!')
        return None

    def fprop_relu(self, layer, x, slope):
        if False:
            return 10
        return self.maximum(x, 0) + slope * self.minimum(0, x)

    def bprop_relu(self, layer, x, error, deltas, slope):
        if False:
            return 10
        return self.greater(x, 0) + slope * self.less(x, 0)

    def fprop_softmax(self, x, axis):
        if False:
            for i in range(10):
                print('nop')
        return self.reciprocal(self.sum(self.exp(x - self.max(x, axis=axis)), axis=axis)) * self.exp(x - self.max(x, axis=axis))

    def batchnorm_layer(self, in_shape):
        if False:
            return 10
        return None

    def fprop_transform(self, ngLayer, transform, inputs, outputs, relu=False):
        if False:
            print('Hello World!')
        outputs[:] = transform(inputs)

    def bprop_transform(self, ngLayer, transform, outputs, error, deltas, relu):
        if False:
            i = 10
            return i + 15
        deltas[:] = transform.bprop(outputs) * error

    def fprop_skipnode(self, x, y, beta):
        if False:
            print('Hello World!')
        y[:] = y * beta + x

    def bprop_skipnode(self, error, deltas, alpha, beta):
        if False:
            print('Hello World!')
        deltas[:] = deltas * beta + alpha * error

    def mergesum_layer(self, layer_num):
        if False:
            i = 10
            return i + 15
        return None

    def fprop_mergesum(self, ngLayer, inputs, inference, layers, outputs, out_shape):
        if False:
            while True:
                i = 10
        for l in layers:
            beta = 0 if l is layers[0] else 1
            l.fprop(inputs, inference, beta=beta)

    def bprop_mergesum(self, ngLayer, alpha, beta, layers, error, deltas):
        if False:
            for i in range(10):
                print('nop')
        for l in reversed(layers):
            b = beta if l is layers[-1] else 1
            l.bprop(error, alpha=alpha, beta=b)

    def mergebroadcast_layer(self, layer_num):
        if False:
            for i in range(10):
                print('nop')
        return None

    def fprop_mergebroadcast(self, ngLayer, inputs, inference, outputs, layers, out_shape):
        if False:
            for i in range(10):
                print('nop')
        for l in layers:
            l.fprop(inputs, inference)

    def bprop_mergebroadcast(self, ngLayer, layers, error_views, error, delta, out_shape, alpha, beta, alphas, betas):
        if False:
            for i in range(10):
                print('nop')
        betas[-1] = beta
        for (l, e, a, b) in reversed(list(zip(layers, error_views, alphas, betas))):
            l.bprop(e, alpha=a * alpha, beta=b)