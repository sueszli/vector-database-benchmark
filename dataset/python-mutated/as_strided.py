import numpy as np
import six
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
index_dtype = {t().itemsize: t for t in np.sctypes['int']}

def _byte2step(iterable, itemsize):
    if False:
        i = 10
        return i + 15
    for i in iterable:
        assert i % itemsize == 0
    return tuple([i // itemsize for i in iterable])

def _step2byte(iterable, itemsize):
    if False:
        for i in range(10):
            print('nop')
    return tuple([i * itemsize for i in iterable])

def _maybe_overlapping_memory(shape, strides):
    if False:
        for i in range(10):
            print('nop')
    'Returns bool value indicating the array with such shape and strides\n    might have overlapping memory.\n\n    Args:\n    shape (tuple of int): The shape of output.\n    strides (tuple of int): The strides of output, given in the unit of steps.\n    storage_offset (int):\n        The offset between the head of allocated memory and the pointer of\n        first element, given in the unit of steps.\n\n    Returns:\n        bool: Existence of the overlapping memory\n    '
    max_ptr_in_slice = 0
    for (stride, size) in sorted(zip([abs(s) for s in strides], shape)):
        if stride <= max_ptr_in_slice:
            return True
        max_ptr_in_slice += stride * (size - 1)
    return False

def _min_index(shape, strides, storage_offset):
    if False:
        for i in range(10):
            print('nop')
    'Returns the leftest index in the array (in the unit-steps)\n\n    Args:\n        shape (tuple of int): The shape of output.\n        strides (tuple of int):\n            The strides of output, given in the unit of steps.\n        storage_offset (int):\n            The offset between the head of allocated memory and the pointer of\n            first element, given in the unit of steps.\n\n    Returns:\n        int: The leftest pointer in the array\n    '
    sh_st_neg = [sh_st for sh_st in zip(shape, strides) if sh_st[1] < 0]
    if not sh_st_neg:
        return storage_offset
    else:
        return storage_offset + six.moves.reduce(lambda base, sh_st: base + (sh_st[0] - 1) * sh_st[1], sh_st_neg, 0)

def _max_index(shape, strides, storage_offset):
    if False:
        i = 10
        return i + 15
    'Returns the rightest index in the array\n\n    Args:\n        shape (tuple of int): The shape of output.\n        strides (tuple of int): The strides of output, given in unit-steps.\n        storage_offset (int):\n            The offset between the head of allocated memory and the pointer of\n            first element, given in the unit of steps.\n\n    Returns:\n        int: The rightest pointer in the array\n    '
    sh_st_pos = [sh_st for sh_st in zip(shape, strides) if sh_st[1] > 0]
    if not sh_st_pos:
        return storage_offset
    else:
        return storage_offset + six.moves.reduce(lambda base, sh_st: base + (sh_st[0] - 1) * sh_st[1], sh_st_pos, 0)

def _index_add(augend, indices, addend):
    if False:
        i = 10
        return i + 15
    'Wrapper of :func:`cupyx.scatter_add` and :func:`numpy.add.at`\n\n    Args:\n        augend (:class:`numpy.ndarray` or :class:`cupy.ndarray`):\n            The array modified in-place.\n        indices (:class:`numpy.ndarray` or :class:`cupy.ndarray`):\n            The indices of ``augend``. The shape is the same to the ``addend``.\n        addend (:class:`numpy.ndarray` or :class:`cupy.ndarray`):\n            The array to be added.\n\n    Returns:\n        None\n    '
    if isinstance(augend, cuda.ndarray):
        cuda.cupyx.scatter_add(augend, indices, addend)
    elif isinstance(augend, np.ndarray):
        np.add.at(augend, indices, addend)

def _get_base_array(array):
    if False:
        while True:
            i = 10
    'Get the founder of :class:`numpy.ndarray`.\n\n    Args:\n        array (:class:`numpy.ndarray`):\n            The view of the base array.\n\n    Returns:\n        :class:`numpy.ndarray`:\n            The base array.\n    '
    base_array_candidate = array
    while base_array_candidate.base is not None:
        base_array_candidate = base_array_candidate.base
    return base_array_candidate

def _stride_array(array, shape, strides, storage_offset):
    if False:
        return 10
    'Wrapper of :func:`numpy.lib.stride_tricks.as_strided`.\n\n    .. note:\n        ``strides`` and ``storage_offset`` is given in the unit of steps\n        instead the unit of bytes. This specification differs from that of\n        :func:`numpy.lib.stride_tricks.as_strided`.\n\n    Args:\n        array (:class:`numpy.ndarray` of :class:`cupy.ndarray`):\n            The base array for the returned view.\n        shape (tuple of int):\n            The shape of the returned view.\n        strides (tuple of int):\n            The strides of the returned view, given in the unit of steps.\n        storage_offset (int):\n            The offset from the leftest pointer of allocated memory to\n            the first element of returned view, given in the unit of steps.\n\n    Returns:\n        :class:`numpy.ndarray` or :class:`cupy.ndarray`:\n            The new view for the base array.\n    '
    min_index = _min_index(shape, strides, storage_offset)
    max_index = _max_index(shape, strides, storage_offset)
    strides = _step2byte(strides, array.itemsize)
    (storage_offset,) = _step2byte((storage_offset,), array.itemsize)
    if min_index < 0:
        raise ValueError('Out of buffer: too small index was specified')
    if isinstance(array, cuda.ndarray):
        pooled_memory = array.data.mem
        if (max_index + 1) * array.itemsize > pooled_memory.size:
            raise ValueError('Out of buffer: too large index was specified')
        memptr = cuda.cupy.cuda.memory.MemoryPointer(pooled_memory, storage_offset)
        return cuda.cupy.ndarray(shape, array.dtype, memptr, strides)
    elif isinstance(array, np.ndarray):
        base_array = _get_base_array(array)
        if (max_index + 1) * base_array.itemsize > base_array.nbytes:
            raise ValueError('Out of buffer: too large index was specified')
        return np.ndarray(shape, base_array.dtype, base_array.data, storage_offset, strides)
    else:
        raise TypeError('Only (np|cp).ndarray is accepted')

class TensorGeometry(object):

    def __init__(self, array):
        if False:
            return 10
        self.shape = array.shape
        self.strides = _byte2step(array.strides, array.itemsize)
        if isinstance(array, np.ndarray):
            base_array = _get_base_array(array)
            array_ptr = array.__array_interface__['data'][0]
            base_array_ptr = base_array.__array_interface__['data'][0]
            offset_bytes = array_ptr - base_array_ptr
        elif isinstance(array, cuda.ndarray):
            offset_bytes = array.data.ptr - array.data.mem.ptr
        else:
            raise ValueError('only (np|cp).ndarray is supported')
        (self.storage_offset,) = _byte2step((offset_bytes,), array.itemsize)
        self.itemsize = array.itemsize

    @property
    def ndim(self):
        if False:
            while True:
                i = 10
        return len(self.shape)

class AsStrided(function_node.FunctionNode):
    """Transportation of :func:`torch.Tensor.as_strided`.
    While :func:`torch.Tensor.as_strided` does not support nagative strides,
    this implementation does support it.
    """

    def __init__(self, shape, strides, storage_offset=None):
        if False:
            while True:
                i = 10
        self.shape = shape
        self.strides = strides
        self.storage_offset = storage_offset
        self.input_geometry = None

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        assert len(inputs) > 0
        x = inputs[0]
        self.input_geometry = TensorGeometry(x)
        if self.storage_offset is None:
            self.storage_offset = self.input_geometry.storage_offset
        return (_stride_array(x, self.shape, self.strides, self.storage_offset),)

    def backward(self, _, grad_outputs):
        if False:
            print('Hello World!')
        'Backward computation which calls :class:`AsStridedGrad`.\n\n        .. note:\n            While this implementation is based on *New-Style Function\n            Implementation*, the backward computation does not support\n            double-backpropagation due to *layout agnostic* algorithm (\n            originally named in the note of pytorch).\n        '
        return AsStridedGrad(self.input_geometry, self.shape, self.strides, self.storage_offset).apply(grad_outputs)

class AsStridedGrad(function_node.FunctionNode):
    """Backward of :func:`~chainer.functions.as_strided`.
    """

    def __init__(self, input_geometry, shape, strides, storage_offset):
        if False:
            print('Hello World!')
        self.input_geometry = input_geometry
        self.shape = shape
        self.strides = strides
        self.storage_offset = storage_offset

    def forward(self, grads):
        if False:
            while True:
                i = 10
        assert len(grads) > 0
        gy = grads[0]
        if gy.dtype not in np.sctypes['float']:
            raise TypeError('Only float is supported for back propagation')
        xp = backend.get_array_module(gy)
        input_geometry = self.input_geometry
        itemsize = input_geometry.itemsize
        if 0 in input_geometry.shape:
            return xp.zeros(input_geometry.shape)
        if 0 in gy.shape:
            return backend.get_array_module(gy).zeros(input_geometry.shape)
        else:
            out_shape = tuple([self.shape[i] for i in six.moves.range(gy.ndim) if self.shape[i] != 1 and self.strides[i] != 0])
            out_strides = tuple([self.strides[i] for i in six.moves.range(gy.ndim) if self.shape[i] != 1 and self.strides[i] != 0])
            gy = gy.sum(tuple([i for i in six.moves.range(gy.ndim) if self.strides[i] == 0]))
            gy = gy.squeeze()
        out_storage_offset = self.storage_offset
        inp_shape = tuple([input_geometry.shape[i] for i in six.moves.range(input_geometry.ndim) if input_geometry.shape[i] != 1])
        inp_strides = tuple([input_geometry.strides[i] for i in six.moves.range(input_geometry.ndim) if input_geometry.shape[i] != 1])
        inp_storage_offset = input_geometry.storage_offset
        inp_min_ptr = _min_index(inp_shape, inp_strides, input_geometry.storage_offset)
        out_min_ptr = _min_index(out_shape, out_strides, self.storage_offset)
        common_min_ptr = min(inp_min_ptr, out_min_ptr)
        inp_max_ptr = _max_index(inp_shape, inp_strides, input_geometry.storage_offset)
        out_max_ptr = _max_index(out_shape, out_strides, self.storage_offset)
        common_max_ptr = max(inp_max_ptr, out_max_ptr)
        base_size = common_max_ptr - common_min_ptr + 1
        storage = xp.zeros(base_size, dtype=gy.dtype)
        flatten_full_indices = xp.arange(base_size, dtype=index_dtype[itemsize])
        out_maybe_overlap = _maybe_overlapping_memory(out_shape, out_strides)
        if out_maybe_overlap:
            out_indices = _stride_array(flatten_full_indices, out_shape, out_strides, out_storage_offset - common_min_ptr)
            _index_add(storage, out_indices, gy)
        else:
            storage_view = _stride_array(storage, out_shape, out_strides, out_storage_offset - common_min_ptr)
            storage_view[:] = gy[:]
        inp_maybe_overlap = _maybe_overlapping_memory(inp_shape, inp_strides)
        if inp_maybe_overlap:
            count = xp.zeros_like(storage)
            inp_indices = _stride_array(flatten_full_indices, inp_shape, inp_strides, inp_storage_offset - common_min_ptr)
            _index_add(count, inp_indices, xp.ones(1))
            with np.errstate(divide='ignore', invalid='ignore'):
                storage /= count
        return (_stride_array(storage, inp_shape, inp_strides, inp_storage_offset - common_min_ptr),)

    def backward(self, target_input_indexes, grad_outputs):
        if False:
            while True:
                i = 10
        raise NotImplementedError

def as_strided(x, shape, strides, storage_offset=None):
    if False:
        while True:
            i = 10
    "Create a new view of array with the given shape, strides, and offset.\n\n    Args:\n        x (tuple of :class:`~chainer.Variable` or :class:`numpy.ndarray` or         :class:`cupy.ndarray`):\n            The array pointing a memory buffer. Its view is totally ignored.\n        shape (tuple of int):\n            The shape of output.\n        strides (tuple of int):\n            The strides of output, given in the unit of steps.\n        storage_offset (int):\n            The offset between the head of allocated memory and the pointer of\n            first element, given in the unit of steps.\n\n    Returns:\n        ~chainer.Variable: The strided variable.\n\n    .. warning::\n        Users should be aware that this function potentially causes unintended\n        side effects. See `numpy.lib.stride_tricks.as_strided`_ for the detail.\n\n    .. note::\n        The backward algorithm is borrowed from `torch.Tensor.as_strided`.\n        Therefore, the returned gradient of ``backward`` is *layout-agnostic*\n        when ``x`` contains memory overlap. See notes in pytorch's source\n        code (as_strided Backward and layout-aware/agnostic autograd) too.\n\n    .. note::\n        In this function ``strides`` and ``storage_offset`` are given in the\n        unit of steps instead of bytes. This specification differs from\n        :func:`numpy.lib.stride_tricks.as_strided`.\n\n    .. admonition:: Example\n\n        >>> from chainer import functions as F, Variable\n        >>> x = Variable(np.arange(4, dtype=np.float32))\n        >>> x\n        variable([0., 1., 2., 3.])\n        >>> y = F.as_strided(x, (3, 2), (1, 1), 0)\n        >>> y\n        variable([[0., 1.],\n                  [1., 2.],\n                  [2., 3.]])\n        >>> y.grad = np.ones((3, 2), dtype=np.float32)\n        >>> y.backward()\n        >>> x.grad\n        array([1., 2., 2., 1.], dtype=float32)\n\n    .. _numpy.lib.stride_tricks.as_strided:\n        https://docs.scipy.org/doc/numpy/reference/generated/        numpy.lib.stride_tricks.as_strided.html\n\n    "
    return AsStrided(shape, strides, storage_offset).apply((x,))[0]