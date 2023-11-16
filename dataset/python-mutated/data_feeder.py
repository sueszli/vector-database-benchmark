import struct
import numpy as np
from ..pir import OpResult
from ..pir.core import ParameterMeta
from . import core
from .framework import Variable, _cpu_num, _cuda_ids, default_main_program, in_dygraph_mode, in_pir_mode
__all__ = []
_PADDLE_DTYPE_2_NUMPY_DTYPE = {core.VarDesc.VarType.BOOL: 'bool', core.VarDesc.VarType.FP16: 'float16', core.VarDesc.VarType.BF16: 'uint16', core.VarDesc.VarType.FP32: 'float32', core.VarDesc.VarType.FP64: 'float64', core.VarDesc.VarType.INT8: 'int8', core.VarDesc.VarType.INT16: 'int16', core.VarDesc.VarType.INT32: 'int32', core.VarDesc.VarType.INT64: 'int64', core.VarDesc.VarType.UINT8: 'uint8', core.VarDesc.VarType.COMPLEX64: 'complex64', core.VarDesc.VarType.COMPLEX128: 'complex128'}
_NUMPY_DTYPE_2_PADDLE_DTYPE = {'bool': core.VarDesc.VarType.BOOL, 'float16': core.VarDesc.VarType.FP16, 'uint16': core.VarDesc.VarType.BF16, 'float32': core.VarDesc.VarType.FP32, 'float64': core.VarDesc.VarType.FP64, 'int8': core.VarDesc.VarType.INT8, 'int16': core.VarDesc.VarType.INT16, 'int32': core.VarDesc.VarType.INT32, 'int64': core.VarDesc.VarType.INT64, 'uint8': core.VarDesc.VarType.UINT8, 'complex64': core.VarDesc.VarType.COMPLEX64, 'complex128': core.VarDesc.VarType.COMPLEX128}
_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE = {core.DataType.BOOL: 'bool', core.DataType.FLOAT16: 'float16', core.DataType.BFLOAT16: 'uint16', core.DataType.FLOAT32: 'float32', core.DataType.FLOAT64: 'float64', core.DataType.INT8: 'int8', core.DataType.INT16: 'int16', core.DataType.INT32: 'int32', core.DataType.INT64: 'int64', core.DataType.UINT8: 'uint8', core.DataType.COMPLEX64: 'complex64', core.DataType.COMPLEX128: 'complex128'}

def convert_float_to_uint16(data, data_format='NCHW'):
    if False:
        while True:
            i = 10
    if data.size == 0:
        return data.view(np.uint16)
    if data_format == 'NHWC':
        data = np.transpose(data, [0, 3, 1, 2])
    new_data = np.vectorize(lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16, otypes=[np.uint16])(data.flat)
    new_data = np.reshape(new_data, data.shape)
    if data_format == 'NHWC':
        new_data = np.transpose(new_data, [0, 2, 3, 1])
    return new_data

def convert_uint16_to_float(data):
    if False:
        i = 10
        return i + 15
    new_data = np.vectorize(lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0], otypes=[np.float32])(data.flat)
    return np.reshape(new_data, data.shape)

def convert_dtype(dtype):
    if False:
        while True:
            i = 10
    if isinstance(dtype, core.VarDesc.VarType):
        if dtype in _PADDLE_DTYPE_2_NUMPY_DTYPE:
            return _PADDLE_DTYPE_2_NUMPY_DTYPE[dtype]
    if isinstance(dtype, core.DataType):
        if dtype in _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE:
            return _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[dtype]
    elif isinstance(dtype, type):
        if dtype in [bool, np.float16, np.uint16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.complex64, np.complex128]:
            return dtype.__name__
    else:
        if dtype in ['bool', 'float16', 'uint16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128']:
            return str(dtype)
        if dtype in ['bfloat16']:
            return 'uint16'
    raise TypeError('dtype must be any of [bool, float16, uint16, float32, float64, int8, int16, int32, int64, uint8, complex64, complex128, bfloat16], but received %s' % dtype)

def check_variable_and_dtype(input, input_name, expected_dtype, op_name, extra_message=''):
    if False:
        while True:
            i = 10
    if in_pir_mode():
        check_type(input, input_name, (OpResult, ParameterMeta), op_name, extra_message)
    else:
        check_type(input, input_name, Variable, op_name, extra_message)
    check_dtype(input.dtype, input_name, expected_dtype, op_name, extra_message)

def check_type(input, input_name, expected_type, op_name, extra_message=''):
    if False:
        print('Hello World!')
    if in_dygraph_mode():
        return
    from .dygraph.base import in_to_static_mode
    if in_to_static_mode():
        if not isinstance(expected_type, tuple):
            expected_type = (expected_type,)
        expected_type += (core.eager.Tensor,)
    elif isinstance(input, core.eager.Tensor):
        raise TypeError(f"Please use `with base.dygraph.guard()` as context or `base.enable_dygraph()` to switch to imperative mode firstly. Because received '{input_name}' in {op_name} is a imperative Variable.")
    if not isinstance(input, expected_type):
        raise TypeError("The type of '{}' in {} must be {}, but received {}. {}".format(input_name, op_name, expected_type, type(input), extra_message))

def check_dtype(input_dtype, input_name, expected_dtype, op_name, extra_message=''):
    if False:
        print('Hello World!')
    if in_dygraph_mode():
        return
    if convert_dtype(input_dtype) not in expected_dtype:
        raise TypeError("The data type of '{}' in {} must be {}, but received {}. {}".format(input_name, op_name, expected_dtype, convert_dtype(input_dtype), extra_message))

def check_shape(shape, op_name, expected_shape_type=(list, tuple, Variable), expected_element_type=(int, Variable), expected_tensor_dtype=('int32', 'int64')):
    if False:
        i = 10
        return i + 15
    if in_dygraph_mode():
        return
    check_type(shape, 'shape', expected_shape_type, op_name)
    if expected_element_type is not None and (not isinstance(shape, Variable)):
        for item in shape:
            check_type(item, 'element of shape', expected_element_type, op_name)
            if expected_tensor_dtype is not None and isinstance(item, Variable):
                check_dtype(item.dtype, 'element of shape', expected_tensor_dtype, op_name, 'If element of shape is Tensor, its data type should be {}'.format(', '.join(expected_tensor_dtype)))
    if expected_tensor_dtype is not None and isinstance(shape, Variable):
        check_dtype(shape.dtype, 'shape', expected_tensor_dtype, op_name)

class DataToLoDTensorConverter:

    def __init__(self, place, lod_level, shape, dtype):
        if False:
            i = 10
            return i + 15
        self.place = place
        self.lod_level = lod_level
        self.shape = shape
        negtive_count = 0
        for s in self.shape:
            if s < 0:
                negtive_count += 1
            if negtive_count > 1:
                self.shape = None
                break
        self.dtype = convert_dtype(dtype)
        self._reset()

    def _reset(self):
        if False:
            print('Hello World!')
        self.data = []
        self.lod = [[] for _ in range(self.lod_level)]

    def feed(self, data):
        if False:
            return 10
        self._feed_impl_(data, self.lod, self.lod_level)

    def _feed_impl_(self, data, lod, lod_level):
        if False:
            print('Hello World!')
        if lod_level == 0:
            self.data.append(data)
        else:
            lod[0].append(len(data))
            for each_data in data:
                self._feed_impl_(each_data, lod[1:], lod_level - 1)

    def _check_shape(self, shape):
        if False:
            return 10
        for (s1, s2) in zip(self.shape, shape):
            if s1 != s2 and s1 >= 0 and (s2 >= 0):
                raise ValueError('Shape not match. What is defined in data layer is {}, but receive {}'.format(self.shape, shape))

    def done(self):
        if False:
            i = 10
            return i + 15
        arr = np.array(self.data, dtype=self.dtype)
        if self.shape:
            if len(arr.shape) != len(self.shape):
                try:
                    arr = arr.reshape(self.shape)
                except ValueError:
                    raise ValueError('Reshape error. What is defined in data layer is {}, but receive {}'.format(self.shape, arr.shape))
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_recursive_sequence_lengths(self.lod)
        self._reset()
        return t

class BatchedTensorProvider:

    def __init__(self, feed_list, place, batch_size, generator, drop_last):
        if False:
            i = 10
            return i + 15
        self.place = place
        self.batch_size = batch_size
        self.generator = generator
        self.converters = []
        self.drop_last = drop_last
        for var in feed_list:
            assert var.lod_level == 0, 'lod_level must be 0'
            self.converters.append(DataToLoDTensorConverter(place=self.place, lod_level=0, shape=var.shape, dtype=var.dtype))

    def _done(self):
        if False:
            print('Hello World!')
        return [c.done() for c in self.converters]

    def __call__(self):
        if False:
            i = 10
            return i + 15
        idx = 0
        for each_sample in self.generator():
            for (each_slot, each_converter) in zip(each_sample, self.converters):
                each_converter.data.append(each_slot)
            idx += 1
            if idx == self.batch_size:
                idx = 0
                yield self._done()
        if not self.drop_last and idx > 0:
            yield self._done()
        else:
            [c._reset() for c in self.converters]

class DataFeeder:
    """
    :api_attr: Static Graph

    DataFeeder converts the data that returned by a reader into a data
    structure that can feed into Executor. The reader is usually a
    python generator that returns a list of mini-batch data entries.

    Parameters:
        feed_list (list): Variables or names of Variables that need
            to feed.
        place (:ref:`api_paddle_CPUPlace` | :ref:`api_paddle_CUDAPlace` ):
            place indicates the device (CPU | GPU) the data will be fed into, if
            you want to feed data into GPU, please using :code:`base.CUDAPlace(i)`
            (:code:`i` represents the GPU id), or if you want to feed data into CPU,
            please using :code:`base.CPUPlace()`.
        program (:ref:`api_paddle_static_Program` , optional): The Program that will
            feed data into, if program is None, it will use default_main_program().
            Default None.

    Raises:
        :code:`ValueError` - If some Variables are not in this Program.

    Example:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle import base

            >>> paddle.enable_static()
            >>> place = paddle.CPUPlace()
            >>> def reader():
            ...     for _ in range(4):
            ...         yield np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32'),
            ...
            >>> main_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()

            >>> with paddle.static.program_guard(main_program, startup_program):
            ...     data_1 = paddle.static.data(name='data_1', shape=[None, 2, 2], dtype='float32')
            ...     data_2 = paddle.static.data(name='data_2', shape=[None, 1, 3], dtype='float32')
            ...     out = paddle.static.nn.fc(x=[data_1, data_2], size=2)
            ...     # ...
            >>> feeder = base.DataFeeder([data_1, data_2], place)

            >>> exe = paddle.static.Executor(place)
            >>> exe.run(startup_program)

            >>> feed_data = feeder.feed(reader())

            >>> # print feed_data to view feed results
            >>> # print(feed_data['data_1'])
            >>> # print(feed_data['data_2'])

            >>> outs = exe.run(
            ...     program=main_program,
            ...     feed=feed_data,
            ...     fetch_list=[out]
            ... )
            >>> print(outs)

    """

    def __init__(self, feed_list, place, program=None):
        if False:
            for i in range(10):
                print('nop')
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        if program is None:
            program = default_main_program()
        for each_var in feed_list:
            if isinstance(each_var, str):
                each_var = program.block(0).var(each_var)
            if not isinstance(each_var, Variable):
                raise TypeError('Feed list should contain a list of variable')
            self.feed_dtypes.append(each_var.dtype)
            self.feed_names.append(each_var.name)
            self.feed_lod_level.append(each_var.lod_level)
            self.feed_shapes.append(each_var.shape)
        self.place = place

    def feed(self, iterable):
        if False:
            i = 10
            return i + 15
        "\n        According to :code:`feed_list` of :code:`DataFeeder` and :code:`iterable` , converts\n        the input into a data structure that can feed into Executor.\n\n        Parameters:\n            iterable (generator): user defined python generator to read the raw input data\n\n        Returns:\n            :code:`dict`: a :code:`dict` that contains (variable name - converted tensor) pairs\n\n        Example:\n            .. code-block:: python\n\n                >>> # In this example, reader - generator will return a list of ndarray of 3 elements\n                >>> # feed API will convert each ndarray input into a tensor\n                >>> # the return result is a dict with keys: data_1, data_2, data_3\n                >>> # result['data_1']  a LoD-Tensor with shape of  [5, 2, 1, 3]. 5 is batch size, and [2, 1, 3] is the real shape of data_1.\n                >>> # result['data_2'], result['data_3'] are similar.\n                >>> import numpy as np\n                >>> import paddle\n                >>> from paddle import base\n\n                >>> paddle.enable_static()\n\n                >>> def reader(limit=5):\n                ...     for i in range(1, limit + 1):\n                ...         yield np.ones([6]).astype('float32') * i , np.ones([1]).astype('int64') * i, np.random.random([9]).astype('float32')\n                ...\n                >>> data_1 = paddle.static.data(name='data_1', shape=[None, 2, 1, 3])\n                >>> data_2 = paddle.static.data(name='data_2', shape=[None, 1], dtype='int64')\n                >>> data_3 = paddle.static.data(name='data_3', shape=[None, 3, 3], dtype='float32')\n                >>> feeder = base.DataFeeder(['data_1','data_2', 'data_3'], paddle.CPUPlace())\n\n                >>> result = feeder.feed(reader())\n                >>> print(result['data_1'])\n                >>> print(result['data_2'])\n                >>> print(result['data_3'])\n\n        "
        converter = []
        for (lod_level, shape, dtype) in zip(self.feed_lod_level, self.feed_shapes, self.feed_dtypes):
            converter.append(DataToLoDTensorConverter(place=self.place, lod_level=lod_level, shape=shape, dtype=dtype))
        for each_sample in iterable:
            assert len(each_sample) == len(converter), ('The number of fields in data (%d) does not match ' + 'len(feed_list) (%d)') % (len(each_sample), len(converter))
            for (each_converter, each_slot) in zip(converter, each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for (each_name, each_converter) in zip(self.feed_names, converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict

    def _get_number_of_places_(self, num_places):
        if False:
            print('Hello World!')
        if num_places is not None:
            return int(num_places)
        elif isinstance(self.place, core.CUDAPlace):
            return len(_cuda_ids())
        else:
            return _cpu_num()