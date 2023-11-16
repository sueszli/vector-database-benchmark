"""An XLA client in Python."""
from __future__ import annotations
import atexit
import contextlib
import enum
import gzip
import inspect
import logging
import os
import threading
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
import ml_dtypes
import numpy as np
from . import xla_extension as _xla
ops = _xla.ops
profiler = _xla.profiler
_version = 216
mlir_api_version = 54
xla_platform_names = {'cpu': 'Host', 'gpu': 'CUDA'}
logger = logging.getLogger(__name__)
_NameValueMapping = Mapping[str, Union[str, int, List[int], float, bool]]

def make_cpu_client(distributed_client=None, node_id=0, num_nodes=1) -> ...:
    if False:
        i = 10
        return i + 15
    register_custom_call_handler('cpu', _xla.register_custom_call_target)
    return _xla.get_tfrt_cpu_client(asynchronous=True, distributed_client=distributed_client, node_id=node_id, num_nodes=num_nodes)

def make_gpu_client(distributed_client=None, node_id=0, num_nodes=1, platform_name=None, allowed_devices=None, mock=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a GPU client. BFC allocator is used by default.'
    options = generate_pjrt_gpu_plugin_options()
    allocator = options['allocator']
    memory_fraction = options['memory_fraction'] if 'memory_fraction' in options else None
    preallocate = options['preallocate'] if 'preallocate' in options else None
    config = _xla.GpuAllocatorConfig()
    if allocator == 'default':
        config.kind = _xla.GpuAllocatorConfig.Kind.DEFAULT
    if allocator == 'platform':
        config.kind = _xla.GpuAllocatorConfig.Kind.PLATFORM
    if allocator == 'bfc':
        config.kind = _xla.GpuAllocatorConfig.Kind.BFC
    if allocator == 'cuda_async':
        config.kind = _xla.GpuAllocatorConfig.Kind.CUDA_ASYNC
    if memory_fraction:
        config.memory_fraction = float(memory_fraction)
    config.preallocate = preallocate not in ('0', 'false', 'False')
    register_custom_call_handler('CUDA', _xla.register_custom_call_target)
    register_custom_call_handler('ROCM', _xla.register_custom_call_target)
    return _xla.get_gpu_client(asynchronous=True, allocator_config=config, distributed_client=distributed_client, node_id=node_id, num_nodes=num_nodes, platform_name=platform_name, allowed_devices=allowed_devices, mock=mock)

def make_tfrt_tpu_c_api_client(options: Optional[_NameValueMapping]=None):
    if False:
        i = 10
        return i + 15
    assert pjrt_plugin_loaded('tpu')
    if not pjrt_plugin_initialized('tpu'):
        initialize_pjrt_plugin('tpu')
    if options is None:
        options = {}
    return _xla.get_c_api_client('tpu', options)
DeviceTopology = _xla.DeviceTopology
get_topology_for_devices = _xla.get_topology_for_devices

def make_tfrt_tpu_c_api_device_topology(topology_name: str='', **kwargs) -> DeviceTopology:
    if False:
        print('Hello World!')
    'Creates a PJRT C API TopologyDescription.'
    return _xla.get_default_c_api_topology('tpu', topology_name, dict(**kwargs))

def pjrt_plugin_loaded(plugin_name: str) -> bool:
    if False:
        while True:
            i = 10
    return _xla.pjrt_plugin_loaded(plugin_name)

def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> Any:
    if False:
        i = 10
        return i + 15
    return _xla.load_pjrt_plugin(plugin_name, library_path)

def pjrt_plugin_initialized(plugin_name: str) -> bool:
    if False:
        return 10
    return _xla.pjrt_plugin_initialized(plugin_name)

def initialize_pjrt_plugin(plugin_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Initializes a PJRT plugin.\n\n  The plugin needs to be loaded first (through load_pjrt_plugin_dynamically or\n  static linking) before this method is called.\n  Args:\n    plugin_name: the name of the PJRT plugin.\n  '
    _xla.initialize_pjrt_plugin(plugin_name)

def make_c_api_client(plugin_name: str, options: Optional[_NameValueMapping]=None, distributed_client: Optional[_xla.DistributedRuntimeClient]=None):
    if False:
        for i in range(10):
            print('nop')
    'Creates a PJRT C API client for a PJRT plugin.\n\n  It is required that load_pjrt_plugin_dynamically is called once with the same\n  plugin_name before this method is called.\n\n  Args:\n     plugin_name: the name of the PJRT plugin.\n     options: extra platform-specific options.\n     distributed_client: distributed client.\n\n  Returns:\n     A PJRT C API client for plugin_name.\n  '
    if options is None:
        options = {}
    return _xla.get_c_api_client(plugin_name, options, distributed_client)

def make_tpu_client(library_path: Optional[str]=None):
    if False:
        return 10
    'Returns a TPU client. Defaults to allowing 32 in-flight computations.'
    if not pjrt_plugin_loaded('tpu'):
        c_api = load_pjrt_plugin_dynamically('tpu', library_path or 'libtpu.so')
        profiler.register_plugin_profiler(c_api)
    return make_tfrt_tpu_c_api_client()

def generate_pjrt_gpu_plugin_options(visible_devices: str='all') -> _NameValueMapping:
    if False:
        for i in range(10):
            print('nop')
    'Generates the PjRt GPU plugin options.\n\n  Args:\n    visible_devices: A string of visible cuda devices.\n\n  Returns:\n    A dictionary of plugin options.\n  '
    options = {}
    if visible_devices != 'all':
        options['visible_devices'] = [int(x) for x in visible_devices.split(',')]
        options['platform_name'] = 'cuda'
    allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
    memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
    preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')
    if allocator not in ('default', 'platform', 'bfc', 'cuda_async'):
        raise ValueError('XLA_PYTHON_CLIENT_ALLOCATOR env var must be "default", "platform", "bfc", or "cuda_async", got "%s"' % allocator)
    options['allocator'] = allocator
    if memory_fraction:
        options['memory_fraction'] = float(memory_fraction)
    if preallocate:
        options['preallocate'] = preallocate not in ('false', 'False', '0')
    return options

class OpMetadata:
    """Python representation of a xla.OpMetadata protobuf."""
    __slots__ = ('op_type', 'op_name', 'source_file', 'source_line')

    def __init__(self, op_type='', op_name='', source_file='', source_line=0):
        if False:
            print('Hello World!')
        self.op_type = op_type
        self.op_name = op_name
        self.source_file = source_file
        self.source_line = source_line

def CurrentSourceInfoMetadata(op_type=None, op_name=None, skip_frames=1):
    if False:
        return 10
    'Helper for use in source mapping that returns an OpMetadata object.'
    (full_filename, lineno) = inspect.stack()[skip_frames][1:3]
    filename = os.path.basename(full_filename)
    return OpMetadata(op_type=op_type, op_name=op_name, source_file=filename, source_line=lineno)
PrimitiveType = _xla.PrimitiveType
bfloat16 = ml_dtypes.bfloat16
float8_e4m3fn = ml_dtypes.float8_e4m3fn
float8_e4m3b11fnuz = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fnuz = ml_dtypes.float8_e4m3fnuz
float8_e5m2 = ml_dtypes.float8_e5m2
float8_e5m2fnuz = ml_dtypes.float8_e5m2fnuz
XLA_ELEMENT_TYPE_TO_DTYPE = {PrimitiveType.PRED: np.dtype('bool'), PrimitiveType.S8: np.dtype('int8'), PrimitiveType.S16: np.dtype('int16'), PrimitiveType.S32: np.dtype('int32'), PrimitiveType.S64: np.dtype('int64'), PrimitiveType.U8: np.dtype('uint8'), PrimitiveType.U16: np.dtype('uint16'), PrimitiveType.U32: np.dtype('uint32'), PrimitiveType.U64: np.dtype('uint64'), PrimitiveType.F8E4M3FN: np.dtype(float8_e4m3fn), PrimitiveType.F8E4M3B11FNUZ: np.dtype(float8_e4m3b11fnuz), PrimitiveType.F8E5M2: np.dtype(float8_e5m2), PrimitiveType.F8E4M3FNUZ: np.dtype(float8_e4m3fnuz), PrimitiveType.F8E5M2FNUZ: np.dtype(float8_e5m2fnuz), PrimitiveType.BF16: np.dtype(bfloat16), PrimitiveType.F16: np.dtype('float16'), PrimitiveType.F32: np.dtype('float32'), PrimitiveType.F64: np.dtype('float64'), PrimitiveType.C64: np.dtype('complex64'), PrimitiveType.C128: np.dtype('complex128'), PrimitiveType.TUPLE: np.dtype(np.object_), PrimitiveType.TOKEN: np.dtype(np.object_)}
DTYPE_TO_XLA_ELEMENT_TYPE = {str(dt): et for (et, dt) in XLA_ELEMENT_TYPE_TO_DTYPE.items()}

def dtype_to_etype(dtype):
    if False:
        return 10
    'Convenience function for reading DTYPE_TO_XLA_ELEMENT_TYPE.'
    return DTYPE_TO_XLA_ELEMENT_TYPE[str(np.dtype(dtype))]
Shape = _xla.Shape
Shape.__doc__ = '\nA Shape is an object defined in C++ that duck types like the following class:\n\nclass Shape:\n  \'\'\'Represents an XLA shape.\n\n  A shape is either an array shape, having rank-many integer\n  dimensions and an element type (represented by a Numpy dtype), or it\n  is a tuple shape, having a shape for every tuple component:\n\n    type shape =\n        TupleShape of shape list\n      | ArrayShape of { dimensions: int list; element_type: dtype }\n  \'\'\'\n\n  @staticmethod\n  def tuple_shape(tuple_shapes) -> Shape:\n    "Construct a tuple shape."\n\n  @staticmethod\n  def array_shape(element_type, dimensions, minor_to_major=None) -> Shape:\n\n  @staticmethod\n  def from_pyval(pyval) -> Shape:\n    "Returns a Shape that describes a tuple-tree of Numpy arrays."\n\n  def __init__(self, str) -> Shape:\n    "Parses a shape string."\n  def __eq__(self, other: Shape) -> bool:\n  def __ne__(self, other: Shape) -> bool:\n  def __hash__(self):\n  def __repr__(self):\n  def is_tuple(self) -> bool:\n  def is_array(self) -> bool:\n  def tuple_shapes(self) -> [Shape]:\n  def numpy_dtype(self) -> np.dtype:\n    "Like element_type(), but returns dtype(\'O\') for a tuple shape."\n  def xla_element_type(self) -> PrimitiveType:\n  def element_type(self) -> np.dtype:\n  def dimensions(self) -> (int, int, ...):\n  def rank(self) -> int:\n  def with_major_to_minor_layout_if_absent(self) -> Shape:\n    "Returns a copy with missing layouts set to major-to-minor."\n\n  def to_serialized_proto(self) -> bytes:\n    "Returns \'shape\' as a serialized proto."\n'
ProgramShape = _xla.ProgramShape
ProgramShape.__doc__ = '\nA ProgramShape is a C++ object that duck types like the following class.\n\nclass ProgramShape:\n  def __init__(self, parameter_shapes, result_shape):\n  def parameter_shapes(self) -> [Shape]:\n  def result_shape(self) -> Shape:\n  def __repr__(self):\n'
ShapeIndex = _xla.ShapeIndex
ShapeIndex.__doc__ = "\nA Shape is an object defined in C++ that duck types like the following class:\n\nclass ShapeIndex:\n  '''Represents an XLA ShapeIndex.\n\n  An index for specifying a particular nested subshape within a shape. Used in\n  ShapeUtil::GetSubshape and other interfaces. ShapeIndex defines a path through\n  the Shape tree where each element of ShapeIndex indexes into a tuple (or\n  nested tuple) within the shape. For a non-nested tuple, an index has a single\n  element.\n  '''\n\n  def __init__(self, List[int]) -> ShapeIndex:\n  def __eq__(self, other: Shape) -> bool:\n  def __ne__(self, other: Shape) -> bool:\n  def __hash__(self):\n  def __repr__(self):\n"

def shape_from_pyval(pyval, layout: Sequence[int] | None=None):
    if False:
        while True:
            i = 10
    'Returns a Shape that describes a tuple-tree of Numpy arrays.'

    def convert(pyval):
        if False:
            print('Hello World!')
        if isinstance(pyval, tuple):
            if layout is not None:
                raise NotImplementedError('shape_from_pyval does not support layouts for tuple shapes')
            return Shape.tuple_shape(tuple((convert(elt) for elt in pyval)))
        else:
            return Shape.array_shape(pyval.dtype, np.shape(pyval), layout)
    return convert(pyval)
DeviceAssignment = _xla.DeviceAssignment
DeviceAssignment.__doc__ = "\nA DeviceAssignment is a C++ object with the following signature.\n\ndef create(assignment):\n  '''Builds a device assignment.\n\n   Args:\n     assignment: a 2D numpy array of device ordinal integers, indexed by\n       [replica][computation_in_replica].\n   Returns:\n     A device assignment.\n  '''\n\ndef replica_count():\n  '''Returns the number of replicas.'''\ndef computation_count():\n  '''Returns the number of computations per replica.'''\n"
Device = _xla.Device
CompileOptions = _xla.CompileOptions
HostBufferSemantics = _xla.HostBufferSemantics

def execute_with_python_values(executable, arguments, backend):
    if False:
        while True:
            i = 10
    'Execute on one replica with Python values as arguments and output.'

    def put(arg):
        if False:
            for i in range(10):
                print('nop')
        return backend.buffer_from_pyval(arg, device=executable.local_devices()[0])
    arguments = [put(arg) for arg in arguments]
    outputs = executable.execute(arguments)
    return [np.asarray(x) for x in outputs]

def execute_with_python_values_replicated(executable, arguments, backend):
    if False:
        print('Hello World!')
    'Execute on many replicas with Python values as arguments and output.\n\n  Args:\n    executable: the program to run.\n    arguments: a list of lists of Python values indexed by `[replica][arg_num]`\n      to pass as inputs.\n    backend: the backend we are targeting.\n\n  Returns:\n    A list of python values, one per replica.\n  '
    devices = executable.local_devices()

    def copy_to_devices(pyvals):
        if False:
            while True:
                i = 10
        return [backend.buffer_from_pyval(v, d) for (v, d) in zip(pyvals, devices)]
    inputs = [copy_to_devices(pyvals) for pyvals in zip(*arguments)]
    outputs = executable.execute_sharded_on_local_devices(inputs)
    return [[np.asarray(x) for x in xs] for xs in zip(*outputs)]

class PaddingType(enum.Enum):
    VALID = 1
    SAME = 2

def window_padding_type_to_pad_values(padding_type, lhs_dims, rhs_dims, window_strides):
    if False:
        return 10
    'Maps PaddingType or string to pad values (list of pairs of ints).'
    if not isinstance(padding_type, (str, PaddingType)):
        msg = 'padding_type must be str or PaddingType, got {}.'
        raise TypeError(msg.format(type(padding_type)))
    if isinstance(padding_type, str):
        if padding_type.upper() == 'VALID':
            padding_type = PaddingType.VALID
        elif padding_type.upper() == 'SAME':
            padding_type = PaddingType.SAME
        else:
            msg = 'Unknown padding type string: expected "VALID" or "SAME", got {}.'
            raise ValueError(msg.format(padding_type))
    if padding_type == PaddingType.VALID:
        return [(0, 0)] * len(window_strides)
    elif padding_type == PaddingType.SAME:
        out_shape = np.ceil(np.true_divide(lhs_dims, window_strides)).astype(int)
        pad_sizes = [max((out_size - 1) * stride + filter_size - in_size, 0) for (out_size, stride, filter_size, in_size) in zip(out_shape, window_strides, rhs_dims, lhs_dims)]
        return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
    else:
        msg = 'Unexpected PaddingType value: {}'
        raise ValueError(msg.format(padding_type))
XlaBuilder = _xla.XlaBuilder
XlaComputation = _xla.XlaComputation
XlaOp = _xla.XlaOp
FftType = _xla.FftType
Client = _xla.Client
Memory = _xla.Memory
ArrayImpl = _xla.ArrayImpl
LoadedExecutable = _xla.LoadedExecutable
DeviceList = _xla.DeviceList
OpSharding = _xla.OpSharding
HloSharding = _xla.HloSharding
Sharding = _xla.Sharding
XLACompatibleSharding = _xla.XLACompatibleSharding
NamedSharding = _xla.NamedSharding
SingleDeviceSharding = _xla.SingleDeviceSharding
PmapSharding = _xla.PmapSharding
GSPMDSharding = _xla.GSPMDSharding

def LoadedExecutable_execute(self, arguments, device=None):
    if False:
        return 10
    del device
    results = self.execute_sharded(arguments)
    return [x[0] for x in results.disassemble_into_single_device_arrays()]

def LoadedExecutable_execute_with_token(self, arguments, device=None):
    if False:
        print('Hello World!')
    del device
    results = self.execute_sharded(arguments, with_tokens=True)
    return ([x[0] for x in results.disassemble_into_single_device_arrays()], results.consume_token().get_token(0))
LoadedExecutable.execute = LoadedExecutable_execute
LoadedExecutable.execute_with_token = LoadedExecutable_execute_with_token
_custom_callback_handler: dict[str, Any] = {}
_custom_callback: dict[str, list[Tuple[str, Any]]] = {}
_custom_callback_lock = threading.Lock()

def register_custom_call_target(name: str, fn: Any, platform: str='cpu') -> None:
    if False:
        print('Hello World!')
    'Registers a custom call target.\n\n  Args:\n    name: bytes containing the name of the function.\n    fn: a PyCapsule object containing the function pointer.\n    platform: the target platform.\n  '
    xla_platform_name = xla_platform_names.get(platform, platform)
    with _custom_callback_lock:
        if xla_platform_name in _custom_callback_handler:
            _custom_callback_handler[xla_platform_name](name, fn, xla_platform_name)
        else:
            _custom_callback.setdefault(xla_platform_name, []).append((name, fn))

def register_custom_call_handler(platform: str, handler: Any) -> None:
    if False:
        i = 10
        return i + 15
    'Registers a custom handler and use it to register existing custom calls.\n\n  If a custom call handler for the platform already exist, calling this method\n  is a no-op and it will not register a new handler.\n  Args:\n    platform: the target platform.\n    handler: the function to register a custom call.\n  '
    xla_platform_name = xla_platform_names.get(platform, platform)
    with _custom_callback_lock:
        if xla_platform_name in _custom_callback_handler:
            logger.debug('Custom call handler for %s is already register. Will not register a new one', xla_platform_name)
            return
        _custom_callback_handler[xla_platform_name] = handler
        if xla_platform_name in _custom_callback:
            for (name, fn) in _custom_callback[xla_platform_name]:
                handler(name, fn, xla_platform_name)
            del _custom_callback[xla_platform_name]
register_custom_call_partitioner = _xla.register_custom_call_partitioner
encode_inspect_sharding_callback = _xla.encode_inspect_sharding_callback
hlo_sharding_util = _xla.hlo_sharding_util

class PaddingConfigDimension:
    """Python representation of a xla.PaddingConfigDimension protobuf."""
    __slots__ = ('edge_padding_low', 'edge_padding_high', 'interior_padding')
    edge_padding_low: int
    edge_padding_high: int
    interior_padding: int

    def __init__(self):
        if False:
            print('Hello World!')
        self.edge_padding_low = 0
        self.edge_padding_high = 0
        self.interior_padding = 0

class PaddingConfig:
    """Python representation of a xla.PaddingConfig protobuf."""
    __slots__ = ('dimensions',)

    def __init__(self):
        if False:
            while True:
                i = 10
        self.dimensions = []

def make_padding_config(padding_config: Union[PaddingConfig, Sequence[Tuple[int, int, int]]]) -> PaddingConfig:
    if False:
        print('Hello World!')
    'Create PaddingConfig proto from list of triples of integers.\n\n  Args:\n    padding_config: either a PaddingConfig or a list of integer triples\n      (edge_padding_low, edge_padding_high, interior_padding) representing the\n      configuration of the padding operation.\n\n  Returns:\n    A `PaddingConfig` object.\n  '
    if not isinstance(padding_config, PaddingConfig):
        triples = padding_config
        padding_config = PaddingConfig()
        for (lo, hi, interior) in triples:
            dimension = PaddingConfigDimension()
            dimension.edge_padding_low = lo
            dimension.edge_padding_high = hi
            dimension.interior_padding = interior
            padding_config.dimensions.append(dimension)
    return padding_config

class DotDimensionNumbers:
    """Python representation of a xla.DotDimensionNumbers protobuf."""
    __slots__ = ('lhs_contracting_dimensions', 'rhs_contracting_dimensions', 'lhs_batch_dimensions', 'rhs_batch_dimensions')

    def __init__(self):
        if False:
            print('Hello World!')
        self.lhs_contracting_dimensions = []
        self.rhs_contracting_dimensions = []
        self.lhs_batch_dimensions = []
        self.rhs_batch_dimensions = []

def make_dot_dimension_numbers(dimension_numbers: Union[DotDimensionNumbers, Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]]) -> DotDimensionNumbers:
    if False:
        print('Hello World!')
    'Builds a DotDimensionNumbers object from a specification.\n\n  Args:\n    dimension_numbers: either a `DotDimensionNumbers` or a nested tuple\n      `((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))` of lists of\n      integers representing the dimensions to treat as contracting dimensions\n      and batch dimensions on each input operand.\n\n  Returns:\n    A `DotDimensionNumbers` object.\n  '
    if isinstance(dimension_numbers, (list, tuple)):
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
        dot_dims_proto = DotDimensionNumbers()
        dot_dims_proto.lhs_contracting_dimensions.extend(lhs_contract)
        dot_dims_proto.rhs_contracting_dimensions.extend(rhs_contract)
        dot_dims_proto.lhs_batch_dimensions.extend(lhs_batch)
        dot_dims_proto.rhs_batch_dimensions.extend(rhs_batch)
        return dot_dims_proto
    else:
        return dimension_numbers

class ConvolutionDimensionNumbers:
    """Python representation of a xla.ConvolutionDimensionNumbers protobuf."""
    __slots__ = ('input_batch_dimension', 'input_feature_dimension', 'input_spatial_dimensions', 'kernel_input_feature_dimension', 'kernel_output_feature_dimension', 'kernel_spatial_dimensions', 'output_batch_dimension', 'output_feature_dimension', 'output_spatial_dimensions')

    def __init__(self):
        if False:
            return 10
        self.input_batch_dimension = 0
        self.input_feature_dimension = 0
        self.input_spatial_dimensions = []
        self.kernel_input_feature_dimension = 0
        self.kernel_output_feature_dimension = 0
        self.kernel_spatial_dimensions = []
        self.output_batch_dimension = 0
        self.output_feature_dimension = 0
        self.output_spatial_dimensions = []

def make_convolution_dimension_numbers(dimension_numbers: Union[None, ConvolutionDimensionNumbers, Tuple[str, str, str]], num_spatial_dimensions: int) -> ConvolutionDimensionNumbers:
    if False:
        print('Hello World!')
    "Builds a ConvolutionDimensionNumbers object from a specification.\n\n  Args:\n    dimension_numbers: optional, either a ConvolutionDimensionNumbers object or\n      a tuple (lhs_spec, rhs_spec, out_spec). Each element is a string of length\n      N+2 identifying by position: (1) batch dimensions in lhs, rhs, and the\n      output with the character 'N', (2) feature dimensions in lhs and the\n      output with the character 'C', (3) input and output feature dimensions in\n      rhs with the characters 'I' and 'O' respectively, and (4) spatial\n      dimension correspondences between lhs, rhs, and the output using any\n      distinct characters. For example, to indicate dimension numbers consistent\n      with the Conv operation with two spatial dimensions, one could use\n      ('NCHW', 'OIHW', 'NCHW'). As another example, to indicate dimension\n      numbers consistent with the TensorFlow Conv2D operation, one could use\n      ('NHWC', 'HWIO', 'NHWC'). When using the latter form of convolution\n      dimension specification, window strides are associated with spatial\n      dimension character labels according to the order in which the labels\n      appear in the rhs_spec string, so that window_strides[0] is matched with\n      the dimension corresponding to the first character appearing in rhs_spec\n      that is not 'I' or 'O'. By default, use the same dimension numbering as\n      Conv and ConvWithGeneralPadding.\n    num_spatial_dimensions: the number of spatial dimensions.\n\n  Returns:\n    A `ConvolutionDimensionNumbers` object.\n  "
    if dimension_numbers is None:
        nd = num_spatial_dimensions
        dimension_numbers = ConvolutionDimensionNumbers()
        dimension_numbers.input_batch_dimension = 0
        dimension_numbers.input_feature_dimension = 1
        dimension_numbers.output_batch_dimension = 0
        dimension_numbers.output_feature_dimension = 1
        dimension_numbers.kernel_output_feature_dimension = 0
        dimension_numbers.kernel_input_feature_dimension = 1
        dimension_numbers.input_spatial_dimensions.extend(range(2, 2 + nd))
        dimension_numbers.kernel_spatial_dimensions.extend(range(2, 2 + nd))
        dimension_numbers.output_spatial_dimensions.extend(range(2, 2 + nd))
    elif isinstance(dimension_numbers, tuple):
        (lhs_spec, rhs_spec, out_spec) = dimension_numbers
        dimension_numbers = ConvolutionDimensionNumbers()
        dimension_numbers.input_batch_dimension = lhs_spec.index('N')
        dimension_numbers.input_feature_dimension = lhs_spec.index('C')
        dimension_numbers.output_batch_dimension = out_spec.index('N')
        dimension_numbers.output_feature_dimension = out_spec.index('C')
        dimension_numbers.kernel_output_feature_dimension = rhs_spec.index('O')
        dimension_numbers.kernel_input_feature_dimension = rhs_spec.index('I')
        dimension_numbers.kernel_spatial_dimensions.extend((i for (i, c) in enumerate(rhs_spec) if c not in {'I', 'O'}))
        dimension_numbers.input_spatial_dimensions.extend(sorted((i for (i, c) in enumerate(lhs_spec) if c not in {'N', 'C'}), key=lambda i: rhs_spec.index(lhs_spec[i])))
        dimension_numbers.output_spatial_dimensions.extend(sorted((i for (i, c) in enumerate(out_spec) if c not in {'N', 'C'}), key=lambda i: rhs_spec.index(out_spec[i])))
    return dimension_numbers

class PrecisionConfig:
    """Python representation of a xla.PrecisionConfig protobuf."""
    __slots__ = ('operand_precision',)
    Precision = _xla.PrecisionConfig_Precision

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.operand_precision = []

class GatherDimensionNumbers:
    """Python representation of a xla.GatherDimensionNumbers protobuf."""
    __slots__ = ('offset_dims', 'collapsed_slice_dims', 'start_index_map', 'index_vector_dim')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.offset_dims = []
        self.collapsed_slice_dims = []
        self.start_index_map = []
        self.index_vector_dim = 0

class ScatterDimensionNumbers:
    """Python representation of a xla.ScatterDimensionNumbers protobuf."""
    __slots__ = ('update_window_dims', 'inserted_window_dims', 'scatter_dims_to_operand_dims', 'index_vector_dim')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.update_window_dims = []
        self.inserted_window_dims = []
        self.scatter_dims_to_operand_dims = []
        self.index_vector_dim = 0

class ReplicaGroup:
    """Python representation of a xla.ReplicaGroup protobuf."""
    __slots__ = ('replica_ids',)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.replica_ids = []

def _make_replica_group_proto(replica_group):
    if False:
        while True:
            i = 10
    replica_group_proto = ReplicaGroup()
    replica_group_proto.replica_ids.extend(replica_group)
    return replica_group_proto

def make_replica_groups(replica_groups):
    if False:
        return 10
    if replica_groups is None:
        replica_groups_protos = []
    else:
        replica_groups = list(replica_groups)
        replica_groups_protos = [_make_replica_group_proto(group) for group in replica_groups]
    return replica_groups_protos
Traceback = _xla.Traceback
Frame = _xla.Frame

@contextlib.contextmanager
def tracebacks(enabled=True):
    if False:
        return 10
    'Context manager that enables or disables traceback collection.'
    saved = Traceback.enabled
    Traceback.enabled = enabled
    try:
        yield
    finally:
        Traceback.enabled = saved

def heap_profile(client: Client) -> bytes:
    if False:
        while True:
            i = 10
    'Returns a gzipped pprof protocol buffer containing a heap profile.'
    return gzip.compress(client.heap_profile())
XlaRuntimeError = _xla.XlaRuntimeError
atexit.register(_xla.collect_garbage)
weakref_lru_cache = _xla.weakref_lru_cache
array_result_handler = _xla.array_result_handler
copy_array_to_devices_with_sharding = _xla.copy_array_to_devices_with_sharding
batched_device_put = _xla.batched_device_put
check_and_canonicalize_memory_kind = _xla.check_and_canonicalize_memory_kind
Layout = _xla.Layout