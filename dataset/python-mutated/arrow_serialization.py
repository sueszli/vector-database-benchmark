from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    import pyarrow
    from ray.data.extensions import ArrowTensorArray
RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION = 'RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION'
RAY_DISABLE_CUSTOM_ARROW_DATA_SERIALIZATION = 'RAY_DISABLE_CUSTOM_ARROW_DATA_SERIALIZATION'
logger = logging.getLogger(__name__)
_serialization_fallback_set = set()
_in_test = None

def _is_in_test():
    if False:
        return 10
    global _in_test
    if _in_test is None:
        _in_test = any((env_var in os.environ for env_var in ('PYTEST_CURRENT_TEST', 'BUILDKITE')))
    return _in_test

def _register_custom_datasets_serializers(serialization_context):
    if False:
        i = 10
        return i + 15
    try:
        import pyarrow as pa
    except ModuleNotFoundError:
        return
    _register_arrow_data_serializer(serialization_context)
    _register_arrow_json_readoptions_serializer(serialization_context)
    _register_arrow_json_parseoptions_serializer(serialization_context)

def _register_arrow_json_readoptions_serializer(serialization_context):
    if False:
        i = 10
        return i + 15
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION, '0') == '1':
        return
    import pyarrow.json as pajson
    serialization_context._register_cloudpickle_serializer(pajson.ReadOptions, custom_serializer=lambda opts: (opts.use_threads, opts.block_size), custom_deserializer=lambda args: pajson.ReadOptions(*args))

def _register_arrow_json_parseoptions_serializer(serialization_context):
    if False:
        while True:
            i = 10
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION, '0') == '1':
        return
    import pyarrow.json as pajson
    serialization_context._register_cloudpickle_serializer(pajson.ParseOptions, custom_serializer=lambda opts: (opts.explicit_schema, opts.newlines_in_values, opts.unexpected_field_behavior), custom_deserializer=lambda args: pajson.ParseOptions(*args))

def _register_arrow_data_serializer(serialization_context):
    if False:
        print('Hello World!')
    "Custom reducer for Arrow data that works around a zero-copy slicing pickling\n    bug by using the Arrow IPC format for the underlying serialization.\n\n    Background:\n        Arrow has both array-level slicing and buffer-level slicing; both are zero-copy,\n        but the former has a serialization bug where the entire buffer is serialized\n        instead of just the slice, while the latter's serialization works as expected\n        and only serializes the slice of the buffer. I.e., array-level slicing doesn't\n        propagate the slice down to the buffer when serializing the array.\n\n        We work around this by registering a custom cloudpickle reducers for Arrow\n        Tables that delegates serialization to the Arrow IPC format; thankfully, Arrow's\n        IPC serialization has fixed this buffer truncation bug.\n\n    See https://issues.apache.org/jira/browse/ARROW-10739.\n    "
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_DATA_SERIALIZATION, '0') == '1':
        return
    import pyarrow as pa
    serialization_context._register_cloudpickle_reducer(pa.Table, _arrow_table_reduce)

def _arrow_table_reduce(t: 'pyarrow.Table'):
    if False:
        while True:
            i = 10
    "Custom reducer for Arrow Tables that works around a zero-copy slice pickling bug.\n    Background:\n        Arrow has both array-level slicing and buffer-level slicing; both are zero-copy,\n        but the former has a serialization bug where the entire buffer is serialized\n        instead of just the slice, while the latter's serialization works as expected\n        and only serializes the slice of the buffer. I.e., array-level slicing doesn't\n        propagate the slice down to the buffer when serializing the array.\n        All that these copy methods do is, at serialization time, take the array-level\n        slicing and translate them to buffer-level slicing, so only the buffer slice is\n        sent over the wire instead of the entire buffer.\n    See https://issues.apache.org/jira/browse/ARROW-10739.\n    "
    global _serialization_fallback_set
    reduced_columns = []
    for column_name in t.column_names:
        column = t[column_name]
        try:
            reduced_column = _arrow_chunked_array_reduce(column)
        except Exception as e:
            if not _is_dense_union(column.type) and _is_in_test():
                raise e from None
            if type(column.type) not in _serialization_fallback_set:
                logger.warning(f"Failed to complete optimized serialization of Arrow Table, serialization of column '{column_name}' of type {column.type} failed, so we're falling back to Arrow IPC serialization for the table. Note that this may result in slower serialization and more worker memory utilization. Serialization error:", exc_info=True)
                _serialization_fallback_set.add(type(column.type))
            return _arrow_table_ipc_reduce(t)
        else:
            reduced_columns.append(reduced_column)
    return (_reconstruct_table, (reduced_columns, t.schema))

def _reconstruct_table(reduced_columns: List[Tuple[List['pyarrow.Array'], 'pyarrow.DataType']], schema: 'pyarrow.Schema') -> 'pyarrow.Table':
    if False:
        return 10
    'Restore a serialized Arrow Table, reconstructing each reduced column.'
    import pyarrow as pa
    columns = []
    for (chunks_payload, type_) in reduced_columns:
        columns.append(_reconstruct_chunked_array(chunks_payload, type_))
    return pa.Table.from_arrays(columns, schema=schema)

def _arrow_chunked_array_reduce(ca: 'pyarrow.ChunkedArray') -> Tuple[List['PicklableArrayPayload'], 'pyarrow.DataType']:
    if False:
        print('Hello World!')
    "Custom reducer for Arrow ChunkedArrays that works around a zero-copy slice\n    pickling bug. This reducer does not return a reconstruction function, since it's\n    expected to be reconstructed by the Arrow Table reconstructor.\n    "
    chunk_payloads = []
    for chunk in ca.chunks:
        chunk_payload = PicklableArrayPayload.from_array(chunk)
        chunk_payloads.append(chunk_payload)
    return (chunk_payloads, ca.type)

def _reconstruct_chunked_array(chunks: List['PicklableArrayPayload'], type_: 'pyarrow.DataType') -> 'pyarrow.ChunkedArray':
    if False:
        i = 10
        return i + 15
    'Restore a serialized Arrow ChunkedArray from chunks and type.'
    import pyarrow as pa
    chunks = [chunk.to_array() for chunk in chunks]
    return pa.chunked_array(chunks, type_)

@dataclass
class PicklableArrayPayload:
    """Picklable array payload, holding data buffers and array metadata.

    This is a helper container for pickling and reconstructing nested Arrow Arrays while
    ensuring that the buffers that underly zero-copy slice views are properly truncated.
    """
    type: 'pyarrow.DataType'
    length: int
    buffers: List['pyarrow.Buffer']
    null_count: int
    offset: int
    children: List['PicklableArrayPayload']

    @classmethod
    def from_array(self, a: 'pyarrow.Array') -> 'PicklableArrayPayload':
        if False:
            for i in range(10):
                print('nop')
        'Create a picklable array payload from an Arrow Array.\n\n        This will recursively accumulate data buffer and metadata payloads that are\n        ready for pickling; namely, the data buffers underlying zero-copy slice views\n        will be properly truncated.\n        '
        return _array_to_array_payload(a)

    def to_array(self) -> 'pyarrow.Array':
        if False:
            for i in range(10):
                print('nop')
        'Reconstruct an Arrow Array from this picklable payload.'
        return _array_payload_to_array(self)

def _array_payload_to_array(payload: 'PicklableArrayPayload') -> 'pyarrow.Array':
    if False:
        i = 10
        return i + 15
    'Reconstruct an Arrow Array from a possibly nested PicklableArrayPayload.'
    import pyarrow as pa
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowVariableShapedTensorType
    children = [child_payload.to_array() for child_payload in payload.children]
    if pa.types.is_dictionary(payload.type):
        assert len(children) == 2, len(children)
        (indices, dictionary) = children
        return pa.DictionaryArray.from_arrays(indices, dictionary)
    elif pa.types.is_map(payload.type) and len(children) > 1:
        assert len(children) == 3, len(children)
        (offsets, keys, items) = children
        return pa.MapArray.from_arrays(offsets, keys, items)
    elif isinstance(payload.type, ArrowTensorType) or isinstance(payload.type, ArrowVariableShapedTensorType):
        assert len(children) == 1, len(children)
        storage = children[0]
        return pa.ExtensionArray.from_storage(payload.type, storage)
    else:
        return pa.Array.from_buffers(type=payload.type, length=payload.length, buffers=payload.buffers, null_count=payload.null_count, offset=payload.offset, children=children)

def _array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    if False:
        while True:
            i = 10
    "Serialize an Arrow Array to an PicklableArrayPayload for later pickling.\n\n    This function's primary purpose is to dispatch to the handler for the input array\n    type.\n    "
    import pyarrow as pa
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowVariableShapedTensorType
    if _is_dense_union(a.type):
        raise NotImplementedError('Custom slice view serialization of dense union arrays is not yet supported.')
    if pa.types.is_null(a.type):
        return _null_array_to_array_payload(a)
    elif _is_primitive(a.type):
        return _primitive_array_to_array_payload(a)
    elif _is_binary(a.type):
        return _binary_array_to_array_payload(a)
    elif pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
        return _list_array_to_array_payload(a)
    elif pa.types.is_fixed_size_list(a.type):
        return _fixed_size_list_array_to_array_payload(a)
    elif pa.types.is_struct(a.type):
        return _struct_array_to_array_payload(a)
    elif pa.types.is_union(a.type):
        return _union_array_to_array_payload(a)
    elif pa.types.is_dictionary(a.type):
        return _dictionary_array_to_array_payload(a)
    elif pa.types.is_map(a.type):
        return _map_array_to_array_payload(a)
    elif isinstance(a.type, ArrowTensorType) or isinstance(a.type, ArrowVariableShapedTensorType):
        return _tensor_array_to_array_payload(a)
    else:
        raise ValueError('Unhandled Arrow array type:', a.type)

def _is_primitive(type_: 'pyarrow.DataType') -> bool:
    if False:
        print('Hello World!')
    'Whether the provided Array type is primitive (boolean, numeric, temporal or\n    fixed-size binary).'
    import pyarrow as pa
    return pa.types.is_integer(type_) or pa.types.is_floating(type_) or pa.types.is_decimal(type_) or pa.types.is_boolean(type_) or pa.types.is_temporal(type_) or pa.types.is_fixed_size_binary(type_)

def _is_binary(type_: 'pyarrow.DataType') -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Whether the provided Array type is a variable-sized binary type.'
    import pyarrow as pa
    return pa.types.is_string(type_) or pa.types.is_large_string(type_) or pa.types.is_binary(type_) or pa.types.is_large_binary(type_)

def _null_array_to_array_payload(a: 'pyarrow.NullArray') -> 'PicklableArrayPayload':
    if False:
        for i in range(10):
            print('nop')
    'Serialize null array to PicklableArrayPayload.'
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[None], null_count=a.null_count, offset=0, children=[])

def _primitive_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    if False:
        i = 10
        return i + 15
    'Serialize primitive (numeric, temporal, boolean) arrays to\n    PicklableArrayPayload.\n    '
    assert _is_primitive(a.type), a.type
    buffers = a.buffers()
    assert len(buffers) == 2, len(buffers)
    bitmap_buf = buffers[0]
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(bitmap_buf, a.offset, len(a))
    else:
        bitmap_buf = None
    data_buf = buffers[1]
    if data_buf is not None:
        data_buf = _copy_buffer_if_needed(buffers[1], a.type, a.offset, len(a))
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, data_buf], null_count=a.null_count, offset=0, children=[])

def _binary_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    if False:
        i = 10
        return i + 15
    'Serialize binary (variable-sized binary, string) arrays to\n    PicklableArrayPayload.\n    '
    assert _is_binary(a.type), a.type
    buffers = a.buffers()
    assert len(buffers) == 3, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    offset_buf = buffers[1]
    (offset_buf, data_offset, data_length) = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    data_buf = buffers[2]
    data_buf = _copy_buffer_if_needed(data_buf, None, data_offset, data_length)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, offset_buf, data_buf], null_count=a.null_count, offset=0, children=[])

def _list_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    if False:
        i = 10
        return i + 15
    'Serialize list (regular and large) arrays to PicklableArrayPayload.'
    buffers = a.buffers()
    assert len(buffers) > 1, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    offset_buf = buffers[1]
    (offset_buf, child_offset, child_length) = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    child = a.values.slice(child_offset, child_length)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, offset_buf], null_count=a.null_count, offset=0, children=[_array_to_array_payload(child)])

def _fixed_size_list_array_to_array_payload(a: 'pyarrow.FixedSizeListArray') -> 'PicklableArrayPayload':
    if False:
        for i in range(10):
            print('nop')
    'Serialize fixed size list arrays to PicklableArrayPayload.'
    buffers = a.buffers()
    assert len(buffers) >= 1, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    child_offset = a.type.list_size * a.offset
    child_length = a.type.list_size * len(a)
    child = a.values.slice(child_offset, child_length)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf], null_count=a.null_count, offset=0, children=[_array_to_array_payload(child)])

def _struct_array_to_array_payload(a: 'pyarrow.StructArray') -> 'PicklableArrayPayload':
    if False:
        for i in range(10):
            print('nop')
    'Serialize struct arrays to PicklableArrayPayload.'
    buffers = a.buffers()
    assert len(buffers) >= 1, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    children = [_array_to_array_payload(a.field(i)) for i in range(a.type.num_fields)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf], null_count=a.null_count, offset=0, children=children)

def _union_array_to_array_payload(a: 'pyarrow.UnionArray') -> 'PicklableArrayPayload':
    if False:
        while True:
            i = 10
    'Serialize union arrays to PicklableArrayPayload.'
    import pyarrow as pa
    assert not _is_dense_union(a.type)
    buffers = a.buffers()
    assert len(buffers) > 1, len(buffers)
    bitmap_buf = buffers[0]
    assert bitmap_buf is None, bitmap_buf
    type_code_buf = buffers[1]
    type_code_buf = _copy_buffer_if_needed(type_code_buf, pa.int8(), a.offset, len(a))
    children = [_array_to_array_payload(a.field(i)) for i in range(a.type.num_fields)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, type_code_buf], null_count=a.null_count, offset=0, children=children)

def _dictionary_array_to_array_payload(a: 'pyarrow.DictionaryArray') -> 'PicklableArrayPayload':
    if False:
        return 10
    'Serialize dictionary arrays to PicklableArrayPayload.'
    indices_payload = _array_to_array_payload(a.indices)
    dictionary_payload = _array_to_array_payload(a.dictionary)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[], null_count=a.null_count, offset=0, children=[indices_payload, dictionary_payload])

def _map_array_to_array_payload(a: 'pyarrow.MapArray') -> 'PicklableArrayPayload':
    if False:
        i = 10
        return i + 15
    'Serialize map arrays to PicklableArrayPayload.'
    import pyarrow as pa
    buffers = a.buffers()
    assert len(buffers) > 0, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    new_buffers = [bitmap_buf]
    offset_buf = buffers[1]
    (offset_buf, data_offset, data_length) = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    if isinstance(a, pa.lib.ListArray):
        new_buffers.append(offset_buf)
        children = [_array_to_array_payload(a.values.slice(data_offset, data_length))]
    else:
        buffers = a.buffers()
        assert len(buffers) > 2, len(buffers)
        offsets = pa.Array.from_buffers(pa.int32(), len(a) + 1, [bitmap_buf, offset_buf])
        keys = a.keys.slice(data_offset, data_length)
        items = a.items.slice(data_offset, data_length)
        children = [_array_to_array_payload(offsets), _array_to_array_payload(keys), _array_to_array_payload(items)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=new_buffers, null_count=a.null_count, offset=0, children=children)

def _tensor_array_to_array_payload(a: 'ArrowTensorArray') -> 'PicklableArrayPayload':
    if False:
        i = 10
        return i + 15
    'Serialize tensor arrays to PicklableArrayPayload.'
    storage_payload = _array_to_array_payload(a.storage)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[], null_count=a.null_count, offset=0, children=[storage_payload])

def _copy_buffer_if_needed(buf: 'pyarrow.Buffer', type_: Optional['pyarrow.DataType'], offset: int, length: int) -> 'pyarrow.Buffer':
    if False:
        print('Hello World!')
    'Copy buffer, if needed.'
    import pyarrow as pa
    if type_ is not None and pa.types.is_boolean(type_):
        buf = _copy_bitpacked_buffer_if_needed(buf, offset, length)
    else:
        type_bytewidth = type_.bit_width // 8 if type_ is not None else 1
        buf = _copy_normal_buffer_if_needed(buf, type_bytewidth, offset, length)
    return buf

def _copy_normal_buffer_if_needed(buf: 'pyarrow.Buffer', byte_width: int, offset: int, length: int) -> 'pyarrow.Buffer':
    if False:
        for i in range(10):
            print('nop')
    'Copy buffer, if needed.'
    byte_offset = offset * byte_width
    byte_length = length * byte_width
    if offset > 0 or byte_length < buf.size:
        buf = buf.slice(byte_offset, byte_length)
    return buf

def _copy_bitpacked_buffer_if_needed(buf: 'pyarrow.Buffer', offset: int, length: int) -> 'pyarrow.Buffer':
    if False:
        print('Hello World!')
    'Copy bit-packed binary buffer, if needed.'
    bit_offset = offset % 8
    byte_offset = offset // 8
    byte_length = _bytes_for_bits(bit_offset + length) // 8
    if offset > 0 or byte_length < buf.size:
        buf = buf.slice(byte_offset, byte_length)
        if bit_offset != 0:
            buf = _align_bit_offset(buf, bit_offset, byte_length)
    return buf

def _copy_offsets_buffer_if_needed(buf: 'pyarrow.Buffer', arr_type: 'pyarrow.DataType', offset: int, length: int) -> Tuple['pyarrow.Buffer', int, int]:
    if False:
        for i in range(10):
            print('nop')
    'Copy the provided offsets buffer, returning the copied buffer and the\n    offset + length of the underlying data.\n    '
    import pyarrow as pa
    import pyarrow.compute as pac
    if pa.types.is_large_list(arr_type) or pa.types.is_large_string(arr_type) or pa.types.is_large_binary(arr_type) or pa.types.is_large_unicode(arr_type):
        offset_type = pa.int64()
    else:
        offset_type = pa.int32()
    buf = _copy_buffer_if_needed(buf, offset_type, offset, length + 1)
    offsets = pa.Array.from_buffers(offset_type, length + 1, [None, buf])
    child_offset = offsets[0].as_py()
    child_length = offsets[-1].as_py() - child_offset
    offsets = pac.subtract(offsets, child_offset)
    if pa.types.is_int32(offset_type):
        offsets = offsets.cast(offset_type, safe=False)
    buf = offsets.buffers()[1]
    return (buf, child_offset, child_length)

def _bytes_for_bits(n: int) -> int:
    if False:
        print('Hello World!')
    'Round up n to the nearest multiple of 8.\n    This is used to get the byte-padded number of bits for n bits.\n    '
    return n + 7 & -8

def _align_bit_offset(buf: 'pyarrow.Buffer', bit_offset: int, byte_length: int) -> 'pyarrow.Buffer':
    if False:
        for i in range(10):
            print('nop')
    'Align the bit offset into the buffer with the front of the buffer by shifting\n    the buffer and eliminating the offset.\n    '
    import pyarrow as pa
    bytes_ = buf.to_pybytes()
    bytes_as_int = int.from_bytes(bytes_, sys.byteorder)
    bytes_as_int >>= bit_offset
    bytes_ = bytes_as_int.to_bytes(byte_length, sys.byteorder)
    return pa.py_buffer(bytes_)

def _arrow_table_ipc_reduce(table: 'pyarrow.Table'):
    if False:
        while True:
            i = 10
    'Custom reducer for Arrow Table that works around a zero-copy slicing pickling\n    bug by using the Arrow IPC format for the underlying serialization.\n\n    This is currently used as a fallback for unsupported types (or unknown bugs) for\n    the manual buffer truncation workaround, e.g. for dense unions.\n    '
    from pyarrow.ipc import RecordBatchStreamWriter
    from pyarrow.lib import BufferOutputStream
    output_stream = BufferOutputStream()
    with RecordBatchStreamWriter(output_stream, schema=table.schema) as wr:
        wr.write_table(table)
    return (_restore_table_from_ipc, (output_stream.getvalue(),))

def _restore_table_from_ipc(buf: bytes) -> 'pyarrow.Table':
    if False:
        return 10
    'Restore an Arrow Table serialized to Arrow IPC format.'
    from pyarrow.ipc import RecordBatchStreamReader
    with RecordBatchStreamReader(buf) as reader:
        return reader.read_all()

def _is_dense_union(type_: 'pyarrow.DataType') -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Whether the provided Arrow type is a dense union.'
    import pyarrow as pa
    return pa.types.is_union(type_) and type_.mode == 'dense'