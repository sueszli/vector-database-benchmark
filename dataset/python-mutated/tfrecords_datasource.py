import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    import pyarrow
    import tensorflow as tf
    from tensorflow_metadata.proto.v0 import schema_pb2

@PublicAPI(stability='alpha')
class TFRecordDatasource(FileBasedDatasource):
    """TFRecord datasource, for reading and writing TFRecord files."""
    _FILE_EXTENSIONS = ['tfrecords']

    def __init__(self, paths: Union[str, List[str]], tf_schema: Optional['schema_pb2.Schema']=None, **file_based_datasource_kwargs):
        if False:
            while True:
                i = 10
        super().__init__(paths, **file_based_datasource_kwargs)
        self.tf_schema = tf_schema

    def _read_stream(self, f: 'pyarrow.NativeFile', path: str) -> Iterator[Block]:
        if False:
            for i in range(10):
                print('nop')
        import pyarrow as pa
        import tensorflow as tf
        from google.protobuf.message import DecodeError
        for record in _read_records(f, path):
            example = tf.train.Example()
            try:
                example.ParseFromString(record)
            except DecodeError as e:
                raise ValueError(f"`TFRecordDatasource` failed to parse `tf.train.Example` record in '{path}'. This error can occur if your TFRecord file contains a message type other than `tf.train.Example`: {e}")
            yield pa.Table.from_pydict(_convert_example_to_dict(example, self.tf_schema))

def _convert_example_to_dict(example: 'tf.train.Example', tf_schema: Optional['schema_pb2.Schema']) -> Dict[str, 'pyarrow.Array']:
    if False:
        while True:
            i = 10
    record = {}
    schema_dict = {}
    if tf_schema is not None:
        for schema_feature in tf_schema.feature:
            schema_dict[schema_feature.name] = schema_feature.type
    for (feature_name, feature) in example.features.feature.items():
        if tf_schema is not None and feature_name not in schema_dict:
            raise ValueError(f'Found extra unexpected feature {feature_name} not in specified schema: {tf_schema}')
        schema_feature_type = schema_dict.get(feature_name)
        record[feature_name] = _get_feature_value(feature, schema_feature_type)
    return record

def _convert_arrow_table_to_examples(arrow_table: 'pyarrow.Table', tf_schema: Optional['schema_pb2.Schema']=None) -> Iterable['tf.train.Example']:
    if False:
        i = 10
        return i + 15
    import tensorflow as tf
    schema_dict = {}
    if tf_schema is not None:
        for schema_feature in tf_schema.feature:
            schema_dict[schema_feature.name] = schema_feature.type
    for i in range(arrow_table.num_rows):
        features: Dict[str, 'tf.train.Feature'] = {}
        for name in arrow_table.column_names:
            if tf_schema is not None and name not in schema_dict:
                raise ValueError(f'Found extra unexpected feature {name} not in specified schema: {tf_schema}')
            schema_feature_type = schema_dict.get(name)
            features[name] = _value_to_feature(arrow_table[name][i], schema_feature_type)
        proto = tf.train.Example(features=tf.train.Features(feature=features))
        yield proto

def _get_single_true_type(dct) -> str:
    if False:
        print('Hello World!')
    'Utility function for getting the single key which has a `True` value in\n    a dict. Used to filter a dict of `{field_type: is_valid}` to get\n    the field type from a schema or data source.'
    filtered_types = iter([_type for _type in dct if dct[_type]])
    return next(filtered_types, None)

def _get_feature_value(feature: 'tf.train.Feature', schema_feature_type: Optional['schema_pb2.FeatureType']=None) -> 'pyarrow.Array':
    if False:
        i = 10
        return i + 15
    import pyarrow as pa
    underlying_feature_type = {'bytes': feature.HasField('bytes_list'), 'float': feature.HasField('float_list'), 'int': feature.HasField('int64_list')}
    assert sum((bool(value) for value in underlying_feature_type.values())) <= 1
    if schema_feature_type is not None:
        try:
            from tensorflow_metadata.proto.v0 import schema_pb2
        except ModuleNotFoundError:
            raise ModuleNotFoundError('To use TensorFlow schemas, please install the tensorflow-metadata package.')
        specified_feature_type = {'bytes': schema_feature_type == schema_pb2.FeatureType.BYTES, 'float': schema_feature_type == schema_pb2.FeatureType.FLOAT, 'int': schema_feature_type == schema_pb2.FeatureType.INT}
        und_type = _get_single_true_type(underlying_feature_type)
        spec_type = _get_single_true_type(specified_feature_type)
        if und_type is not None and und_type != spec_type:
            raise ValueError(f'Schema field type mismatch during read: specified type is {spec_type}, but underlying type is {und_type}')
        underlying_feature_type = specified_feature_type
    if underlying_feature_type['bytes']:
        value = feature.bytes_list.value
        type_ = pa.binary()
    elif underlying_feature_type['float']:
        value = feature.float_list.value
        type_ = pa.float32()
    elif underlying_feature_type['int']:
        value = feature.int64_list.value
        type_ = pa.int64()
    else:
        value = []
        type_ = pa.null()
    value = list(value)
    if len(value) == 1 and schema_feature_type is None:
        value = value[0]
    else:
        if len(value) == 0:
            type_ = pa.null()
        type_ = pa.list_(type_)
    return pa.array([value], type=type_)

def _value_to_feature(value: Union['pyarrow.Scalar', 'pyarrow.Array'], schema_feature_type: Optional['schema_pb2.FeatureType']=None) -> 'tf.train.Feature':
    if False:
        return 10
    import pyarrow as pa
    import tensorflow as tf
    if isinstance(value, pa.ListScalar):
        value_type = value.type.value_type
        value = value.as_py()
    else:
        value_type = value.type
        value = value.as_py()
        if value is None:
            value = []
        else:
            value = [value]
    underlying_value_type = {'bytes': pa.types.is_binary(value_type), 'string': pa.types.is_string(value_type), 'float': pa.types.is_floating(value_type), 'int': pa.types.is_integer(value_type)}
    assert sum((bool(value) for value in underlying_value_type.values())) <= 1
    if schema_feature_type is not None:
        try:
            from tensorflow_metadata.proto.v0 import schema_pb2
        except ModuleNotFoundError:
            raise ModuleNotFoundError('To use TensorFlow schemas, please install the tensorflow-metadata package.')
        specified_feature_type = {'bytes': schema_feature_type == schema_pb2.FeatureType.BYTES and (not underlying_value_type['string']), 'string': schema_feature_type == schema_pb2.FeatureType.BYTES and underlying_value_type['string'], 'float': schema_feature_type == schema_pb2.FeatureType.FLOAT, 'int': schema_feature_type == schema_pb2.FeatureType.INT}
        und_type = _get_single_true_type(underlying_value_type)
        spec_type = _get_single_true_type(specified_feature_type)
        if und_type is not None and und_type != spec_type:
            raise ValueError(f'Schema field type mismatch during write: specified type is {spec_type}, but underlying type is {und_type}')
        underlying_value_type = specified_feature_type
    if underlying_value_type['int']:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    if underlying_value_type['float']:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    if underlying_value_type['bytes']:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    if underlying_value_type['string']:
        value = [v.encode() for v in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    if pa.types.is_null(value_type):
        raise ValueError('Unable to infer type from partially missing column. Try setting read parallelism = 1, or use an input data source which explicitly specifies the schema.')
    raise ValueError(f'Value is of type {value_type}, which we cannot convert to a supported tf.train.Feature storage type (bytes, float, or int).')

def _read_records(file: 'pyarrow.NativeFile', path: str) -> Iterable[memoryview]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Read records from TFRecord file.\n\n    A TFRecord file contains a sequence of records. The file can only be read\n    sequentially. Each record is stored in the following formats:\n        uint64 length\n        uint32 masked_crc32_of_length\n        byte   data[length]\n        uint32 masked_crc32_of_data\n\n    See https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details\n    for more details.\n    '
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    row_count = 0
    while True:
        try:
            num_length_bytes_read = file.readinto(length_bytes)
            if num_length_bytes_read == 0:
                break
            elif num_length_bytes_read != 8:
                raise ValueError('Failed to read the length of record data. Expected 8 bytes but got {num_length_bytes_read} bytes.')
            num_length_crc_bytes_read = file.readinto(crc_bytes)
            if num_length_crc_bytes_read != 4:
                raise ValueError('Failed to read the length of CRC-32C hashes. Expected 4 bytes but got {num_length_crc_bytes_read} bytes.')
            (data_length,) = struct.unpack('<Q', length_bytes)
            if data_length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(data_length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:data_length]
            num_datum_bytes_read = file.readinto(datum_bytes_view)
            if num_datum_bytes_read != data_length:
                raise ValueError(f'Failed to read the record. Exepcted {data_length} bytes but got {num_datum_bytes_read} bytes.')
            num_crc_bytes_read = file.readinto(crc_bytes)
            if num_crc_bytes_read != 4:
                raise ValueError(f'Failed to read the CRC-32C hashes. Expected 4 bytes but got {num_crc_bytes_read} bytes.')
            yield datum_bytes_view
            row_count += 1
            data_length = None
        except Exception as e:
            error_message = f'Failed to read TFRecord file {path}. Please ensure that the TFRecord file has correct format. Already read {row_count} rows.'
            if data_length is not None:
                error_message += f' Byte size of current record data is {data_length}.'
            raise RuntimeError(error_message) from e

def _write_record(file: 'pyarrow.NativeFile', example: 'tf.train.Example') -> None:
    if False:
        return 10
    record = example.SerializeToString()
    length = len(record)
    length_bytes = struct.pack('<Q', length)
    file.write(length_bytes)
    file.write(_masked_crc(length_bytes))
    file.write(record)
    file.write(_masked_crc(record))

def _masked_crc(data: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    'CRC checksum.'
    import crc32c
    mask = 2726488792
    crc = crc32c.crc32(data)
    masked = (crc >> 15 | crc << 17) + mask
    masked = np.uint32(masked & np.iinfo(np.uint32).max)
    masked_bytes = struct.pack('<I', masked)
    return masked_bytes