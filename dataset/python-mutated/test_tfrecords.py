import json
import os
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.api.types import is_float_dtype, is_int64_dtype, is_object_dtype
import ray
from ray.tests.conftest import *
if TYPE_CHECKING:
    import tensorflow as tf
    from tensorflow_metadata.proto.v0 import schema_pb2

def tf_records_partial():
    if False:
        return 10
    'Underlying data corresponds to `data_partial` fixture.'
    import tensorflow as tf
    return [tf.train.Example(features=tf.train.Features(feature={'int_item': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])), 'int_list': tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 2, 3])), 'int_partial': tf.train.Feature(int64_list=tf.train.Int64List(value=[])), 'float_item': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])), 'float_list': tf.train.Feature(float_list=tf.train.FloatList(value=[2.0, 3.0, 4.0])), 'float_partial': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])), 'bytes_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'abc'])), 'bytes_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'def', b'1234'])), 'bytes_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])), 'string_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'uvw'])), 'string_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'xyz', b'999'])), 'string_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))})), tf.train.Example(features=tf.train.Features(feature={'int_item': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])), 'int_list': tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 3, 4])), 'int_partial': tf.train.Feature(int64_list=tf.train.Int64List(value=[9, 2])), 'float_item': tf.train.Feature(float_list=tf.train.FloatList(value=[2.0])), 'float_list': tf.train.Feature(float_list=tf.train.FloatList(value=[5.0, 6.0, 7.0])), 'float_partial': tf.train.Feature(float_list=tf.train.FloatList(value=[])), 'bytes_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'ghi'])), 'bytes_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jkl', b'5678'])), 'bytes_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'hello'])), 'string_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'mno'])), 'string_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'pqr', b'111'])), 'string_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'world']))}))]

def data_partial(with_tf_schema):
    if False:
        for i in range(10):
            print('nop')
    'TFRecords generated from this corresponds to `tf_records_partial`.'
    return [{'int_item': [1] if with_tf_schema else 1, 'int_list': [2, 2, 3], 'int_partial': [], 'float_item': [1.0] if with_tf_schema else 1.0, 'float_list': [2.0, 3.0, 4.0], 'float_partial': [1.0] if with_tf_schema else 1.0, 'bytes_item': [b'abc'] if with_tf_schema else b'abc', 'bytes_list': [b'def', b'1234'], 'bytes_partial': [] if with_tf_schema else None, 'string_item': ['uvw'] if with_tf_schema else 'uvw', 'string_list': ['xyz', '999'], 'string_partial': [] if with_tf_schema else None}, {'int_item': [2] if with_tf_schema else 2, 'int_list': [3, 3, 4], 'int_partial': [9, 2], 'float_item': [2.0] if with_tf_schema else 2.0, 'float_list': [5.0, 6.0, 7.0], 'float_partial': [] if with_tf_schema else None, 'bytes_item': [b'ghi'] if with_tf_schema else b'ghi', 'bytes_list': [b'jkl', b'5678'], 'bytes_partial': [b'hello'] if with_tf_schema else b'hello', 'string_item': ['mno'] if with_tf_schema else 'mno', 'string_list': ['pqr', '111'], 'string_partial': ['world'] if with_tf_schema else 'world'}]

def tf_records_empty():
    if False:
        for i in range(10):
            print('nop')
    'Underlying data corresponds to `data_empty` fixture.'
    import tensorflow as tf
    return [tf.train.Example(features=tf.train.Features(feature={'int_item': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])), 'int_list': tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 2, 3])), 'int_partial': tf.train.Feature(int64_list=tf.train.Int64List(value=[])), 'int_empty': tf.train.Feature(int64_list=tf.train.Int64List(value=[])), 'float_item': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])), 'float_list': tf.train.Feature(float_list=tf.train.FloatList(value=[2.0, 3.0, 4.0])), 'float_partial': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])), 'float_empty': tf.train.Feature(float_list=tf.train.FloatList(value=[])), 'bytes_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'abc'])), 'bytes_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'def', b'1234'])), 'bytes_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])), 'bytes_empty': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])), 'string_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'uvw'])), 'string_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'xyz', b'999'])), 'string_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])), 'string_empty': tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))})), tf.train.Example(features=tf.train.Features(feature={'int_item': tf.train.Feature(int64_list=tf.train.Int64List(value=[2])), 'int_list': tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 3, 4])), 'int_partial': tf.train.Feature(int64_list=tf.train.Int64List(value=[9, 2])), 'int_empty': tf.train.Feature(int64_list=tf.train.Int64List(value=[])), 'float_item': tf.train.Feature(float_list=tf.train.FloatList(value=[2.0])), 'float_list': tf.train.Feature(float_list=tf.train.FloatList(value=[5.0, 6.0, 7.0])), 'float_partial': tf.train.Feature(float_list=tf.train.FloatList(value=[])), 'float_empty': tf.train.Feature(float_list=tf.train.FloatList(value=[])), 'bytes_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'ghi'])), 'bytes_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jkl', b'5678'])), 'bytes_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'hello'])), 'bytes_empty': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])), 'string_item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'mno'])), 'string_list': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'pqr', b'111'])), 'string_partial': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'world'])), 'string_empty': tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))}))]

def data_empty(with_tf_schema):
    if False:
        while True:
            i = 10
    'TFRecords generated from this corresponds to\n    the `tf_records_empty` fixture.'
    return [{'int_item': [1] if with_tf_schema else 1, 'int_list': [2, 2, 3], 'int_partial': [], 'int_empty': [], 'float_item': [1.0] if with_tf_schema else 1.0, 'float_list': [2.0, 3.0, 4.0], 'float_partial': [1.0] if with_tf_schema else 1.0, 'float_empty': [], 'bytes_item': [b'abc'] if with_tf_schema else b'abc', 'bytes_list': [b'def', b'1234'], 'bytes_partial': [], 'bytes_empty': [], 'string_item': ['uvw'] if with_tf_schema else 'uvw', 'string_list': ['xyz', '999'], 'string_partial': [] if with_tf_schema else None, 'string_empty': []}, {'int_item': [2] if with_tf_schema else 2, 'int_list': [3, 3, 4], 'int_partial': [9, 2], 'int_empty': [], 'float_item': [2.0] if with_tf_schema else 2.0, 'float_list': [5.0, 6.0, 7.0], 'float_partial': [], 'float_empty': [], 'bytes_item': [b'ghi'] if with_tf_schema else b'ghi', 'bytes_list': [b'jkl', b'5678'], 'bytes_partial': [b'hello'] if with_tf_schema else b'hello', 'bytes_empty': [], 'string_item': ['mno'] if with_tf_schema else 'mno', 'string_list': ['pqr', '111'], 'string_partial': ['world'] if with_tf_schema else 'world', 'string_empty': []}]

def _features_to_schema(features: 'tf.train.Features') -> 'schema_pb2.Schema':
    if False:
        return 10
    from tensorflow_metadata.proto.v0 import schema_pb2
    tf_schema = schema_pb2.Schema()
    for (feature_name, feature_msg) in features.feature.items():
        schema_feature = tf_schema.feature.add()
        schema_feature.name = feature_name
        if feature_msg.HasField('bytes_list'):
            schema_feature.type = schema_pb2.FeatureType.BYTES
        elif feature_msg.HasField('float_list'):
            schema_feature.type = schema_pb2.FeatureType.FLOAT
        elif feature_msg.HasField('int64_list'):
            schema_feature.type = schema_pb2.FeatureType.INT
    return tf_schema

def _ds_eq_streaming(ds_expected, ds_actual) -> bool:
    if False:
        print('Hello World!')

    def _str2bytes(d):
        if False:
            print('Hello World!')
        for (k, v) in d.items():
            if 'string' in k:
                if isinstance(v, list):
                    d[k] = [vv.encode() for vv in v]
                elif isinstance(v, str):
                    d[k] = v.encode()
        return d
    ds_expected = ds_expected.map(_str2bytes)
    assert ds_expected.take() == ds_actual.take()

@pytest.mark.parametrize('with_tf_schema', (True, False))
def test_read_tfrecords(with_tf_schema, ray_start_regular_shared, tmp_path):
    if False:
        i = 10
        return i + 15
    import pandas as pd
    import tensorflow as tf
    example = tf_records_empty()[0]
    tf_schema = None
    if with_tf_schema:
        tf_schema = _features_to_schema(example.features)
    path = os.path.join(tmp_path, 'data.tfrecords')
    with tf.io.TFRecordWriter(path=path) as writer:
        writer.write(example.SerializeToString())
    ds = ray.data.read_tfrecords(path, tf_schema=tf_schema)
    df = ds.to_pandas()
    if with_tf_schema:
        assert is_object_dtype(dict(df.dtypes)['int_item'])
    else:
        assert is_int64_dtype(dict(df.dtypes)['int_item'])
    assert is_object_dtype(dict(df.dtypes)['int_list'])
    assert is_object_dtype(dict(df.dtypes)['int_partial'])
    assert is_object_dtype(dict(df.dtypes)['int_empty'])
    if with_tf_schema:
        assert is_object_dtype(dict(df.dtypes)['float_item'])
        assert is_object_dtype(dict(df.dtypes)['float_partial'])
    else:
        assert is_float_dtype(dict(df.dtypes)['float_item'])
        assert is_float_dtype(dict(df.dtypes)['float_partial'])
    assert is_object_dtype(dict(df.dtypes)['float_list'])
    assert is_object_dtype(dict(df.dtypes)['float_empty'])
    assert is_object_dtype(dict(df.dtypes)['bytes_item'])
    assert is_object_dtype(dict(df.dtypes)['bytes_partial'])
    assert is_object_dtype(dict(df.dtypes)['bytes_list'])
    assert is_object_dtype(dict(df.dtypes)['bytes_empty'])
    assert is_object_dtype(dict(df.dtypes)['string_item'])
    assert is_object_dtype(dict(df.dtypes)['string_partial'])
    assert is_object_dtype(dict(df.dtypes)['string_list'])
    assert is_object_dtype(dict(df.dtypes)['string_empty'])
    if with_tf_schema:
        assert isinstance(df['int_item'], pd.Series)
        assert df['int_item'].tolist() == [[1]]
    else:
        assert list(df['int_item']) == [1]
    assert np.array_equal(df['int_list'][0], np.array([2, 2, 3]))
    assert np.array_equal(df['int_partial'][0], np.array([], dtype=np.int64))
    assert np.array_equal(df['int_empty'][0], np.array([], dtype=np.int64))
    if with_tf_schema:
        assert isinstance(df['float_item'], pd.Series)
        assert df['float_item'].tolist() == [[1.0]]
    else:
        assert list(df['float_item']) == [1.0]
    assert np.array_equal(df['float_list'][0], np.array([2.0, 3.0, 4.0]))
    assert list(df['float_partial']) == [1.0]
    assert np.array_equal(df['float_empty'][0], np.array([], dtype=np.float32))
    if with_tf_schema:
        assert isinstance(df['bytes_item'], pd.Series)
        assert df['bytes_item'].tolist() == [[b'abc']]
        assert isinstance(df['string_item'], pd.Series)
        assert df['string_item'].tolist() == [[b'uvw']]
    else:
        assert list(df['bytes_item']) == [b'abc']
        assert list(df['string_item']) == [b'uvw']
    assert np.array_equal(df['bytes_list'][0], np.array([b'def', b'1234']))
    assert np.array_equal(df['bytes_partial'][0], np.array([], dtype=np.bytes_))
    assert np.array_equal(df['bytes_empty'][0], np.array([], dtype=np.bytes_))
    assert np.array_equal(df['string_list'][0], np.array([b'xyz', b'999']))
    assert np.array_equal(df['string_partial'][0], np.array([], dtype=np.bytes_))
    assert np.array_equal(df['string_empty'][0], np.array([], dtype=np.bytes_))

@pytest.mark.parametrize('ignore_missing_paths', [True, False])
def test_read_tfrecords_ignore_missing_paths(ray_start_regular_shared, tmp_path, ignore_missing_paths):
    if False:
        return 10
    import tensorflow as tf
    example = tf_records_empty()[0]
    path = os.path.join(tmp_path, 'data.tfrecords')
    with tf.io.TFRecordWriter(path=path) as writer:
        writer.write(example.SerializeToString())
    paths = [path, 'missing.tfrecords']
    if ignore_missing_paths:
        ds = ray.data.read_tfrecords(paths, ignore_missing_paths=ignore_missing_paths)
        assert ds.input_files() == [path]
    else:
        with pytest.raises(FileNotFoundError):
            ds = ray.data.read_tfrecords(paths, ignore_missing_paths=ignore_missing_paths)
            ds.materialize()

@pytest.mark.parametrize('with_tf_schema', (True, False))
def test_write_tfrecords(with_tf_schema, ray_start_regular_shared, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that write_tfrecords writes TFRecords correctly.\n\n    Test this by writing a Dataset to a TFRecord (function under test),\n    reading it back out into a tf.train.Example,\n    and checking that the result is analogous to the original Dataset.\n    '
    import tensorflow as tf
    ds = ray.data.from_items(data_partial(with_tf_schema), parallelism=1)
    expected_records = tf_records_partial()
    tf_schema = None
    if with_tf_schema:
        features = expected_records[0].features
        tf_schema = _features_to_schema(features)
    ds.write_tfrecords(tmp_path, tf_schema=tf_schema)
    filenames = sorted(os.listdir(tmp_path))
    filepaths = [os.path.join(tmp_path, filename) for filename in filenames]
    raw_dataset = tf.data.TFRecordDataset(filepaths)
    tfrecords = []
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        tfrecords.append(example)
    assert tfrecords == expected_records

@pytest.mark.parametrize('with_tf_schema', (True, False))
def test_write_tfrecords_empty_features(with_tf_schema, ray_start_regular_shared, tmp_path):
    if False:
        return 10
    'Test that write_tfrecords writes TFRecords with completely empty features\n    correctly (i.e. the case where type inference from partially filled features\n    is not possible). We expect this to succeed when passing an explicit `tf_schema`\n    param, and otherwise will raise a `ValueError`.\n\n    Test this by writing a Dataset to a TFRecord (function under test),\n    reading it back out into a tf.train.Example,\n    and checking that the result is analogous to the original Dataset.\n    '
    import tensorflow as tf
    ds = ray.data.from_items(data_empty(with_tf_schema))
    expected_records = tf_records_empty()
    if not with_tf_schema:
        with pytest.raises(ValueError):
            ds.write_tfrecords(tmp_path)
    else:
        features = expected_records[0].features
        tf_schema = _features_to_schema(features)
        ds.write_tfrecords(tmp_path, tf_schema=tf_schema)
        filenames = sorted(os.listdir(tmp_path))
        filepaths = [os.path.join(tmp_path, filename) for filename in filenames]
        raw_dataset = tf.data.TFRecordDataset(filepaths)
        tfrecords = []
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            tfrecords.append(example)
        assert tfrecords == expected_records

@pytest.mark.parametrize('with_tf_schema', (True, False))
def test_readback_tfrecords(ray_start_regular_shared, tmp_path, with_tf_schema):
    if False:
        while True:
            i = 10
    '\n    Test reading back TFRecords written using datasets.\n    The dataset we read back should be the same that we wrote.\n    '
    ds = ray.data.from_items(data_partial(with_tf_schema), parallelism=1)
    expected_records = tf_records_partial()
    tf_schema = None
    if with_tf_schema:
        features = expected_records[0].features
        tf_schema = _features_to_schema(features)
    ds.write_tfrecords(tmp_path, tf_schema=tf_schema)
    readback_ds = ray.data.read_tfrecords(tmp_path, tf_schema=tf_schema, parallelism=1)
    _ds_eq_streaming(ds, readback_ds)

@pytest.mark.parametrize('with_tf_schema', (True, False))
def test_readback_tfrecords_empty_features(ray_start_regular_shared, tmp_path, with_tf_schema):
    if False:
        return 10
    '\n    Test reading back TFRecords written using datasets.\n    The dataset we read back should be the same that we wrote.\n    '
    ds = ray.data.from_items(data_empty(with_tf_schema))
    if not with_tf_schema:
        with pytest.raises(ValueError):
            ds.write_tfrecords(tmp_path)
    else:
        ds = ray.data.from_items(data_empty(with_tf_schema), parallelism=1)
        expected_records = tf_records_empty()
        features = expected_records[0].features
        tf_schema = _features_to_schema(features)
        ds.write_tfrecords(tmp_path, tf_schema=tf_schema)
        readback_ds = ray.data.read_tfrecords(tmp_path, tf_schema=tf_schema, parallelism=1)
        _ds_eq_streaming(ds, readback_ds)

def test_write_invalid_tfrecords(ray_start_regular_shared, tmp_path):
    if False:
        return 10
    '\n    If we try to write a dataset with invalid TFRecord datatypes,\n    ValueError should be raised.\n    '
    ds = ray.data.from_items([{'item': None}])
    with pytest.raises(ValueError):
        ds.write_tfrecords(tmp_path)

def test_read_invalid_tfrecords(ray_start_regular_shared, tmp_path):
    if False:
        i = 10
        return i + 15
    file_path = os.path.join(tmp_path, 'file.json')
    with open(file_path, 'w') as file:
        json.dump({'number': 0, 'string': 'foo'}, file)
    with pytest.raises(RuntimeError, match='Failed to read TFRecord file'):
        ray.data.read_tfrecords(file_path).schema()

def test_read_with_invalid_schema(ray_start_regular_shared, tmp_path):
    if False:
        while True:
            i = 10
    from tensorflow_metadata.proto.v0 import schema_pb2
    ds = ray.data.from_items(data_partial(True), parallelism=1)
    expected_records = tf_records_partial()
    tf_schema_wrong_name = schema_pb2.Schema()
    schema_feature = tf_schema_wrong_name.feature.add()
    schema_feature.name = 'wrong_name'
    schema_feature.type = schema_pb2.FeatureType.INT
    tf_schema_wrong_type = _features_to_schema(expected_records[0].features)
    for schema_feature in tf_schema_wrong_type.feature:
        if schema_feature.name == 'bytes_item':
            schema_feature.type = schema_pb2.FeatureType.INT
            break
    with pytest.raises(ValueError) as e:
        ds.write_tfrecords(tmp_path, tf_schema=tf_schema_wrong_name)
    assert 'Found extra unexpected feature' in str(e.value.args[0])
    with pytest.raises(ValueError) as e:
        ds.write_tfrecords(tmp_path, tf_schema=tf_schema_wrong_type)
    assert str(e.value.args[0]) == 'Schema field type mismatch during write: specified type is int, but underlying type is bytes'
    ds.write_tfrecords(tmp_path)
    with pytest.raises(ValueError) as e:
        ray.data.read_tfrecords(tmp_path, tf_schema=tf_schema_wrong_name).materialize()
    assert 'Found extra unexpected feature' in str(e.value.args[0])
    with pytest.raises(ValueError) as e:
        ray.data.read_tfrecords(tmp_path, tf_schema=tf_schema_wrong_type).materialize()
    assert str(e.value.args[0]) == 'Schema field type mismatch during read: specified type is int, but underlying type is bytes'
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))