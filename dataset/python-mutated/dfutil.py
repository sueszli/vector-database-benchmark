"""A collection of utility functions for loading/saving TensorFlow TFRecords files as Spark DataFrames."""
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, LongType, StringType, StructField, StructType
loadedDF = {}

def isLoadedDF(df):
    if False:
        while True:
            i = 10
    'Returns True if the input DataFrame was produced by the loadTFRecords() method.\n\n  This is primarily used by the Spark ML Pipelines APIs.\n\n  Args:\n    :df: Spark Dataframe\n  '
    return df in loadedDF

def saveAsTFRecords(df, output_dir):
    if False:
        i = 10
        return i + 15
    'Save a Spark DataFrame as TFRecords.\n\n  This will convert the DataFrame rows to TFRecords prior to saving.\n\n  Args:\n    :df: Spark DataFrame\n    :output_dir: Path to save TFRecords\n  '
    tf_rdd = df.rdd.mapPartitions(toTFExample(df.dtypes))
    tf_rdd.saveAsNewAPIHadoopFile(output_dir, 'org.tensorflow.hadoop.io.TFRecordFileOutputFormat', keyClass='org.apache.hadoop.io.BytesWritable', valueClass='org.apache.hadoop.io.NullWritable')

def loadTFRecords(sc, input_dir, binary_features=[]):
    if False:
        while True:
            i = 10
    'Load TFRecords from disk into a Spark DataFrame.\n\n  This will attempt to automatically convert the tf.train.Example features into Spark DataFrame columns of equivalent types.\n\n  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to\n  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"\n  from the caller in the ``binary_features`` argument.\n\n  Args:\n    :sc: SparkContext\n    :input_dir: location of TFRecords on disk.\n    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.\n\n  Returns:\n    A Spark DataFrame mirroring the tf.train.Example schema.\n  '
    import tensorflow as tf
    tfr_rdd = sc.newAPIHadoopFile(input_dir, 'org.tensorflow.hadoop.io.TFRecordFileInputFormat', keyClass='org.apache.hadoop.io.BytesWritable', valueClass='org.apache.hadoop.io.NullWritable')
    record = tfr_rdd.take(1)[0]
    example = tf.train.Example()
    example.ParseFromString(bytes(record[0]))
    schema = infer_schema(example, binary_features)
    example_rdd = tfr_rdd.mapPartitions(lambda x: fromTFExample(x, binary_features))
    df = example_rdd.toDF(schema)
    loadedDF[df] = input_dir
    return df

def toTFExample(dtypes):
    if False:
        return 10
    'mapPartition function to convert a Spark RDD of Row into an RDD of serialized tf.train.Example bytestring.\n\n  Note that tf.train.Example is a fairly flat structure with limited datatypes, e.g. tf.train.FloatList,\n  tf.train.Int64List, and tf.train.BytesList, so most DataFrame types will be coerced into one of these types.\n\n  Args:\n    :dtypes: the DataFrame.dtypes of the source DataFrame.\n\n  Returns:\n    A mapPartition function which converts the source DataFrame into tf.train.Example bytestrings.\n  '

    def _toTFExample(iter):
        if False:
            return 10
        float_dtypes = ['float', 'double']
        int64_dtypes = ['boolean', 'tinyint', 'smallint', 'int', 'bigint', 'long']
        bytes_dtypes = ['binary', 'string']
        float_list_dtypes = ['array<float>', 'array<double>']
        int64_list_dtypes = ['array<boolean>', 'array<tinyint>', 'array<smallint>', 'array<int>', 'array<bigint>', 'array<long>']

        def _toTFFeature(name, dtype, row):
            if False:
                while True:
                    i = 10
            feature = None
            if dtype in float_dtypes:
                feature = (name, tf.train.Feature(float_list=tf.train.FloatList(value=[row[name]])))
            elif dtype in int64_dtypes:
                feature = (name, tf.train.Feature(int64_list=tf.train.Int64List(value=[row[name]])))
            elif dtype in bytes_dtypes:
                if dtype == 'binary':
                    feature = (name, tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(row[name])])))
                else:
                    feature = (name, tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(row[name]).encode('utf-8')])))
            elif dtype in float_list_dtypes:
                feature = (name, tf.train.Feature(float_list=tf.train.FloatList(value=row[name])))
            elif dtype in int64_list_dtypes:
                feature = (name, tf.train.Feature(int64_list=tf.train.Int64List(value=row[name])))
            else:
                raise Exception('Unsupported dtype: {0}'.format(dtype))
            return feature
        results = []
        for row in iter:
            features = dict([_toTFFeature(name, dtype, row) for (name, dtype) in dtypes])
            example = tf.train.Example(features=tf.train.Features(feature=features))
            results.append((bytearray(example.SerializeToString()), None))
        return results
    return _toTFExample

def infer_schema(example, binary_features=[]):
    if False:
        while True:
            i = 10
    'Given a tf.train.Example, infer the Spark DataFrame schema (StructFields).\n\n  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to\n  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"\n  from the caller in the ``binary_features`` argument.\n\n  Args:\n    :example: a tf.train.Example\n    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.\n\n  Returns:\n    A DataFrame StructType schema\n  '

    def _infer_sql_type(k, v):
        if False:
            while True:
                i = 10
        if k in binary_features:
            return BinaryType()
        if v.int64_list.value:
            result = v.int64_list.value
            sql_type = LongType()
        elif v.float_list.value:
            result = v.float_list.value
            sql_type = DoubleType()
        else:
            result = v.bytes_list.value
            sql_type = StringType()
        if len(result) > 1:
            return ArrayType(sql_type)
        else:
            return sql_type
    return StructType([StructField(k, _infer_sql_type(k, v), True) for (k, v) in sorted(example.features.feature.items())])

def fromTFExample(iter, binary_features=[]):
    if False:
        print('Hello World!')
    'mapPartition function to convert an RDD of serialized tf.train.Example bytestring into an RDD of Row.\n\n  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to\n  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"\n  from the caller in the ``binary_features`` argument.\n\n  Args:\n    :iter: the RDD partition iterator\n    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.\n\n  Returns:\n    An array/iterator of DataFrame Row with features converted into columns.\n  '

    def _get_value(k, v):
        if False:
            return 10
        if v.int64_list.value:
            result = v.int64_list.value
        elif v.float_list.value:
            result = v.float_list.value
        elif k in binary_features:
            return bytearray(v.bytes_list.value[0])
        else:
            return v.bytes_list.value[0].decode('utf-8')
        if len(result) > 1:
            return list(result)
        elif len(result) == 1:
            return result[0]
        else:
            return None
    results = []
    for record in iter:
        example = tf.train.Example()
        example.ParseFromString(bytes(record[0]))
        d = {k: _get_value(k, v) for (k, v) in sorted(example.features.feature.items())}
        row = Row(**d)
        results.append(row)
    return results