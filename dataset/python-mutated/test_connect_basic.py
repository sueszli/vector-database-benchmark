import array
import datetime
import os
import unittest
import random
import shutil
import string
import tempfile
import uuid
from collections import defaultdict
from pyspark.errors import PySparkAttributeError, PySparkTypeError, PySparkException, PySparkValueError
from pyspark.errors.exceptions.base import SessionNotSameException
from pyspark.sql import SparkSession as PySparkSession, Row
from pyspark.sql.connect.client.retries import RetryPolicy, RetriesExceeded
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, MapType, ArrayType, Row
from pyspark.testing.sqlutils import MyObject, SQLTestUtils, PythonOnlyUDT, ExamplePoint, PythonOnlyPoint
from pyspark.testing.connectutils import should_test_connect, ReusedConnectTestCase, connect_requirement_message
from pyspark.testing.pandasutils import PandasOnSparkTestUtils
from pyspark.errors.exceptions.connect import AnalysisException, ParseException, SparkConnectException, SparkUpgradeException
if should_test_connect:
    import grpc
    import pandas as pd
    import numpy as np
    from pyspark.sql.connect.proto import Expression as ProtoExpression
    from pyspark.sql.connect.session import SparkSession as RemoteSparkSession
    from pyspark.sql.connect.client import ChannelBuilder
    from pyspark.sql.connect.column import Column
    from pyspark.sql.connect.readwriter import DataFrameWriterV2
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as CDataFrame
    from pyspark.sql import functions as SF
    from pyspark.sql.connect import functions as CF
    from pyspark.sql.connect.client.core import Retrying, SparkConnectClient

class SparkConnectSQLTestCase(ReusedConnectTestCase, SQLTestUtils, PandasOnSparkTestUtils):
    """Parent test fixture class for all Spark Connect related
    test cases."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(SparkConnectSQLTestCase, cls).setUpClass()
        os.environ['PYSPARK_NO_NAMESPACE_SHARE'] = '1'
        cls.connect = cls.spark
        cls.spark = PySparkSession._instantiatedSession
        assert cls.spark is not None
        cls.testData = [Row(key=i, value=str(i)) for i in range(100)]
        cls.testDataStr = [Row(key=str(i)) for i in range(100)]
        cls.df = cls.spark.sparkContext.parallelize(cls.testData).toDF()
        cls.df_text = cls.spark.sparkContext.parallelize(cls.testDataStr).toDF()
        cls.tbl_name = 'test_connect_basic_table_1'
        cls.tbl_name2 = 'test_connect_basic_table_2'
        cls.tbl_name3 = 'test_connect_basic_table_3'
        cls.tbl_name4 = 'test_connect_basic_table_4'
        cls.tbl_name_empty = 'test_connect_basic_table_empty'
        cls.spark_connect_clean_up_test_data()
        cls.spark_connect_load_test_data()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        try:
            cls.spark_connect_clean_up_test_data()
            cls.spark = cls.connect
            del os.environ['PYSPARK_NO_NAMESPACE_SHARE']
        finally:
            super(SparkConnectSQLTestCase, cls).tearDownClass()

    @classmethod
    def spark_connect_load_test_data(cls):
        if False:
            for i in range(10):
                print('nop')
        df = cls.spark.createDataFrame([(x, f'{x}') for x in range(100)], ['id', 'name'])
        df.write.saveAsTable(cls.tbl_name)
        df2 = cls.spark.createDataFrame([(x, f'{x}', 2 * x) for x in range(100)], ['col1', 'col2', 'col3'])
        df2.write.saveAsTable(cls.tbl_name2)
        df3 = cls.spark.createDataFrame([(x, f'{x}') for x in range(100)], ['id', 'test\n_column'])
        df3.write.saveAsTable(cls.tbl_name3)
        df4 = cls.spark.createDataFrame([(x, {'a': x}, [x, x * 2]) for x in range(100)], ['id', 'map_column', 'array_column'])
        df4.write.saveAsTable(cls.tbl_name4)
        empty_table_schema = StructType([StructField('firstname', StringType(), True), StructField('middlename', StringType(), True), StructField('lastname', StringType(), True)])
        emptyRDD = cls.spark.sparkContext.emptyRDD()
        empty_df = cls.spark.createDataFrame(emptyRDD, empty_table_schema)
        empty_df.write.saveAsTable(cls.tbl_name_empty)

    @classmethod
    def spark_connect_clean_up_test_data(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.spark.sql('DROP TABLE IF EXISTS {}'.format(cls.tbl_name))
        cls.spark.sql('DROP TABLE IF EXISTS {}'.format(cls.tbl_name2))
        cls.spark.sql('DROP TABLE IF EXISTS {}'.format(cls.tbl_name3))
        cls.spark.sql('DROP TABLE IF EXISTS {}'.format(cls.tbl_name4))
        cls.spark.sql('DROP TABLE IF EXISTS {}'.format(cls.tbl_name_empty))

class SparkConnectBasicTests(SparkConnectSQLTestCase):

    def test_recursion_handling_for_plan_logging(self):
        if False:
            i = 10
            return i + 15
        'SPARK-45852 - Test that we can handle recursion in plan logging.'
        cdf = self.connect.range(1)
        for x in range(400):
            cdf = cdf.withColumn(f'col_{x}', CF.lit(x))
        self.assertIsNotNone(cdf.schema)
        result = self.connect._client._proto_to_string(cdf._plan.to_proto(self.connect._client))
        self.assertIn('recursion', result)

    def test_df_getattr_behavior(self):
        if False:
            print('Hello World!')
        cdf = self.connect.range(10)
        sdf = self.spark.range(10)
        sdf._simple_extension = 10
        cdf._simple_extension = 10
        self.assertEqual(sdf._simple_extension, cdf._simple_extension)
        self.assertEqual(type(sdf._simple_extension), type(cdf._simple_extension))
        self.assertTrue(hasattr(cdf, '_simple_extension'))
        self.assertFalse(hasattr(cdf, '_simple_extension_does_not_exsit'))

    def test_df_get_item(self):
        if False:
            while True:
                i = 10
        query = '\n            SELECT * FROM VALUES\n            (true, 1, NULL), (false, NULL, 2.0), (NULL, 3, 3.0)\n            AS tab(a, b, c)\n            '
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assert_eq(cdf[cdf.a].toPandas(), sdf[sdf.a].toPandas())
        self.assert_eq(cdf[cdf.b.isin(2, 3)].toPandas(), sdf[sdf.b.isin(2, 3)].toPandas())
        self.assert_eq(cdf[cdf.c > 1.5].toPandas(), sdf[sdf.c > 1.5].toPandas())
        self.assert_eq(cdf[[cdf.a, 'b', cdf.c]].toPandas(), sdf[[sdf.a, 'b', sdf.c]].toPandas())
        self.assert_eq(cdf[cdf.a, 'b', cdf.c].toPandas(), sdf[sdf.a, 'b', sdf.c].toPandas())
        self.assertTrue(isinstance(cdf[0], Column))
        self.assertTrue(isinstance(cdf[1], Column))
        self.assertTrue(isinstance(cdf[2], Column))
        self.assert_eq(cdf[[cdf[0], cdf[1], cdf[2]]].toPandas(), sdf[[sdf[0], sdf[1], sdf[2]]].toPandas())
        with self.assertRaises(PySparkTypeError) as pe:
            cdf[1.5]
        self.check_error(exception=pe.exception, error_class='NOT_COLUMN_OR_INT_OR_LIST_OR_STR_OR_TUPLE', message_parameters={'arg_name': 'item', 'arg_type': 'float'})
        with self.assertRaises(PySparkTypeError) as pe:
            cdf[None]
        self.check_error(exception=pe.exception, error_class='NOT_COLUMN_OR_INT_OR_LIST_OR_STR_OR_TUPLE', message_parameters={'arg_name': 'item', 'arg_type': 'NoneType'})
        with self.assertRaises(PySparkTypeError) as pe:
            cdf[cdf]
        self.check_error(exception=pe.exception, error_class='NOT_COLUMN_OR_INT_OR_LIST_OR_STR_OR_TUPLE', message_parameters={'arg_name': 'item', 'arg_type': 'DataFrame'})

    def test_error_handling(self):
        if False:
            i = 10
            return i + 15
        df = self.connect.range(10).select('id2')
        with self.assertRaises(AnalysisException):
            df.collect()

    def test_simple_read(self):
        if False:
            return 10
        df = self.connect.read.table(self.tbl_name)
        data = df.limit(10).toPandas()
        self.assertEqual(len(data.index), 10)

    def test_json(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as d:
            self.spark.createDataFrame([{'age': 100, 'name': 'Hyukjin Kwon'}]).write.mode('overwrite').format('json').save(d)
            self.assert_eq(self.connect.read.json(d).toPandas(), self.spark.read.json(d).toPandas())
            for schema in ['age INT, name STRING', StructType([StructField('age', IntegerType()), StructField('name', StringType())])]:
                self.assert_eq(self.connect.read.json(path=d, schema=schema).toPandas(), self.spark.read.json(path=d, schema=schema).toPandas())
            self.assert_eq(self.connect.read.json(path=d, primitivesAsString=True).toPandas(), self.spark.read.json(path=d, primitivesAsString=True).toPandas())

    def test_xml(self):
        if False:
            while True:
                i = 10
        tmpPath = tempfile.mkdtemp()
        shutil.rmtree(tmpPath)
        xsdPath = tempfile.mkdtemp()
        xsdString = '<?xml version="1.0" encoding="UTF-8" ?>\n          <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">\n            <xs:element name="person">\n              <xs:complexType>\n                <xs:sequence>\n                  <xs:element name="name" type="xs:string" />\n                  <xs:element name="age" type="xs:long" />\n                </xs:sequence>\n              </xs:complexType>\n            </xs:element>\n          </xs:schema>'
        try:
            xsdFile = os.path.join(xsdPath, 'people.xsd')
            with open(xsdFile, 'w') as f:
                _ = f.write(xsdString)
            df = self.spark.createDataFrame([('Hyukjin', 100), ('Aria', 101), ('Arin', 102)]).toDF('name', 'age')
            df.write.xml(tmpPath, rootTag='people', rowTag='person')
            people = self.spark.read.xml(tmpPath, rowTag='person', rowValidationXSDPath=xsdFile)
            peopleConnect = self.connect.read.xml(tmpPath, rowTag='person', rowValidationXSDPath=xsdFile)
            self.assert_eq(people.toPandas(), peopleConnect.toPandas())
            expected = [Row(age=100, name='Hyukjin'), Row(age=101, name='Aria'), Row(age=102, name='Arin')]
            expectedSchema = StructType([StructField('age', LongType(), True), StructField('name', StringType(), True)])
            self.assertEqual(people.sort('age').collect(), expected)
            self.assertEqual(people.schema, expectedSchema)
            for schema in ['age INT, name STRING', expectedSchema]:
                people = self.spark.read.xml(tmpPath, rowTag='person', rowValidationXSDPath=xsdFile, schema=schema)
                peopleConnect = self.connect.read.xml(tmpPath, rowTag='person', rowValidationXSDPath=xsdFile, schema=schema)
                self.assert_eq(people.toPandas(), peopleConnect.toPandas())
        finally:
            shutil.rmtree(tmpPath)
            shutil.rmtree(xsdPath)

    def test_parquet(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as d:
            self.spark.createDataFrame([{'age': 100, 'name': 'Hyukjin Kwon'}]).write.mode('overwrite').format('parquet').save(d)
            self.assert_eq(self.connect.read.parquet(d).toPandas(), self.spark.read.parquet(d).toPandas())

    def test_text(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as d:
            self.spark.createDataFrame([{'name': 'Sandeep Singh'}, {'name': 'Hyukjin Kwon'}]).write.mode('overwrite').format('text').save(d)
            self.assert_eq(self.connect.read.text(d).toPandas(), self.spark.read.text(d).toPandas())

    def test_csv(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as d:
            self.spark.createDataFrame([{'name': 'Sandeep Singh'}, {'name': 'Hyukjin Kwon'}]).write.mode('overwrite').format('csv').save(d)
            self.assert_eq(self.connect.read.csv(d).toPandas(), self.spark.read.csv(d).toPandas())

    def test_multi_paths(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as d:
            text_files = []
            for i in range(0, 3):
                text_file = f'{d}/text-{i}.text'
                shutil.copyfile('python/test_support/sql/text-test.txt', text_file)
                text_files.append(text_file)
            self.assertEqual(self.connect.read.text(text_files).collect(), self.spark.read.text(text_files).collect())
        with tempfile.TemporaryDirectory() as d:
            json_files = []
            for i in range(0, 5):
                json_file = f'{d}/json-{i}.json'
                shutil.copyfile('python/test_support/sql/people.json', json_file)
                json_files.append(json_file)
            self.assertEqual(self.connect.read.json(json_files).collect(), self.spark.read.json(json_files).collect())

    def test_orc(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as d:
            self.spark.createDataFrame([{'name': 'Sandeep Singh'}, {'name': 'Hyukjin Kwon'}]).write.mode('overwrite').format('orc').save(d)
            self.assert_eq(self.connect.read.orc(d).toPandas(), self.spark.read.orc(d).toPandas())

    def test_join_condition_column_list_columns(self):
        if False:
            for i in range(10):
                print('nop')
        left_connect_df = self.connect.read.table(self.tbl_name)
        right_connect_df = self.connect.read.table(self.tbl_name2)
        left_spark_df = self.spark.read.table(self.tbl_name)
        right_spark_df = self.spark.read.table(self.tbl_name2)
        joined_plan = left_connect_df.join(other=right_connect_df, on=left_connect_df.id == right_connect_df.col1, how='inner')
        joined_plan2 = left_spark_df.join(other=right_spark_df, on=left_spark_df.id == right_spark_df.col1, how='inner')
        self.assert_eq(joined_plan.toPandas(), joined_plan2.toPandas())
        joined_plan3 = left_connect_df.join(other=right_connect_df, on=[left_connect_df.id == right_connect_df.col1, left_connect_df.name == right_connect_df.col2], how='inner')
        joined_plan4 = left_spark_df.join(other=right_spark_df, on=[left_spark_df.id == right_spark_df.col1, left_spark_df.name == right_spark_df.col2], how='inner')
        self.assert_eq(joined_plan3.toPandas(), joined_plan4.toPandas())

    def test_join_ambiguous_cols(self):
        if False:
            return 10
        data1 = [Row(id=1, value='foo'), Row(id=2, value=None)]
        cdf1 = self.connect.createDataFrame(data1)
        sdf1 = self.spark.createDataFrame(data1)
        data2 = [Row(value='bar'), Row(value=None), Row(value='foo')]
        cdf2 = self.connect.createDataFrame(data2)
        sdf2 = self.spark.createDataFrame(data2)
        cdf3 = cdf1.join(cdf2, cdf1['value'] == cdf2['value'])
        sdf3 = sdf1.join(sdf2, sdf1['value'] == sdf2['value'])
        self.assertEqual(cdf3.schema, sdf3.schema)
        self.assertEqual(cdf3.collect(), sdf3.collect())
        cdf4 = cdf1.join(cdf2, cdf1['value'].eqNullSafe(cdf2['value']))
        sdf4 = sdf1.join(sdf2, sdf1['value'].eqNullSafe(sdf2['value']))
        self.assertEqual(cdf4.schema, sdf4.schema)
        self.assertEqual(cdf4.collect(), sdf4.collect())
        cdf5 = cdf1.join(cdf2, (cdf1['value'] == cdf2['value']) & cdf1['value'].eqNullSafe(cdf2['value']))
        sdf5 = sdf1.join(sdf2, (sdf1['value'] == sdf2['value']) & sdf1['value'].eqNullSafe(sdf2['value']))
        self.assertEqual(cdf5.schema, sdf5.schema)
        self.assertEqual(cdf5.collect(), sdf5.collect())
        cdf6 = cdf1.join(cdf2, cdf1['value'] == cdf2['value']).select(cdf1.value)
        sdf6 = sdf1.join(sdf2, sdf1['value'] == sdf2['value']).select(sdf1.value)
        self.assertEqual(cdf6.schema, sdf6.schema)
        self.assertEqual(cdf6.collect(), sdf6.collect())
        cdf7 = cdf1.join(cdf2, cdf1['value'] == cdf2['value']).select(cdf2.value)
        sdf7 = sdf1.join(sdf2, sdf1['value'] == sdf2['value']).select(sdf2.value)
        self.assertEqual(cdf7.schema, sdf7.schema)
        self.assertEqual(cdf7.collect(), sdf7.collect())

    def test_invalid_column(self):
        if False:
            return 10
        data1 = [Row(a=1, b=2, c=3)]
        cdf1 = self.connect.createDataFrame(data1)
        data2 = [Row(a=2, b=0)]
        cdf2 = self.connect.createDataFrame(data2)
        with self.assertRaises(AnalysisException):
            cdf1.select(cdf2.a).schema
        with self.assertRaises(AnalysisException):
            cdf2.withColumn('x', cdf1.a + 1).schema
        with self.assertRaisesRegex(AnalysisException, 'attribute.*missing'):
            cdf3 = cdf1.select(cdf1.a)
            cdf3.select(cdf1.b).schema

    def test_collect(self):
        if False:
            for i in range(10):
                print('nop')
        cdf = self.connect.read.table(self.tbl_name)
        sdf = self.spark.read.table(self.tbl_name)
        data = cdf.limit(10).collect()
        self.assertEqual(len(data), 10)
        self.assertTrue('name' in data[0])
        self.assertTrue('id' in data[0])
        cdf = cdf.select(CF.log('id'), CF.log('id'), CF.struct('id', 'name'), CF.struct('id', 'name')).limit(10)
        sdf = sdf.select(SF.log('id'), SF.log('id'), SF.struct('id', 'name'), SF.struct('id', 'name')).limit(10)
        self.assertEqual(cdf.collect(), sdf.collect())

    def test_collect_timestamp(self):
        if False:
            return 10
        query = "\n            SELECT * FROM VALUES\n            (TIMESTAMP('2022-12-25 10:30:00'), 1),\n            (TIMESTAMP('2022-12-25 10:31:00'), 2),\n            (TIMESTAMP('2022-12-25 10:32:00'), 1),\n            (TIMESTAMP('2022-12-25 10:33:00'), 2),\n            (TIMESTAMP('2022-12-26 09:30:00'), 1),\n            (TIMESTAMP('2022-12-26 09:35:00'), 3)\n            AS tab(date, val)\n            "
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assertEqual(cdf.schema, sdf.schema)
        self.assertEqual(cdf.collect(), sdf.collect())
        self.assertEqual(cdf.select(CF.date_trunc('year', cdf.date).alias('year')).collect(), sdf.select(SF.date_trunc('year', sdf.date).alias('year')).collect())

    def test_with_columns_renamed(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.connect.read.table(self.tbl_name).withColumnRenamed('id', 'id_new').schema, self.spark.read.table(self.tbl_name).withColumnRenamed('id', 'id_new').schema)
        self.assertEqual(self.connect.read.table(self.tbl_name).withColumnsRenamed({'id': 'id_new', 'name': 'name_new'}).schema, self.spark.read.table(self.tbl_name).withColumnsRenamed({'id': 'id_new', 'name': 'name_new'}).schema)

    def test_with_local_data(self):
        if False:
            while True:
                i = 10
        'SPARK-41114: Test creating a dataframe using local data'
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
        df = self.connect.createDataFrame(pdf)
        rows = df.filter(df.a == CF.lit(3)).collect()
        self.assertTrue(len(rows) == 1)
        self.assertEqual(rows[0][0], 3)
        self.assertEqual(rows[0][1], 'c')
        pdf = pd.DataFrame({'a': []})
        with self.assertRaises(ValueError):
            self.connect.createDataFrame(pdf)

    def test_with_local_ndarray(self):
        if False:
            while True:
                i = 10
        'SPARK-41446: Test creating a dataframe using local list'
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        sdf = self.spark.createDataFrame(data)
        cdf = self.connect.createDataFrame(data)
        self.assertEqual(sdf.schema, cdf.schema)
        self.assert_eq(sdf.toPandas(), cdf.toPandas())
        for schema in [StructType([StructField('col1', IntegerType(), True), StructField('col2', IntegerType(), True), StructField('col3', IntegerType(), True), StructField('col4', IntegerType(), True)]), 'struct<col1 int, col2 int, col3 int, col4 int>', 'col1 int, col2 int, col3 int, col4 int', 'col1 int, col2 long, col3 string, col4 long', 'col1 int, col2 string, col3 short, col4 long', ['a', 'b', 'c', 'd'], ('x1', 'x2', 'x3', 'x4')]:
            with self.subTest(schema=schema):
                sdf = self.spark.createDataFrame(data, schema=schema)
                cdf = self.connect.createDataFrame(data, schema=schema)
                self.assertEqual(sdf.schema, cdf.schema)
                self.assert_eq(sdf.toPandas(), cdf.toPandas())
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.createDataFrame(data, ['a', 'b', 'c', 'd', 'e'])
        self.check_error(exception=pe.exception, error_class='AXIS_LENGTH_MISMATCH', message_parameters={'expected_length': '5', 'actual_length': '4'})
        with self.assertRaises(ParseException):
            self.connect.createDataFrame(data, 'col1 magic_type, col2 int, col3 int, col4 int')
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.createDataFrame(data, 'col1 int, col2 int, col3 int')
        self.check_error(exception=pe.exception, error_class='AXIS_LENGTH_MISMATCH', message_parameters={'expected_length': '3', 'actual_length': '4'})
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, float('NaN'), 5.0])
        self.assertEqual(data.ndim, 1)
        sdf = self.spark.createDataFrame(data)
        cdf = self.connect.createDataFrame(data)
        self.assertEqual(sdf.schema, cdf.schema)
        self.assert_eq(sdf.toPandas(), cdf.toPandas())

    def test_with_local_list(self):
        if False:
            for i in range(10):
                print('nop')
        'SPARK-41446: Test creating a dataframe using local list'
        data = [[1, 2, 3, 4]]
        sdf = self.spark.createDataFrame(data)
        cdf = self.connect.createDataFrame(data)
        self.assertEqual(sdf.schema, cdf.schema)
        self.assert_eq(sdf.toPandas(), cdf.toPandas())
        for schema in ['struct<col1 int, col2 int, col3 int, col4 int>', 'col1 int, col2 int, col3 int, col4 int', 'col1 int, col2 long, col3 string, col4 long', 'col1 int, col2 string, col3 short, col4 long', ['a', 'b', 'c', 'd'], ('x1', 'x2', 'x3', 'x4')]:
            sdf = self.spark.createDataFrame(data, schema=schema)
            cdf = self.connect.createDataFrame(data, schema=schema)
            self.assertEqual(sdf.schema, cdf.schema)
            self.assert_eq(sdf.toPandas(), cdf.toPandas())
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.createDataFrame(data, ['a', 'b', 'c', 'd', 'e'])
        self.check_error(exception=pe.exception, error_class='AXIS_LENGTH_MISMATCH', message_parameters={'expected_length': '5', 'actual_length': '4'})
        with self.assertRaises(ParseException):
            self.connect.createDataFrame(data, 'col1 magic_type, col2 int, col3 int, col4 int')
        with self.assertRaisesRegex(ValueError, 'Length mismatch: Expected axis has 3 elements, new values have 4 elements'):
            self.connect.createDataFrame(data, 'col1 int, col2 int, col3 int')

    def test_with_local_rows(self):
        if False:
            i = 10
            return i + 15
        rows = [Row(course='dotNET', year=2012, earnings=10000), Row(course='Java', year=2012, earnings=20000), Row(course='dotNET', year=2012, earnings=5000), Row(course='dotNET', year=2013, earnings=48000), Row(course='Java', year=2013, earnings=30000), Row(course='Scala', year=2022, earnings=None)]
        dicts = [row.asDict() for row in rows]
        for data in [rows, dicts]:
            sdf = self.spark.createDataFrame(data)
            cdf = self.connect.createDataFrame(data)
            self.assertEqual(sdf.schema, cdf.schema)
            self.assert_eq(sdf.toPandas(), cdf.toPandas())
            sdf = self.spark.createDataFrame(data, schema=['a', 'b', 'c'])
            cdf = self.connect.createDataFrame(data, schema=['a', 'b', 'c'])
            self.assertEqual(sdf.schema, cdf.schema)
            self.assert_eq(sdf.toPandas(), cdf.toPandas())

    def test_streaming_local_relation(self):
        if False:
            i = 10
            return i + 15
        threshold_conf = 'spark.sql.session.localRelationCacheThreshold'
        old_threshold = self.connect.conf.get(threshold_conf)
        threshold = 1024 * 1024
        self.connect.conf.set(threshold_conf, threshold)
        try:
            suffix = 'abcdef'
            letters = string.ascii_lowercase
            str = ''.join((random.choice(letters) for i in range(threshold))) + suffix
            data = [[0, str], [1, str]]
            for i in range(0, 2):
                cdf = self.connect.createDataFrame(data, ['a', 'b'])
                self.assert_eq(cdf.count(), len(data))
                self.assert_eq(cdf.filter(f"endsWith(b, '{suffix}')").isEmpty(), False)
        finally:
            self.connect.conf.set(threshold_conf, old_threshold)

    def test_with_atom_type(self):
        if False:
            return 10
        for data in [[1, 2, 3], [1, 2, 3]]:
            for schema in ['long', 'int', 'short']:
                sdf = self.spark.createDataFrame(data, schema=schema)
                cdf = self.connect.createDataFrame(data, schema=schema)
                self.assertEqual(sdf.schema, cdf.schema)
                self.assert_eq(sdf.toPandas(), cdf.toPandas())

    def test_with_none_and_nan(self):
        if False:
            for i in range(10):
                print('nop')
        data1 = [Row(id=1, value=float('NaN')), Row(id=2, value=42.0), Row(id=3, value=None)]
        data2 = [Row(id=1, value=np.nan), Row(id=2, value=42.0), Row(id=3, value=None)]
        data3 = [{'id': 1, 'value': float('NaN')}, {'id': 2, 'value': 42.0}, {'id': 3, 'value': None}]
        data4 = [{'id': 1, 'value': np.nan}, {'id': 2, 'value': 42.0}, {'id': 3, 'value': None}]
        data5 = [(1, float('NaN')), (2, 42.0), (3, None)]
        data6 = [(1, np.nan), (2, 42.0), (3, None)]
        data7 = np.array([[1, float('NaN')], [2, 42.0], [3, None]])
        data8 = np.array([[1, np.nan], [2, 42.0], [3, None]])
        for data in [data1, data2, data3, data4, data5, data6, data7, data8]:
            if isinstance(data[0], (Row, dict)):
                cdf = self.connect.createDataFrame(data)
                sdf = self.spark.createDataFrame(data)
            else:
                cdf = self.connect.createDataFrame(data, schema=['id', 'value'])
                sdf = self.spark.createDataFrame(data, schema=['id', 'value'])
            self.assert_eq(cdf.toPandas(), sdf.toPandas())
            self.assert_eq(cdf.select(cdf['value'].eqNullSafe(None), cdf['value'].eqNullSafe(float('NaN')), cdf['value'].eqNullSafe(42.0)).toPandas(), sdf.select(sdf['value'].eqNullSafe(None), sdf['value'].eqNullSafe(float('NaN')), sdf['value'].eqNullSafe(42.0)).toPandas())
        data = [(1.0, float('nan')), (float('nan'), 2.0)]
        cdf = self.connect.createDataFrame(data, ('a', 'b'))
        sdf = self.spark.createDataFrame(data, ('a', 'b'))
        self.assert_eq(cdf.toPandas(), sdf.toPandas())
        self.assert_eq(cdf.select(CF.nanvl('a', 'b').alias('r1'), CF.nanvl(cdf.a, cdf.b).alias('r2')).toPandas(), sdf.select(SF.nanvl('a', 'b').alias('r1'), SF.nanvl(sdf.a, sdf.b).alias('r2')).toPandas())
        data = [(1.0, float('nan')), (float('nan'), 2.0), (10.0, 3.0), (float('nan'), float('nan')), (-3.0, 4.0), (-10.0, 3.0), (-5.0, -6.0), (7.0, -8.0), (1.0, 2.0)]
        cdf = self.connect.createDataFrame(data, ('a', 'b'))
        sdf = self.spark.createDataFrame(data, ('a', 'b'))
        self.assert_eq(cdf.toPandas(), sdf.toPandas())
        self.assert_eq(cdf.select(CF.pmod('a', 'b')).toPandas(), sdf.select(SF.pmod('a', 'b')).toPandas())

    def test_cast_with_ddl(self):
        if False:
            while True:
                i = 10
        data = [Row(date=datetime.date(2021, 12, 27), add=2)]
        cdf = self.connect.createDataFrame(data, 'date date, add integer')
        sdf = self.spark.createDataFrame(data, 'date date, add integer')
        self.assertEqual(cdf.schema, sdf.schema)

    def test_create_empty_df(self):
        if False:
            return 10
        for schema in ['STRING', 'x STRING', 'x STRING, y INTEGER', StringType(), StructType([StructField('x', StringType(), True), StructField('y', IntegerType(), True)])]:
            cdf = self.connect.createDataFrame(data=[], schema=schema)
            sdf = self.spark.createDataFrame(data=[], schema=schema)
            self.assert_eq(cdf.toPandas(), sdf.toPandas())
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.createDataFrame(data=[])
        self.check_error(exception=pe.exception, error_class='CANNOT_INFER_EMPTY_SCHEMA', message_parameters={})

    def test_create_dataframe_from_arrays(self):
        if False:
            return 10
        data1 = [Row(a=1, b=array.array('i', [1, 2, 3]), c=array.array('d', [4, 5, 6]))]
        data2 = [(array.array('d', [1, 2, 3]), 2, '3')]
        data3 = [{'a': 1, 'b': array.array('i', [1, 2, 3])}]
        for data in [data1, data2, data3]:
            cdf = self.connect.createDataFrame(data)
            sdf = self.spark.createDataFrame(data)
            self.assertEqual(cdf.collect(), sdf.collect())

    def test_timestampe_create_from_rows(self):
        if False:
            return 10
        data = [(datetime.datetime(2016, 3, 11, 9, 0, 7), 1)]
        cdf = self.connect.createDataFrame(data, ['date', 'val'])
        sdf = self.spark.createDataFrame(data, ['date', 'val'])
        self.assertEqual(cdf.schema, sdf.schema)
        self.assertEqual(cdf.collect(), sdf.collect())

    def test_create_dataframe_with_coercion(self):
        if False:
            while True:
                i = 10
        data1 = [[1.33, 1], ['2.1', 1]]
        data2 = [[True, 1], ['false', 1]]
        for data in [data1, data2]:
            cdf = self.connect.createDataFrame(data, ['a', 'b'])
            sdf = self.spark.createDataFrame(data, ['a', 'b'])
            self.assertEqual(cdf.schema, sdf.schema)
            self.assertEqual(cdf.collect(), sdf.collect())

    def test_nested_type_create_from_rows(self):
        if False:
            print('Hello World!')
        data1 = [Row(a=1, b=Row(c=2, d=Row(e=3, f=Row(g=4, h=Row(i=5)))))]
        data2 = [(1, 'a', Row(a=1, b=[1, 2, 3], c={'a': 'b'}, d=Row(x=1, y='y', z=Row(o=1, p=2, q=Row(g=1.5)))))]
        data3 = [Row(a=1, b=[1, 2, 3], c={'a': 'b'}, d=Row(x=1, y='y', z=Row(1, 2, 3)), e=list('hello connect'))]
        data4 = [{'a': 1, 'b': Row(x=1, y=Row(z=2)), 'c': {'x': -1, 'y': 2}, 'd': [1, 2, 3, 4, 5]}]
        data5 = [{'a': [Row(x=1, y='2'), Row(x=-1, y='-2')], 'b': [[1, 2, 3], [4, 5], [6]], 'c': {3: {4: {5: 6}}, 7: {8: {9: 0}}}}]
        for data in [data1, data2, data3, data4, data5]:
            with self.subTest(data=data):
                cdf = self.connect.createDataFrame(data)
                sdf = self.spark.createDataFrame(data)
                self.assertEqual(cdf.schema, sdf.schema)
                self.assertEqual(cdf.collect(), sdf.collect())

    def test_create_df_from_objects(self):
        if False:
            return 10
        data = [MyObject(1, '1'), MyObject(2, '2')]
        cdf = self.connect.createDataFrame(data)
        sdf = self.spark.createDataFrame(data)
        self.assertEqual(cdf.schema, sdf.schema)
        self.assertEqual(cdf.collect(), sdf.collect())

    def test_simple_explain_string(self):
        if False:
            return 10
        df = self.connect.read.table(self.tbl_name).limit(10)
        result = df._explain_string()
        self.assertGreater(len(result), 0)

    def test_schema(self):
        if False:
            i = 10
            return i + 15
        schema = self.connect.read.table(self.tbl_name).schema
        self.assertEqual(StructType([StructField('id', LongType(), True), StructField('name', StringType(), True)]), schema)
        query = '\n            SELECT * FROM VALUES\n            (float(1.0), double(1.0), 1.0, "1", true, NULL),\n            (float(2.0), double(2.0), 2.0, "2", false, NULL),\n            (float(3.0), double(3.0), NULL, "3", false, NULL)\n            AS tab(a, b, c, d, e, f)\n            '
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)
        query = "\n            SELECT * FROM VALUES\n            (TIMESTAMP('2019-04-12 15:50:00'), DATE('2022-02-22')),\n            (TIMESTAMP('2019-04-12 15:50:00'), NULL),\n            (NULL, DATE('2022-02-22'))\n            AS tab(a, b)\n            "
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)
        query = " SELECT INTERVAL '100 10:30' DAY TO MINUTE AS interval "
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)
        query = "\n            SELECT * FROM VALUES\n            (MAP('a', 'ab'), MAP('a', 'ab'), MAP(1, 2, 3, 4)),\n            (MAP('x', 'yz'), MAP('x', NULL), NULL),\n            (MAP('c', 'de'), NULL, MAP(-1, NULL, -3, -4))\n            AS tab(a, b, c)\n            "
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)
        query = "\n            SELECT * FROM VALUES\n            (ARRAY('a', 'ab'), ARRAY(1, 2, 3), ARRAY(1, NULL, 3)),\n            (ARRAY('x', NULL), NULL, ARRAY(1, 3)),\n            (NULL, ARRAY(-1, -2, -3), Array())\n            AS tab(a, b, c)\n            "
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)
        query = '\n            SELECT STRUCT(a, b, c, d), STRUCT(e, f, g), STRUCT(STRUCT(a, b), STRUCT(h)) FROM VALUES\n            (float(1.0), double(1.0), 1.0, "1", true, NULL, ARRAY(1, NULL, 3), MAP(1, 2, 3, 4)),\n            (float(2.0), double(2.0), 2.0, "2", false, NULL, ARRAY(1, 3), MAP(1, NULL, 3, 4)),\n            (float(3.0), double(3.0), NULL, "3", false, NULL, ARRAY(NULL), NULL)\n            AS tab(a, b, c, d, e, f, g, h)\n            '
        self.assertEqual(self.spark.sql(query).schema, self.connect.sql(query).schema)

    def test_to(self):
        if False:
            print('Hello World!')
        cdf = self.connect.read.table(self.tbl_name)
        df = self.spark.read.table(self.tbl_name)

        def assert_eq_schema(cdf: CDataFrame, df: DataFrame, schema: StructType):
            if False:
                i = 10
                return i + 15
            cdf_to = cdf.to(schema)
            df_to = df.to(schema)
            self.assertEqual(cdf_to.schema, df_to.schema)
            self.assert_eq(cdf_to.toPandas(), df_to.toPandas())
        schema = StructType([StructField('id', IntegerType(), True), StructField('name', StringType(), True)])
        assert_eq_schema(cdf, df, schema)
        schema2 = StructType([StructField('struct', schema, False)])
        cdf_to = cdf.select(CF.struct('id', 'name').alias('struct')).to(schema2)
        df_to = df.select(SF.struct('id', 'name').alias('struct')).to(schema2)
        self.assertEqual(cdf_to.schema, df_to.schema)
        schema = StructType([StructField('col1', IntegerType(), True), StructField('col2', StringType(), True)])
        assert_eq_schema(cdf, df, schema)
        schema = StructType([StructField('id', StringType(), True), StructField('name', StringType(), True)])
        assert_eq_schema(cdf, df, schema)
        schema = StructType([StructField('id', LongType(), True)])
        assert_eq_schema(cdf, df, schema)
        schema = StructType([StructField('id', LongType(), False)])
        self.assertRaisesRegex(AnalysisException, 'NULLABLE_COLUMN_OR_FIELD', lambda : cdf.to(schema).toPandas())
        schema = StructType([StructField('name', LongType())])
        self.assertRaisesRegex(AnalysisException, 'INVALID_COLUMN_OR_FIELD_DATA_TYPE', lambda : cdf.to(schema).toPandas())
        schema = StructType([StructField('id', IntegerType(), True), StructField('name', IntegerType(), True)])
        self.assertRaisesRegex(AnalysisException, 'INVALID_COLUMN_OR_FIELD_DATA_TYPE', lambda : cdf.to(schema).toPandas())
        schema = StructType([StructField('id', StringType(), True), StructField('my_map', MapType(StringType(), IntegerType(), False), True), StructField('my_array', ArrayType(IntegerType(), False), True)])
        cdf = self.connect.read.table(self.tbl_name4)
        df = self.spark.read.table(self.tbl_name4)
        assert_eq_schema(cdf, df, schema)

    def test_toDF(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.connect.read.table(self.tbl_name).toDF('col1', 'col2').schema, self.spark.read.table(self.tbl_name).toDF('col1', 'col2').schema)

    def test_print_schema(self):
        if False:
            i = 10
            return i + 15
        tree_str = self.connect.sql('SELECT 1 AS X, 2 AS Y')._tree_string()
        expected = 'root\n |-- X: integer (nullable = false)\n |-- Y: integer (nullable = false)\n'
        self.assertEqual(tree_str, expected)

    def test_is_local(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.connect.sql('SHOW DATABASES').isLocal())
        self.assertFalse(self.connect.read.table(self.tbl_name).isLocal())

    def test_is_streaming(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.connect.read.table(self.tbl_name).isStreaming)
        self.assertFalse(self.connect.sql('SELECT 1 AS X LIMIT 0').isStreaming)

    def test_input_files(self):
        if False:
            print('Hello World!')
        tmpPath = tempfile.mkdtemp()
        shutil.rmtree(tmpPath)
        try:
            self.df_text.write.text(tmpPath)
            input_files_list1 = self.spark.read.format('text').schema('id STRING').load(path=tmpPath).inputFiles()
            input_files_list2 = self.connect.read.format('text').schema('id STRING').load(path=tmpPath).inputFiles()
            self.assertTrue(len(input_files_list1) > 0)
            self.assertEqual(len(input_files_list1), len(input_files_list2))
            for file_path in input_files_list2:
                self.assertTrue(file_path in input_files_list1)
        finally:
            shutil.rmtree(tmpPath)

    def test_limit_offset(self):
        if False:
            i = 10
            return i + 15
        df = self.connect.read.table(self.tbl_name)
        pd = df.limit(10).offset(1).toPandas()
        self.assertEqual(9, len(pd.index))
        pd2 = df.offset(98).limit(10).toPandas()
        self.assertEqual(2, len(pd2.index))

    def test_tail(self):
        if False:
            i = 10
            return i + 15
        df = self.connect.read.table(self.tbl_name)
        df2 = self.spark.read.table(self.tbl_name)
        self.assertEqual(df.tail(10), df2.tail(10))

    def test_sql(self):
        if False:
            print('Hello World!')
        pdf = self.connect.sql('SELECT 1').toPandas()
        self.assertEqual(1, len(pdf.index))

    def test_sql_with_named_args(self):
        if False:
            return 10
        sqlText = "SELECT *, element_at(:m, 'a') FROM range(10) WHERE id > :minId"
        df = self.connect.sql(sqlText, args={'minId': 7, 'm': CF.create_map(CF.lit('a'), CF.lit(1))})
        df2 = self.spark.sql(sqlText, args={'minId': 7, 'm': SF.create_map(SF.lit('a'), SF.lit(1))})
        self.assert_eq(df.toPandas(), df2.toPandas())

    def test_sql_with_pos_args(self):
        if False:
            return 10
        sqlText = 'SELECT *, element_at(?, 1) FROM range(10) WHERE id > ?'
        df = self.connect.sql(sqlText, args=[CF.array(CF.lit(1)), 7])
        df2 = self.spark.sql(sqlText, args=[SF.array(SF.lit(1)), 7])
        self.assert_eq(df.toPandas(), df2.toPandas())

    def test_head(self):
        if False:
            return 10
        df = self.connect.read.table(self.tbl_name)
        self.assertIsNotNone(len(df.head()))
        self.assertIsNotNone(len(df.head(1)))
        self.assertIsNotNone(len(df.head(5)))
        df2 = self.connect.read.table(self.tbl_name_empty)
        self.assertIsNone(df2.head())

    def test_deduplicate(self):
        if False:
            print('Hello World!')
        df = self.connect.read.table(self.tbl_name)
        df2 = self.spark.read.table(self.tbl_name)
        self.assert_eq(df.distinct().toPandas(), df2.distinct().toPandas())
        self.assert_eq(df.dropDuplicates().toPandas(), df2.dropDuplicates().toPandas())
        self.assert_eq(df.dropDuplicates(['name']).toPandas(), df2.dropDuplicates(['name']).toPandas())

    def test_deduplicate_within_watermark_in_batch(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.connect.read.table(self.tbl_name)
        with self.assertRaisesRegex(AnalysisException, 'dropDuplicatesWithinWatermark is not supported with batch DataFrames/DataSets'):
            df.dropDuplicatesWithinWatermark().toPandas()

    def test_first(self):
        if False:
            return 10
        df = self.connect.read.table(self.tbl_name)
        self.assertIsNotNone(len(df.first()))
        df2 = self.connect.read.table(self.tbl_name_empty)
        self.assertIsNone(df2.first())

    def test_take(self) -> None:
        if False:
            while True:
                i = 10
        df = self.connect.read.table(self.tbl_name)
        self.assertEqual(5, len(df.take(5)))
        df2 = self.connect.read.table(self.tbl_name_empty)
        self.assertEqual(0, len(df2.take(5)))

    def test_drop(self):
        if False:
            for i in range(10):
                print('nop')
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL), (false, NULL, 2), (NULL, 3, 3)\n            AS tab(a, b, c)\n            '
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assert_eq(cdf.drop('a').toPandas(), sdf.drop('a').toPandas())
        self.assert_eq(cdf.drop('a', 'b').toPandas(), sdf.drop('a', 'b').toPandas())
        self.assert_eq(cdf.drop('a', 'x').toPandas(), sdf.drop('a', 'x').toPandas())
        self.assert_eq(cdf.drop(cdf.a, 'x').toPandas(), sdf.drop(sdf.a, 'x').toPandas())

    def test_subquery_alias(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        plan_text = self.connect.read.table(self.tbl_name).alias('special_alias')._explain_string(extended=True)
        self.assertTrue('special_alias' in plan_text)

    def test_sort(self):
        if False:
            for i in range(10):
                print('nop')
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL), (false, NULL, 2.0), (NULL, 3, 3.0)\n            AS tab(a, b, c)\n            '
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assert_eq(cdf.sort('a').toPandas(), sdf.sort('a').toPandas())
        self.assert_eq(cdf.sort('c').toPandas(), sdf.sort('c').toPandas())
        self.assert_eq(cdf.sort('b').toPandas(), sdf.sort('b').toPandas())
        self.assert_eq(cdf.sort(cdf.c, 'b').toPandas(), sdf.sort(sdf.c, 'b').toPandas())
        self.assert_eq(cdf.sort(cdf.c.desc(), 'b').toPandas(), sdf.sort(sdf.c.desc(), 'b').toPandas())
        self.assert_eq(cdf.sort(cdf.c.desc(), cdf.a.asc()).toPandas(), sdf.sort(sdf.c.desc(), sdf.a.asc()).toPandas())

    def test_range(self):
        if False:
            return 10
        self.assert_eq(self.connect.range(start=0, end=10).toPandas(), self.spark.range(start=0, end=10).toPandas())
        self.assert_eq(self.connect.range(start=0, end=10, step=3).toPandas(), self.spark.range(start=0, end=10, step=3).toPandas())
        self.assert_eq(self.connect.range(start=0, end=10, step=3, numPartitions=2).toPandas(), self.spark.range(start=0, end=10, step=3, numPartitions=2).toPandas())
        self.assert_eq(self.connect.range(10).toPandas(), self.connect.range(start=0, end=10).toPandas())

    def test_create_global_temp_view(self):
        if False:
            for i in range(10):
                print('nop')
        with self.tempView('view_1'):
            self.connect.sql('SELECT 1 AS X LIMIT 0').createGlobalTempView('view_1')
            self.connect.sql('SELECT 2 AS X LIMIT 1').createOrReplaceGlobalTempView('view_1')
            self.assertTrue(self.spark.catalog.tableExists('global_temp.view_1'))
            self.assertTrue(self.spark.catalog.tableExists('global_temp.view_1'))
            with self.assertRaises(AnalysisException):
                self.connect.sql('SELECT 1 AS X LIMIT 0').createGlobalTempView('view_1')

    def test_create_session_local_temp_view(self):
        if False:
            while True:
                i = 10
        with self.tempView('view_local_temp'):
            self.connect.sql('SELECT 1 AS X').createTempView('view_local_temp')
            self.assertEqual(self.connect.sql('SELECT * FROM view_local_temp').count(), 1)
            self.connect.sql('SELECT 1 AS X LIMIT 0').createOrReplaceTempView('view_local_temp')
            self.assertEqual(self.connect.sql('SELECT * FROM view_local_temp').count(), 0)
            with self.assertRaises(AnalysisException):
                self.connect.sql('SELECT 1 AS X LIMIT 0').createTempView('view_local_temp')

    def test_to_pandas(self):
        if False:
            while True:
                i = 10
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL),\n            (false, NULL, float(2.0)),\n            (NULL, 3, float(3.0))\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).toPandas(), self.spark.sql(query).toPandas())
        query = '\n            SELECT * FROM VALUES\n            (1, 1, NULL),\n            (2, NULL, float(2.0)),\n            (3, 3, float(3.0))\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).toPandas(), self.spark.sql(query).toPandas())
        query = '\n            SELECT * FROM VALUES\n            (double(1.0), 1, "1"),\n            (NULL, NULL, NULL),\n            (double(2.0), 3, "3")\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).toPandas(), self.spark.sql(query).toPandas())
        query = '\n            SELECT * FROM VALUES\n            (float(1.0), double(1.0), 1, "1"),\n            (float(2.0), double(2.0), 2, "2"),\n            (float(3.0), double(3.0), 3, "3")\n            AS tab(a, b, c, d)\n            '
        self.assert_eq(self.connect.sql(query).toPandas(), self.spark.sql(query).toPandas())

    def test_create_dataframe_from_pandas_with_ns_timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        'Truncate the timestamps for nanoseconds.'
        from datetime import datetime, timezone, timedelta
        from pandas import Timestamp
        import pandas as pd
        pdf = pd.DataFrame({'naive': [datetime(2019, 1, 1, 0)], 'aware': [Timestamp(year=2019, month=1, day=1, nanosecond=500, tz=timezone(timedelta(hours=-8)))]})
        with self.sql_conf({'spark.sql.execution.arrow.pyspark.enabled': False}):
            self.assertEqual(self.connect.createDataFrame(pdf).collect(), self.spark.createDataFrame(pdf).collect())
        with self.sql_conf({'spark.sql.execution.arrow.pyspark.enabled': True}):
            self.assertEqual(self.connect.createDataFrame(pdf).collect(), self.spark.createDataFrame(pdf).collect())

    def test_select_expr(self):
        if False:
            while True:
                i = 10
        self.assert_eq(self.connect.read.table(self.tbl_name).selectExpr('id * 2').toPandas(), self.spark.read.table(self.tbl_name).selectExpr('id * 2').toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).selectExpr(['id * 2', 'cast(name as long) as name']).toPandas(), self.spark.read.table(self.tbl_name).selectExpr(['id * 2', 'cast(name as long) as name']).toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).selectExpr('id * 2', 'cast(name as long) as name').toPandas(), self.spark.read.table(self.tbl_name).selectExpr('id * 2', 'cast(name as long) as name').toPandas())

    def test_select_star(self):
        if False:
            for i in range(10):
                print('nop')
        data = [Row(a=1, b=Row(c=2, d=Row(e=3)))]
        cdf = self.connect.createDataFrame(data=data)
        sdf = self.spark.createDataFrame(data=data)
        self.assertEqual(cdf.select('*').collect(), sdf.select('*').collect())
        self.assertEqual(cdf.select('a', '*').collect(), sdf.select('a', '*').collect())
        self.assertEqual(cdf.select('a', 'b').collect(), sdf.select('a', 'b').collect())
        self.assertEqual(cdf.select('a', 'b.*').collect(), sdf.select('a', 'b.*').collect())

    def test_fill_na(self):
        if False:
            for i in range(10):
                print('nop')
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL), (false, NULL, 2.0), (NULL, 3, 3.0)\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).fillna(True).toPandas(), self.spark.sql(query).fillna(True).toPandas())
        self.assert_eq(self.connect.sql(query).fillna(2).toPandas(), self.spark.sql(query).fillna(2).toPandas())
        self.assert_eq(self.connect.sql(query).fillna(2, ['a', 'b']).toPandas(), self.spark.sql(query).fillna(2, ['a', 'b']).toPandas())
        self.assert_eq(self.connect.sql(query).na.fill({'a': True, 'b': 2}).toPandas(), self.spark.sql(query).na.fill({'a': True, 'b': 2}).toPandas())

    def test_drop_na(self):
        if False:
            while True:
                i = 10
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL), (false, NULL, 2.0), (NULL, 3, 3.0)\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).dropna().toPandas(), self.spark.sql(query).dropna().toPandas())
        self.assert_eq(self.connect.sql(query).na.drop(how='all', thresh=1).toPandas(), self.spark.sql(query).na.drop(how='all', thresh=1).toPandas())
        self.assert_eq(self.connect.sql(query).dropna(thresh=1, subset=('a', 'b')).toPandas(), self.spark.sql(query).dropna(thresh=1, subset=('a', 'b')).toPandas())
        self.assert_eq(self.connect.sql(query).na.drop(how='any', thresh=2, subset='a').toPandas(), self.spark.sql(query).na.drop(how='any', thresh=2, subset='a').toPandas())

    def test_replace(self):
        if False:
            print('Hello World!')
        query = '\n            SELECT * FROM VALUES\n            (false, 1, NULL), (false, NULL, 2.0), (NULL, 3, 3.0)\n            AS tab(a, b, c)\n            '
        self.assert_eq(self.connect.sql(query).replace(2, 3).toPandas(), self.spark.sql(query).replace(2, 3).toPandas())
        self.assert_eq(self.connect.sql(query).na.replace(False, True).toPandas(), self.spark.sql(query).na.replace(False, True).toPandas())
        self.assert_eq(self.connect.sql(query).replace({1: 2, 3: -1}, subset=('a', 'b')).toPandas(), self.spark.sql(query).replace({1: 2, 3: -1}, subset=('a', 'b')).toPandas())
        self.assert_eq(self.connect.sql(query).na.replace((1, 2), (3, 1)).toPandas(), self.spark.sql(query).na.replace((1, 2), (3, 1)).toPandas())
        self.assert_eq(self.connect.sql(query).na.replace((1, 2), (3, 1), subset=('c', 'b')).toPandas(), self.spark.sql(query).na.replace((1, 2), (3, 1), subset=('c', 'b')).toPandas())
        with self.assertRaises(ValueError) as context:
            self.connect.sql(query).replace({None: 1}, subset='a').toPandas()
            self.assertTrue('Mixed type replacements are not supported' in str(context.exception))
        with self.assertRaises(AnalysisException) as context:
            self.connect.sql(query).replace({1: 2, 3: -1}, subset=('a', 'x')).toPandas()
            self.assertIn('Cannot resolve column name "x" among (a, b, c)', str(context.exception))

    def test_unpivot(self):
        if False:
            return 10
        self.assert_eq(self.connect.read.table(self.tbl_name).filter('id > 3').unpivot(['id'], ['name'], 'variable', 'value').toPandas(), self.spark.read.table(self.tbl_name).filter('id > 3').unpivot(['id'], ['name'], 'variable', 'value').toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).filter('id > 3').unpivot('id', None, 'variable', 'value').toPandas(), self.spark.read.table(self.tbl_name).filter('id > 3').unpivot('id', None, 'variable', 'value').toPandas())

    def test_union_by_name(self):
        if False:
            while True:
                i = 10
        data1 = [(1, 2, 3)]
        data2 = [(6, 2, 5)]
        df1_connect = self.connect.createDataFrame(data1, ['a', 'b', 'c'])
        df2_connect = self.connect.createDataFrame(data2, ['a', 'b', 'c'])
        union_df_connect = df1_connect.unionByName(df2_connect)
        df1_spark = self.spark.createDataFrame(data1, ['a', 'b', 'c'])
        df2_spark = self.spark.createDataFrame(data2, ['a', 'b', 'c'])
        union_df_spark = df1_spark.unionByName(df2_spark)
        self.assert_eq(union_df_connect.toPandas(), union_df_spark.toPandas())
        df2_connect = self.connect.createDataFrame(data2, ['a', 'B', 'C'])
        union_df_connect = df1_connect.unionByName(df2_connect, allowMissingColumns=True)
        df2_spark = self.spark.createDataFrame(data2, ['a', 'B', 'C'])
        union_df_spark = df1_spark.unionByName(df2_spark, allowMissingColumns=True)
        self.assert_eq(union_df_connect.toPandas(), union_df_spark.toPandas())

    def test_random_split(self):
        if False:
            for i in range(10):
                print('nop')
        relations = self.connect.read.table(self.tbl_name).filter('id > 3').randomSplit([1.0, 2.0, 3.0], 2)
        datasets = self.spark.read.table(self.tbl_name).filter('id > 3').randomSplit([1.0, 2.0, 3.0], 2)
        self.assertTrue(len(relations) == len(datasets))
        i = 0
        while i < len(relations):
            self.assert_eq(relations[i].toPandas(), datasets[i].toPandas())
            i += 1

    def test_observe(self):
        if False:
            print('Hello World!')
        observation_name = 'my_metric'
        self.assert_eq(self.connect.read.table(self.tbl_name).filter('id > 3').observe(observation_name, CF.min('id'), CF.max('id'), CF.sum('id')).toPandas(), self.spark.read.table(self.tbl_name).filter('id > 3').observe(observation_name, SF.min('id'), SF.max('id'), SF.sum('id')).toPandas())
        from pyspark.sql.connect.observation import Observation as ConnectObservation
        from pyspark.sql.observation import Observation
        cobservation = ConnectObservation(observation_name)
        observation = Observation(observation_name)
        cdf = self.connect.read.table(self.tbl_name).filter('id > 3').observe(cobservation, CF.min('id'), CF.max('id'), CF.sum('id')).toPandas()
        df = self.spark.read.table(self.tbl_name).filter('id > 3').observe(observation, SF.min('id'), SF.max('id'), SF.sum('id')).toPandas()
        self.assert_eq(cdf, df)
        self.assertEquals(cobservation.get, observation.get)
        observed_metrics = cdf.attrs['observed_metrics']
        self.assert_eq(len(observed_metrics), 1)
        self.assert_eq(observed_metrics[0].name, observation_name)
        self.assert_eq(len(observed_metrics[0].metrics), 3)
        for metric in observed_metrics[0].metrics:
            self.assertIsInstance(metric, ProtoExpression.Literal)
        values = list(map(lambda metric: metric.long, observed_metrics[0].metrics))
        self.assert_eq(values, [4, 99, 4944])
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.read.table(self.tbl_name).observe(observation_name)
        self.check_error(exception=pe.exception, error_class='CANNOT_BE_EMPTY', message_parameters={'item': 'exprs'})
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name).observe(observation_name, CF.lit(1), 'id')
        self.check_error(exception=pe.exception, error_class='NOT_LIST_OF_COLUMN', message_parameters={'arg_name': 'exprs'})

    def test_with_columns(self):
        if False:
            i = 10
            return i + 15
        self.assert_eq(self.connect.read.table(self.tbl_name).withColumn('id', CF.lit(False)).toPandas(), self.spark.read.table(self.tbl_name).withColumn('id', SF.lit(False)).toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).withColumns({'id': CF.lit(False), 'col_not_exist': CF.lit(False)}).toPandas(), self.spark.read.table(self.tbl_name).withColumns({'id': SF.lit(False), 'col_not_exist': SF.lit(False)}).toPandas())

    def test_hint(self):
        if False:
            while True:
                i = 10
        self.assert_eq(self.connect.read.table(self.tbl_name).hint('COALESCE', 3000).toPandas(), self.spark.read.table(self.tbl_name).hint('COALESCE', 3000).toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).hint('illegal').toPandas(), self.spark.read.table(self.tbl_name).hint('illegal').toPandas())
        such_a_nice_list = ['itworks1', 'itworks2', 'itworks3']
        self.assert_eq(self.connect.read.table(self.tbl_name).hint('my awesome hint', 1.2345, 2).toPandas(), self.spark.read.table(self.tbl_name).hint('my awesome hint', 1.2345, 2).toPandas())
        with self.assertRaises(AnalysisException):
            self.connect.read.table(self.tbl_name).hint('REPARTITION', 'id+1').toPandas()
        with self.assertRaises(TypeError):
            self.connect.read.table(self.tbl_name).hint('REPARTITION', range(5)).toPandas()
        with self.assertRaises(TypeError):
            self.connect.read.table(self.tbl_name).hint('my awesome hint', 1.2345, 2, such_a_nice_list, range(6)).toPandas()
        with self.assertRaises(AnalysisException):
            self.connect.read.table(self.tbl_name).hint('REPARTITION', 'id', 3).toPandas()

    def test_join_hint(self):
        if False:
            while True:
                i = 10
        cdf1 = self.connect.createDataFrame([(2, 'Alice'), (5, 'Bob')], schema=['age', 'name'])
        cdf2 = self.connect.createDataFrame([Row(height=80, name='Tom'), Row(height=85, name='Bob')])
        self.assertTrue('BroadcastHashJoin' in cdf1.join(cdf2.hint('BROADCAST'), 'name')._explain_string())
        self.assertTrue('SortMergeJoin' in cdf1.join(cdf2.hint('MERGE'), 'name')._explain_string())
        self.assertTrue('ShuffledHashJoin' in cdf1.join(cdf2.hint('SHUFFLE_HASH'), 'name')._explain_string())

    def test_different_spark_session_join_or_union(self):
        if False:
            i = 10
            return i + 15
        df = self.connect.range(10).limit(3)
        spark2 = RemoteSparkSession(connection='sc://localhost')
        df2 = spark2.range(10).limit(3)
        with self.assertRaises(SessionNotSameException) as e1:
            df.union(df2).collect()
        self.check_error(exception=e1.exception, error_class='SESSION_NOT_SAME', message_parameters={})
        with self.assertRaises(SessionNotSameException) as e2:
            df.unionByName(df2).collect()
        self.check_error(exception=e2.exception, error_class='SESSION_NOT_SAME', message_parameters={})
        with self.assertRaises(SessionNotSameException) as e3:
            df.join(df2).collect()
        self.check_error(exception=e3.exception, error_class='SESSION_NOT_SAME', message_parameters={})

    def test_extended_hint_types(self):
        if False:
            for i in range(10):
                print('nop')
        cdf = self.connect.range(100).toDF('id')
        cdf.hint('my awesome hint', 1.2345, 'what', ['itworks1', 'itworks2', 'itworks3']).show()
        with self.assertRaises(PySparkTypeError) as pe:
            cdf.hint('my awesome hint', 1.2345, 'what', {'itworks1': 'itworks2'}).show()
        self.check_error(exception=pe.exception, error_class='INVALID_ITEM_FOR_CONTAINER', message_parameters={'arg_name': 'parameters', 'allowed_types': 'str, float, int, Column, list[str], list[float], list[int]', 'item_type': 'dict'})

    def test_empty_dataset(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.connect.sql('SELECT 1 AS X LIMIT 0').toPandas().equals(self.spark.sql('SELECT 1 AS X LIMIT 0').toPandas()))
        pdf = self.connect.sql('SELECT 1 AS X LIMIT 0').toPandas()
        self.assertEqual(0, len(pdf))
        self.assertEqual(1, len(pdf.columns))
        self.assertEqual('X', pdf.columns[0])

    def test_is_empty(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.connect.sql('SELECT 1 AS X').isEmpty())
        self.assertTrue(self.connect.sql('SELECT 1 AS X LIMIT 0').isEmpty())

    def test_session(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.connect, self.connect.sql('SELECT 1').sparkSession)

    def test_show(self):
        if False:
            for i in range(10):
                print('nop')
        show_str = self.connect.sql('SELECT 1 AS X, 2 AS Y')._show_string()
        expected = '+---+---+\n|  X|  Y|\n+---+---+\n|  1|  2|\n+---+---+\n'
        self.assertEqual(show_str, expected)

    def test_describe(self):
        if False:
            return 10
        self.assert_eq(self.connect.read.table(self.tbl_name).describe('id').toPandas(), self.spark.read.table(self.tbl_name).describe('id').toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).describe('id', 'name').toPandas(), self.spark.read.table(self.tbl_name).describe('id', 'name').toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).describe(['id', 'name']).toPandas(), self.spark.read.table(self.tbl_name).describe(['id', 'name']).toPandas())

    def test_stat_cov(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.connect.read.table(self.tbl_name2).stat.cov('col1', 'col3'), self.spark.read.table(self.tbl_name2).stat.cov('col1', 'col3'))

    def test_stat_corr(self):
        if False:
            return 10
        self.assertEqual(self.connect.read.table(self.tbl_name2).stat.corr('col1', 'col3'), self.spark.read.table(self.tbl_name2).stat.corr('col1', 'col3'))
        self.assertEqual(self.connect.read.table(self.tbl_name2).stat.corr('col1', 'col3', 'pearson'), self.spark.read.table(self.tbl_name2).stat.corr('col1', 'col3', 'pearson'))
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.corr(1, 'col3', 'pearson')
        self.check_error(exception=pe.exception, error_class='NOT_STR', message_parameters={'arg_name': 'col1', 'arg_type': 'int'})
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name).stat.corr('col1', 1, 'pearson')
        self.check_error(exception=pe.exception, error_class='NOT_STR', message_parameters={'arg_name': 'col2', 'arg_type': 'int'})
        with self.assertRaises(ValueError) as context:
            (self.connect.read.table(self.tbl_name2).stat.corr('col1', 'col3', 'spearman'),)
            self.assertTrue('Currently only the calculation of the Pearson Correlation ' + 'coefficient is supported.' in str(context.exception))

    def test_stat_approx_quantile(self):
        if False:
            print('Hello World!')
        result = self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1', 'col3'], [0.1, 0.5, 0.9], 0.1)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(len(result[1]), 3)
        result = self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1'], [0.1, 0.5, 0.9], 0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.approxQuantile(1, [0.1, 0.5, 0.9], 0.1)
        self.check_error(exception=pe.exception, error_class='NOT_LIST_OR_STR_OR_TUPLE', message_parameters={'arg_name': 'col', 'arg_type': 'int'})
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1', 'col3'], 0.1, 0.1)
        self.check_error(exception=pe.exception, error_class='NOT_LIST_OR_TUPLE', message_parameters={'arg_name': 'probabilities', 'arg_type': 'float'})
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1', 'col3'], [-0.1], 0.1)
        self.check_error(exception=pe.exception, error_class='NOT_LIST_OF_FLOAT_OR_INT', message_parameters={'arg_name': 'probabilities', 'arg_type': 'float'})
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1', 'col3'], [0.1, 0.5, 0.9], 'str')
        self.check_error(exception=pe.exception, error_class='NOT_FLOAT_OR_INT', message_parameters={'arg_name': 'relativeError', 'arg_type': 'str'})
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.read.table(self.tbl_name2).stat.approxQuantile(['col1', 'col3'], [0.1, 0.5, 0.9], -0.1)
        self.check_error(exception=pe.exception, error_class='NEGATIVE_VALUE', message_parameters={'arg_name': 'relativeError', 'arg_value': '-0.1'})

    def test_stat_freq_items(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_eq(self.connect.read.table(self.tbl_name2).stat.freqItems(['col1', 'col3']).toPandas(), self.spark.read.table(self.tbl_name2).stat.freqItems(['col1', 'col3']).toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name2).stat.freqItems(['col1', 'col3'], 0.4).toPandas(), self.spark.read.table(self.tbl_name2).stat.freqItems(['col1', 'col3'], 0.4).toPandas())
        with self.assertRaises(PySparkTypeError) as pe:
            self.connect.read.table(self.tbl_name2).stat.freqItems('col1')
        self.check_error(exception=pe.exception, error_class='NOT_LIST_OR_TUPLE', message_parameters={'arg_name': 'cols', 'arg_type': 'str'})

    def test_stat_sample_by(self):
        if False:
            return 10
        cdf = self.connect.range(0, 100).select((CF.col('id') % 3).alias('key'))
        sdf = self.spark.range(0, 100).select((SF.col('id') % 3).alias('key'))
        self.assert_eq(cdf.sampleBy(cdf.key, fractions={0: 0.1, 1: 0.2}, seed=0).groupBy('key').agg(CF.count(CF.lit(1))).orderBy('key').toPandas(), sdf.sampleBy(sdf.key, fractions={0: 0.1, 1: 0.2}, seed=0).groupBy('key').agg(SF.count(SF.lit(1))).orderBy('key').toPandas())
        with self.assertRaises(PySparkTypeError) as pe:
            cdf.stat.sampleBy(cdf.key, fractions={0: 0.1, None: 0.2}, seed=0)
        self.check_error(exception=pe.exception, error_class='DISALLOWED_TYPE_FOR_CONTAINER', message_parameters={'arg_name': 'fractions', 'arg_type': 'dict', 'allowed_types': 'float, int, str', 'return_type': 'NoneType'})
        with self.assertRaises(SparkConnectException):
            cdf.sampleBy(cdf.key, fractions={0: 0.1, 1: 1.2}, seed=0).show()

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        query = 'SELECT * FROM VALUES (1L, NULL), (3L, "Z") AS tab(a, b)'
        self.assertEqual(self.connect.sql(query).__repr__(), self.spark.sql(query).__repr__())

    def test_explain_string(self):
        if False:
            return 10
        plan_str = self.connect.sql('SELECT 1')._explain_string(extended=True)
        self.assertTrue('Parsed Logical Plan' in plan_str)
        self.assertTrue('Analyzed Logical Plan' in plan_str)
        self.assertTrue('Optimized Logical Plan' in plan_str)
        self.assertTrue('Physical Plan' in plan_str)
        with self.assertRaises(PySparkValueError) as pe:
            self.connect.sql('SELECT 1')._explain_string(mode='unknown')
        self.check_error(exception=pe.exception, error_class='UNKNOWN_EXPLAIN_MODE', message_parameters={'explain_mode': 'unknown'})

    def test_simple_datasource_read(self) -> None:
        if False:
            i = 10
            return i + 15
        writeDf = self.df_text
        tmpPath = tempfile.mkdtemp()
        shutil.rmtree(tmpPath)
        writeDf.write.text(tmpPath)
        for schema in ['id STRING', StructType([StructField('id', StringType())])]:
            readDf = self.connect.read.format('text').schema(schema).load(path=tmpPath)
            expectResult = writeDf.collect()
            pandasResult = readDf.toPandas()
            if pandasResult is None:
                self.assertTrue(False, 'Empty pandas dataframe')
            else:
                actualResult = pandasResult.values.tolist()
                self.assertEqual(len(expectResult), len(actualResult))

    def test_simple_read_without_schema(self) -> None:
        if False:
            print('Hello World!')
        'SPARK-41300: Schema not set when reading CSV.'
        writeDf = self.df_text
        tmpPath = tempfile.mkdtemp()
        shutil.rmtree(tmpPath)
        writeDf.write.csv(tmpPath, header=True)
        readDf = self.connect.read.format('csv').option('header', True).load(path=tmpPath)
        expectResult = set(writeDf.collect())
        pandasResult = set(readDf.collect())
        self.assertEqual(expectResult, pandasResult)

    def test_count(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self.connect.read.table(self.tbl_name).count(), self.spark.read.table(self.tbl_name).count())

    def test_simple_transform(self) -> None:
        if False:
            return 10
        'SPARK-41203: Support DF.transform'

        def transform_df(input_df: CDataFrame) -> CDataFrame:
            if False:
                for i in range(10):
                    print('nop')
            return input_df.select((CF.col('id') + CF.lit(10)).alias('id'))
        df = self.connect.range(1, 100)
        result_left = df.transform(transform_df).collect()
        result_right = self.connect.range(11, 110).collect()
        self.assertEqual(result_right, result_left)
        with self.assertRaises(AssertionError):
            df.transform(lambda x: 2)

    def test_alias(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Testing supported and unsupported alias'
        col0 = self.connect.range(1, 10).select(CF.col('id').alias('name', metadata={'max': 99})).schema.names[0]
        self.assertEqual('name', col0)
        with self.assertRaises(SparkConnectException) as exc:
            self.connect.range(1, 10).select(CF.col('id').alias('this', 'is', 'not')).collect()
        self.assertIn('(this, is, not)', str(exc.exception))

    def test_column_regexp(self) -> None:
        if False:
            print('Hello World!')
        ndf = self.connect.read.table(self.tbl_name3)
        df = self.spark.read.table(self.tbl_name3)
        self.assert_eq(ndf.select(ndf.colRegex('`tes.*\n.*mn`')).toPandas(), df.select(df.colRegex('`tes.*\n.*mn`')).toPandas())

    def test_repartition(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_eq(self.connect.read.table(self.tbl_name).repartition(10).toPandas(), self.spark.read.table(self.tbl_name).repartition(10).toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).coalesce(10).toPandas(), self.spark.read.table(self.tbl_name).coalesce(10).toPandas())

    def test_repartition_by_expression(self) -> None:
        if False:
            print('Hello World!')
        self.assert_eq(self.connect.read.table(self.tbl_name).repartition(10, 'id').toPandas(), self.spark.read.table(self.tbl_name).repartition(10, 'id').toPandas())
        self.assert_eq(self.connect.read.table(self.tbl_name).repartition('id').toPandas(), self.spark.read.table(self.tbl_name).repartition('id').toPandas())
        with self.assertRaises(AnalysisException):
            self.connect.read.table(self.tbl_name).repartition('id+1').toPandas()

    def test_repartition_by_range(self) -> None:
        if False:
            while True:
                i = 10
        cdf = self.connect.read.table(self.tbl_name)
        sdf = self.spark.read.table(self.tbl_name)
        self.assert_eq(cdf.repartitionByRange(10, 'id').toPandas(), sdf.repartitionByRange(10, 'id').toPandas())
        self.assert_eq(cdf.repartitionByRange('id').toPandas(), sdf.repartitionByRange('id').toPandas())
        self.assert_eq(cdf.repartitionByRange(cdf.id.desc()).toPandas(), sdf.repartitionByRange(sdf.id.desc()).toPandas())
        with self.assertRaises(AnalysisException):
            self.connect.read.table(self.tbl_name).repartitionByRange('id+1').toPandas()

    def test_agg_with_two_agg_exprs(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_eq(self.connect.read.table(self.tbl_name).agg({'name': 'min', 'id': 'max'}).toPandas(), self.spark.read.table(self.tbl_name).agg({'name': 'min', 'id': 'max'}).toPandas())

    def test_subtract(self):
        if False:
            while True:
                i = 10
        ndf1 = self.connect.read.table(self.tbl_name)
        ndf2 = ndf1.filter('id > 3')
        df1 = self.spark.read.table(self.tbl_name)
        df2 = df1.filter('id > 3')
        self.assert_eq(ndf1.subtract(ndf2).toPandas(), df1.subtract(df2).toPandas())

    def test_write_operations(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as d:
            df = self.connect.range(50)
            df.write.mode('overwrite').format('csv').save(d)
            ndf = self.connect.read.schema('id int').load(d, format='csv')
            self.assertEqual(50, len(ndf.collect()))
            cd = ndf.collect()
            self.assertEqual(set(df.collect()), set(cd))
        with tempfile.TemporaryDirectory() as d:
            df = self.connect.range(50)
            df.write.mode('overwrite').csv(d, lineSep='|')
            ndf = self.connect.read.schema('id int').load(d, format='csv', lineSep='|')
            self.assertEqual(set(df.collect()), set(ndf.collect()))
        df = self.connect.range(50)
        df.write.format('parquet').saveAsTable('parquet_test')
        ndf = self.connect.read.table('parquet_test')
        self.assertEqual(set(df.collect()), set(ndf.collect()))

    def test_writeTo_operations(self):
        if False:
            while True:
                i = 10
        import datetime
        from pyspark.sql.connect.functions import col, years, months, days, hours, bucket
        df = self.connect.createDataFrame([(1, datetime.datetime(2000, 1, 1), 'foo')], ('id', 'ts', 'value'))
        writer = df.writeTo('table1')
        self.assertIsInstance(writer.option('property', 'value'), DataFrameWriterV2)
        self.assertIsInstance(writer.options(property='value'), DataFrameWriterV2)
        self.assertIsInstance(writer.using('source'), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(col('id')), DataFrameWriterV2)
        self.assertIsInstance(writer.tableProperty('foo', 'bar'), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(years('ts')), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(months('ts')), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(days('ts')), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(hours('ts')), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(bucket(11, 'id')), DataFrameWriterV2)
        self.assertIsInstance(writer.partitionedBy(bucket(3, 'id'), hours('ts')), DataFrameWriterV2)

    def test_agg_with_avg(self):
        if False:
            while True:
                i = 10
        df = self.connect.range(10).groupBy((CF.col('id') % CF.lit(2)).alias('moded')).avg('id').sort('moded')
        res = df.collect()
        self.assertEqual(2, len(res))
        self.assertEqual(4.0, res[0][1])
        self.assertEqual(5.0, res[1][1])
        df_a = self.connect.range(10).groupBy((CF.col('id') % CF.lit(3)).alias('moded'))
        df_b = self.spark.range(10).groupBy((SF.col('id') % SF.lit(3)).alias('moded'))
        self.assertEqual(set(df_b.agg(SF.sum('id')).collect()), set(df_a.agg(CF.sum('id')).collect()))
        measures = {'id': 'sum'}
        self.assertEqual(set(df_a.agg(measures).select('sum(id)').collect()), set(df_b.agg(measures).select('sum(id)').collect()))

    def test_column_cannot_be_constructed_from_string(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            Column('col')

    def test_crossjoin(self):
        if False:
            print('Hello World!')
        connect_df = self.connect.read.table(self.tbl_name)
        spark_df = self.spark.read.table(self.tbl_name)
        self.assert_eq(set(connect_df.select('id').join(other=connect_df.select('name'), how='cross').toPandas()), set(spark_df.select('id').join(other=spark_df.select('name'), how='cross').toPandas()))
        self.assert_eq(set(connect_df.select('id').crossJoin(other=connect_df.select('name')).toPandas()), set(spark_df.select('id').crossJoin(other=spark_df.select('name')).toPandas()))

    def test_grouped_data(self):
        if False:
            i = 10
            return i + 15
        query = "\n            SELECT * FROM VALUES\n                ('James', 'Sales', 3000, 2020),\n                ('Michael', 'Sales', 4600, 2020),\n                ('Robert', 'Sales', 4100, 2020),\n                ('Maria', 'Finance', 3000, 2020),\n                ('James', 'Sales', 3000, 2019),\n                ('Scott', 'Finance', 3300, 2020),\n                ('Jen', 'Finance', 3900, 2020),\n                ('Jeff', 'Marketing', 3000, 2020),\n                ('Kumar', 'Marketing', 2000, 2020),\n                ('Saif', 'Sales', 4100, 2020)\n            AS T(name, department, salary, year)\n            "
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assert_eq(cdf.groupBy('name').agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy('name').agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy('name', cdf.department).agg(CF.max('year'), CF.min(cdf.salary)).toPandas(), sdf.groupBy('name', sdf.department).agg(SF.max('year'), SF.min(sdf.salary)).toPandas())
        self.assert_eq(cdf.rollup('name').agg(CF.sum(cdf.salary)).toPandas(), sdf.rollup('name').agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.rollup('name', cdf.department).agg(CF.max('year'), CF.min(cdf.salary)).toPandas(), sdf.rollup('name', sdf.department).agg(SF.max('year'), SF.min(sdf.salary)).toPandas())
        self.assert_eq(cdf.cube('name').agg(CF.sum(cdf.salary)).toPandas(), sdf.cube('name').agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.cube('name', cdf.department).agg(CF.max('year'), CF.min(cdf.salary)).toPandas(), sdf.cube('name', sdf.department).agg(SF.max('year'), SF.min(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department', ['Sales', 'Finance', 'Marketing']).agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy(sdf.name).pivot('department', ['Sales', 'Finance', 'Marketing']).agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy(sdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department').agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy('name').pivot('department').agg(SF.sum(sdf.salary)).toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('year').agg(CF.sum(cdf.salary)).toPandas(), sdf.groupBy('name').pivot('year').agg(SF.sum(sdf.salary)).toPandas())
        with self.assertRaisesRegex(Exception, 'PIVOT after ROLLUP is not supported'):
            cdf.rollup('name').pivot('department').agg(CF.sum(cdf.salary))
        with self.assertRaisesRegex(Exception, 'PIVOT after CUBE is not supported'):
            cdf.cube('name').pivot('department').agg(CF.sum(cdf.salary))
        with self.assertRaisesRegex(Exception, 'Repeated PIVOT operation is not supported'):
            cdf.groupBy('name').pivot('year').pivot('year').agg(CF.sum(cdf.salary))
        with self.assertRaises(PySparkTypeError) as pe:
            cdf.groupBy('name').pivot('department', ['Sales', b'Marketing']).agg(CF.sum(cdf.salary))
        self.check_error(exception=pe.exception, error_class='NOT_BOOL_OR_FLOAT_OR_INT_OR_STR', message_parameters={'arg_name': 'value', 'arg_type': 'bytes'})

    def test_numeric_aggregation(self):
        if False:
            while True:
                i = 10
        query = "\n                SELECT * FROM VALUES\n                    ('James', 'Sales', 3000, 2020),\n                    ('Michael', 'Sales', 4600, 2020),\n                    ('Robert', 'Sales', 4100, 2020),\n                    ('Maria', 'Finance', 3000, 2020),\n                    ('James', 'Sales', 3000, 2019),\n                    ('Scott', 'Finance', 3300, 2020),\n                    ('Jen', 'Finance', 3900, 2020),\n                    ('Jeff', 'Marketing', 3000, 2020),\n                    ('Kumar', 'Marketing', 2000, 2020),\n                    ('Saif', 'Sales', 4100, 2020)\n                AS T(name, department, salary, year)\n                "
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assert_eq(cdf.groupBy('name').min().toPandas(), sdf.groupBy('name').min().toPandas())
        self.assert_eq(cdf.groupBy('name').min('salary').toPandas(), sdf.groupBy('name').min('salary').toPandas())
        self.assert_eq(cdf.groupBy('name').max('salary').toPandas(), sdf.groupBy('name').max('salary').toPandas())
        self.assert_eq(cdf.groupBy('name', cdf.department).avg('salary', 'year').toPandas(), sdf.groupBy('name', sdf.department).avg('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy('name', cdf.department).mean('salary', 'year').toPandas(), sdf.groupBy('name', sdf.department).mean('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy('name', cdf.department).sum('salary', 'year').toPandas(), sdf.groupBy('name', sdf.department).sum('salary', 'year').toPandas())
        self.assert_eq(cdf.rollup('name').max().toPandas(), sdf.rollup('name').max().toPandas())
        self.assert_eq(cdf.rollup('name').min('salary').toPandas(), sdf.rollup('name').min('salary').toPandas())
        self.assert_eq(cdf.rollup('name').max('salary').toPandas(), sdf.rollup('name').max('salary').toPandas())
        self.assert_eq(cdf.rollup('name', cdf.department).avg('salary', 'year').toPandas(), sdf.rollup('name', sdf.department).avg('salary', 'year').toPandas())
        self.assert_eq(cdf.rollup('name', cdf.department).mean('salary', 'year').toPandas(), sdf.rollup('name', sdf.department).mean('salary', 'year').toPandas())
        self.assert_eq(cdf.rollup('name', cdf.department).sum('salary', 'year').toPandas(), sdf.rollup('name', sdf.department).sum('salary', 'year').toPandas())
        self.assert_eq(cdf.cube('name').avg().toPandas(), sdf.cube('name').avg().toPandas())
        self.assert_eq(cdf.cube('name').mean().toPandas(), sdf.cube('name').mean().toPandas())
        self.assert_eq(cdf.cube('name').min('salary').toPandas(), sdf.cube('name').min('salary').toPandas())
        self.assert_eq(cdf.cube('name').max('salary').toPandas(), sdf.cube('name').max('salary').toPandas())
        self.assert_eq(cdf.cube('name', cdf.department).avg('salary', 'year').toPandas(), sdf.cube('name', sdf.department).avg('salary', 'year').toPandas())
        self.assert_eq(cdf.cube('name', cdf.department).sum('salary', 'year').toPandas(), sdf.cube('name', sdf.department).sum('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).sum().toPandas(), sdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).sum().toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).min('salary').toPandas(), sdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).min('salary').toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).max('salary').toPandas(), sdf.groupBy('name').pivot('department', ['Sales', 'Marketing']).max('salary').toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).avg('salary', 'year').toPandas(), sdf.groupBy(sdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).avg('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).sum('salary', 'year').toPandas(), sdf.groupBy(sdf.name).pivot('department', ['Sales', 'Finance', 'Unknown']).sum('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department').min().toPandas(), sdf.groupBy('name').pivot('department').min().toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department').min('salary').toPandas(), sdf.groupBy('name').pivot('department').min('salary').toPandas())
        self.assert_eq(cdf.groupBy('name').pivot('department').max('salary').toPandas(), sdf.groupBy('name').pivot('department').max('salary').toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department').avg('salary', 'year').toPandas(), sdf.groupBy(sdf.name).pivot('department').avg('salary', 'year').toPandas())
        self.assert_eq(cdf.groupBy(cdf.name).pivot('department').sum('salary', 'year').toPandas(), sdf.groupBy(sdf.name).pivot('department').sum('salary', 'year').toPandas())
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.groupBy('name').min('department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.groupBy('name').max('salary', 'department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.rollup('name').avg('department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.rollup('name').sum('salary', 'department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.cube('name').min('department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.cube('name').max('salary', 'department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.groupBy('name').pivot('department').avg('department').show()
        with self.assertRaisesRegex(TypeError, 'Numeric aggregation function can only be applied on numeric columns'):
            cdf.groupBy('name').pivot('department').sum('salary', 'department').show()

    def test_with_metadata(self):
        if False:
            i = 10
            return i + 15
        cdf = self.connect.createDataFrame(data=[(2, 'Alice'), (5, 'Bob')], schema=['age', 'name'])
        self.assertEqual(cdf.schema['age'].metadata, {})
        self.assertEqual(cdf.schema['name'].metadata, {})
        cdf1 = cdf.withMetadata(columnName='age', metadata={'max_age': 5})
        self.assertEqual(cdf1.schema['age'].metadata, {'max_age': 5})
        cdf2 = cdf.withMetadata(columnName='name', metadata={'names': ['Alice', 'Bob']})
        self.assertEqual(cdf2.schema['name'].metadata, {'names': ['Alice', 'Bob']})
        with self.assertRaises(PySparkTypeError) as pe:
            cdf.withMetadata(columnName='name', metadata=['magic'])
        self.check_error(exception=pe.exception, error_class='NOT_DICT', message_parameters={'arg_name': 'metadata', 'arg_type': 'list'})

    def test_collect_nested_type(self):
        if False:
            i = 10
            return i + 15
        query = '\n            SELECT * FROM VALUES\n            (1, 4, 0, 8, true, true, ARRAY(1, NULL, 3), MAP(1, 2, 3, 4)),\n            (2, 5, -1, NULL, false, NULL, ARRAY(1, 3), MAP(1, NULL, 3, 4)),\n            (3, 6, NULL, 0, false, NULL, ARRAY(NULL), NULL)\n            AS tab(a, b, c, d, e, f, g, h)\n            '
        cdf = self.connect.sql(query)
        sdf = self.spark.sql(query)
        self.assertEqual(cdf.select(CF.array('a', 'b', 'c'), CF.array('e', 'f'), CF.col('g')).collect(), sdf.select(SF.array('a', 'b', 'c'), SF.array('e', 'f'), SF.col('g')).collect())
        self.assertEqual(cdf.select(CF.array(CF.array('a'), CF.array('b'), CF.array('c')), CF.array(CF.array('e'), CF.array('f'))).collect(), sdf.select(SF.array(SF.array('a'), SF.array('b'), SF.array('c')), SF.array(SF.array('e'), SF.array('f'))).collect())
        self.assertEqual(cdf.select(CF.array(CF.struct('a')), CF.array('h')).collect(), sdf.select(SF.array(SF.struct('a')), SF.array('h')).collect())
        self.assertEqual(cdf.select(CF.col('h'), CF.create_map('a', 'b', 'b', 'c')).collect(), sdf.select(SF.col('h'), SF.create_map('a', 'b', 'b', 'c')).collect())
        self.assertEqual(cdf.select(CF.create_map('a', 'g'), CF.create_map('a', CF.struct('b', 'g'))).collect(), sdf.select(SF.create_map('a', 'g'), SF.create_map('a', SF.struct('b', 'g'))).collect())
        self.assertEqual(cdf.select(CF.struct('a', 'b', 'c', 'd'), CF.struct('e', 'f', 'g')).collect(), sdf.select(SF.struct('a', 'b', 'c', 'd'), SF.struct('e', 'f', 'g')).collect())
        self.assertEqual(cdf.select(CF.struct('a', CF.struct('a', CF.struct('c', CF.struct('d')))), CF.struct('a', 'b', CF.struct('c', 'd')), CF.struct('e', 'f', CF.struct('g'))).collect(), sdf.select(SF.struct('a', SF.struct('a', SF.struct('c', SF.struct('d')))), SF.struct('a', 'b', SF.struct('c', 'd')), SF.struct('e', 'f', SF.struct('g'))).collect())
        self.assertEqual(cdf.select(CF.struct('a', CF.struct('a', CF.struct('g', CF.struct('h'))))).collect(), sdf.select(SF.struct('a', SF.struct('a', SF.struct('g', SF.struct('h'))))).collect())

    def test_simple_udt(self):
        if False:
            i = 10
            return i + 15
        from pyspark.ml.linalg import MatrixUDT, VectorUDT
        for schema in [StructType().add('key', LongType()).add('val', PythonOnlyUDT()), StructType().add('key', LongType()).add('val', ArrayType(PythonOnlyUDT())), StructType().add('key', LongType()).add('val', MapType(LongType(), PythonOnlyUDT())), StructType().add('key', LongType()).add('val', PythonOnlyUDT()), StructType().add('key', LongType()).add('vec', VectorUDT()), StructType().add('key', LongType()).add('mat', MatrixUDT())]:
            cdf = self.connect.createDataFrame(data=[], schema=schema)
            sdf = self.spark.createDataFrame(data=[], schema=schema)
            self.assertEqual(cdf.schema, sdf.schema)

    def test_simple_udt_from_read(self):
        if False:
            print('Hello World!')
        from pyspark.ml.linalg import Matrices, Vectors
        with tempfile.TemporaryDirectory() as d:
            path1 = f'{d}/df1.parquet'
            self.spark.createDataFrame([(i % 3, PythonOnlyPoint(float(i), float(i))) for i in range(10)], schema=StructType().add('key', LongType()).add('val', PythonOnlyUDT())).write.parquet(path1)
            path2 = f'{d}/df2.parquet'
            self.spark.createDataFrame([(i % 3, [PythonOnlyPoint(float(i), float(i))]) for i in range(10)], schema=StructType().add('key', LongType()).add('val', ArrayType(PythonOnlyUDT()))).write.parquet(path2)
            path3 = f'{d}/df3.parquet'
            self.spark.createDataFrame([(i % 3, {i % 3: PythonOnlyPoint(float(i + 1), float(i + 1))}) for i in range(10)], schema=StructType().add('key', LongType()).add('val', MapType(LongType(), PythonOnlyUDT()))).write.parquet(path3)
            path4 = f'{d}/df4.parquet'
            self.spark.createDataFrame([(i % 3, PythonOnlyPoint(float(i), float(i))) for i in range(10)], schema=StructType().add('key', LongType()).add('val', PythonOnlyUDT())).write.parquet(path4)
            path5 = f'{d}/df5.parquet'
            self.spark.createDataFrame([Row(label=1.0, point=ExamplePoint(1.0, 2.0))]).write.parquet(path5)
            path6 = f'{d}/df6.parquet'
            self.spark.createDataFrame([(Vectors.dense(1.0, 2.0, 3.0),), (Vectors.sparse(3, {1: 1.0, 2: 5.5}),)], ['vec']).write.parquet(path6)
            path7 = f'{d}/df7.parquet'
            self.spark.createDataFrame([(Matrices.dense(3, 2, [0, 1, 4, 5, 9, 10]),), (Matrices.sparse(1, 1, [0, 1], [0], [2.0]),)], ['mat']).write.parquet(path7)
            for path in [path1, path2, path3, path4, path5, path6, path7]:
                self.assertEqual(self.connect.read.parquet(path).schema, self.spark.read.parquet(path).schema)

    def test_version(self):
        if False:
            return 10
        self.assertEqual(self.connect.version, self.spark.version)

    def test_same_semantics(self):
        if False:
            print('Hello World!')
        plan = self.connect.sql('SELECT 1')
        other = self.connect.sql('SELECT 1')
        self.assertTrue(plan.sameSemantics(other))

    def test_semantic_hash(self):
        if False:
            for i in range(10):
                print('nop')
        plan = self.connect.sql('SELECT 1')
        other = self.connect.sql('SELECT 1')
        self.assertEqual(plan.semanticHash(), other.semanticHash())

    def test_unsupported_functions(self):
        if False:
            print('Hello World!')
        df = self.connect.read.table(self.tbl_name)
        for f in ('checkpoint', 'localCheckpoint'):
            with self.assertRaises(NotImplementedError):
                getattr(df, f)()

    def test_sql_with_command(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.connect.sql('show functions').collect(), self.spark.sql('show functions').collect())

    def test_schema_has_nullable(self):
        if False:
            print('Hello World!')
        schema_false = StructType().add('id', IntegerType(), False)
        cdf1 = self.connect.createDataFrame([[1]], schema=schema_false)
        sdf1 = self.spark.createDataFrame([[1]], schema=schema_false)
        self.assertEqual(cdf1.schema, sdf1.schema)
        self.assertEqual(cdf1.collect(), sdf1.collect())
        schema_true = StructType().add('id', IntegerType(), True)
        cdf2 = self.connect.createDataFrame([[1]], schema=schema_true)
        sdf2 = self.spark.createDataFrame([[1]], schema=schema_true)
        self.assertEqual(cdf2.schema, sdf2.schema)
        self.assertEqual(cdf2.collect(), sdf2.collect())
        pdf1 = cdf1.toPandas()
        cdf3 = self.connect.createDataFrame(pdf1, cdf1.schema)
        sdf3 = self.spark.createDataFrame(pdf1, sdf1.schema)
        self.assertEqual(cdf3.schema, sdf3.schema)
        self.assertEqual(cdf3.collect(), sdf3.collect())
        pdf2 = cdf2.toPandas()
        cdf4 = self.connect.createDataFrame(pdf2, cdf2.schema)
        sdf4 = self.spark.createDataFrame(pdf2, sdf2.schema)
        self.assertEqual(cdf4.schema, sdf4.schema)
        self.assertEqual(cdf4.collect(), sdf4.collect())

    def test_array_has_nullable(self):
        if False:
            while True:
                i = 10
        for (schemas, data) in [([StructType().add('arr', ArrayType(IntegerType(), False), True)], [Row([1, 2]), Row([3]), Row(None)]), ([StructType().add('arr', ArrayType(IntegerType(), True), True), 'arr array<integer>'], [Row([1, None]), Row([3]), Row(None)]), ([StructType().add('arr', ArrayType(IntegerType(), False), False)], [Row([1, 2]), Row([3])]), ([StructType().add('arr', ArrayType(IntegerType(), True), False), 'arr array<integer> not null'], [Row([1, None]), Row([3])])]:
            for schema in schemas:
                with self.subTest(schema=schema):
                    cdf = self.connect.createDataFrame(data, schema=schema)
                    sdf = self.spark.createDataFrame(data, schema=schema)
                    self.assertEqual(cdf.schema, sdf.schema)
                    self.assertEqual(cdf.collect(), sdf.collect())

    def test_map_has_nullable(self):
        if False:
            for i in range(10):
                print('nop')
        for (schemas, data) in [([StructType().add('map', MapType(StringType(), IntegerType(), False), True)], [Row({'a': 1, 'b': 2}), Row({'a': 3}), Row(None)]), ([StructType().add('map', MapType(StringType(), IntegerType(), True), True), 'map map<string, integer>'], [Row({'a': 1, 'b': None}), Row({'a': 3}), Row(None)]), ([StructType().add('map', MapType(StringType(), IntegerType(), False), False)], [Row({'a': 1, 'b': 2}), Row({'a': 3})]), ([StructType().add('map', MapType(StringType(), IntegerType(), True), False), 'map map<string, integer> not null'], [Row({'a': 1, 'b': None}), Row({'a': 3})])]:
            for schema in schemas:
                with self.subTest(schema=schema):
                    cdf = self.connect.createDataFrame(data, schema=schema)
                    sdf = self.spark.createDataFrame(data, schema=schema)
                    self.assertEqual(cdf.schema, sdf.schema)
                    self.assertEqual(cdf.collect(), sdf.collect())

    def test_struct_has_nullable(self):
        if False:
            i = 10
            return i + 15
        for (schemas, data) in [([StructType().add('struct', StructType().add('i', IntegerType(), False), True), 'struct struct<i: integer not null>'], [Row(Row(1)), Row(Row(2)), Row(None)]), ([StructType().add('struct', StructType().add('i', IntegerType(), True), True), 'struct struct<i: integer>'], [Row(Row(1)), Row(Row(2)), Row(Row(None)), Row(None)]), ([StructType().add('struct', StructType().add('i', IntegerType(), False), False), 'struct struct<i: integer not null> not null'], [Row(Row(1)), Row(Row(2))]), ([StructType().add('struct', StructType().add('i', IntegerType(), True), False), 'struct struct<i: integer> not null'], [Row(Row(1)), Row(Row(2)), Row(Row(None))])]:
            for schema in schemas:
                with self.subTest(schema=schema):
                    cdf = self.connect.createDataFrame(data, schema=schema)
                    sdf = self.spark.createDataFrame(data, schema=schema)
                    self.assertEqual(cdf.schema, sdf.schema)
                    self.assertEqual(cdf.collect(), sdf.collect())

    def test_large_client_data(self):
        if False:
            i = 10
            return i + 15
        cols = ['abcdefghijklmnoprstuvwxyz' for x in range(10)]
        row_count = 100 * 1000
        rows = [cols] * row_count
        self.assertEqual(row_count, self.connect.createDataFrame(data=rows).count())

    def test_unsupported_jvm_attribute(self):
        if False:
            print('Hello World!')
        unsupported_attrs = ['_jsc', '_jconf', '_jvm', '_jsparkSession']
        spark_session = self.connect
        for attr in unsupported_attrs:
            with self.assertRaises(PySparkAttributeError) as pe:
                getattr(spark_session, attr)
            self.check_error(exception=pe.exception, error_class='JVM_ATTRIBUTE_NOT_SUPPORTED', message_parameters={'attr_name': attr})
        unsupported_attrs = ['_jseq', '_jdf', '_jmap', '_jcols']
        cdf = self.connect.range(10)
        for attr in unsupported_attrs:
            with self.assertRaises(PySparkAttributeError) as pe:
                getattr(cdf, attr)
            self.check_error(exception=pe.exception, error_class='JVM_ATTRIBUTE_NOT_SUPPORTED', message_parameters={'attr_name': attr})
        with self.assertRaises(PySparkAttributeError) as pe:
            getattr(cdf.id, '_jc')
        self.check_error(exception=pe.exception, error_class='JVM_ATTRIBUTE_NOT_SUPPORTED', message_parameters={'attr_name': '_jc'})
        with self.assertRaises(PySparkAttributeError) as pe:
            getattr(spark_session.read, '_jreader')
        self.check_error(exception=pe.exception, error_class='JVM_ATTRIBUTE_NOT_SUPPORTED', message_parameters={'attr_name': '_jreader'})

    def test_df_caache(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.connect.range(10)
        df.cache()
        self.assert_eq(10, df.count())
        self.assertTrue(df.is_cached)

class SparkConnectSessionTests(ReusedConnectTestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark = PySparkSession.builder.config(conf=self.conf()).appName(self.__class__.__name__).remote('local[4]').getOrCreate()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.spark.stop()

    def _check_no_active_session_error(self, e: PySparkException):
        if False:
            print('Hello World!')
        self.check_error(exception=e, error_class='NO_ACTIVE_SESSION', message_parameters=dict())

    def test_stop_session(self):
        if False:
            print('Hello World!')
        df = self.spark.sql('select 1 as a, 2 as b')
        catalog = self.spark.catalog
        self.spark.stop()
        with self.assertRaises(SparkConnectException) as e:
            self.spark.sql('select 1')
        self._check_no_active_session_error(e.exception)
        with self.assertRaises(SparkConnectException) as e:
            catalog.tableExists('table')
        self._check_no_active_session_error(e.exception)
        with self.assertRaises(SparkConnectException) as e:
            self.spark.udf.register('test_func', lambda x: x + 1)
        self._check_no_active_session_error(e.exception)
        with self.assertRaises(SparkConnectException) as e:
            df._explain_string(extended=True)
        self._check_no_active_session_error(e.exception)
        with self.assertRaises(SparkConnectException) as e:
            self.spark.conf.get('some.conf')
        self._check_no_active_session_error(e.exception)

    def test_error_enrichment_message(self):
        if False:
            print('Hello World!')
        with self.sql_conf({'spark.sql.connect.enrichError.enabled': True, 'spark.sql.connect.serverStacktrace.enabled': False, 'spark.sql.pyspark.jvmStacktrace.enabled': False}):
            name = 'test' * 10000
            with self.assertRaises(AnalysisException) as e:
                self.spark.sql('select ' + name).collect()
            self.assertTrue(name in e.exception._message)
            self.assertFalse('JVM stacktrace' in e.exception._message)

    def test_error_enrichment_jvm_stacktrace(self):
        if False:
            print('Hello World!')
        with self.sql_conf({'spark.sql.connect.enrichError.enabled': True, 'spark.sql.pyspark.jvmStacktrace.enabled': False}):
            with self.sql_conf({'spark.sql.connect.serverStacktrace.enabled': False}):
                with self.assertRaises(SparkUpgradeException) as e:
                    self.spark.sql('select from_json(\n                            \'{"d": "02-29"}\', \'d date\', map(\'dateFormat\', \'MM-dd\'))').collect()
                self.assertFalse('JVM stacktrace' in e.exception._message)
            with self.sql_conf({'spark.sql.connect.serverStacktrace.enabled': True}):
                with self.assertRaises(SparkUpgradeException) as e:
                    self.spark.sql('select from_json(\n                            \'{"d": "02-29"}\', \'d date\', map(\'dateFormat\', \'MM-dd\'))').collect()
                self.assertTrue('JVM stacktrace' in str(e.exception))
                self.assertTrue('org.apache.spark.SparkUpgradeException' in str(e.exception))
                self.assertTrue('at org.apache.spark.sql.errors.ExecutionErrors.failToParseDateTimeInNewParserError' in str(e.exception))
                self.assertTrue('Caused by: java.time.DateTimeException:' in str(e.exception))

    def test_not_hitting_netty_header_limit(self):
        if False:
            for i in range(10):
                print('nop')
        with self.sql_conf({'spark.sql.pyspark.jvmStacktrace.enabled': True}):
            with self.assertRaises(AnalysisException):
                self.spark.sql('select ' + 'test' * 1).collect()

    def test_error_stack_trace(self):
        if False:
            return 10
        with self.sql_conf({'spark.sql.connect.enrichError.enabled': False}):
            with self.sql_conf({'spark.sql.pyspark.jvmStacktrace.enabled': True}):
                with self.assertRaises(AnalysisException) as e:
                    self.spark.sql('select x').collect()
                self.assertTrue('JVM stacktrace' in str(e.exception))
                self.assertIsNotNone(e.exception.getStackTrace())
                self.assertTrue('at org.apache.spark.sql.catalyst.analysis.CheckAnalysis' in str(e.exception))
            with self.sql_conf({'spark.sql.pyspark.jvmStacktrace.enabled': False}):
                with self.assertRaises(AnalysisException) as e:
                    self.spark.sql('select x').collect()
                self.assertFalse('JVM stacktrace' in str(e.exception))
                self.assertIsNone(e.exception.getStackTrace())
                self.assertFalse('at org.apache.spark.sql.catalyst.analysis.CheckAnalysis' in str(e.exception))
        self.spark.stop()
        spark = PySparkSession.builder.config(conf=self.conf()).config('spark.connect.jvmStacktrace.maxSize', 128).remote('local[4]').getOrCreate()
        spark.conf.set('spark.sql.connect.enrichError.enabled', False)
        spark.conf.set('spark.sql.pyspark.jvmStacktrace.enabled', True)
        with self.assertRaises(AnalysisException) as e:
            spark.sql('select x').collect()
        self.assertTrue('JVM stacktrace' in str(e.exception))
        self.assertIsNotNone(e.exception.getStackTrace())
        self.assertFalse('at org.apache.spark.sql.catalyst.analysis.CheckAnalysis' in str(e.exception))
        spark.stop()

    def test_can_create_multiple_sessions_to_different_remotes(self):
        if False:
            while True:
                i = 10
        self.spark.stop()
        self.assertIsNotNone(self.spark._client)
        other = PySparkSession.builder.remote('sc://other.remote:114/').create()
        self.assertNotEquals(self.spark, other)
        same = PySparkSession.builder.remote('sc://other.remote.host:114/').getOrCreate()
        self.assertEquals(other, same)
        same.release_session_on_close = False
        same.stop()
        self.spark.stop()
        with self.assertRaises(RuntimeError) as e:
            PySparkSession.builder.create()
            self.assertIn('Create a new SparkSession is only supported with SparkConnect.', str(e))

class SparkConnectSessionWithOptionsTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.spark = PySparkSession.builder.config('string', 'foo').config('integer', 1).config('boolean', False).appName(self.__class__.__name__).remote('local[4]').getOrCreate()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.spark.stop()

    def test_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.spark.conf.get('string'), 'foo')
        self.assertEqual(self.spark.conf.get('boolean'), 'false')
        self.assertEqual(self.spark.conf.get('integer'), '1')

class TestError(grpc.RpcError, Exception):

    def __init__(self, code: grpc.StatusCode):
        if False:
            for i in range(10):
                print('nop')
        self._code = code

    def code(self):
        if False:
            for i in range(10):
                print('nop')
        return self._code

class TestPolicy(RetryPolicy):

    def __init__(self, initial_backoff=10, **kwargs):
        if False:
            return 10
        super().__init__(initial_backoff=initial_backoff, **kwargs)

    def can_retry(self, exception: BaseException):
        if False:
            while True:
                i = 10
        return isinstance(exception, TestError)

class TestPolicySpecificError(TestPolicy):

    def __init__(self, specific_code: grpc.StatusCode, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.specific_code = specific_code

    def can_retry(self, exception: BaseException):
        if False:
            return 10
        return exception.code() == self.specific_code

@unittest.skipIf(not should_test_connect, connect_requirement_message)
class RetryTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.call_wrap = defaultdict(int)

    def stub(self, retries, code):
        if False:
            i = 10
            return i + 15
        self.call_wrap['attempts'] += 1
        if self.call_wrap['attempts'] < retries:
            self.call_wrap['raised'] += 1
            raise TestError(code)

    def test_simple(self):
        if False:
            while True:
                i = 10
        for attempt in Retrying(TestPolicy(max_retries=1)):
            with attempt:
                self.stub(2, grpc.StatusCode.INTERNAL)
        self.assertEqual(2, self.call_wrap['attempts'])
        self.assertEqual(1, self.call_wrap['raised'])

    def test_below_limit(self):
        if False:
            for i in range(10):
                print('nop')
        for attempt in Retrying(TestPolicy(max_retries=4)):
            with attempt:
                self.stub(2, grpc.StatusCode.INTERNAL)
        self.assertLess(self.call_wrap['attempts'], 4)
        self.assertEqual(self.call_wrap['raised'], 1)

    def test_exceed_retries(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(RetriesExceeded):
            for attempt in Retrying(TestPolicy(max_retries=2)):
                with attempt:
                    self.stub(5, grpc.StatusCode.INTERNAL)
        self.assertLess(self.call_wrap['attempts'], 5)
        self.assertEqual(self.call_wrap['raised'], 3)

    def test_throw_not_retriable_error(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            for attempt in Retrying(TestPolicy(max_retries=2)):
                with attempt:
                    raise ValueError

    def test_specific_exception(self):
        if False:
            i = 10
            return i + 15
        policy = TestPolicySpecificError(max_retries=4, specific_code=grpc.StatusCode.UNAVAILABLE)
        for attempt in Retrying(policy):
            with attempt:
                self.stub(2, grpc.StatusCode.UNAVAILABLE)
        self.assertLess(self.call_wrap['attempts'], 4)
        self.assertEqual(self.call_wrap['raised'], 1)

    def test_specific_exception_exceed_retries(self):
        if False:
            while True:
                i = 10
        policy = TestPolicySpecificError(max_retries=2, specific_code=grpc.StatusCode.UNAVAILABLE)
        with self.assertRaises(RetriesExceeded):
            for attempt in Retrying(policy):
                with attempt:
                    self.stub(5, grpc.StatusCode.UNAVAILABLE)
        self.assertLess(self.call_wrap['attempts'], 4)
        self.assertEqual(self.call_wrap['raised'], 3)

    def test_rejected_by_policy(self):
        if False:
            i = 10
            return i + 15
        policy = TestPolicySpecificError(max_retries=4, specific_code=grpc.StatusCode.UNAVAILABLE)
        with self.assertRaises(TestError):
            for attempt in Retrying(policy):
                with attempt:
                    self.stub(5, grpc.StatusCode.INTERNAL)
        self.assertEqual(self.call_wrap['attempts'], 1)
        self.assertEqual(self.call_wrap['raised'], 1)

    def test_multiple_policies(self):
        if False:
            for i in range(10):
                print('nop')
        policy1 = TestPolicySpecificError(max_retries=2, specific_code=grpc.StatusCode.UNAVAILABLE)
        policy2 = TestPolicySpecificError(max_retries=4, specific_code=grpc.StatusCode.INTERNAL)
        error_suply = iter([grpc.StatusCode.UNAVAILABLE] * 2 + [grpc.StatusCode.INTERNAL] * 4)
        for attempt in Retrying([policy1, policy2]):
            with attempt:
                error = next(error_suply, None)
                if error:
                    raise TestError(error)
        self.assertEqual(next(error_suply, None), None)

    def test_multiple_policies_exceed(self):
        if False:
            for i in range(10):
                print('nop')
        policy1 = TestPolicySpecificError(max_retries=2, specific_code=grpc.StatusCode.INTERNAL)
        policy2 = TestPolicySpecificError(max_retries=4, specific_code=grpc.StatusCode.INTERNAL)
        with self.assertRaises(RetriesExceeded):
            for attempt in Retrying([policy1, policy2]):
                with attempt:
                    self.stub(10, grpc.StatusCode.INTERNAL)
        self.assertEqual(self.call_wrap['attempts'], 7)
        self.assertEqual(self.call_wrap['raised'], 7)

@unittest.skipIf(not should_test_connect, connect_requirement_message)
class ChannelBuilderTests(unittest.TestCase):

    def test_invalid_connection_strings(self):
        if False:
            i = 10
            return i + 15
        invalid = ['scc://host:12', 'http://host', 'sc:/host:1234/path', 'sc://host/path', 'sc://host/;parm1;param2']
        for i in invalid:
            self.assertRaises(PySparkValueError, ChannelBuilder, i)

    def test_sensible_defaults(self):
        if False:
            while True:
                i = 10
        chan = ChannelBuilder('sc://host')
        self.assertFalse(chan.secure, 'Default URL is not secure')
        chan = ChannelBuilder('sc://host/;token=abcs')
        self.assertTrue(chan.secure, 'specifying a token must set the channel to secure')
        self.assertRegex(chan.userAgent, '^_SPARK_CONNECT_PYTHON spark/[^ ]+ os/[^ ]+ python/[^ ]+$')
        chan = ChannelBuilder('sc://host/;use_ssl=abcs')
        self.assertFalse(chan.secure, 'Garbage in, false out')

    def test_user_agent(self):
        if False:
            return 10
        chan = ChannelBuilder('sc://host/;user_agent=Agent123%20%2F3.4')
        self.assertIn('Agent123 /3.4', chan.userAgent)

    def test_user_agent_len(self):
        if False:
            i = 10
            return i + 15
        user_agent = 'x' * 2049
        chan = ChannelBuilder(f'sc://host/;user_agent={user_agent}')
        with self.assertRaises(SparkConnectException) as err:
            chan.userAgent
        self.assertRegex(err.exception._message, "'user_agent' parameter should not exceed")
        user_agent = '%C3%A4' * 341
        expected = 'ä' * 341
        chan = ChannelBuilder(f'sc://host/;user_agent={user_agent}')
        self.assertIn(expected, chan.userAgent)

    def test_valid_channel_creation(self):
        if False:
            while True:
                i = 10
        chan = ChannelBuilder('sc://host').toChannel()
        self.assertIsInstance(chan, grpc.Channel)
        chan = ChannelBuilder('sc://host/;use_ssl=true;token=abc').toChannel()
        self.assertIsInstance(chan, grpc.Channel)
        chan = ChannelBuilder('sc://host/;use_ssl=true').toChannel()
        self.assertIsInstance(chan, grpc.Channel)

    def test_channel_properties(self):
        if False:
            print('Hello World!')
        chan = ChannelBuilder('sc://host/;use_ssl=true;token=abc;user_agent=foo;param1=120%2021')
        self.assertEqual('host:15002', chan.endpoint)
        self.assertIn('foo', chan.userAgent.split(' '))
        self.assertEqual(True, chan.secure)
        self.assertEqual('120 21', chan.get('param1'))

    def test_metadata(self):
        if False:
            i = 10
            return i + 15
        chan = ChannelBuilder('sc://host/;use_ssl=true;token=abc;param1=120%2021;x-my-header=abcd')
        md = chan.metadata()
        self.assertEqual([('param1', '120 21'), ('x-my-header', 'abcd')], md)

    def test_metadata(self):
        if False:
            i = 10
            return i + 15
        id = str(uuid.uuid4())
        chan = ChannelBuilder(f'sc://host/;session_id={id}')
        self.assertEqual(id, chan.session_id)
        chan = ChannelBuilder(f'sc://host/;session_id={id};user_agent=acbd;token=abcd;use_ssl=true')
        md = chan.metadata()
        for kv in md:
            self.assertNotIn(kv[0], [ChannelBuilder.PARAM_SESSION_ID, ChannelBuilder.PARAM_TOKEN, ChannelBuilder.PARAM_USER_ID, ChannelBuilder.PARAM_USER_AGENT, ChannelBuilder.PARAM_USE_SSL], 'Metadata must not contain fixed params')
        with self.assertRaises(ValueError) as ve:
            chan = ChannelBuilder('sc://host/;session_id=abcd')
            SparkConnectClient(chan)
        self.assertIn("Parameter value 'session_id' must be a valid UUID format.", str(ve.exception))
        chan = ChannelBuilder('sc://host/')
        self.assertIsNone(chan.session_id)
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_connect_basic import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)