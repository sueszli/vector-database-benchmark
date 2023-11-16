import os
import unittest
from pyspark.sql.datasource import DataSource, DataSourceReader
from pyspark.sql.types import Row
from pyspark.testing import assertDataFrameEqual
from pyspark.testing.sqlutils import ReusedSQLTestCase
from pyspark.testing.utils import SPARK_HOME

class BasePythonDataSourceTestsMixin:

    def test_basic_data_source_class(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDataSource(DataSource):
            ...
        options = dict(a=1, b=2)
        ds = MyDataSource(paths=[], userSpecifiedSchema=None, options=options)
        self.assertEqual(ds.options, options)
        self.assertEqual(ds.name(), 'MyDataSource')
        with self.assertRaises(NotImplementedError):
            ds.schema()
        with self.assertRaises(NotImplementedError):
            ds.reader(None)
        with self.assertRaises(NotImplementedError):
            ds.writer(None, None)

    def test_basic_data_source_reader_class(self):
        if False:
            print('Hello World!')

        class MyDataSourceReader(DataSourceReader):

            def read(self, partition):
                if False:
                    while True:
                        i = 10
                yield (None,)
        reader = MyDataSourceReader()
        self.assertEqual(list(reader.partitions()), [None])
        self.assertEqual(list(reader.read(None)), [(None,)])

    def test_in_memory_data_source(self):
        if False:
            print('Hello World!')

        class InMemDataSourceReader(DataSourceReader):
            DEFAULT_NUM_PARTITIONS: int = 3

            def __init__(self, paths, options):
                if False:
                    while True:
                        i = 10
                self.paths = paths
                self.options = options

            def partitions(self):
                if False:
                    print('Hello World!')
                if 'num_partitions' in self.options:
                    num_partitions = int(self.options['num_partitions'])
                else:
                    num_partitions = self.DEFAULT_NUM_PARTITIONS
                return range(num_partitions)

            def read(self, partition):
                if False:
                    for i in range(10):
                        print('nop')
                yield (partition, str(partition))

        class InMemoryDataSource(DataSource):

            @classmethod
            def name(cls):
                if False:
                    print('Hello World!')
                return 'memory'

            def schema(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'x INT, y STRING'

            def reader(self, schema) -> 'DataSourceReader':
                if False:
                    for i in range(10):
                        print('nop')
                return InMemDataSourceReader(self.paths, self.options)
        self.spark.dataSource.register(InMemoryDataSource)
        df = self.spark.read.format('memory').load()
        self.assertEqual(df.rdd.getNumPartitions(), 3)
        assertDataFrameEqual(df, [Row(x=0, y='0'), Row(x=1, y='1'), Row(x=2, y='2')])
        df = self.spark.read.format('memory').option('num_partitions', 2).load()
        assertDataFrameEqual(df, [Row(x=0, y='0'), Row(x=1, y='1')])
        self.assertEqual(df.rdd.getNumPartitions(), 2)

    def test_custom_json_data_source(self):
        if False:
            print('Hello World!')
        import json

        class JsonDataSourceReader(DataSourceReader):

            def __init__(self, paths, options):
                if False:
                    while True:
                        i = 10
                self.paths = paths
                self.options = options

            def partitions(self):
                if False:
                    return 10
                return iter(self.paths)

            def read(self, path):
                if False:
                    while True:
                        i = 10
                with open(path, 'r') as file:
                    for line in file.readlines():
                        if line.strip():
                            data = json.loads(line)
                            yield (data.get('name'), data.get('age'))

        class JsonDataSource(DataSource):

            @classmethod
            def name(cls):
                if False:
                    i = 10
                    return i + 15
                return 'my-json'

            def schema(self):
                if False:
                    return 10
                return 'name STRING, age INT'

            def reader(self, schema) -> 'DataSourceReader':
                if False:
                    return 10
                return JsonDataSourceReader(self.paths, self.options)
        self.spark.dataSource.register(JsonDataSource)
        path1 = os.path.join(SPARK_HOME, 'python/test_support/sql/people.json')
        path2 = os.path.join(SPARK_HOME, 'python/test_support/sql/people1.json')
        df1 = self.spark.read.format('my-json').load(path1)
        self.assertEqual(df1.rdd.getNumPartitions(), 1)
        assertDataFrameEqual(df1, [Row(name='Michael', age=None), Row(name='Andy', age=30), Row(name='Justin', age=19)])
        df2 = self.spark.read.format('my-json').load([path1, path2])
        self.assertEqual(df2.rdd.getNumPartitions(), 2)
        assertDataFrameEqual(df2, [Row(name='Michael', age=None), Row(name='Andy', age=30), Row(name='Justin', age=19), Row(name='Jonathan', age=None)])

class PythonDataSourceTests(BasePythonDataSourceTestsMixin, ReusedSQLTestCase):
    ...
if __name__ == '__main__':
    from pyspark.sql.tests.test_python_datasource import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)