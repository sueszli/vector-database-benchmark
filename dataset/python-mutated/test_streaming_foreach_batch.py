import time
from pyspark.sql.dataframe import DataFrame
from pyspark.testing.sqlutils import ReusedSQLTestCase

def my_test_function_1():
    if False:
        print('Hello World!')
    return 1

class StreamingTestsForeachBatchMixin:

    def test_streaming_foreach_batch(self):
        if False:
            print('Hello World!')
        q = None

        def collectBatch(batch_df, batch_id):
            if False:
                print('Hello World!')
            batch_df.createOrReplaceGlobalTempView('test_view')
        try:
            df = self.spark.readStream.format('text').load('python/test_support/sql/streaming')
            q = df.writeStream.foreachBatch(collectBatch).start()
            q.processAllAvailable()
            collected = self.spark.sql('select * from global_temp.test_view').collect()
            self.assertTrue(len(collected), 2)
        finally:
            if q:
                q.stop()

    def test_streaming_foreach_batch_tempview(self):
        if False:
            return 10
        q = None

        def collectBatch(batch_df, batch_id):
            if False:
                return 10
            batch_df.createOrReplaceTempView('updates')
            assert len(batch_df.sparkSession.sql('SELECT * FROM updates').collect()) == 2
            batch_df.createOrReplaceGlobalTempView('temp_view')
        try:
            df = self.spark.readStream.format('text').load('python/test_support/sql/streaming')
            q = df.writeStream.foreachBatch(collectBatch).start()
            q.processAllAvailable()
            collected = self.spark.sql('SELECT * FROM global_temp.temp_view').collect()
            self.assertTrue(len(collected[0]), 2)
        finally:
            if q:
                q.stop()

    def test_streaming_foreach_batch_propagates_python_errors(self):
        if False:
            print('Hello World!')
        from pyspark.errors import StreamingQueryException
        q = None

        def collectBatch(df, id):
            if False:
                print('Hello World!')
            raise RuntimeError('this should fail the query')
        try:
            df = self.spark.readStream.format('text').load('python/test_support/sql/streaming')
            q = df.writeStream.foreachBatch(collectBatch).start()
            q.processAllAvailable()
            self.fail('Expected a failure')
        except StreamingQueryException as e:
            self.assertTrue('this should fail' in str(e))
        finally:
            if q:
                q.stop()

    def test_streaming_foreach_batch_graceful_stop(self):
        if False:
            i = 10
            return i + 15

        def func(batch_df, _):
            if False:
                i = 10
                return i + 15
            batch_df.sparkSession._jvm.java.lang.Thread.sleep(10000)
        q = self.spark.readStream.format('rate').load().writeStream.foreachBatch(func).start()
        time.sleep(3)
        q.stop()
        self.assertIsNone(q.exception(), 'No exception has to be propagated.')

    def test_streaming_foreach_batch_spark_session(self):
        if False:
            i = 10
            return i + 15
        table_name = 'testTable_foreach_batch'

        def func(df: DataFrame, batch_id: int):
            if False:
                while True:
                    i = 10
            if batch_id > 0:
                return
            spark = df.sparkSession
            df1 = spark.createDataFrame([('structured',), ('streaming',)])
            df1.union(df).write.mode('append').saveAsTable(table_name)
        df = self.spark.readStream.format('text').load('python/test_support/sql/streaming')
        q = df.writeStream.foreachBatch(func).start()
        q.processAllAvailable()
        q.stop()
        actual = self.spark.read.table(table_name)
        df = self.spark.read.format('text').load(path='python/test_support/sql/streaming/').union(self.spark.createDataFrame([('structured',), ('streaming',)]))
        self.assertEqual(sorted(df.collect()), sorted(actual.collect()))

    def test_streaming_foreach_batch_path_access(self):
        if False:
            i = 10
            return i + 15
        table_name = 'testTable_foreach_batch_path'

        def func(df: DataFrame, batch_id: int):
            if False:
                print('Hello World!')
            if batch_id > 0:
                return
            spark = df.sparkSession
            df1 = spark.read.format('text').load('python/test_support/sql/streaming')
            df1.union(df).write.mode('append').saveAsTable(table_name)
        df = self.spark.readStream.format('text').load('python/test_support/sql/streaming')
        q = df.writeStream.foreachBatch(func).start()
        q.processAllAvailable()
        q.stop()
        actual = self.spark.read.table(table_name)
        df = self.spark.read.format('text').load(path='python/test_support/sql/streaming/')
        df = df.union(df)
        self.assertEqual(sorted(df.collect()), sorted(actual.collect()))

    @staticmethod
    def my_test_function_2():
        if False:
            while True:
                i = 10
        return 2

    def test_streaming_foreach_batch_fuction_calling(self):
        if False:
            while True:
                i = 10

        def my_test_function_3():
            if False:
                i = 10
                return i + 15
            return 3
        table_name = 'testTable_foreach_batch_function'

        def func(df: DataFrame, batch_id: int):
            if False:
                while True:
                    i = 10
            if batch_id > 0:
                return
            spark = df.sparkSession
            df1 = spark.createDataFrame([(my_test_function_1(),), (StreamingTestsForeachBatchMixin.my_test_function_2(),), (my_test_function_3(),)])
            df1.write.mode('append').saveAsTable(table_name)
        df = self.spark.readStream.format('rate').load()
        q = df.writeStream.foreachBatch(func).start()
        q.processAllAvailable()
        q.stop()
        actual = self.spark.read.table(table_name)
        df = self.spark.createDataFrame([(my_test_function_1(),), (StreamingTestsForeachBatchMixin.my_test_function_2(),), (my_test_function_3(),)])
        self.assertEqual(sorted(df.collect()), sorted(actual.collect()))

    def test_streaming_foreach_batch_import(self):
        if False:
            i = 10
            return i + 15
        import time
        table_name = 'testTable_foreach_batch_import'

        def func(df: DataFrame, batch_id: int):
            if False:
                return 10
            if batch_id > 0:
                return
            time.sleep(1)
            spark = df.sparkSession
            df1 = spark.read.format('text').load('python/test_support/sql/streaming')
            df1.write.mode('append').saveAsTable(table_name)
        df = self.spark.readStream.format('rate').load()
        q = df.writeStream.foreachBatch(func).start()
        q.processAllAvailable()
        q.stop()
        actual = self.spark.read.table(table_name)
        df = self.spark.read.format('text').load('python/test_support/sql/streaming')
        self.assertEqual(sorted(df.collect()), sorted(actual.collect()))

class StreamingTestsForeachBatch(StreamingTestsForeachBatchMixin, ReusedSQLTestCase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.streaming.test_streaming_foreach_batch import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)