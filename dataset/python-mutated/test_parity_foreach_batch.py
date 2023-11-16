import unittest
from pyspark.sql.tests.streaming.test_streaming_foreach_batch import StreamingTestsForeachBatchMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.errors import PySparkPicklingError

class StreamingForeachBatchParityTests(StreamingTestsForeachBatchMixin, ReusedConnectTestCase):

    def test_streaming_foreach_batch_propagates_python_errors(self):
        if False:
            print('Hello World!')
        super().test_streaming_foreach_batch_propagates_python_errors()

    @unittest.skip('This seems specific to py4j and pinned threads. The intention is unclear')
    def test_streaming_foreach_batch_graceful_stop(self):
        if False:
            return 10
        super().test_streaming_foreach_batch_graceful_stop()

    def test_accessing_spark_session(self):
        if False:
            while True:
                i = 10
        spark = self.spark

        def func(df, _):
            if False:
                for i in range(10):
                    print('nop')
            spark.createDataFrame([('do', 'not'), ('serialize', 'spark')]).collect()
        error_thrown = False
        try:
            self.spark.readStream.format('rate').load().writeStream.foreachBatch(func).start()
        except PySparkPicklingError as e:
            self.assertEqual(e.getErrorClass(), 'STREAMING_CONNECT_SERIALIZATION_ERROR')
            error_thrown = True
        self.assertTrue(error_thrown)

    def test_accessing_spark_session_through_df(self):
        if False:
            i = 10
            return i + 15
        dataframe = self.spark.createDataFrame([('do', 'not'), ('serialize', 'dataframe')])

        def func(df, _):
            if False:
                return 10
            dataframe.collect()
        error_thrown = False
        try:
            self.spark.readStream.format('rate').load().writeStream.foreachBatch(func).start()
        except PySparkPicklingError as e:
            self.assertEqual(e.getErrorClass(), 'STREAMING_CONNECT_SERIALIZATION_ERROR')
            error_thrown = True
        self.assertTrue(error_thrown)
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.streaming.test_parity_foreach_batch import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)