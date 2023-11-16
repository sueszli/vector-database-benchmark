import unittest
import time
import pyspark.cloudpickle
from pyspark.errors import PySparkPicklingError
from pyspark.sql.tests.streaming.test_streaming_listener import StreamingListenerTestsMixin
from pyspark.sql.streaming.listener import StreamingQueryListener
from pyspark.sql.functions import count, lit
from pyspark.testing.connectutils import ReusedConnectTestCase

class TestListener(StreamingQueryListener):

    def onQueryStarted(self, event):
        if False:
            for i in range(10):
                print('nop')
        e = pyspark.cloudpickle.dumps(event)
        df = self.spark.createDataFrame(data=[(e,)])
        df.write.mode('append').saveAsTable('listener_start_events')

    def onQueryProgress(self, event):
        if False:
            return 10
        e = pyspark.cloudpickle.dumps(event)
        df = self.spark.createDataFrame(data=[(e,)])
        df.write.mode('append').saveAsTable('listener_progress_events')

    def onQueryIdle(self, event):
        if False:
            i = 10
            return i + 15
        pass

    def onQueryTerminated(self, event):
        if False:
            while True:
                i = 10
        e = pyspark.cloudpickle.dumps(event)
        df = self.spark.createDataFrame(data=[(e,)])
        df.write.mode('append').saveAsTable('listener_terminated_events')

class StreamingListenerParityTests(StreamingListenerTestsMixin, ReusedConnectTestCase):

    def test_listener_events(self):
        if False:
            i = 10
            return i + 15
        test_listener = TestListener()
        try:
            self.spark.streams.addListener(test_listener)
            time.sleep(30)
            df = self.spark.readStream.format('rate').option('rowsPerSecond', 10).load()
            df_observe = df.observe('my_event', count(lit(1)).alias('rc'))
            df_stateful = df_observe.groupBy().count()
            q = df_stateful.writeStream.format('noop').queryName('test').outputMode('complete').start()
            self.assertTrue(q.isActive)
            time.sleep(10)
            self.assertTrue(q.lastProgress['batchId'] > 0)
            q.stop()
            self.assertFalse(q.isActive)
            start_event = pyspark.cloudpickle.loads(self.spark.read.table('listener_start_events').collect()[0][0])
            progress_event = pyspark.cloudpickle.loads(self.spark.read.table('listener_progress_events').collect()[0][0])
            terminated_event = pyspark.cloudpickle.loads(self.spark.read.table('listener_terminated_events').collect()[0][0])
            self.check_start_event(start_event)
            self.check_progress_event(progress_event)
            self.check_terminated_event(terminated_event)
        finally:
            self.spark.streams.removeListener(test_listener)
            self.spark.streams.removeListener(test_listener)

    def test_accessing_spark_session(self):
        if False:
            for i in range(10):
                print('nop')
        spark = self.spark

        class TestListener(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    while True:
                        i = 10
                spark.createDataFrame([('do', 'not'), ('serialize', 'spark')]).collect()

            def onQueryProgress(self, event):
                if False:
                    print('Hello World!')
                pass

            def onQueryIdle(self, event):
                if False:
                    i = 10
                    return i + 15
                pass

            def onQueryTerminated(self, event):
                if False:
                    return 10
                pass
        error_thrown = False
        try:
            self.spark.streams.addListener(TestListener())
        except PySparkPicklingError as e:
            self.assertEqual(e.getErrorClass(), 'STREAMING_CONNECT_SERIALIZATION_ERROR')
            error_thrown = True
        self.assertTrue(error_thrown)

    def test_accessing_spark_session_through_df(self):
        if False:
            print('Hello World!')
        dataframe = self.spark.createDataFrame([('do', 'not'), ('serialize', 'dataframe')])

        class TestListener(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    while True:
                        i = 10
                dataframe.collect()

            def onQueryProgress(self, event):
                if False:
                    return 10
                pass

            def onQueryIdle(self, event):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def onQueryTerminated(self, event):
                if False:
                    i = 10
                    return i + 15
                pass
        error_thrown = False
        try:
            self.spark.streams.addListener(TestListener())
        except PySparkPicklingError as e:
            self.assertEqual(e.getErrorClass(), 'STREAMING_CONNECT_SERIALIZATION_ERROR')
            error_thrown = True
        self.assertTrue(error_thrown)
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.streaming.test_parity_listener import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)