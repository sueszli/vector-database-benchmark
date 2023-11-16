import json
import time
import uuid
from datetime import datetime
from pyspark import Row
from pyspark.sql.streaming import StreamingQueryListener
from pyspark.sql.streaming.listener import QueryStartedEvent, QueryProgressEvent, QueryTerminatedEvent, SinkProgress, SourceProgress, StateOperatorProgress, StreamingQueryProgress
from pyspark.testing.sqlutils import ReusedSQLTestCase

class StreamingListenerTestsMixin:

    def check_start_event(self, event):
        if False:
            print('Hello World!')
        'Check QueryStartedEvent'
        self.assertTrue(isinstance(event, QueryStartedEvent))
        self.assertTrue(isinstance(event.id, uuid.UUID))
        self.assertTrue(isinstance(event.runId, uuid.UUID))
        self.assertTrue(event.name is None or event.name == 'test')
        try:
            datetime.strptime(event.timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            self.fail("'%s' is not in ISO 8601 format.")

    def check_progress_event(self, event):
        if False:
            print('Hello World!')
        'Check QueryProgressEvent'
        self.assertTrue(isinstance(event, QueryProgressEvent))
        self.check_streaming_query_progress(event.progress)

    def check_terminated_event(self, event, exception=None, error_class=None):
        if False:
            return 10
        'Check QueryTerminatedEvent'
        self.assertTrue(isinstance(event, QueryTerminatedEvent))
        self.assertTrue(isinstance(event.id, uuid.UUID))
        self.assertTrue(isinstance(event.runId, uuid.UUID))
        if exception:
            self.assertTrue(exception in event.exception)
        else:
            self.assertEquals(event.exception, None)
        if error_class:
            self.assertTrue(error_class in event.errorClassOnException)
        else:
            self.assertEquals(event.errorClassOnException, None)

    def check_streaming_query_progress(self, progress):
        if False:
            while True:
                i = 10
        'Check StreamingQueryProgress'
        self.assertTrue(isinstance(progress, StreamingQueryProgress))
        self.assertTrue(isinstance(progress.id, uuid.UUID))
        self.assertTrue(isinstance(progress.runId, uuid.UUID))
        self.assertEquals(progress.name, 'test')
        try:
            json.loads(progress.json)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(progress.prettyJson)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(str(progress))
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            datetime.strptime(progress.timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        except Exception:
            self.fail("'%s' is not in ISO 8601 format.")
        self.assertTrue(isinstance(progress.batchId, int))
        self.assertTrue(isinstance(progress.batchDuration, int))
        self.assertTrue(isinstance(progress.durationMs, dict))
        self.assertTrue(set(progress.durationMs.keys()).issubset({'triggerExecution', 'queryPlanning', 'getBatch', 'commitOffsets', 'latestOffset', 'addBatch', 'walCommit'}))
        self.assertTrue(all(map(lambda v: isinstance(v, int), progress.durationMs.values())))
        self.assertTrue(all(map(lambda v: isinstance(v, str), progress.eventTime.values())))
        self.assertTrue(isinstance(progress.stateOperators, list))
        self.assertTrue(len(progress.stateOperators) >= 1)
        for so in progress.stateOperators:
            self.check_state_operator_progress(so)
        self.assertTrue(isinstance(progress.sources, list))
        self.assertTrue(len(progress.sources) >= 1)
        for so in progress.sources:
            self.check_source_progress(so)
        self.assertTrue(isinstance(progress.sink, SinkProgress))
        self.check_sink_progress(progress.sink)
        self.assertTrue(isinstance(progress.observedMetrics, dict))

    def check_state_operator_progress(self, progress):
        if False:
            return 10
        'Check StateOperatorProgress'
        self.assertTrue(isinstance(progress, StateOperatorProgress))
        try:
            json.loads(progress.json)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(progress.prettyJson)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(str(progress))
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        self.assertTrue(isinstance(progress.operatorName, str))
        self.assertTrue(isinstance(progress.numRowsTotal, int))
        self.assertTrue(isinstance(progress.numRowsUpdated, int))
        self.assertTrue(isinstance(progress.allUpdatesTimeMs, int))
        self.assertTrue(isinstance(progress.numRowsRemoved, int))
        self.assertTrue(isinstance(progress.allRemovalsTimeMs, int))
        self.assertTrue(isinstance(progress.commitTimeMs, int))
        self.assertTrue(isinstance(progress.memoryUsedBytes, int))
        self.assertTrue(isinstance(progress.numRowsDroppedByWatermark, int))
        self.assertTrue(isinstance(progress.numShufflePartitions, int))
        self.assertTrue(isinstance(progress.numStateStoreInstances, int))
        self.assertTrue(isinstance(progress.customMetrics, dict))

    def check_source_progress(self, progress):
        if False:
            print('Hello World!')
        'Check SourceProgress'
        self.assertTrue(isinstance(progress, SourceProgress))
        try:
            json.loads(progress.json)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(progress.prettyJson)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(str(progress))
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        self.assertTrue(isinstance(progress.description, str))
        self.assertTrue(isinstance(progress.startOffset, (str, type(None))))
        self.assertTrue(isinstance(progress.endOffset, (str, type(None))))
        self.assertTrue(isinstance(progress.latestOffset, (str, type(None))))
        self.assertTrue(isinstance(progress.numInputRows, int))
        self.assertTrue(isinstance(progress.inputRowsPerSecond, float))
        self.assertTrue(isinstance(progress.processedRowsPerSecond, float))
        self.assertTrue(isinstance(progress.metrics, dict))

    def check_sink_progress(self, progress):
        if False:
            for i in range(10):
                print('nop')
        'Check SinkProgress'
        self.assertTrue(isinstance(progress, SinkProgress))
        try:
            json.loads(progress.json)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(progress.prettyJson)
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        try:
            json.loads(str(progress))
        except Exception:
            self.fail("'%s' is not a valid JSON.")
        self.assertTrue(isinstance(progress.description, str))
        self.assertTrue(isinstance(progress.numOutputRows, int))
        self.assertTrue(isinstance(progress.metrics, dict))

class StreamingListenerTests(StreamingListenerTestsMixin, ReusedSQLTestCase):

    def test_number_of_public_methods(self):
        if False:
            i = 10
            return i + 15
        msg = 'New field or method was detected in JVM side. If you added a new public field or method, implement that in the corresponding Python class too.Otherwise, fix the number on the assert here.'

        def get_number_of_public_methods(clz):
            if False:
                return 10
            return len(self.spark.sparkContext._jvm.org.apache.spark.util.Utils.classForName(clz, True, False).getMethods())
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.StreamingQueryListener$QueryStartedEvent'), 15, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.StreamingQueryListener$QueryProgressEvent'), 12, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.StreamingQueryListener$QueryTerminatedEvent'), 15, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.StreamingQueryProgress'), 38, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.StateOperatorProgress'), 27, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.SourceProgress'), 21, msg)
        self.assertEquals(get_number_of_public_methods('org.apache.spark.sql.streaming.SinkProgress'), 19, msg)

    def test_listener_events(self):
        if False:
            for i in range(10):
                print('nop')
        start_event = None
        progress_event = None
        terminated_event = None

        class TestListenerV1(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    i = 10
                    return i + 15
                nonlocal start_event
                start_event = event

            def onQueryProgress(self, event):
                if False:
                    while True:
                        i = 10
                nonlocal progress_event
                progress_event = event

            def onQueryTerminated(self, event):
                if False:
                    i = 10
                    return i + 15
                nonlocal terminated_event
                terminated_event = event

        class TestListenerV2(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal start_event
                start_event = event

            def onQueryProgress(self, event):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal progress_event
                progress_event = event

            def onQueryIdle(self, event):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def onQueryTerminated(self, event):
                if False:
                    i = 10
                    return i + 15
                nonlocal terminated_event
                terminated_event = event

        def verify(test_listener):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal start_event
            nonlocal progress_event
            nonlocal terminated_event
            start_event = None
            progress_event = None
            terminated_event = None
            try:
                self.spark.streams.addListener(test_listener)
                df = self.spark.readStream.format('rate').option('rowsPerSecond', 10).load()
                df_stateful = df.groupBy().count()
                q = df_stateful.writeStream.format('noop').queryName('test').outputMode('complete').start()
                self.assertTrue(q.isActive)
                time.sleep(10)
                q.stop()
                self.spark.sparkContext._jsc.sc().listenerBus().waitUntilEmpty()
                self.check_start_event(start_event)
                self.check_progress_event(progress_event)
                self.check_terminated_event(terminated_event)
                from pyspark.sql.functions import col, udf
                bad_udf = udf(lambda x: 1 / 0)
                q = df.select(bad_udf(col('value'))).writeStream.format('noop').start()
                time.sleep(5)
                q.stop()
                self.spark.sparkContext._jsc.sc().listenerBus().waitUntilEmpty()
                self.check_terminated_event(terminated_event, 'ZeroDivisionError')
            finally:
                self.spark.streams.removeListener(test_listener)
        verify(TestListenerV1())
        verify(TestListenerV2())

    def test_remove_listener(self):
        if False:
            print('Hello World!')

        class TestListenerV1(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    while True:
                        i = 10
                pass

            def onQueryProgress(self, event):
                if False:
                    print('Hello World!')
                pass

            def onQueryTerminated(self, event):
                if False:
                    while True:
                        i = 10
                pass

        class TestListenerV2(StreamingQueryListener):

            def onQueryStarted(self, event):
                if False:
                    print('Hello World!')
                pass

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
                    while True:
                        i = 10
                pass

        def verify(test_listener):
            if False:
                for i in range(10):
                    print('nop')
            num_listeners = len(self.spark.streams._jsqm.listListeners())
            self.spark.streams.addListener(test_listener)
            self.assertEqual(num_listeners + 1, len(self.spark.streams._jsqm.listListeners()))
            self.spark.streams.removeListener(test_listener)
            self.assertEqual(num_listeners, len(self.spark.streams._jsqm.listListeners()))
        verify(TestListenerV1())
        verify(TestListenerV2())

    def test_query_started_event_fromJson(self):
        if False:
            for i in range(10):
                print('nop')
        start_event = '\n            {\n                "id" : "78923ec2-8f4d-4266-876e-1f50cf3c283b",\n                "runId" : "55a95d45-e932-4e08-9caa-0a8ecd9391e8",\n                "name" : null,\n                "timestamp" : "2023-06-09T18:13:29.741Z"\n            }\n        '
        start_event = QueryStartedEvent.fromJson(json.loads(start_event))
        self.check_start_event(start_event)
        self.assertEqual(start_event.id, uuid.UUID('78923ec2-8f4d-4266-876e-1f50cf3c283b'))
        self.assertEqual(start_event.runId, uuid.UUID('55a95d45-e932-4e08-9caa-0a8ecd9391e8'))
        self.assertIsNone(start_event.name)
        self.assertEqual(start_event.timestamp, '2023-06-09T18:13:29.741Z')

    def test_query_terminated_event_fromJson(self):
        if False:
            return 10
        terminated_json = '\n            {\n                "id" : "78923ec2-8f4d-4266-876e-1f50cf3c283b",\n                "runId" : "55a95d45-e932-4e08-9caa-0a8ecd9391e8",\n                "exception" : "org.apache.spark.SparkException: Job aborted due to stage failure",\n                "errorClassOnException" : null}\n        '
        terminated_event = QueryTerminatedEvent.fromJson(json.loads(terminated_json))
        self.check_terminated_event(terminated_event, 'SparkException')
        self.assertEqual(terminated_event.id, uuid.UUID('78923ec2-8f4d-4266-876e-1f50cf3c283b'))
        self.assertEqual(terminated_event.runId, uuid.UUID('55a95d45-e932-4e08-9caa-0a8ecd9391e8'))
        self.assertIn('SparkException', terminated_event.exception)
        self.assertIsNone(terminated_event.errorClassOnException)

    def test_streaming_query_progress_fromJson(self):
        if False:
            return 10
        progress_json = '\n            {\n              "id" : "00000000-0000-0001-0000-000000000001",\n              "runId" : "00000000-0000-0001-0000-000000000002",\n              "name" : "test",\n              "timestamp" : "2016-12-05T20:54:20.827Z",\n              "batchId" : 2,\n              "numInputRows" : 678,\n              "inputRowsPerSecond" : 10.0,\n              "processedRowsPerSecond" : 5.4,\n              "batchDuration": 5,\n              "durationMs" : {\n                "getBatch" : 0\n              },\n              "eventTime" : {\n                "min" : "2016-12-05T20:54:20.827Z",\n                "avg" : "2016-12-05T20:54:20.827Z",\n                "watermark" : "2016-12-05T20:54:20.827Z",\n                "max" : "2016-12-05T20:54:20.827Z"\n              },\n              "stateOperators" : [ {\n                "operatorName" : "op1",\n                "numRowsTotal" : 0,\n                "numRowsUpdated" : 1,\n                "allUpdatesTimeMs" : 1,\n                "numRowsRemoved" : 2,\n                "allRemovalsTimeMs" : 34,\n                "commitTimeMs" : 23,\n                "memoryUsedBytes" : 3,\n                "numRowsDroppedByWatermark" : 0,\n                "numShufflePartitions" : 2,\n                "numStateStoreInstances" : 2,\n                "customMetrics" : {\n                  "loadedMapCacheHitCount" : 1,\n                  "loadedMapCacheMissCount" : 0,\n                  "stateOnCurrentVersionSizeBytes" : 2\n                }\n              } ],\n              "sources" : [ {\n                "description" : "source",\n                "startOffset" : 123,\n                "endOffset" : 456,\n                "latestOffset" : 789,\n                "numInputRows" : 678,\n                "inputRowsPerSecond" : 10.0,\n                "processedRowsPerSecond" : 5.4,\n                "metrics": {}\n              } ],\n              "sink" : {\n                "description" : "sink",\n                "numOutputRows" : -1,\n                "metrics": {}\n              },\n              "observedMetrics" : {\n                "event1" : {\n                  "c1" : 1,\n                  "c2" : 3.0\n                },\n                "event2" : {\n                  "rc" : 1,\n                  "min_q" : "hello",\n                  "max_q" : "world"\n                }\n              }\n            }\n        '
        progress = StreamingQueryProgress.fromJson(json.loads(progress_json))
        self.check_streaming_query_progress(progress)
        self.assertEqual(progress.id, uuid.UUID('00000000-0000-0001-0000-000000000001'))
        self.assertEqual(progress.runId, uuid.UUID('00000000-0000-0001-0000-000000000002'))
        self.assertEqual(progress.name, 'test')
        self.assertEqual(progress.timestamp, '2016-12-05T20:54:20.827Z')
        self.assertEqual(progress.batchId, 2)
        self.assertEqual(progress.numInputRows, 678)
        self.assertEqual(progress.inputRowsPerSecond, 10.0)
        self.assertEqual(progress.batchDuration, 5)
        self.assertEqual(progress.durationMs, {'getBatch': 0})
        self.assertEqual(progress.eventTime, {'min': '2016-12-05T20:54:20.827Z', 'avg': '2016-12-05T20:54:20.827Z', 'watermark': '2016-12-05T20:54:20.827Z', 'max': '2016-12-05T20:54:20.827Z'})
        self.assertEqual(progress.observedMetrics, {'event1': Row('c1', 'c2')(1, 3.0), 'event2': Row('rc', 'min_q', 'max_q')(1, 'hello', 'world')})
        self.assertEqual(len(progress.stateOperators), 1)
        state_operator = progress.stateOperators[0]
        self.assertTrue(isinstance(state_operator, StateOperatorProgress))
        self.assertEqual(state_operator.operatorName, 'op1')
        self.assertEqual(state_operator.numRowsTotal, 0)
        self.assertEqual(state_operator.numRowsUpdated, 1)
        self.assertEqual(state_operator.allUpdatesTimeMs, 1)
        self.assertEqual(state_operator.numRowsRemoved, 2)
        self.assertEqual(state_operator.allRemovalsTimeMs, 34)
        self.assertEqual(state_operator.commitTimeMs, 23)
        self.assertEqual(state_operator.memoryUsedBytes, 3)
        self.assertEqual(state_operator.numRowsDroppedByWatermark, 0)
        self.assertEqual(state_operator.numShufflePartitions, 2)
        self.assertEqual(state_operator.numStateStoreInstances, 2)
        self.assertEqual(state_operator.customMetrics, {'loadedMapCacheHitCount': 1, 'loadedMapCacheMissCount': 0, 'stateOnCurrentVersionSizeBytes': 2})
        self.assertEqual(len(progress.sources), 1)
        source = progress.sources[0]
        self.assertTrue(isinstance(source, SourceProgress))
        self.assertEqual(source.description, 'source')
        self.assertEqual(source.startOffset, '123')
        self.assertEqual(source.endOffset, '456')
        self.assertEqual(source.latestOffset, '789')
        self.assertEqual(source.numInputRows, 678)
        self.assertEqual(source.inputRowsPerSecond, 10.0)
        self.assertEqual(source.processedRowsPerSecond, 5.4)
        self.assertEqual(source.metrics, {})
        sink = progress.sink
        self.assertTrue(isinstance(sink, SinkProgress))
        self.assertEqual(sink.description, 'sink')
        self.assertEqual(sink.numOutputRows, -1)
        self.assertEqual(sink.metrics, {})
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.streaming.test_streaming_listener import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)