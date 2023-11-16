import os
import tempfile
from pyspark.testing.sqlutils import ReusedSQLTestCase

class StreamingTestsForeachMixin:

    class ForeachWriterTester:

        def __init__(self, spark):
            if False:
                while True:
                    i = 10
            self.spark = spark

        def write_open_event(self, partitionId, epochId):
            if False:
                print('Hello World!')
            self._write_event(self.open_events_dir, {'partition': partitionId, 'epoch': epochId})

        def write_process_event(self, row):
            if False:
                while True:
                    i = 10
            self._write_event(self.process_events_dir, {'value': 'text'})

        def write_close_event(self, error):
            if False:
                return 10
            self._write_event(self.close_events_dir, {'error': str(error)})

        def write_input_file(self):
            if False:
                i = 10
                return i + 15
            self._write_event(self.input_dir, 'text')

        def open_events(self):
            if False:
                print('Hello World!')
            return self._read_events(self.open_events_dir, 'partition INT, epoch INT')

        def process_events(self):
            if False:
                i = 10
                return i + 15
            return self._read_events(self.process_events_dir, 'value STRING')

        def close_events(self):
            if False:
                print('Hello World!')
            return self._read_events(self.close_events_dir, 'error STRING')

        def run_streaming_query_on_writer(self, writer, num_files):
            if False:
                print('Hello World!')
            self._reset()
            try:
                sdf = self.spark.readStream.format('text').load(self.input_dir)
                sq = sdf.writeStream.foreach(writer).start()
                for i in range(num_files):
                    self.write_input_file()
                    sq.processAllAvailable()
            finally:
                self.stop_all()

        def assert_invalid_writer(self, writer, msg=None):
            if False:
                while True:
                    i = 10
            self._reset()
            try:
                sdf = self.spark.readStream.format('text').load(self.input_dir)
                sq = sdf.writeStream.foreach(writer).start()
                self.write_input_file()
                sq.processAllAvailable()
                self.fail('invalid writer %s did not fail the query' % str(writer))
            except Exception as e:
                if msg:
                    assert msg in str(e), '%s not in %s' % (msg, str(e))
            finally:
                self.stop_all()

        def stop_all(self):
            if False:
                for i in range(10):
                    print('nop')
            for q in self.spark.streams.active:
                q.stop()

        def _reset(self):
            if False:
                while True:
                    i = 10
            self.input_dir = tempfile.mkdtemp()
            self.open_events_dir = tempfile.mkdtemp()
            self.process_events_dir = tempfile.mkdtemp()
            self.close_events_dir = tempfile.mkdtemp()

        def _read_events(self, dir, json):
            if False:
                while True:
                    i = 10
            rows = self.spark.read.schema(json).json(dir).collect()
            dicts = [row.asDict() for row in rows]
            return dicts

        def _write_event(self, dir, event):
            if False:
                return 10
            import uuid
            with open(os.path.join(dir, str(uuid.uuid4())), 'w') as f:
                f.write('%s\n' % str(event))

        def __getstate__(self):
            if False:
                while True:
                    i = 10
            return (self.open_events_dir, self.process_events_dir, self.close_events_dir)

        def __setstate__(self, state):
            if False:
                return 10
            (self.open_events_dir, self.process_events_dir, self.close_events_dir) = state

    def test_streaming_foreach_with_simple_function(self):
        if False:
            while True:
                i = 10
        tester = self.ForeachWriterTester(self.spark)

        def foreach_func(row):
            if False:
                print('Hello World!')
            tester.write_process_event(row)
        tester.run_streaming_query_on_writer(foreach_func, 2)
        self.assertEqual(len(tester.process_events()), 2)

    def test_streaming_foreach_with_basic_open_process_close(self):
        if False:
            i = 10
            return i + 15
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def open(self, partitionId, epochId):
                if False:
                    for i in range(10):
                        print('nop')
                tester.write_open_event(partitionId, epochId)
                return True

            def process(self, row):
                if False:
                    return 10
                tester.write_process_event(row)

            def close(self, error):
                if False:
                    print('Hello World!')
                tester.write_close_event(error)
        tester.run_streaming_query_on_writer(ForeachWriter(), 2)
        open_events = tester.open_events()
        self.assertEqual(len(open_events), 2)
        self.assertSetEqual(set([e['epoch'] for e in open_events]), {0, 1})
        self.assertEqual(len(tester.process_events()), 2)
        close_events = tester.close_events()
        self.assertEqual(len(close_events), 2)
        self.assertSetEqual(set([e['error'] for e in close_events]), {'None'})

    def test_streaming_foreach_with_open_returning_false(self):
        if False:
            print('Hello World!')
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def open(self, partition_id, epoch_id):
                if False:
                    for i in range(10):
                        print('nop')
                tester.write_open_event(partition_id, epoch_id)
                return False

            def process(self, row):
                if False:
                    i = 10
                    return i + 15
                tester.write_process_event(row)

            def close(self, error):
                if False:
                    return 10
                tester.write_close_event(error)
        tester.run_streaming_query_on_writer(ForeachWriter(), 2)
        self.assertEqual(len(tester.open_events()), 2)
        self.assertEqual(len(tester.process_events()), 0)
        close_events = tester.close_events()
        self.assertEqual(len(close_events), 2)
        self.assertSetEqual(set([e['error'] for e in close_events]), {'None'})

    def test_streaming_foreach_without_open_method(self):
        if False:
            while True:
                i = 10
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def process(self, row):
                if False:
                    return 10
                tester.write_process_event(row)

            def close(self, error):
                if False:
                    while True:
                        i = 10
                tester.write_close_event(error)
        tester.run_streaming_query_on_writer(ForeachWriter(), 2)
        self.assertEqual(len(tester.open_events()), 0)
        self.assertEqual(len(tester.process_events()), 2)
        self.assertEqual(len(tester.close_events()), 2)

    def test_streaming_foreach_without_close_method(self):
        if False:
            print('Hello World!')
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def open(self, partition_id, epoch_id):
                if False:
                    return 10
                tester.write_open_event(partition_id, epoch_id)
                return True

            def process(self, row):
                if False:
                    i = 10
                    return i + 15
                tester.write_process_event(row)
        tester.run_streaming_query_on_writer(ForeachWriter(), 2)
        self.assertEqual(len(tester.open_events()), 2)
        self.assertEqual(len(tester.process_events()), 2)
        self.assertEqual(len(tester.close_events()), 0)

    def test_streaming_foreach_without_open_and_close_methods(self):
        if False:
            return 10
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def process(self, row):
                if False:
                    while True:
                        i = 10
                tester.write_process_event(row)
        tester.run_streaming_query_on_writer(ForeachWriter(), 2)
        self.assertEqual(len(tester.open_events()), 0)
        self.assertEqual(len(tester.process_events()), 2)
        self.assertEqual(len(tester.close_events()), 0)

    def test_streaming_foreach_with_process_throwing_error(self):
        if False:
            i = 10
            return i + 15
        from pyspark.errors import StreamingQueryException
        tester = self.ForeachWriterTester(self.spark)

        class ForeachWriter:

            def process(self, row):
                if False:
                    while True:
                        i = 10
                raise RuntimeError('test error')

            def close(self, error):
                if False:
                    return 10
                tester.write_close_event(error)
        try:
            tester.run_streaming_query_on_writer(ForeachWriter(), 1)
            self.fail('bad writer did not fail the query')
        except StreamingQueryException:
            pass
        self.assertEqual(len(tester.process_events()), 0)
        close_events = tester.close_events()
        self.assertEqual(len(close_events), 1)

    def test_streaming_foreach_with_invalid_writers(self):
        if False:
            for i in range(10):
                print('nop')
        tester = self.ForeachWriterTester(self.spark)

        def func_with_iterator_input(iter):
            if False:
                print('Hello World!')
            for x in iter:
                print(x)
        tester.assert_invalid_writer(func_with_iterator_input)

        class WriterWithoutProcess:

            def open(self, partition):
                if False:
                    print('Hello World!')
                pass
        tester.assert_invalid_writer(WriterWithoutProcess(), 'ATTRIBUTE_NOT_CALLABLE')

        class WriterWithNonCallableProcess:
            process = True
        tester.assert_invalid_writer(WriterWithNonCallableProcess(), 'ATTRIBUTE_NOT_CALLABLE')

        class WriterWithNoParamProcess:

            def process(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        tester.assert_invalid_writer(WriterWithNoParamProcess())

        class WithProcess:

            def process(self, row):
                if False:
                    while True:
                        i = 10
                pass

        class WriterWithNonCallableOpen(WithProcess):
            open = True
        tester.assert_invalid_writer(WriterWithNonCallableOpen(), 'ATTRIBUTE_NOT_CALLABLE')

        class WriterWithNoParamOpen(WithProcess):

            def open(self):
                if False:
                    print('Hello World!')
                pass
        tester.assert_invalid_writer(WriterWithNoParamOpen())

        class WriterWithNonCallableClose(WithProcess):
            close = True
        tester.assert_invalid_writer(WriterWithNonCallableClose(), 'ATTRIBUTE_NOT_CALLABLE')

class StreamingTestsForeach(StreamingTestsForeachMixin, ReusedSQLTestCase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.streaming.test_streaming_foreach import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)