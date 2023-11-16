"""Unit tests for the write transform."""
import logging
import unittest
import apache_beam as beam
from apache_beam.io import iobase
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import is_empty
from apache_beam.transforms.ptransform import PTransform

class _TestSink(iobase.Sink):
    TEST_INIT_RESULT = 'test_init_result'

    def __init__(self, return_init_result=True, return_write_results=True):
        if False:
            print('Hello World!')
        self.return_init_result = return_init_result
        self.return_write_results = return_write_results

    def initialize_write(self):
        if False:
            return 10
        if self.return_init_result:
            return _TestSink.TEST_INIT_RESULT

    def pre_finalize(self, init_result, writer_results):
        if False:
            print('Hello World!')
        pass

    def finalize_write(self, init_result, writer_results, unused_pre_finalize_result):
        if False:
            for i in range(10):
                print('nop')
        self.init_result_at_finalize = init_result
        self.write_results_at_finalize = writer_results

    def open_writer(self, init_result, uid):
        if False:
            return 10
        writer = _TestWriter(init_result, uid, self.return_write_results)
        return writer

class _TestWriter(iobase.Writer):
    (STATE_UNSTARTED, STATE_WRITTEN, STATE_CLOSED) = (0, 1, 2)
    TEST_WRITE_RESULT = 'test_write_result'

    def __init__(self, init_result, uid, return_write_results=True):
        if False:
            while True:
                i = 10
        self.state = _TestWriter.STATE_UNSTARTED
        self.init_result = init_result
        self.uid = uid
        self.write_output = []
        self.return_write_results = return_write_results

    def close(self):
        if False:
            print('Hello World!')
        assert self.state in (_TestWriter.STATE_WRITTEN, _TestWriter.STATE_UNSTARTED)
        self.state = _TestWriter.STATE_CLOSED
        if self.return_write_results:
            return _TestWriter.TEST_WRITE_RESULT

    def write(self, value):
        if False:
            print('Hello World!')
        if self.write_output:
            assert self.state == _TestWriter.STATE_WRITTEN
        else:
            assert self.state == _TestWriter.STATE_UNSTARTED
        self.state = _TestWriter.STATE_WRITTEN
        self.write_output.append(value)

class WriteToTestSink(PTransform):

    def __init__(self, return_init_result=True, return_write_results=True):
        if False:
            while True:
                i = 10
        self.return_init_result = return_init_result
        self.return_write_results = return_write_results
        self.last_sink = None
        self.label = 'write_to_test_sink'

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        self.last_sink = _TestSink(return_init_result=self.return_init_result, return_write_results=self.return_write_results)
        return pcoll | beam.io.Write(self.last_sink)

class WriteTest(unittest.TestCase):
    DATA = ['some data', 'more data', 'another data', 'yet another data']

    def _run_write_test(self, data, return_init_result=True, return_write_results=True):
        if False:
            print('Hello World!')
        write_to_test_sink = WriteToTestSink(return_init_result, return_write_results)
        with TestPipeline() as p:
            result = p | beam.Create(data) | write_to_test_sink | beam.Map(list)
            assert_that(result, is_empty())
        sink = write_to_test_sink.last_sink
        self.assertIsNotNone(sink)

    def test_write(self):
        if False:
            while True:
                i = 10
        self._run_write_test(WriteTest.DATA)

    def test_write_with_empty_pcollection(self):
        if False:
            i = 10
            return i + 15
        data = []
        self._run_write_test(data)

    def test_write_no_init_result(self):
        if False:
            return 10
        self._run_write_test(WriteTest.DATA, return_init_result=False)

    def test_write_no_write_results(self):
        if False:
            while True:
                i = 10
        self._run_write_test(WriteTest.DATA, return_write_results=False)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()