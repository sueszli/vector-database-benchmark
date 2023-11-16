"""Tests for the SWIG-wrapped events writer."""
import os.path
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat

class PywrapeventsWriterTest(test_util.TensorFlowTestCase):

    def testWriteEvents(self):
        if False:
            while True:
                i = 10
        file_prefix = os.path.join(self.get_temp_dir(), 'events')
        writer = _pywrap_events_writer.EventsWriter(compat.as_bytes(file_prefix))
        filename = compat.as_text(writer.FileName())
        event_written = event_pb2.Event(wall_time=123.45, step=67, summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag='foo', simple_value=89.0)]))
        writer.WriteEvent(event_written)
        writer.Flush()
        writer.Close()
        with self.assertRaises(errors.NotFoundError):
            for r in tf_record.tf_record_iterator(filename + 'DOES_NOT_EXIST'):
                self.assertTrue(False)
        reader = tf_record.tf_record_iterator(filename)
        event_read = event_pb2.Event()
        event_read.ParseFromString(next(reader))
        self.assertTrue(event_read.HasField('file_version'))
        event_read.ParseFromString(next(reader))
        self.assertProtoEquals("\n    wall_time: 123.45 step: 67\n    summary { value { tag: 'foo' simple_value: 89.0 } }\n    ", event_read)
        with self.assertRaises(StopIteration):
            next(reader)

    def testWriteEventInvalidType(self):
        if False:
            i = 10
            return i + 15

        class _Invalid(object):

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                return 'Invalid'
        with self.assertRaisesRegex(TypeError, 'Invalid'):
            _pywrap_events_writer.EventsWriter(b'foo').WriteEvent(_Invalid())
if __name__ == '__main__':
    googletest.main()