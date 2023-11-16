"""Tests for tensorflow.python.summary.summary_iterator."""
import glob
import os.path
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer

class SummaryIteratorTestCase(test.TestCase):

    @test_util.run_deprecated_v1
    def testSummaryIteratorEventsAddedAfterEndOfFile(self):
        if False:
            print('Hello World!')
        test_dir = os.path.join(self.get_temp_dir(), 'events')
        with writer.FileWriter(test_dir) as w:
            session_log_start = event_pb2.SessionLog.START
            w.add_session_log(event_pb2.SessionLog(status=session_log_start), 1)
            w.flush()
            path = glob.glob(os.path.join(test_dir, 'event*'))[0]
            rr = summary_iterator.summary_iterator(path)
            ev = next(rr)
            self.assertEqual('brain.Event:2', ev.file_version)
            ev = next(rr)
            self.assertEqual(1, ev.step)
            self.assertEqual(session_log_start, ev.session_log.status)
            self.assertRaises(StopIteration, lambda : next(rr))
            w.add_session_log(event_pb2.SessionLog(status=session_log_start), 2)
            w.flush()
            ev = next(rr)
            self.assertEqual(2, ev.step)
            self.assertEqual(session_log_start, ev.session_log.status)
            self.assertRaises(StopIteration, lambda : next(rr))
if __name__ == '__main__':
    test.main()