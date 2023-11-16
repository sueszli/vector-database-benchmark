"""Tests for checking the checkpoint reading and writing metrics."""
import os
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.eager import context
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics

class CheckpointMetricTests(test.TestCase):

    def _get_write_histogram_proto(self, api_label):
        if False:
            for i in range(10):
                print('nop')
        proto_bytes = metrics.GetCheckpointWriteDurations(api_label=api_label)
        histogram_proto = summary_pb2.HistogramProto()
        histogram_proto.ParseFromString(proto_bytes)
        return histogram_proto

    def _get_read_histogram_proto(self, api_label):
        if False:
            print('Hello World!')
        proto_bytes = metrics.GetCheckpointReadDurations(api_label=api_label)
        histogram_proto = summary_pb2.HistogramProto()
        histogram_proto.ParseFromString(proto_bytes)
        return histogram_proto

    def _get_time_saved(self, api_label):
        if False:
            while True:
                i = 10
        return metrics.GetTrainingTimeSaved(api_label=api_label)

    def _get_checkpoint_size(self, api_label, filesize):
        if False:
            return 10
        return metrics.GetCheckpointSize(api_label=api_label, filesize=filesize)

    def test_metrics_v2(self):
        if False:
            for i in range(10):
                print('nop')
        api_label = util._CHECKPOINT_V2
        prefix = os.path.join(self.get_temp_dir(), 'ckpt')
        with context.eager_mode():
            ckpt = util.Checkpoint(v=variables_lib.Variable(1.0))
            self.assertEqual(self._get_time_saved(api_label), 0.0)
            self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)
            for i in range(3):
                time_saved = self._get_time_saved(api_label)
                time.sleep(1)
                ckpt_path = ckpt.write(file_prefix=prefix)
                filesize = util._get_checkpoint_size(ckpt_path)
                self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
                self.assertGreater(self._get_time_saved(api_label), time_saved)
        self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)
        self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)
        time_saved = self._get_time_saved(api_label)
        with context.eager_mode():
            ckpt.restore(ckpt_path)
        self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
        self.assertEqual(self._get_time_saved(api_label), time_saved)

    def test_metrics_v1(self):
        if False:
            i = 10
            return i + 15
        api_label = util._CHECKPOINT_V1
        prefix = os.path.join(self.get_temp_dir(), 'ckpt')
        with self.cached_session():
            ckpt = util.CheckpointV1()
            v = variables_lib.Variable(1.0)
            self.evaluate(v.initializer)
            ckpt.v = v
            self.assertEqual(self._get_time_saved(api_label), 0.0)
            self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)
            for i in range(3):
                time_saved = self._get_time_saved(api_label)
                time.sleep(1)
                ckpt_path = ckpt.write(file_prefix=prefix)
                filesize = util._get_checkpoint_size(ckpt_path)
                self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
                self.assertGreater(self._get_time_saved(api_label), time_saved)
        self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)
        self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)
        time_saved = self._get_time_saved(api_label)
        ckpt.restore(ckpt_path)
        self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
        self.assertEqual(self._get_time_saved(api_label), time_saved)
if __name__ == '__main__':
    test.main()