"""Tests for the actual serialized proto output of the V1 tf.summary ops.

The tensor, audio, and image ops have dedicated tests in adjacent files. The
overall tf.summary API surface also has its own tests in summary_test.py that
check calling the API methods but not the exact serialized proto output.
"""
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary

class SummaryV1OpsTest(test.TestCase):

    def _AsSummary(self, s):
        if False:
            return 10
        summ = summary_pb2.Summary()
        summ.ParseFromString(s)
        return summ

    def testScalarSummary(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            const = constant_op.constant([10.0, 20.0])
            summ = logging_ops.scalar_summary(['c1', 'c2'], const, name='mysumm')
            value = self.evaluate(summ)
        self.assertEqual([], summ.get_shape())
        self.assertProtoEquals('\n      value { tag: "c1" simple_value: 10.0 }\n      value { tag: "c2" simple_value: 20.0 }\n      ', self._AsSummary(value))

    def testScalarSummaryDefaultName(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            const = constant_op.constant([10.0, 20.0])
            summ = logging_ops.scalar_summary(['c1', 'c2'], const)
            value = self.evaluate(summ)
        self.assertEqual([], summ.get_shape())
        self.assertProtoEquals('\n      value { tag: "c1" simple_value: 10.0 }\n      value { tag: "c2" simple_value: 20.0 }\n      ', self._AsSummary(value))

    @test_util.run_deprecated_v1
    def testMergeSummary(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            const = constant_op.constant(10.0)
            summ1 = summary.histogram('h', const)
            summ2 = logging_ops.scalar_summary('c', const)
            merge = summary.merge([summ1, summ2])
            value = self.evaluate(merge)
        self.assertEqual([], merge.get_shape())
        self.assertProtoEquals('\n      value {\n        tag: "h"\n        histo {\n          min: 10.0\n          max: 10.0\n          num: 1.0\n          sum: 10.0\n          sum_squares: 100.0\n          bucket_limit: 9.93809490288\n          bucket_limit: 10.9319043932\n          bucket_limit: 1.7976931348623157e+308\n          bucket: 0.0\n          bucket: 1.0\n          bucket: 0.0\n        }\n      }\n      value { tag: "c" simple_value: 10.0 }\n    ', self._AsSummary(value))

    def testMergeAllSummaries(self):
        if False:
            return 10
        with ops.Graph().as_default():
            const = constant_op.constant(10.0)
            summ1 = summary.histogram('h', const)
            summ2 = summary.scalar('o', const, collections=['foo_key'])
            summ3 = summary.scalar('c', const)
            merge = summary.merge_all()
            self.assertEqual('MergeSummary', merge.op.type)
            self.assertEqual(2, len(merge.op.inputs))
            self.assertEqual(summ1, merge.op.inputs[0])
            self.assertEqual(summ3, merge.op.inputs[1])
            merge = summary.merge_all('foo_key')
            self.assertEqual('MergeSummary', merge.op.type)
            self.assertEqual(1, len(merge.op.inputs))
            self.assertEqual(summ2, merge.op.inputs[0])
            self.assertTrue(summary.merge_all('bar_key') is None)
if __name__ == '__main__':
    test.main()