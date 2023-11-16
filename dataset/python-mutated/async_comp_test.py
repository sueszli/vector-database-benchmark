"""Tests for asynchronous compilation on the CPU and GPU devices."""
import os
import unittest
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

def RunMetadataLabels(run_metadata):
    if False:
        i = 10
        return i + 15
    'Returns all labels in run_metadata.'
    labels = []
    for dev_stats in run_metadata.step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
            labels.append(node_stats.timeline_label)
    return labels

def InLabels(labels, substr):
    if False:
        return 10
    'Returns true iff one of the labels contains substr.'
    return any((substr in x for x in labels))

def MetadataHasXlaRunOp(run_metadata):
    if False:
        return 10
    "Returns true if there are XlaRun kernels in run_metadata's timeline."
    return InLabels(RunMetadataLabels(run_metadata), '_XlaRun')

class AsyncCompilationTest(test.TestCase):

    @unittest.skip('b/263146341 - flaky Kokoro build.')
    def testAsyncCompilationJit(self):
        if False:
            while True:
                i = 10

        @function.Defun(compiled=True)
        def CompiledFunction(x):
            if False:
                i = 10
                return i + 15
            return math_ops.log(x)
        with session_lib.Session() as sess:
            x = array_ops.placeholder(dtypes.float32)
            y = CompiledFunction(x)
            run_metadata = config_pb2.RunMetadata()
            sess.run(y, feed_dict={x: [0.0] * 60}, run_metadata=run_metadata, options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE))
            hasXlaRunOp = MetadataHasXlaRunOp(run_metadata)
            self.assertFalse(hasXlaRunOp)
            while not hasXlaRunOp:
                run_metadata = config_pb2.RunMetadata()
                sess.run(y, feed_dict={x: [0.0] * 60}, run_metadata=run_metadata, options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE))
                hasXlaRunOp = MetadataHasXlaRunOp(run_metadata)
if __name__ == '__main__':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_async_compilation=true ' + '--tf_xla_enable_lazy_compilation=true ' + os.environ.get('TF_XLA_FLAGS', '')
    ops.disable_eager_execution()
    test.main()