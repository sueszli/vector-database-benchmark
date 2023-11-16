"""Protobuf related tests."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

class ProtoTest(test.TestCase):

    def _testLargeProto(self):
        if False:
            print('Hello World!')
        a = constant_op.constant(np.zeros([1024, 1024, 17]))
        gdef = a.op.graph.as_graph_def()
        serialized = gdef.SerializeToString()
        unserialized = ops.Graph().as_graph_def()
        unserialized.ParseFromString(serialized)
        self.assertProtoEquals(unserialized, gdef)
if __name__ == '__main__':
    test.main()