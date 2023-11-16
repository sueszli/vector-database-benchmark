"""Tests for tf.GrpcServer."""
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SameVariablesNoClearTest(test.TestCase):

    @test_util.run_v1_only('This exercises tensor lookup via names which is not supported in V2.')
    def testSameVariablesNoClear(self):
        if False:
            i = 10
            return i + 15
        server = server_lib.Server.create_local_server()
        with session.Session(server.target) as sess_1:
            v0 = variable_v1.VariableV1([[2, 1]], name='v0')
            v1 = variable_v1.VariableV1([[1], [2]], name='v1')
            v2 = math_ops.matmul(v0, v1)
            sess_1.run([v0.initializer, v1.initializer])
            self.assertAllEqual([[4]], sess_1.run(v2))
        with session.Session(server.target) as sess_2:
            new_v0 = ops.get_default_graph().get_tensor_by_name('v0:0')
            new_v1 = ops.get_default_graph().get_tensor_by_name('v1:0')
            new_v2 = math_ops.matmul(new_v0, new_v1)
            self.assertAllEqual([[4]], sess_2.run(new_v2))
if __name__ == '__main__':
    test.main()