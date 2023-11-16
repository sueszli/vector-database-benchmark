"""Tests for tf.GrpcServer."""
from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SameVariablesClearTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSameVariablesClear(self):
        if False:
            print('Hello World!')
        server = server_lib.Server.create_local_server()
        v0 = variables.Variable([[2, 1]], name='v0')
        v1 = variables.Variable([[1], [2]], name='v1')
        v2 = math_ops.matmul(v0, v1)
        sess_1 = session.Session(server.target)
        sess_2 = session.Session(server.target)
        sess_1.run(variables.global_variables_initializer())
        self.assertAllEqual([[4]], sess_1.run(v2))
        self.assertAllEqual([[4]], sess_2.run(v2))
        session.Session.reset(server.target)
        with self.assertRaises(errors_impl.AbortedError):
            self.assertAllEqual([[4]], sess_2.run(v2))
        sess_2 = session.Session(server.target)
        with self.assertRaises(errors_impl.FailedPreconditionError):
            sess_2.run(v2)
        sess_2.run(variables.global_variables_initializer())
        self.assertAllEqual([[4]], sess_2.run(v2))
        sess_2.close()
if __name__ == '__main__':
    test.main()