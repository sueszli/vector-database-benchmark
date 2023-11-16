"""Tests for tf.GrpcServer."""
from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class MultipleContainersTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testMultipleContainers(self):
        if False:
            print('Hello World!')
        with ops.container('test0'):
            v0 = variables.Variable(1.0, name='v0')
        with ops.container('test1'):
            v1 = variables.Variable(2.0, name='v0')
        server = server_lib.Server.create_local_server()
        sess = session.Session(server.target)
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(1.0, sess.run(v0))
        self.assertAllEqual(2.0, sess.run(v1))
        session.Session.reset(server.target, ['test0'])
        with self.assertRaises(errors_impl.AbortedError):
            sess.run(v1)
        sess = session.Session(server.target)
        with self.assertRaises(errors_impl.FailedPreconditionError):
            sess.run(v0)
        self.assertAllEqual(2.0, sess.run(v1))
if __name__ == '__main__':
    test.main()