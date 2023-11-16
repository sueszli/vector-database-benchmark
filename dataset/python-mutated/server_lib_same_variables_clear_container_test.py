"""Tests for tf.GrpcServer."""
from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SameVariablesClearContainerTest(test.TestCase):

    def testSameVariablesClearContainer(self):
        if False:
            while True:
                i = 10
        server0 = server_lib.Server({'local0': ['localhost:0']}, protocol='grpc', start=True)
        server1 = server_lib.Server({'local1': ['localhost:0']}, protocol='grpc', start=True)
        with ops.Graph().as_default():
            v0 = variables.Variable(1.0, name='v0')
            v1 = variables.Variable(2.0, name='v0')
            sess_0 = session.Session(server0.target)
            sess_1 = session.Session(server1.target)
            sess_0.run(v0.initializer)
            sess_1.run(v1.initializer)
            self.assertAllEqual(1.0, sess_0.run(v0))
            self.assertAllEqual(2.0, sess_1.run(v1))
            session.Session.reset(server0.target, ['local0'])
            _ = session.Session(server0.target)
            with self.assertRaises(errors_impl.FailedPreconditionError):
                self.evaluate(v0)
            self.evaluate(v0.initializer)
            self.assertAllEqual(2.0, sess_1.run(v1))
            session.Session.reset(server1.target, ['local1'])
            _ = session.Session(server1.target)
            with self.assertRaises(errors_impl.FailedPreconditionError):
                self.evaluate(v1)
            _ = session.Session(server0.target)
            self.assertAllEqual(1.0, self.evaluate(v0))
if __name__ == '__main__':
    test.main()