"""Tests for tf.GrpcServer."""
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SparseJobTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSparseJob(self):
        if False:
            print('Hello World!')
        server = server_lib.Server({'local': {37: 'localhost:0'}})
        with ops.device('/job:local/task:37'):
            a = constant_op.constant(1.0)
        with session.Session(server.target) as sess:
            self.assertEqual(1.0, self.evaluate(a))
if __name__ == '__main__':
    test.main()