"""Test that user ops can be used as expected."""
import tensorflow as tf
from tensorflow.python.framework import test_util

class FactTest(tf.test.TestCase):

    @test_util.run_deprecated_v1
    def test(self):
        if False:
            return 10
        with self.cached_session():
            print(tf.compat.v1.user_ops.my_fact().eval())
if __name__ == '__main__':
    tf.test.main()