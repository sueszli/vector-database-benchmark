import tensorflow as tf

class SquareTest(tf.test.TestCase):

    def testSquare(self):
        if False:
            return 10
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])
if __name__ == '__main__':
    tf.test.main()