"""Identity Tests."""
from tensorflow.python.framework import test_util
from tensorflow.python.ops.distributions import bijector_test_util
from tensorflow.python.ops.distributions import identity_bijector
from tensorflow.python.platform import test

class IdentityBijectorTest(test.TestCase):
    """Tests correctness of the Y = g(X) = X transformation."""

    def testBijector(self):
        if False:
            for i in range(10):
                print('nop')
        bijector = identity_bijector.Identity(validate_args=True)
        self.assertEqual('identity', bijector.name)
        x = [[[0.0], [1.0]]]
        self.assertAllEqual(x, self.evaluate(bijector.forward(x)))
        self.assertAllEqual(x, self.evaluate(bijector.inverse(x)))
        self.assertAllEqual(0.0, self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=3)))
        self.assertAllEqual(0.0, self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=3)))

    @test_util.run_deprecated_v1
    def testScalarCongruency(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            bijector = identity_bijector.Identity()
            bijector_test_util.assert_scalar_congruency(bijector, lower_x=-2.0, upper_x=2.0)
if __name__ == '__main__':
    test.main()