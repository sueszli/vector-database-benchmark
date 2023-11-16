"""Identity bijector."""
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation
__all__ = ['Identity']

class Identity(bijector.Bijector):
    """Compute Y = g(X) = X.

    Example Use:

    ```python
    # Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
    # ndim and 1 event ndim (i.e., vector of vectors).
    identity = Identity()
    x = [[1., 2],
         [3, 4]]
    x == identity.forward(x) == identity.inverse(x)
    ```

  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, validate_args=False, name='identity'):
        if False:
            print('Hello World!')
        super(Identity, self).__init__(forward_min_event_ndims=0, is_constant_jacobian=True, validate_args=validate_args, name=name)

    def _forward(self, x):
        if False:
            i = 10
            return i + 15
        return x

    def _inverse(self, y):
        if False:
            return 10
        return y

    def _inverse_log_det_jacobian(self, y):
        if False:
            print('Hello World!')
        return constant_op.constant(0.0, dtype=y.dtype)

    def _forward_log_det_jacobian(self, x):
        if False:
            i = 10
            return i + 15
        return constant_op.constant(0.0, dtype=x.dtype)