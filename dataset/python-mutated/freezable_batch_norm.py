"""A freezable batch norm layer that uses Keras batch normalization."""
import tensorflow as tf

class FreezableBatchNorm(tf.keras.layers.BatchNormalization):
    """Batch normalization layer (Ioffe and Szegedy, 2014).

  This is a `freezable` batch norm layer that supports setting the `training`
  parameter in the __init__ method rather than having to set it either via
  the Keras learning phase or via the `call` method parameter. This layer will
  forward all other parameters to the default Keras `BatchNormalization`
  layer

  This is class is necessary because Object Detection model training sometimes
  requires batch normalization layers to be `frozen` and used as if it was
  evaluation time, despite still training (and potentially using dropout layers)

  Like the default Keras BatchNormalization layer, this will normalize the
  activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
    training: If False, the layer will normalize using the moving average and
      std. dev, without updating the learned avg and std. dev.
      If None or True, the layer will follow the keras BatchNormalization layer
      strategy of checking the Keras learning phase at `call` time to decide
      what to do.
    **kwargs: The keyword arguments to forward to the keras BatchNormalization
        layer constructor.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

    def __init__(self, training=None, **kwargs):
        if False:
            print('Hello World!')
        super(FreezableBatchNorm, self).__init__(**kwargs)
        self._training = training

    def call(self, inputs, training=None):
        if False:
            for i in range(10):
                print('nop')
        if self._training is False:
            training = self._training
        return super(FreezableBatchNorm, self).call(inputs, training=training)