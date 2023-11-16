from keras import regularizers
from keras.api_export import keras_export
from keras.layers.layer import Layer

@keras_export('keras.layers.ActivityRegularization')
class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity.

    Args:
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return inputs

    def compute_output_shape(self, input_shape):
        if False:
            i = 10
            return i + 15
        return input_shape

    def get_config(self):
        if False:
            while True:
                i = 10
        base_config = super().get_config()
        config = {'l1': self.l1, 'l2': self.l2}
        return {**base_config, **config}