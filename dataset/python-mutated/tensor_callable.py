"""`Callable` class used for checkpointing."""
from tensorflow.python.training.saving import saveable_object

class Callable(saveable_object.SaveSpec):
    """A callable that represents a Tensor that should be saved to checkpoint.

  This can be returned from `_serialize_to_tensor` in place of a Tensor. The
  callable will be executed on the specified device when the checkpoint is
  about to be written.

  Any class can use `Callable` for checkpointing, but for SavedModel export,
  only resource-type variables* are supported.

  * `resource_variable_ops.is_resource_variable(obj)` must return True.
  """

    def __init__(self, tensor_callable, dtype, device):
        if False:
            while True:
                i = 10
        'Initializes a `Callable` object.\n\n    Args:\n      tensor_callable: A callable that takes no arguments and returns a Tensor.\n      dtype: Dtype of the tensor returned by the callable.\n      device: Device of the tensor returned by the callable.\n    '
        super().__init__(tensor_callable, None, None, dtype, device)