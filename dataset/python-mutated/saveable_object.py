"""Types for specifying saving and loading behavior."""

class SaveSpec:
    """Class used to describe tensor slices that need to be saved."""

    def __init__(self, tensor, slice_spec, name, dtype=None, device=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `SaveSpec` object.\n\n    Args:\n      tensor: the tensor to save or callable that produces a tensor to save.\n        If the value is `None`, the `SaveSpec` is ignored.\n      slice_spec: the slice to be saved. See `Variable.SaveSliceInfo`.\n      name: the name to save the tensor under.\n      dtype: The data type of the Tensor. Required if `tensor` is callable.\n        Used for error checking in the restore op.\n      device: The device generating and consuming this tensor. Required if\n        `tensor` is callable. Used to group objects to save by device.\n    '
        self._tensor = tensor
        self.slice_spec = slice_spec
        self.name = name
        if callable(self._tensor):
            if dtype is None or device is None:
                raise AssertionError('When passing a callable `tensor` to a SaveSpec, an explicit dtype and device must be provided.')
            self.dtype = dtype
            self.device = device
        else:
            self.dtype = tensor.dtype
            if device is not None:
                self.device = device
            else:
                self.device = tensor.device

    @property
    def tensor(self):
        if False:
            while True:
                i = 10
        return self._tensor() if callable(self._tensor) else self._tensor

class SaveableObject:
    """Base class for saving and restoring saveable objects."""

    def __init__(self, op, specs, name):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `SaveableObject` object.\n\n    Args:\n      op: the "producer" object that this class wraps; it produces a list of\n        tensors to save.  E.g., a "Variable" object saving its backing tensor.\n      specs: a list of SaveSpec, each element of which describes one tensor to\n        save under this object. All Tensors must be on the same device.\n      name: the name to save the object under.\n    '
        self.op = op
        self.specs = specs
        self.name = name

    @property
    def device(self):
        if False:
            while True:
                i = 10
        'The device for SaveSpec Tensors.'
        return self.specs[0].device

    def restore(self, restored_tensors, restored_shapes):
        if False:
            while True:
                i = 10
        "Restores this object from 'restored_tensors'.\n\n    Args:\n      restored_tensors: the tensors that were loaded from a checkpoint\n      restored_shapes: the shapes this object should conform to after\n        restore, or None.\n\n    Returns:\n      An operation that restores the state of the object.\n\n    Raises:\n      ValueError: If the object cannot be restored using the provided\n        parameters.\n    "
        raise ValueError('Calling an abstract method.')