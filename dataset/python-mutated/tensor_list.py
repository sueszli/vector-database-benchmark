"""A typed list in Python."""
from tensorflow.python.framework import tensor
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops

def dynamic_list_append(target, element):
    if False:
        print('Hello World!')
    'Converts a list append call inline.'
    if isinstance(target, tensor_array_ops.TensorArray):
        return target.write(target.size(), element)
    if isinstance(target, tensor.Tensor):
        return list_ops.tensor_list_push_back(target, element)
    target.append(element)
    return target

class TensorList(object):
    """Tensor list wrapper API-compatible with Python built-in list."""

    def __init__(self, shape, dtype):
        if False:
            return 10
        self.dtype = dtype
        self.shape = shape
        self.clear()

    def append(self, value):
        if False:
            print('Hello World!')
        self.list_ = list_ops.tensor_list_push_back(self.list_, value)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        (self.list_, value) = list_ops.tensor_list_pop_back(self.list_, self.dtype)
        return value

    def clear(self):
        if False:
            return 10
        self.list_ = list_ops.empty_tensor_list(self.shape, self.dtype)

    def count(self):
        if False:
            while True:
                i = 10
        return list_ops.tensor_list_length(self.list_)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return list_ops.tensor_list_get_item(self.list_, key, self.dtype)

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self.list_ = list_ops.tensor_list_set_item(self.list_, key, value)