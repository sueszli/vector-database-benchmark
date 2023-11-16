import collections.abc
import re
import numpy as np
np_str_obj_array_pattern = re.compile('[aO]')
default_collate_err_msg_format = 'default_collator: inputs must contain numpy arrays, numbers, Unicode strings, bytes, dicts or lists; found {}'

class Collator:
    """Used for merging a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a dataset.
    Modified from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    """

    def apply(self, inputs):
        if False:
            i = 10
            return i + 15
        elem = inputs[0]
        elem_type = type(elem)
        if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and (elem_type.__name__ != 'string_'):
            elem = inputs[0]
            if elem_type.__name__ == 'ndarray':
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return np.ascontiguousarray(np.stack(inputs))
            elif elem.shape == ():
                return np.array(inputs)
        elif isinstance(elem, float):
            return np.array(inputs, dtype=np.float64)
        elif isinstance(elem, int):
            return np.array(inputs)
        elif isinstance(elem, (str, bytes)):
            return inputs
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.apply([d[key] for d in inputs]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return elem_type(*(self.apply(samples) for samples in zip(*inputs)))
        elif isinstance(elem, collections.abc.Sequence):
            transposed = zip(*inputs)
            return [self.apply(samples) for samples in transposed]
        raise TypeError(default_collate_err_msg_format.format(elem_type))