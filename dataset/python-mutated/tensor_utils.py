from typing import Sized, Any, List
import numbers
import numpy as np

class TensorType:
    Torch = 'torch'
    TF = 'tf'
    Numeric = 'numeric'
    Numpy = 'numpy'
    Other = 'other'

def tensor_type(item: Any) -> str:
    if False:
        while True:
            i = 10
    module_name = type(item).__module__
    class_name = type(item).__name__
    if module_name == 'torch' and class_name == 'Tensor':
        return TensorType.Torch
    elif module_name.startswith('tensorflow') and class_name == 'EagerTensor':
        return TensorType.TF
    elif isinstance(item, numbers.Number):
        return TensorType.Numeric
    elif isinstance(item, np.ndarray):
        return TensorType.Numpy
    else:
        return TensorType.Other

def tensor2scaler(item: Any) -> numbers.Number:
    if False:
        for i in range(10):
            print('nop')
    tt = tensor_type(item)
    if item is None or tt == TensorType.Numeric:
        return item
    if tt == TensorType.TF:
        item = item.numpy()
    return item.item()

def tensor2np(item: Any) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    tt = tensor_type(item)
    if item is None or tt == TensorType.Numpy:
        return item
    elif tt == TensorType.TF:
        return item.numpy()
    elif tt == TensorType.Torch:
        return item.data.cpu().numpy()
    else:
        return np.array(item)

def to_scaler_list(l: Sized) -> List[numbers.Number]:
    if False:
        print('Hello World!')
    'Create list of scalers for given list of tensors where each element is 0-dim tensor\n    '
    if l is not None and len(l):
        tt = tensor_type(l[0])
        if tt == TensorType.Torch or tt == TensorType.Numpy:
            return [i.item() for i in l]
        elif tt == TensorType.TF:
            return [i.numpy().item() for i in l]
        elif tt == TensorType.Numeric:
            return [i for i in l]
        else:
            raise ValueError('Cannot convert tensor list to scaler list because list element are of unsupported type ' + tt)
    else:
        return None if l is None else []

def to_mean_list(l: Sized) -> List[float]:
    if False:
        for i in range(10):
            print('nop')
    'Create list of scalers for given list of tensors where each element is 0-dim tensor\n    '
    if l is not None and len(l):
        tt = tensor_type(l[0])
        if tt == TensorType.Torch or tt == TensorType.Numpy:
            return [i.mean() for i in l]
        elif tt == TensorType.TF:
            return [i.numpy().mean() for i in l]
        elif tt == TensorType.Numeric:
            return [float(i) for i in l]
        else:
            raise ValueError('Cannot convert tensor list to scaler list because list element are of unsupported type ' + tt)
    else:
        return None if l is None else []

def to_np_list(l: Sized) -> List[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    if l is not None and len(l):
        tt = tensor_type(l[0])
        if tt == TensorType.Numeric:
            return [np.array(i) for i in l]
        if tt == TensorType.TF:
            return [i.numpy() for i in l]
        if tt == TensorType.Torch:
            return [i.data.cpu().numpy() for i in l]
        if tt == TensorType.Numpy:
            return [i for i in l]
        raise ValueError('Cannot convert tensor list to scaler list because list element are of unsupported type ' + tt)
    else:
        return None if l is None else []