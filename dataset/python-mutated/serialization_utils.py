import io
import pickle
import numpy as np
import paddle

def convert_object_to_tensor(obj):
    if False:
        print('Hello World!')
    _pickler = pickle.Pickler
    f = io.BytesIO()
    _pickler(f).dump(obj)
    data = np.frombuffer(f.getvalue(), dtype=np.uint8)
    tensor = paddle.to_tensor(data)
    return (tensor, tensor.numel())

def convert_tensor_to_object(tensor, len_of_tensor):
    if False:
        while True:
            i = 10
    _unpickler = pickle.Unpickler
    return _unpickler(io.BytesIO(tensor.numpy()[:len_of_tensor])).load()