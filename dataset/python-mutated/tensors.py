from nvidia.dali.backend import TensorCPU, TensorListCPU, TensorGPU, TensorListGPU

def _transfer_to_cpu(data, device):
    if False:
        i = 10
        return i + 15
    if device.lower() == 'gpu':
        return data.as_cpu()
    return data

def _join_string(data, crop, edgeitems, sep=', '):
    if False:
        for i in range(10):
            print('nop')
    if crop:
        data = data[:edgeitems] + ['...'] + data[-edgeitems:]
    return sep.join(data)
np = None

def import_numpy():
    if False:
        return 10
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('Could not import numpy. Numpy is required for Tensor and TensorList printing. Please make sure you have numpy installed.')

def _tensor_to_string(self):
    if False:
        while True:
            i = 10
    ' Returns string representation of Tensor.'
    import_numpy()
    type_name = type(self).__name__
    indent = ' ' * 4
    layout = self.layout()
    data = np.array(_transfer_to_cpu(self, type_name[-3:]))
    data_str = np.array2string(data, prefix=indent, edgeitems=2)
    params = [f'{type_name}(\n{indent}{data_str}', f'dtype={self.dtype}'] + ([f'layout={layout}'] if layout else []) + [f'shape={self.shape()})']
    return _join_string(params, False, 0, ',\n' + indent)

def _tensorlist_to_string(self, indent=''):
    if False:
        print('Hello World!')
    ' Returns string representation of TensorList.'
    import_numpy()
    edgeitems = 2
    spaces_indent = indent + ' ' * 4
    type_name = type(self).__name__
    layout = self.layout()
    data = _transfer_to_cpu(self, type_name[-3:])
    data_str = '[]'
    crop = False
    if data:
        if data.is_dense_tensor():
            data_str = np.array2string(np.array(data.as_tensor()), prefix=spaces_indent, edgeitems=edgeitems)
        else:
            data = list(map(np.array, data))
            crop = len(data) > 2 * edgeitems + 1 and sum((max(arr.size, 1) for arr in data)) > 1000
            if crop:
                data = data[:edgeitems] + data[-edgeitems:]
            sep = '\n' * data[0].ndim + spaces_indent
            data = [np.array2string(tensor, prefix=spaces_indent, edgeitems=edgeitems) for tensor in data]
            data_str = f'[{_join_string(data, crop, edgeitems, sep)}]'
    shape = self.shape()
    shape_len = len(shape)
    shape_prefix = 'shape=['
    shape_crop = shape_len > 16 or (shape_len > 2 * edgeitems + 1 and shape_len * len(shape[0]) > 100)
    shape = list(map(str, shape))
    shape_str = _join_string(shape, shape_crop, edgeitems)
    if len(shape_str) > 75:
        shape_str = _join_string(shape, shape_crop, edgeitems, ', \n' + spaces_indent + ' ' * len(shape_prefix))
    params = [f'{type_name}(\n{spaces_indent}{data_str}', f'dtype={self.dtype}'] + ([f'layout="{layout}"'] if layout else []) + [f'num_samples={len(self)}', f'{shape_prefix}{shape_str}])']
    return _join_string(params, False, 0, ',\n' + spaces_indent)