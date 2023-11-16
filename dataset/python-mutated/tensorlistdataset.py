from torch.utils.data import Dataset

class TensorListDataset(Dataset):
    """Dataset wrapping tensors, tensor dicts and tensor lists.

    Arguments:
        *data (Tensor or dict or list of Tensors): tensors that have the same size
        of the first dimension.
    """

    def __init__(self, *data):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].size(0)
        elif isinstance(data[0], list):
            size = data[0][0].size(0)
        else:
            size = data[0].size(0)
        for element in data:
            if isinstance(element, dict):
                assert all((size == tensor.size(0) for (name, tensor) in element.items()))
            elif isinstance(element, list):
                assert all((size == tensor.size(0) for tensor in element))
            else:
                assert size == element.size(0)
        self.size = size
        self.data = data

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append({k: v[index] for (k, v) in element.items()})
            elif isinstance(element, list):
                result.append((v[index] for v in element))
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.size