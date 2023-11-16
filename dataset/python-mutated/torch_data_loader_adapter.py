import tree
from keras.trainers.data_adapters.data_adapter import DataAdapter

class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader):
        if False:
            return 10
        import torch
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(f'Expected argument `dataloader` to be an instance of`torch.utils.data.DataLoader`. Received: {dataloader}')
        self._dataloader = dataloader
        self._batch_size = dataloader.batch_size
        self._size = len(dataloader)
        self._partial_batch_size = len(dataloader.dataset) % self._batch_size

    def get_numpy_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        for batch in self._dataloader:
            yield tuple(tree.map_structure(lambda x: x.cpu().numpy(), batch))

    def get_torch_dataloader(self):
        if False:
            return 10
        return self._dataloader

    def get_tf_dataset(self):
        if False:
            while True:
                i = 10
        from keras.utils.module_utils import tensorflow as tf
        output_signature = self.peek_and_get_tensor_spec()
        return tf.data.Dataset.from_generator(self.get_numpy_iterator, output_signature=output_signature)

    def peek_and_get_tensor_spec(self):
        if False:
            print('Hello World!')
        from keras.utils.module_utils import tensorflow as tf
        batch_data = next(iter(self._dataloader))

        def get_tensor_spec(x):
            if False:
                print('Hello World!')
            shape = x.shape
            if len(shape) < 1:
                raise ValueError(f'When passing a Pytorch DataLoader to a Keras model, the arrays returned by the generator must be at least rank 1. Received: {x} of rank {len(x.shape)}')
            shape = list(shape)
            shape[0] = None
            dtype = str(x.dtype).replace('torch.', '')
            return tf.TensorSpec(shape=shape, dtype=dtype)
        return tuple(tree.map_structure(get_tensor_spec, batch_data))

    @property
    def num_batches(self):
        if False:
            i = 10
            return i + 15
        return self._size

    @property
    def batch_size(self):
        if False:
            print('Hello World!')
        return self._batch_size

    @property
    def has_partial_batch(self):
        if False:
            print('Hello World!')
        if self._partial_batch_size:
            return self._partial_batch_size > 0
        else:
            return None

    @property
    def partial_batch_size(self):
        if False:
            return 10
        return self._partial_batch_size