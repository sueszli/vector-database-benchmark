from typing import Optional
import torch
from apache_beam.typehints import typehints
from apache_beam.typehints.batch import BatchConverter
from apache_beam.typehints.batch import N

class PytorchBatchConverter(BatchConverter):

    def __init__(self, batch_type, element_type, dtype, element_shape=(), partition_dimension=0):
        if False:
            while True:
                i = 10
        super().__init__(batch_type, element_type)
        self.dtype = dtype
        self.element_shape = element_shape
        self.partition_dimension = partition_dimension

    @staticmethod
    @BatchConverter.register(name='pytorch')
    def from_typehints(element_type, batch_type) -> Optional['PytorchBatchConverter']:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(element_type, PytorchTypeHint.PytorchTypeConstraint):
            element_type = PytorchTensor[element_type, ()]
        if not isinstance(batch_type, PytorchTypeHint.PytorchTypeConstraint):
            if not batch_type == torch.Tensor:
                raise TypeError('batch type must be torch.Tensor or beam.typehints.pytorch_type_compatibility.PytorchTensor[..]')
            batch_type = PytorchTensor[element_type.dtype, (N,)]
        if not batch_type.dtype == element_type.dtype:
            raise TypeError(f'batch type and element type must have equivalent dtypes (batch={batch_type.dtype}, element={element_type.dtype})')
        computed_element_shape = list(batch_type.shape)
        partition_dimension = computed_element_shape.index(N)
        computed_element_shape.pop(partition_dimension)
        if not tuple(computed_element_shape) == element_type.shape:
            raise TypeError(f"Could not align batch type's batch dimension with element type. (batch type dimensions: {batch_type.shape}, element type dimenstions: {element_type.shape}")
        return PytorchBatchConverter(batch_type, element_type, batch_type.dtype, element_type.shape, partition_dimension)

    def produce_batch(self, elements):
        if False:
            return 10
        return torch.stack(elements, dim=self.partition_dimension)

    def explode_batch(self, batch):
        if False:
            for i in range(10):
                print('nop')
        'Convert an instance of B to Generator[E].'
        yield from torch.swapaxes(batch, self.partition_dimension, 0)

    def combine_batches(self, batches):
        if False:
            while True:
                i = 10
        return torch.cat(batches, dim=self.partition_dimension)

    def get_length(self, batch):
        if False:
            for i in range(10):
                print('nop')
        return batch.size(dim=self.partition_dimension)

    def estimate_byte_size(self, batch):
        if False:
            while True:
                i = 10
        return batch.nelement() * batch.element_size()

class PytorchTypeHint:

    class PytorchTypeConstraint(typehints.TypeConstraint):

        def __init__(self, dtype, shape=()):
            if False:
                print('Hello World!')
            self.dtype = dtype
            self.shape = shape

        def type_check(self, batch):
            if False:
                print('Hello World!')
            if not isinstance(batch, torch.Tensor):
                raise TypeError(f'Batch {batch!r} is not an instance of torch.Tensor')
            if not batch.dtype == self.dtype:
                raise TypeError(f'Batch {batch!r} does not have expected dtype: {self.dtype!r}')
            for dim in range(len(self.shape)):
                if not self.shape[dim] == N and (not batch.shape[dim] == self.shape[dim]):
                    raise TypeError(f'Batch {batch!r} does not have expected shape: {self.shape!r}')

        def _consistent_with_check_(self, sub):
            if False:
                while True:
                    i = 10
            return True

        def __key(self):
            if False:
                print('Hello World!')
            return (self.dtype, self.shape)

        def __eq__(self, other) -> bool:
            if False:
                i = 10
                return i + 15
            if isinstance(other, PytorchTypeHint.PytorchTypeConstraint):
                return self.__key() == other.__key()
            return NotImplemented

        def __hash__(self) -> int:
            if False:
                print('Hello World!')
            return hash(self.__key())

        def __repr__(self):
            if False:
                print('Hello World!')
            if self.shape == (N,):
                return f'PytorchTensor[{self.dtype!r}]'
            else:
                return f'PytorchTensor[{self.dtype!r}, {self.shape!r}]'

    def __getitem__(self, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, tuple):
            if len(value) == 2:
                (dtype, shape) = value
                return self.PytorchTypeConstraint(dtype, shape=shape)
            else:
                raise ValueError
        else:
            dtype = value
            return self.PytorchTypeConstraint(dtype, shape=(N,))
PytorchTensor = PytorchTypeHint()