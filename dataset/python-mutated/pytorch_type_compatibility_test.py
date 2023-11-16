"""Unit tests for pytorch_type_compabitility."""
import unittest
from typing import Any
import pytest
from parameterized import parameterized
from parameterized import parameterized_class
from apache_beam.typehints import typehints
from apache_beam.typehints.batch import BatchConverter
from apache_beam.typehints.batch import N
try:
    import torch
    from apache_beam.typehints.pytorch_type_compatibility import PytorchTensor
except ImportError:
    raise unittest.SkipTest('PyTorch dependencies are not installed')

@parameterized_class([{'batch_typehint': torch.Tensor, 'element_typehint': PytorchTensor[torch.int32, ()], 'batch': torch.tensor(range(100), dtype=torch.int32)}, {'batch_typehint': PytorchTensor[torch.int64, (N, 10)], 'element_typehint': PytorchTensor[torch.int64, (10,)], 'batch': torch.tensor([list(range(i, i + 10)) for i in range(100)], dtype=torch.int64)}])
@pytest.mark.uses_pytorch
class PytorchBatchConverterTest(unittest.TestCase):

    def create_batch_converter(self):
        if False:
            for i in range(10):
                print('nop')
        return BatchConverter.from_typehints(element_type=self.element_typehint, batch_type=self.batch_typehint)

    def setUp(self):
        if False:
            return 10
        self.converter = self.create_batch_converter()
        self.normalized_batch_typehint = typehints.normalize(self.batch_typehint)
        self.normalized_element_typehint = typehints.normalize(self.element_typehint)

    def equality_check(self, left, right):
        if False:
            i = 10
            return i + 15
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        else:
            raise TypeError(f'Encountered unexpected type, left is a {type(left)!r}')

    def test_typehint_validates(self):
        if False:
            return 10
        typehints.validate_composite_type_param(self.batch_typehint, '')
        typehints.validate_composite_type_param(self.element_typehint, '')

    def test_type_check_batch(self):
        if False:
            print('Hello World!')
        typehints.check_constraint(self.normalized_batch_typehint, self.batch)

    def test_type_check_element(self):
        if False:
            i = 10
            return i + 15
        for element in self.converter.explode_batch(self.batch):
            typehints.check_constraint(self.normalized_element_typehint, element)

    def test_explode_rebatch(self):
        if False:
            while True:
                i = 10
        exploded = list(self.converter.explode_batch(self.batch))
        rebatched = self.converter.produce_batch(exploded)
        typehints.check_constraint(self.normalized_batch_typehint, rebatched)
        self.equality_check(self.batch, rebatched)

    def _split_batch_into_n_partitions(self, N):
        if False:
            i = 10
            return i + 15
        elements = list(self.converter.explode_batch(self.batch))
        element_batches = [elements[len(elements) * i // N:len(elements) * (i + 1) // N] for i in range(N)]
        lengths = [len(element_batch) for element_batch in element_batches]
        batches = [self.converter.produce_batch(element_batch) for element_batch in element_batches]
        return (batches, lengths)

    @parameterized.expand([(2,), (3,), (10,)])
    def test_combine_batches(self, N):
        if False:
            return 10
        (batches, _) = self._split_batch_into_n_partitions(N)
        combined = self.converter.combine_batches(batches)
        self.equality_check(self.batch, combined)

    @parameterized.expand([(2,), (3,), (10,)])
    def test_get_length(self, N):
        if False:
            i = 10
            return i + 15
        (batches, lengths) = self._split_batch_into_n_partitions(N)
        for (batch, expected_length) in zip(batches, lengths):
            self.assertEqual(self.converter.get_length(batch), expected_length)

    def test_equals(self):
        if False:
            return 10
        self.assertTrue(self.converter == self.create_batch_converter())
        self.assertTrue(self.create_batch_converter() == self.converter)

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(hash(self.create_batch_converter()), hash(self.converter))

class PytorchBatchConverterErrorsTest(unittest.TestCase):

    @parameterized.expand([(Any, PytorchTensor[torch.int64, ()], 'batch type must be torch\\.Tensor or beam\\.typehints\\.pytorch_type_compatibility.PytorchTensor'), (PytorchTensor[torch.int64, (3, N, 2)], PytorchTensor[torch.int64, (3, 7)], "Could not align batch type\\'s batch dimension"), (PytorchTensor[torch.int64, (N, 10)], PytorchTensor[torch.float32, (10,)], 'batch type and element type must have equivalent dtypes')])
    def test_construction_errors(self, batch_typehint, element_typehint, error_regex):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, error_regex):
            BatchConverter.from_typehints(element_type=element_typehint, batch_type=batch_typehint)
if __name__ == '__main__':
    unittest.main()