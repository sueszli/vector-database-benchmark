"""UnitTests for Batched DoFn (process_batch) API."""
import unittest
from typing import Iterator
from typing import List
from typing import Tuple
from typing import no_type_check
from parameterized import parameterized_class
import apache_beam as beam

class ElementDoFn(beam.DoFn):

    def process(self, element: int, *args, **kwargs) -> Iterator[float]:
        if False:
            print('Hello World!')
        yield (element / 2)

class BatchDoFn(beam.DoFn):

    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[List[float]]:
        if False:
            for i in range(10):
                print('nop')
        yield [element / 2 for element in batch]

class NoReturnAnnotation(beam.DoFn):

    def process_batch(self, batch: List[int], *args, **kwargs):
        if False:
            while True:
                i = 10
        yield [element * 2 for element in batch]

class OverrideTypeInference(beam.DoFn):

    def process_batch(self, batch, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        yield [element * 2 for element in batch]

    def get_input_batch_type(self, input_element_type):
        if False:
            print('Hello World!')
        return List[input_element_type]

    def get_output_batch_type(self, input_element_type):
        if False:
            print('Hello World!')
        return List[input_element_type]

class EitherDoFn(beam.DoFn):

    def process(self, element: int, *args, **kwargs) -> Iterator[float]:
        if False:
            return 10
        yield (element / 2)

    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[List[float]]:
        if False:
            for i in range(10):
                print('nop')
        yield [element / 2 for element in batch]

class ElementToBatchDoFn(beam.DoFn):

    @beam.DoFn.yields_batches
    def process(self, element: int, *args, **kwargs) -> Iterator[List[int]]:
        if False:
            while True:
                i = 10
        yield ([element] * element)

    def infer_output_type(self, input_element_type):
        if False:
            print('Hello World!')
        return input_element_type

class BatchToElementDoFn(beam.DoFn):

    @beam.DoFn.yields_elements
    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[Tuple[int, int]]:
        if False:
            print('Hello World!')
        yield (sum(batch), len(batch))

def get_test_class_name(cls, num, params_dict):
    if False:
        i = 10
        return i + 15
    return '%s_%s' % (cls.__name__, params_dict['dofn'].__class__.__name__)

@parameterized_class([{'dofn': ElementDoFn(), 'input_element_type': int, 'expected_process_defined': True, 'expected_process_batch_defined': False, 'expected_input_batch_type': None, 'expected_output_batch_type': None}, {'dofn': BatchDoFn(), 'input_element_type': int, 'expected_process_defined': False, 'expected_process_batch_defined': True, 'expected_input_batch_type': beam.typehints.List[int], 'expected_output_batch_type': beam.typehints.List[float]}, {'dofn': NoReturnAnnotation(), 'input_element_type': int, 'expected_process_defined': False, 'expected_process_batch_defined': True, 'expected_input_batch_type': beam.typehints.List[int], 'expected_output_batch_type': beam.typehints.List[int]}, {'dofn': OverrideTypeInference(), 'input_element_type': int, 'expected_process_defined': False, 'expected_process_batch_defined': True, 'expected_input_batch_type': beam.typehints.List[int], 'expected_output_batch_type': beam.typehints.List[int]}, {'dofn': EitherDoFn(), 'input_element_type': int, 'expected_process_defined': True, 'expected_process_batch_defined': True, 'expected_input_batch_type': beam.typehints.List[int], 'expected_output_batch_type': beam.typehints.List[float]}, {'dofn': ElementToBatchDoFn(), 'input_element_type': int, 'expected_process_defined': True, 'expected_process_batch_defined': False, 'expected_input_batch_type': None, 'expected_output_batch_type': beam.typehints.List[int]}, {'dofn': BatchToElementDoFn(), 'input_element_type': int, 'expected_process_defined': False, 'expected_process_batch_defined': True, 'expected_input_batch_type': beam.typehints.List[int], 'expected_output_batch_type': None}], class_name_func=get_test_class_name)
class BatchDoFnParameterizedTest(unittest.TestCase):

    def test_process_defined(self):
        if False:
            return 10
        self.assertEqual(self.dofn._process_defined, self.expected_process_defined)

    def test_process_batch_defined(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.dofn._process_batch_defined, self.expected_process_batch_defined)

    def test_get_input_batch_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.dofn._get_input_batch_type_normalized(self.input_element_type), self.expected_input_batch_type)

    def test_get_output_batch_type(self):
        if False:
            return 10
        self.assertEqual(self.dofn._get_output_batch_type_normalized(self.input_element_type), self.expected_output_batch_type)

    def test_can_yield_batches(self):
        if False:
            print('Hello World!')
        expected = self.expected_output_batch_type is not None
        self.assertEqual(self.dofn._can_yield_batches, expected)

class NoInputAnnotation(beam.DoFn):

    def process_batch(self, batch, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        yield [element * 2 for element in batch]

class MismatchedBatchProducingDoFn(beam.DoFn):
    """A DoFn that produces batches from both process and process_batch, with
  mismatched return types (one yields floats, the other ints). Should yield
  a construction time error when applied."""

    @beam.DoFn.yields_batches
    def process(self, element: int, *args, **kwargs) -> Iterator[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        yield [element]

    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[List[float]]:
        if False:
            return 10
        yield [element / 2 for element in batch]

class MismatchedElementProducingDoFn(beam.DoFn):
    """A DoFn that produces elements from both process and process_batch, with
  mismatched return types (one yields floats, the other ints). Should yield
  a construction time error when applied."""

    def process(self, element: int, *args, **kwargs) -> Iterator[float]:
        if False:
            print('Hello World!')
        yield (element / 2)

    @beam.DoFn.yields_elements
    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[int]:
        if False:
            print('Hello World!')
        yield batch[0]

class NoElementOutputAnnotation(beam.DoFn):

    def process_batch(self, batch: List[int], *args, **kwargs) -> Iterator[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        yield [element * 2 for element in batch]

class BatchDoFnTest(unittest.TestCase):

    def test_map_pardo(self):
        if False:
            while True:
                i = 10
        dofn = beam.Map(lambda x: x * 2).dofn
        self.assertTrue(dofn._process_defined)
        self.assertFalse(dofn._process_batch_defined)
        self.assertEqual(dofn._get_input_batch_type_normalized(int), None)
        self.assertEqual(dofn._get_output_batch_type_normalized(int), None)

    def test_no_input_annotation_raises(self):
        if False:
            return 10
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'NoInputAnnotation.process_batch'):
            _ = pc | beam.ParDo(NoInputAnnotation())

    def test_unsupported_dofn_param_raises(self):
        if False:
            return 10

        class BadParam(beam.DoFn):

            @no_type_check
            def process_batch(self, batch: List[int], key=beam.DoFn.KeyParam):
                if False:
                    return 10
                yield (batch * key)
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(NotImplementedError, 'BadParam.*KeyParam'):
            _ = pc | beam.ParDo(BadParam())

    def test_mismatched_batch_producer_raises(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(TypeError, "(?ms)MismatchedBatchProducingDoFn.*process: List\\[<class \\'int\\'>\\].*process_batch: List\\[<class \\'float\\'>\\]"):
            _ = pc | beam.ParDo(MismatchedBatchProducingDoFn())

    def test_mismatched_element_producer_raises(self):
        if False:
            return 10
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(TypeError, '(?ms)MismatchedElementProducingDoFn.*process:.*process_batch:'):
            _ = pc | beam.ParDo(MismatchedElementProducingDoFn())

    def test_cant_infer_batchconverter_input_raises(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline()
        pc = p | beam.Create(['a', 'b', 'c'])
        with self.assertRaisesRegex(TypeError, 'input.*BatchDoFn.*'):
            _ = pc | beam.ParDo(BatchDoFn())

    def test_cant_infer_batchconverter_output_raises(self):
        if False:
            return 10
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'output.*NoElementOutputAnnotation.*DoFn\\.infer_output_type'):
            _ = pc | beam.ParDo(NoElementOutputAnnotation())

    def test_element_to_batch_dofn_typehint(self):
        if False:
            i = 10
            return i + 15
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3]) | beam.ParDo(ElementToBatchDoFn())
        self.assertEqual(pc.element_type, int)

    def test_batch_to_element_dofn_typehint(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        pc = p | beam.Create([1, 2, 3]) | beam.ParDo(BatchToElementDoFn())
        self.assertEqual(pc.element_type, beam.typehints.Tuple[int, int])
if __name__ == '__main__':
    unittest.main()