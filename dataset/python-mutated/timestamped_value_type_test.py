import unittest
from typing import Any
from typing import Dict
from typing import List
from typing import TypeVar
import apache_beam as beam
from apache_beam.transforms.window import TimestampedValue
from apache_beam.typehints.decorators import TypeCheckError
T = TypeVar('T')

def ConvertToTimestampedValue(plant: Dict[str, Any]) -> TimestampedValue[str]:
    if False:
        print('Hello World!')
    return TimestampedValue[str](plant['name'], plant['season'])

def ConvertToTimestampedValue_1(plant: Dict[str, Any]) -> TimestampedValue:
    if False:
        print('Hello World!')
    return TimestampedValue(plant['name'], plant['season'])

def ConvertToTimestampedValue_2(plant: Dict[str, Any]) -> TimestampedValue[List[str]]:
    if False:
        while True:
            i = 10
    return TimestampedValue[List[str]](plant['name'], plant['season'])

def ConvertToTimestampedValue_3(plant: Dict[str, Any]) -> TimestampedValue[T]:
    if False:
        for i in range(10):
            print('nop')
    return TimestampedValue[T](plant['name'], plant['season'])

class TypeCheckTimestampedValueTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.opts = beam.options.pipeline_options.PipelineOptions(runtime_type_check=True)
        self.data = [{'name': 'Strawberry', 'season': 1585699200}]
        self.data_1 = [{'name': 1234, 'season': 1585699200}]
        self.data_2 = [{'name': ['abc', 'cde'], 'season': 1585699200}]
        self.data_3 = [{'name': [123, 'cde'], 'season': 1585699200}]

    def test_pcoll_default_hints(self):
        if False:
            while True:
                i = 10
        for fn in (ConvertToTimestampedValue, ConvertToTimestampedValue_1):
            pc = beam.Map(fn)
            ht = pc.default_type_hints()
            assert len(ht) == 3
            assert ht.output_types[0][0]

    def test_pcoll_with_output_hints(self):
        if False:
            return 10
        pc = beam.Map(ConvertToTimestampedValue).with_output_types(str)
        ht = pc.get_type_hints()
        assert len(ht) == 3
        assert ht.output_types[0][0] == str

    def test_opts_with_check(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(options=self.opts) as p:
            _ = p | 'Garden plants' >> beam.Create(self.data) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue) | beam.Map(print)

    def test_opts_with_check_list_str(self):
        if False:
            for i in range(10):
                print('nop')
        with beam.Pipeline(options=self.opts) as p:
            _ = p | 'Garden plants' >> beam.Create(self.data_2) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue_2) | beam.Map(print)

    def test_opts_with_check_wrong_data(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeCheckError):
            with beam.Pipeline(options=self.opts) as p:
                _ = p | 'Garden plants' >> beam.Create(self.data_1) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue) | beam.Map(print)

    def test_opts_with_check_wrong_data_list_str(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeCheckError):
            with beam.Pipeline(options=self.opts) as p:
                _ = p | 'Garden plants' >> beam.Create(self.data_1) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue_2) | beam.Map(print)
        with self.assertRaises(TypeCheckError):
            with beam.Pipeline(options=self.opts) as p:
                _ = p | 'Garden plants' >> beam.Create(self.data_3) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue_2) | beam.Map(print)

    def test_opts_with_check_typevar(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(RuntimeError):
            with beam.Pipeline(options=self.opts) as p:
                _ = p | 'Garden plants' >> beam.Create(self.data_2) | 'With timestamps' >> beam.Map(ConvertToTimestampedValue_3) | beam.Map(print)
if __name__ == '__main__':
    unittest.main()