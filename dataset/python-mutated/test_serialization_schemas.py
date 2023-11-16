from pyflink.common.serialization import SimpleStringSchema
from pyflink.testing.test_case_utils import PyFlinkTestCase

class SimpleStringSchemaTests(PyFlinkTestCase):

    def test_simple_string_schema(self):
        if False:
            return 10
        expected_string = 'test string'
        simple_string_schema = SimpleStringSchema()
        self.assertEqual(expected_string.encode(encoding='utf-8'), simple_string_schema._j_serialization_schema.serialize(expected_string))
        self.assertEqual(expected_string, simple_string_schema._j_deserialization_schema.deserialize(expected_string.encode(encoding='utf-8')))