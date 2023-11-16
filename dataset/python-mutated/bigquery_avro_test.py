"""
These are the unit tests for the BigQueryLoadAvro class.
"""
import unittest
import avro
import avro.schema
from luigi.contrib.bigquery_avro import BigQueryLoadAvro

class BigQueryAvroTest(unittest.TestCase):

    def test_writer_schema_method_existence(self):
        if False:
            i = 10
            return i + 15
        schema_json = '\n        {\n            "namespace": "example.avro",\n            "type": "record",\n            "name": "User",\n            "fields": [\n                {"name": "name", "type": "string"},\n                {"name": "favorite_number",  "type": ["int", "null"]},\n                {"name": "favorite_color", "type": ["string", "null"]}\n            ]\n        }\n        '
        avro_schema = avro.schema.Parse(schema_json)
        reader = avro.io.DatumReader(avro_schema, avro_schema)
        actual_schema = BigQueryLoadAvro._get_writer_schema(reader)
        self.assertEqual(actual_schema, avro_schema, 'writer(s) avro_schema attribute not found')