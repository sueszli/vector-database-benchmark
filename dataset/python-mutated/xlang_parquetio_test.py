"""Unit tests for cross-language parquet io read/write."""
import logging
import os
import re
import unittest
import apache_beam as beam
from apache_beam import coders
from apache_beam.coders.avro_record import AvroRecord
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
PARQUET_WRITE_URN = 'beam:transforms:xlang:test:parquet_write'

@unittest.skipUnless(os.environ.get('EXPANSION_JAR'), 'EXPANSION_JAR environment variable is not set.')
@unittest.skipUnless(os.environ.get('EXPANSION_PORT'), 'EXPANSION_PORT environment var is not provided.')
class XlangParquetIOTest(unittest.TestCase):

    def test_xlang_parquetio_write(self):
        if False:
            for i in range(10):
                print('nop')
        expansion_jar = os.environ.get('EXPANSION_JAR')
        port = os.environ.get('EXPANSION_PORT')
        address = 'localhost:%s' % port
        try:
            with TestPipeline() as p:
                p.get_pipeline_options().view_as(DebugOptions).experiments.append('jar_packages=' + expansion_jar)
                p.not_use_test_runner_api = True
                _ = p | beam.Create([AvroRecord({'name': 'abc'}), AvroRecord({'name': 'def'}), AvroRecord({'name': 'ghi'})]) | beam.ExternalTransform(PARQUET_WRITE_URN, ImplicitSchemaPayloadBuilder({'data': '/tmp/test.parquet'}), address)
        except RuntimeError as e:
            if re.search(PARQUET_WRITE_URN, str(e)):
                print('looks like URN not implemented in expansion service, skipping.')
            else:
                raise e

class AvroTestCoder(coders.AvroGenericCoder):
    SCHEMA = '\n  {\n    "type": "record", "name": "testrecord",\n    "fields": [ {"name": "name", "type": "string"} ]\n  }\n  '

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(self.SCHEMA)
coders.registry.register_coder(AvroRecord, AvroTestCoder)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()