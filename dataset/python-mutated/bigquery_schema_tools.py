"""Tools used tool work with Schema types in the context of BigQuery.
Classes, constants and functions in this file are experimental and have no
backwards compatibility guarantees.
NOTHING IN THIS FILE HAS BACKWARDS COMPATIBILITY GUARANTEES.
"""
import datetime
from typing import Optional
from typing import Sequence
import numpy as np
import apache_beam as beam
import apache_beam.io.gcp.bigquery_tools
import apache_beam.typehints.schemas
import apache_beam.utils.proto_utils
import apache_beam.utils.timestamp
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.portability.api import schema_pb2
from apache_beam.transforms import DoFn
BIG_QUERY_TO_PYTHON_TYPES = {'STRING': str, 'INTEGER': np.int64, 'FLOAT64': np.float64, 'FLOAT': np.float64, 'BOOLEAN': bool, 'BYTES': bytes, 'TIMESTAMP': apache_beam.utils.timestamp.Timestamp}

def generate_user_type_from_bq_schema(the_table_schema, selected_fields=None):
    if False:
        while True:
            i = 10
    'Convert a schema of type TableSchema into a pcollection element.\n      Args:\n        the_table_schema: A BQ schema of type TableSchema\n        selected_fields: if not None, the subset of fields to consider\n      Returns:\n        type: type that can be used to work with pCollections.\n  '
    the_schema = beam.io.gcp.bigquery_tools.get_dict_table_schema(the_table_schema)
    if the_schema == {}:
        raise ValueError('Encountered an empty schema')
    field_names_and_types = []
    for field in the_schema['fields']:
        if selected_fields is not None and field['name'] not in selected_fields:
            continue
        if field['type'] in BIG_QUERY_TO_PYTHON_TYPES:
            typ = bq_field_to_type(field['type'], field['mode'])
        else:
            raise ValueError(f"Encountered an unsupported type: {field['type']!r}")
        field_names_and_types.append((field['name'], typ))
    sample_schema = beam.typehints.schemas.named_fields_to_schema(field_names_and_types)
    usertype = beam.typehints.schemas.named_tuple_from_schema(sample_schema)
    return usertype

def bq_field_to_type(field, mode):
    if False:
        return 10
    if mode == 'NULLABLE' or mode is None or mode == '':
        return Optional[BIG_QUERY_TO_PYTHON_TYPES[field]]
    elif mode == 'REPEATED':
        return Sequence[BIG_QUERY_TO_PYTHON_TYPES[field]]
    elif mode == 'REQUIRED':
        return BIG_QUERY_TO_PYTHON_TYPES[field]
    else:
        raise ValueError(f'Encountered an unsupported mode: {mode!r}')

def convert_to_usertype(table_schema, selected_fields=None):
    if False:
        i = 10
        return i + 15
    usertype = generate_user_type_from_bq_schema(table_schema, selected_fields)
    return beam.ParDo(BeamSchemaConversionDoFn(usertype))

class BeamSchemaConversionDoFn(DoFn):

    def __init__(self, pcoll_val_ctor):
        if False:
            while True:
                i = 10
        self._pcoll_val_ctor = pcoll_val_ctor

    def process(self, dict_of_tuples):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in dict_of_tuples.items():
            if isinstance(v, datetime.datetime):
                dict_of_tuples[k] = beam.utils.timestamp.Timestamp.from_utc_datetime(v)
        yield self._pcoll_val_ctor(**dict_of_tuples)

    def infer_output_type(self, input_type):
        if False:
            print('Hello World!')
        return self._pcoll_val_ctor

    @classmethod
    def _from_serialized_schema(cls, schema_str):
        if False:
            return 10
        return cls(apache_beam.typehints.schemas.named_tuple_from_schema(apache_beam.utils.proto_utils.parse_Bytes(schema_str, schema_pb2.Schema)))

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (self._from_serialized_schema, (beam.typehints.schemas.named_tuple_to_schema(self._pcoll_val_ctor).SerializeToString(),))