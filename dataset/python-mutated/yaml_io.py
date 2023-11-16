"""This module contains the Python implementations for the builtin IOs.

They are referenced from standard_io.py.

Note that in the case that they overlap with other (likely Java)
implementations of the same transforms, the configs must be kept in sync.
"""
import io
import os
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
import fastavro
import yaml
import apache_beam as beam
import apache_beam.io as beam_io
from apache_beam.io import ReadFromBigQuery
from apache_beam.io import WriteToBigQuery
from apache_beam.io import avroio
from apache_beam.io.gcp.bigquery import BigQueryDisposition
from apache_beam.portability.api import schema_pb2
from apache_beam.typehints import schemas
from apache_beam.yaml import json_utils
from apache_beam.yaml import yaml_mapping
from apache_beam.yaml import yaml_provider

def read_from_text(path: str):
    if False:
        return 10
    return beam_io.ReadFromText(path) | beam.Map(lambda s: beam.Row(line=s))

@beam.ptransform_fn
def write_to_text(pcoll, path: str):
    if False:
        i = 10
        return i + 15
    try:
        field_names = [name for (name, _) in schemas.named_fields_from_element_type(pcoll.element_type)]
    except Exception as exn:
        raise ValueError('WriteToText requires an input schema with exactly one field.') from exn
    if len(field_names) != 1:
        raise ValueError('WriteToText requires an input schema with exactly one field, got %s' % field_names)
    (sole_field_name,) = field_names
    return pcoll | beam.Map(lambda x: str(getattr(x, sole_field_name))) | beam.io.WriteToText(path)

def read_from_bigquery(query=None, table=None, row_restriction=None, fields=None):
    if False:
        i = 10
        return i + 15
    if query is None:
        assert table is not None
    else:
        assert table is None and row_restriction is None and (fields is None)
    return ReadFromBigQuery(query=query, table=table, row_restriction=row_restriction, selected_fields=fields, method='DIRECT_READ', output_type='BEAM_ROW')

def write_to_bigquery(table, *, create_disposition=BigQueryDisposition.CREATE_IF_NEEDED, write_disposition=BigQueryDisposition.WRITE_APPEND, error_handling=None):
    if False:
        for i in range(10):
            print('nop')

    class WriteToBigQueryHandlingErrors(beam.PTransform):

        def default_label(self):
            if False:
                while True:
                    i = 10
            return 'WriteToBigQuery'

        def expand(self, pcoll):
            if False:
                for i in range(10):
                    print('nop')
            write_result = pcoll | WriteToBigQuery(table, method=WriteToBigQuery.Method.STORAGE_WRITE_API if error_handling else None, create_disposition=create_disposition, write_disposition=write_disposition, temp_file_format='AVRO')
            if error_handling and 'output' in error_handling:
                return {'post_write': write_result.failed_rows_with_errors | beam.FlatMap(lambda x: None), error_handling['output']: write_result.failed_rows_with_errors}
            elif write_result._method == WriteToBigQuery.Method.FILE_LOADS:
                return {'post_write': write_result.destination_load_jobid_pairs | beam.FlatMap(lambda x: None)}
            else:

                def raise_exception(failed_row_with_error):
                    if False:
                        for i in range(10):
                            print('nop')
                    raise RuntimeError(failed_row_with_error.error_message)
                _ = write_result.failed_rows_with_errors | beam.Map(raise_exception)
                return {'post_write': write_result.failed_rows_with_errors | beam.FlatMap(lambda x: None)}
    return WriteToBigQueryHandlingErrors()

def _create_parser(format, schema: Any) -> Tuple[schema_pb2.Schema, Callable[[bytes], beam.Row]]:
    if False:
        print('Hello World!')
    if format == 'raw':
        if schema:
            raise ValueError('raw format does not take a schema')
        return (schema_pb2.Schema(fields=[schemas.schema_field('payload', bytes)]), lambda payload: beam.Row(payload=payload))
    elif format == 'json':
        beam_schema = json_utils.json_schema_to_beam_schema(schema)
        return (beam_schema, json_utils.json_parser(beam_schema, schema))
    elif format == 'avro':
        beam_schema = avroio.avro_schema_to_beam_schema(schema)
        covert_to_row = avroio.avro_dict_to_beam_row(schema, beam_schema)
        return (beam_schema, lambda record: covert_to_row(fastavro.schemaless_reader(io.BytesIO(record), schema)))
    else:
        raise ValueError(f'Unknown format: {format}')

def _create_formatter(format, schema: Any, beam_schema: schema_pb2.Schema) -> Callable[[beam.Row], bytes]:
    if False:
        print('Hello World!')
    if format == 'raw':
        if schema:
            raise ValueError('raw format does not take a schema')
        field_names = [field.name for field in beam_schema.fields]
        if len(field_names) != 1:
            raise ValueError(f'Expecting exactly one field, found {field_names}')
        return lambda row: getattr(row, field_names[0])
    elif format == 'json':
        return json_utils.json_formater(beam_schema)
    elif format == 'avro':
        avro_schema = schema or avroio.beam_schema_to_avro_schema(beam_schema)
        from_row = avroio.beam_row_to_avro_dict(avro_schema, beam_schema)

        def formatter(row):
            if False:
                for i in range(10):
                    print('nop')
            buffer = io.BytesIO()
            fastavro.schemaless_writer(buffer, avro_schema, from_row(row))
            buffer.seek(0)
            return buffer.read()
        return formatter
    else:
        raise ValueError(f'Unknown format: {format}')

@beam.ptransform_fn
@yaml_mapping.maybe_with_exception_handling_transform_fn
def read_from_pubsub(root, *, topic: Optional[str]=None, subscription: Optional[str]=None, format: str, schema: Optional[Any]=None, attributes: Optional[Iterable[str]]=None, attributes_map: Optional[str]=None, id_attribute: Optional[str]=None, timestamp_attribute: Optional[str]=None):
    if False:
        print('Hello World!')
    'Reads messages from Cloud Pub/Sub.\n\n  Args:\n    topic: Cloud Pub/Sub topic in the form\n      "projects/<project>/topics/<topic>". If provided, subscription must be\n      None.\n    subscription: Existing Cloud Pub/Sub subscription to use in the\n      form "projects/<project>/subscriptions/<subscription>". If not\n      specified, a temporary subscription will be created from the specified\n      topic. If provided, topic must be None.\n    format: The expected format of the message payload.  Currently suported\n      formats are\n\n        - raw: Produces records with a single `payload` field whose contents\n            are the raw bytes of the pubsub message.\n        - avro: Parses records with a given avro schema.\n        - json: Parses records with a given json schema.\n\n    schema: Schema specification for the given format.\n    attributes: List of attribute keys whose values will be flattened into the\n      output message as additional fields.  For example, if the format is `raw`\n      and attributes is `["a", "b"]` then this read will produce elements of\n      the form `Row(payload=..., a=..., b=...)`.\n    attribute_map: Name of a field in which to store the full set of attributes\n      associated with this message.  For example, if the format is `raw` and\n      `attribute_map` is set to `"attrs"` then this read will produce elements\n      of the form `Row(payload=..., attrs=...)` where `attrs` is a Map type\n      of string to string.\n      If both `attributes` and `attribute_map` are set, the overlapping\n      attribute values will be present in both the flattened structure and the\n      attribute map.\n    id_attribute: The attribute on incoming Pub/Sub messages to use as a unique\n      record identifier. When specified, the value of this attribute (which\n      can be any string that uniquely identifies the record) will be used for\n      deduplication of messages. If not provided, we cannot guarantee\n      that no duplicate data will be delivered on the Pub/Sub stream. In this\n      case, deduplication of the stream will be strictly best effort.\n    timestamp_attribute: Message value to use as element timestamp. If None,\n      uses message publishing time as the timestamp.\n\n      Timestamp values should be in one of two formats:\n\n      - A numerical value representing the number of milliseconds since the\n        Unix epoch.\n      - A string in RFC 3339 format, UTC timezone. Example:\n        ``2015-10-29T23:41:41.123Z``. The sub-second component of the\n        timestamp is optional, and digits beyond the first three (i.e., time\n        units smaller than milliseconds) may be ignored.\n  '
    if topic and subscription:
        raise TypeError('Only one of topic and subscription may be specified.')
    elif not topic and (not subscription):
        raise TypeError('One of topic or subscription may be specified.')
    (payload_schema, parser) = _create_parser(format, schema)
    extra_fields: List[schema_pb2.Field] = []
    if not attributes and (not attributes_map):
        mapper = lambda msg: parser(msg)
    else:
        if isinstance(attributes, str):
            attributes = [attributes]
        if attributes:
            extra_fields.extend([schemas.schema_field(attr, str) for attr in attributes])
        if attributes_map:
            extra_fields.append(schemas.schema_field(attributes_map, Mapping[str, str]))

        def mapper(msg):
            if False:
                for i in range(10):
                    print('nop')
            values = parser(msg.data).as_dict()
            if attributes:
                for attr in attributes:
                    values[attr] = msg.attributes[attr]
            if attributes_map:
                values[attributes_map] = msg.attributes
            return beam.Row(**values)
    output = root | beam.io.ReadFromPubSub(topic=topic, subscription=subscription, with_attributes=bool(attributes or attributes_map), id_label=id_attribute, timestamp_attribute=timestamp_attribute) | 'ParseMessage' >> beam.Map(mapper)
    output.element_type = schemas.named_tuple_from_schema(schema_pb2.Schema(fields=list(payload_schema.fields) + extra_fields))
    return output

@beam.ptransform_fn
@yaml_mapping.maybe_with_exception_handling_transform_fn
def write_to_pubsub(pcoll, *, topic: str, format: str, schema: Optional[Any]=None, attributes: Optional[Iterable[str]]=None, attributes_map: Optional[str]=None, id_attribute: Optional[str]=None, timestamp_attribute: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    'Writes messages from Cloud Pub/Sub.\n\n  Args:\n    topic: Cloud Pub/Sub topic in the form "/topics/<project>/<topic>".\n    format: How to format the message payload.  Currently suported\n      formats are\n\n        - raw: Expects a message with a single field (excluding\n            attribute-related fields) whose contents are used as the raw bytes\n            of the pubsub message.\n        - avro: Encodes records with a given avro schema, which may be inferred\n            from the input PCollection schema.\n        - json: Formats records with a given json schema, which may be inferred\n            from the input PCollection schema.\n\n    schema: Schema specification for the given format.\n    attributes: List of attribute keys whose values will be pulled out as\n      PubSub message attributes.  For example, if the format is `raw`\n      and attributes is `["a", "b"]` then elements of the form\n      `Row(any_field=..., a=..., b=...)` will result in PubSub messages whose\n      payload has the contents of any_field and whose attribute will be\n      populated with the values of `a` and `b`.\n    attribute_map: Name of a string-to-string map field in which to pull a set\n      of attributes associated with this message.  For example, if the format\n      is `raw` and `attribute_map` is set to `"attrs"` then elements of the form\n      `Row(any_field=..., attrs=...)` will result in PubSub messages whose\n      payload has the contents of any_field and whose attribute will be\n      populated with the values from attrs.\n      If both `attributes` and `attribute_map` are set, the union of attributes\n      from these two sources will be used to populate the PubSub message\n      attributes.\n    id_attribute: If set, will set an attribute for each Cloud Pub/Sub message\n      with the given name and a unique value. This attribute can then be used\n      in a ReadFromPubSub PTransform to deduplicate messages.\n    timestamp_attribute: If set, will set an attribute for each Cloud Pub/Sub\n      message with the given name and the message\'s publish time as the value.\n  '
    input_schema = schemas.schema_from_element_type(pcoll.element_type)
    extra_fields: List[str] = []
    if isinstance(attributes, str):
        attributes = [attributes]
    if attributes:
        extra_fields.extend(attributes)
    if attributes_map:
        extra_fields.append(attributes_map)

    def attributes_extractor(row):
        if False:
            for i in range(10):
                print('nop')
        if attributes_map:
            attribute_values = dict(getattr(row, attributes_map))
        else:
            attribute_values = {}
        if attributes:
            attribute_values.update({attr: getattr(row, attr) for attr in attributes})
        return attribute_values
    schema_names = set((f.name for f in input_schema.fields))
    missing_attribute_names = set(extra_fields) - schema_names
    if missing_attribute_names:
        raise ValueError(f'Attribute fields {missing_attribute_names} not found in schema fields {schema_names}')
    payload_schema = schema_pb2.Schema(fields=[field for field in input_schema.fields if field.name not in extra_fields])
    formatter = _create_formatter(format, schema, payload_schema)
    return pcoll | beam.Map(lambda row: beam.io.gcp.pubsub.PubsubMessage(formatter(row), attributes_extractor(row))) | beam.io.WriteToPubSub(topic, with_attributes=True, id_label=id_attribute, timestamp_attribute=timestamp_attribute)

def io_providers():
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(os.path.dirname(__file__), 'standard_io.yaml')) as fin:
        return yaml_provider.parse_providers(yaml.load(fin, Loader=yaml.SafeLoader))