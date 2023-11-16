"""Tools used tool work with Avro files in the context of BigQuery.

Classes, constants and functions in this file are experimental and have no
backwards compatibility guarantees.

NOTHING IN THIS FILE HAS BACKWARDS COMPATIBILITY GUARANTEES.
"""
BIG_QUERY_TO_AVRO_TYPES = {'STRUCT': 'record', 'RECORD': 'record', 'STRING': 'string', 'BOOL': 'boolean', 'BOOLEAN': 'boolean', 'BYTES': 'bytes', 'FLOAT64': 'double', 'FLOAT': 'double', 'INT64': 'long', 'INTEGER': 'long', 'TIME': {'type': 'long', 'logicalType': 'time-micros'}, 'TIMESTAMP': {'type': 'long', 'logicalType': 'timestamp-micros'}, 'DATE': {'type': 'int', 'logicalType': 'date'}, 'DATETIME': 'string', 'NUMERIC': {'type': 'bytes', 'logicalType': 'decimal', 'precision': 38, 'scale': 9}, 'GEOGRAPHY': 'string'}

def get_record_schema_from_dict_table_schema(schema_name, table_schema, namespace='apache_beam.io.gcp.bigquery'):
    if False:
        print('Hello World!')
    'Convert a table schema into an Avro schema.\n\n  Args:\n    schema_name (Text): The name of the record.\n    table_schema (Dict[Text, Any]): A BigQuery table schema in dict form.\n    namespace (Text): The namespace of the Avro schema.\n\n  Returns:\n    Dict[Text, Any]: The schema as an Avro RecordSchema.\n  '
    avro_fields = [table_field_to_avro_field(field, '.'.join((namespace, schema_name))) for field in table_schema['fields']]
    return {'type': 'record', 'name': schema_name, 'fields': avro_fields, 'doc': 'Translated Avro Schema for {}'.format(schema_name), 'namespace': namespace}

def table_field_to_avro_field(table_field, namespace):
    if False:
        print('Hello World!')
    'Convert a BigQuery field to an avro field.\n\n  Args:\n    table_field (Dict[Text, Any]): A BigQuery field in dict form.\n\n  Returns:\n    Dict[Text, Any]: An equivalent Avro field in dict form.\n  '
    assert 'type' in table_field, 'Unable to get type for table field {}'.format(table_field)
    assert table_field['type'] in BIG_QUERY_TO_AVRO_TYPES, 'Unable to map BigQuery field type {} to avro type'.format(table_field['type'])
    avro_type = BIG_QUERY_TO_AVRO_TYPES[table_field['type']]
    if avro_type == 'record':
        element_type = get_record_schema_from_dict_table_schema(table_field['name'], table_field, namespace='.'.join((namespace, table_field['name'])))
    else:
        element_type = avro_type
    field_mode = table_field.get('mode', 'NULLABLE')
    if field_mode in (None, 'NULLABLE'):
        field_type = ['null', element_type]
    elif field_mode == 'REQUIRED':
        field_type = element_type
    elif field_mode == 'REPEATED':
        field_type = {'type': 'array', 'items': element_type}
    else:
        raise ValueError('Unkown BigQuery field mode: {}'.format(field_mode))
    avro_field = {'type': field_type, 'name': table_field['name']}
    doc = table_field.get('description')
    if doc:
        avro_field['doc'] = doc
    return avro_field