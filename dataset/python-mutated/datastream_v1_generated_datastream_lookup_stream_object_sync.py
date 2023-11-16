from google.cloud import datastream_v1

def sample_lookup_stream_object():
    if False:
        return 10
    client = datastream_v1.DatastreamClient()
    source_object_identifier = datastream_v1.SourceObjectIdentifier()
    source_object_identifier.oracle_identifier.schema = 'schema_value'
    source_object_identifier.oracle_identifier.table = 'table_value'
    request = datastream_v1.LookupStreamObjectRequest(parent='parent_value', source_object_identifier=source_object_identifier)
    response = client.lookup_stream_object(request=request)
    print(response)