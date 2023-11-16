from google.cloud import datastream_v1

def sample_create_stream():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    stream = datastream_v1.Stream()
    stream.display_name = 'display_name_value'
    stream.source_config.source_connection_profile = 'source_connection_profile_value'
    stream.destination_config.destination_connection_profile = 'destination_connection_profile_value'
    request = datastream_v1.CreateStreamRequest(parent='parent_value', stream_id='stream_id_value', stream=stream)
    operation = client.create_stream(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)