from google.cloud import datastream_v1

def sample_update_stream():
    if False:
        i = 10
        return i + 15
    client = datastream_v1.DatastreamClient()
    stream = datastream_v1.Stream()
    stream.display_name = 'display_name_value'
    stream.source_config.source_connection_profile = 'source_connection_profile_value'
    stream.destination_config.destination_connection_profile = 'destination_connection_profile_value'
    request = datastream_v1.UpdateStreamRequest(stream=stream)
    operation = client.update_stream(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)