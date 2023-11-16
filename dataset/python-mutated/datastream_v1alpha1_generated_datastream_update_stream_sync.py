from google.cloud import datastream_v1alpha1

def sample_update_stream():
    if False:
        return 10
    client = datastream_v1alpha1.DatastreamClient()
    stream = datastream_v1alpha1.Stream()
    stream.display_name = 'display_name_value'
    stream.source_config.source_connection_profile_name = 'source_connection_profile_name_value'
    stream.destination_config.destination_connection_profile_name = 'destination_connection_profile_name_value'
    request = datastream_v1alpha1.UpdateStreamRequest(stream=stream)
    operation = client.update_stream(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)