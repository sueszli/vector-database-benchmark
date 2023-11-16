from google.cloud import datastream_v1

def sample_delete_stream():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.DeleteStreamRequest(name='name_value')
    operation = client.delete_stream(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)