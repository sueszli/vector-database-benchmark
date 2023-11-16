from google.cloud import datastream_v1

def sample_get_stream_object():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.GetStreamObjectRequest(name='name_value')
    response = client.get_stream_object(request=request)
    print(response)