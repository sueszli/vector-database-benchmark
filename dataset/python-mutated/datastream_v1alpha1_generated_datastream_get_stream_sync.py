from google.cloud import datastream_v1alpha1

def sample_get_stream():
    if False:
        while True:
            i = 10
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.GetStreamRequest(name='name_value')
    response = client.get_stream(request=request)
    print(response)