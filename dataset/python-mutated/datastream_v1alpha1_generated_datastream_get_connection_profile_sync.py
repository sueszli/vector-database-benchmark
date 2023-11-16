from google.cloud import datastream_v1alpha1

def sample_get_connection_profile():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.GetConnectionProfileRequest(name='name_value')
    response = client.get_connection_profile(request=request)
    print(response)