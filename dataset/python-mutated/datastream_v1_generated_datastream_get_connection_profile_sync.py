from google.cloud import datastream_v1

def sample_get_connection_profile():
    if False:
        return 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.GetConnectionProfileRequest(name='name_value')
    response = client.get_connection_profile(request=request)
    print(response)