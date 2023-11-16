from google.cloud import datastream_v1

def sample_get_private_connection():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.GetPrivateConnectionRequest(name='name_value')
    response = client.get_private_connection(request=request)
    print(response)