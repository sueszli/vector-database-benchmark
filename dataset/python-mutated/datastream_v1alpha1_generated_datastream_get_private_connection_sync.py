from google.cloud import datastream_v1alpha1

def sample_get_private_connection():
    if False:
        print('Hello World!')
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.GetPrivateConnectionRequest(name='name_value')
    response = client.get_private_connection(request=request)
    print(response)