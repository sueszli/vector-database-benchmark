from google.cloud import ids_v1

def sample_get_endpoint():
    if False:
        print('Hello World!')
    client = ids_v1.IDSClient()
    request = ids_v1.GetEndpointRequest(name='name_value')
    response = client.get_endpoint(request=request)
    print(response)