from google.cloud import ids_v1

def sample_delete_endpoint():
    if False:
        for i in range(10):
            print('nop')
    client = ids_v1.IDSClient()
    request = ids_v1.DeleteEndpointRequest(name='name_value')
    operation = client.delete_endpoint(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)