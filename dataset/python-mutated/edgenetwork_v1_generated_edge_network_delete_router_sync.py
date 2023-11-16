from google.cloud import edgenetwork_v1

def sample_delete_router():
    if False:
        i = 10
        return i + 15
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DeleteRouterRequest(name='name_value')
    operation = client.delete_router(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)