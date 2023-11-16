from google.cloud import edgenetwork_v1

def sample_delete_network():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DeleteNetworkRequest(name='name_value')
    operation = client.delete_network(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)