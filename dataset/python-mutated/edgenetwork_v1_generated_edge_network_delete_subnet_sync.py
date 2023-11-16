from google.cloud import edgenetwork_v1

def sample_delete_subnet():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DeleteSubnetRequest(name='name_value')
    operation = client.delete_subnet(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)