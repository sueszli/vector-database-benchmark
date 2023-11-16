from google.cloud import edgecontainer_v1

def sample_delete_vpn_connection():
    if False:
        for i in range(10):
            print('nop')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.DeleteVpnConnectionRequest(name='name_value')
    operation = client.delete_vpn_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)