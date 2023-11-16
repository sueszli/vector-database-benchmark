from google.cloud import edgecontainer_v1

def sample_create_vpn_connection():
    if False:
        i = 10
        return i + 15
    client = edgecontainer_v1.EdgeContainerClient()
    vpn_connection = edgecontainer_v1.VpnConnection()
    vpn_connection.name = 'name_value'
    request = edgecontainer_v1.CreateVpnConnectionRequest(parent='parent_value', vpn_connection_id='vpn_connection_id_value', vpn_connection=vpn_connection)
    operation = client.create_vpn_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)