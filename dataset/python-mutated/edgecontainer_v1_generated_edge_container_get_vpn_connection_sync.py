from google.cloud import edgecontainer_v1

def sample_get_vpn_connection():
    if False:
        print('Hello World!')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.GetVpnConnectionRequest(name='name_value')
    response = client.get_vpn_connection(request=request)
    print(response)