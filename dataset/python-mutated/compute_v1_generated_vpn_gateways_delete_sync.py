from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.DeleteVpnGatewayRequest(project='project_value', region='region_value', vpn_gateway='vpn_gateway_value')
    response = client.delete(request=request)
    print(response)