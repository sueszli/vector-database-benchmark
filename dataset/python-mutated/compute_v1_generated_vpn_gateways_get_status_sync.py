from google.cloud import compute_v1

def sample_get_status():
    if False:
        i = 10
        return i + 15
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.GetStatusVpnGatewayRequest(project='project_value', region='region_value', vpn_gateway='vpn_gateway_value')
    response = client.get_status(request=request)
    print(response)