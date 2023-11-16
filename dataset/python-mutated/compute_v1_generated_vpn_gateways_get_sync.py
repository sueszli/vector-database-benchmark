from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.GetVpnGatewayRequest(project='project_value', region='region_value', vpn_gateway='vpn_gateway_value')
    response = client.get(request=request)
    print(response)