from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ExternalVpnGatewaysClient()
    request = compute_v1.GetExternalVpnGatewayRequest(external_vpn_gateway='external_vpn_gateway_value', project='project_value')
    response = client.get(request=request)
    print(response)