from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetVpnGatewaysClient()
    request = compute_v1.GetTargetVpnGatewayRequest(project='project_value', region='region_value', target_vpn_gateway='target_vpn_gateway_value')
    response = client.get(request=request)
    print(response)