from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.TargetVpnGatewaysClient()
    request = compute_v1.DeleteTargetVpnGatewayRequest(project='project_value', region='region_value', target_vpn_gateway='target_vpn_gateway_value')
    response = client.delete(request=request)
    print(response)