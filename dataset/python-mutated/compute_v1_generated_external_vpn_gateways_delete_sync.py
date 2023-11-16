from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.ExternalVpnGatewaysClient()
    request = compute_v1.DeleteExternalVpnGatewayRequest(external_vpn_gateway='external_vpn_gateway_value', project='project_value')
    response = client.delete(request=request)
    print(response)