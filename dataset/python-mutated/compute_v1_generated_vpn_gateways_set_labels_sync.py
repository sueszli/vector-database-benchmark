from google.cloud import compute_v1

def sample_set_labels():
    if False:
        return 10
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.SetLabelsVpnGatewayRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)