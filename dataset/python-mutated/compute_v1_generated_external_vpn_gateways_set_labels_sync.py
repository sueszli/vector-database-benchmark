from google.cloud import compute_v1

def sample_set_labels():
    if False:
        print('Hello World!')
    client = compute_v1.ExternalVpnGatewaysClient()
    request = compute_v1.SetLabelsExternalVpnGatewayRequest(project='project_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)