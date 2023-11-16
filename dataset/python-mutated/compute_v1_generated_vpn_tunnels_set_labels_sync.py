from google.cloud import compute_v1

def sample_set_labels():
    if False:
        print('Hello World!')
    client = compute_v1.VpnTunnelsClient()
    request = compute_v1.SetLabelsVpnTunnelRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)