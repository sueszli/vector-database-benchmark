from google.cloud import compute_v1

def sample_switch_to_custom_mode():
    if False:
        return 10
    client = compute_v1.NetworksClient()
    request = compute_v1.SwitchToCustomModeNetworkRequest(network='network_value', project='project_value')
    response = client.switch_to_custom_mode(request=request)
    print(response)