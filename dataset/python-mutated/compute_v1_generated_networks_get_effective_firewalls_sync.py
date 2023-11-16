from google.cloud import compute_v1

def sample_get_effective_firewalls():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworksClient()
    request = compute_v1.GetEffectiveFirewallsNetworkRequest(network='network_value', project='project_value')
    response = client.get_effective_firewalls(request=request)
    print(response)