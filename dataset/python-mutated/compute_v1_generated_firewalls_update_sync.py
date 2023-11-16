from google.cloud import compute_v1

def sample_update():
    if False:
        i = 10
        return i + 15
    client = compute_v1.FirewallsClient()
    request = compute_v1.UpdateFirewallRequest(firewall='firewall_value', project='project_value')
    response = client.update(request=request)
    print(response)