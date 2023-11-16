from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.FirewallsClient()
    request = compute_v1.PatchFirewallRequest(firewall='firewall_value', project='project_value')
    response = client.patch(request=request)
    print(response)