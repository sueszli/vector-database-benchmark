from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallsClient()
    request = compute_v1.InsertFirewallRequest(project='project_value')
    response = client.insert(request=request)
    print(response)