from google.cloud import compute_v1

def sample_set_default_network_tier():
    if False:
        while True:
            i = 10
    client = compute_v1.ProjectsClient()
    request = compute_v1.SetDefaultNetworkTierProjectRequest(project='project_value')
    response = client.set_default_network_tier(request=request)
    print(response)