from google.cloud import compute_v1

def sample_enable_xpn_resource():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ProjectsClient()
    request = compute_v1.EnableXpnResourceProjectRequest(project='project_value')
    response = client.enable_xpn_resource(request=request)
    print(response)