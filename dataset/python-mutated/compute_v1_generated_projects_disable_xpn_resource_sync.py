from google.cloud import compute_v1

def sample_disable_xpn_resource():
    if False:
        while True:
            i = 10
    client = compute_v1.ProjectsClient()
    request = compute_v1.DisableXpnResourceProjectRequest(project='project_value')
    response = client.disable_xpn_resource(request=request)
    print(response)