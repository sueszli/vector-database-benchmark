from google.cloud import compute_v1

def sample_disable_xpn_host():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ProjectsClient()
    request = compute_v1.DisableXpnHostProjectRequest(project='project_value')
    response = client.disable_xpn_host(request=request)
    print(response)