from google.cloud import compute_v1

def sample_enable_xpn_host():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ProjectsClient()
    request = compute_v1.EnableXpnHostProjectRequest(project='project_value')
    response = client.enable_xpn_host(request=request)
    print(response)