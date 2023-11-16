from google.cloud import compute_v1

def sample_get_xpn_host():
    if False:
        print('Hello World!')
    client = compute_v1.ProjectsClient()
    request = compute_v1.GetXpnHostProjectRequest(project='project_value')
    response = client.get_xpn_host(request=request)
    print(response)