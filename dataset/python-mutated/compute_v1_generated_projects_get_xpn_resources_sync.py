from google.cloud import compute_v1

def sample_get_xpn_resources():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ProjectsClient()
    request = compute_v1.GetXpnResourcesProjectsRequest(project='project_value')
    page_result = client.get_xpn_resources(request=request)
    for response in page_result:
        print(response)