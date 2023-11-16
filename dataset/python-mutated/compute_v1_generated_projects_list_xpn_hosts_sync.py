from google.cloud import compute_v1

def sample_list_xpn_hosts():
    if False:
        while True:
            i = 10
    client = compute_v1.ProjectsClient()
    request = compute_v1.ListXpnHostsProjectsRequest(project='project_value')
    page_result = client.list_xpn_hosts(request=request)
    for response in page_result:
        print(response)