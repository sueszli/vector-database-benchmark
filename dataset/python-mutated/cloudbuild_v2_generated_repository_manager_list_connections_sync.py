from google.cloud.devtools import cloudbuild_v2

def sample_list_connections():
    if False:
        return 10
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.ListConnectionsRequest(parent='parent_value')
    page_result = client.list_connections(request=request)
    for response in page_result:
        print(response)