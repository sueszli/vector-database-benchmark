from google.cloud import resourcemanager_v3

def sample_list_projects():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.ListProjectsRequest(parent='parent_value')
    page_result = client.list_projects(request=request)
    for response in page_result:
        print(response)