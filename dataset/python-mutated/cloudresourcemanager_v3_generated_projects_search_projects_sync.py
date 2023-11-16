from google.cloud import resourcemanager_v3

def sample_search_projects():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.SearchProjectsRequest()
    page_result = client.search_projects(request=request)
    for response in page_result:
        print(response)