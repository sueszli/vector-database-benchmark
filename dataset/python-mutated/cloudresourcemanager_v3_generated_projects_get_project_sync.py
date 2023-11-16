from google.cloud import resourcemanager_v3

def sample_get_project():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.GetProjectRequest(name='name_value')
    response = client.get_project(request=request)
    print(response)