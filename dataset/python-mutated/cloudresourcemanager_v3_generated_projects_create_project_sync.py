from google.cloud import resourcemanager_v3

def sample_create_project():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.CreateProjectRequest()
    operation = client.create_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)