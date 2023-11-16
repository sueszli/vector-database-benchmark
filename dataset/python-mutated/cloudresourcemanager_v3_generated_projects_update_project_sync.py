from google.cloud import resourcemanager_v3

def sample_update_project():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.UpdateProjectRequest()
    operation = client.update_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)