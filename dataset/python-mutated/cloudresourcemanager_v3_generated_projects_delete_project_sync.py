from google.cloud import resourcemanager_v3

def sample_delete_project():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.DeleteProjectRequest(name='name_value')
    operation = client.delete_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)