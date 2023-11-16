from google.cloud import resourcemanager_v3

def sample_undelete_project():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.UndeleteProjectRequest(name='name_value')
    operation = client.undelete_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)