from google.cloud import resourcemanager_v3

def sample_move_project():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.ProjectsClient()
    request = resourcemanager_v3.MoveProjectRequest(name='name_value', destination_parent='destination_parent_value')
    operation = client.move_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)