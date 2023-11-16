from google.cloud import compute_v1

def sample_move_instance():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.ProjectsClient()
    request = compute_v1.MoveInstanceProjectRequest(project='project_value')
    response = client.move_instance(request=request)
    print(response)