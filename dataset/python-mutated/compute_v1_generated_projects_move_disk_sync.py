from google.cloud import compute_v1

def sample_move_disk():
    if False:
        print('Hello World!')
    client = compute_v1.ProjectsClient()
    request = compute_v1.MoveDiskProjectRequest(project='project_value')
    response = client.move_disk(request=request)
    print(response)