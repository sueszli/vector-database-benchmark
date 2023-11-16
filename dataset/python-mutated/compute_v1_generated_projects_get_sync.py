from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ProjectsClient()
    request = compute_v1.GetProjectRequest(project='project_value')
    response = client.get(request=request)
    print(response)