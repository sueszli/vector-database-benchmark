from google.cloud import compute_v1

def sample_insert():
    if False:
        return 10
    client = compute_v1.UrlMapsClient()
    request = compute_v1.InsertUrlMapRequest(project='project_value')
    response = client.insert(request=request)
    print(response)