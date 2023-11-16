from google.cloud import compute_v1

def sample_insert():
    if False:
        return 10
    client = compute_v1.RoutersClient()
    request = compute_v1.InsertRouterRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)