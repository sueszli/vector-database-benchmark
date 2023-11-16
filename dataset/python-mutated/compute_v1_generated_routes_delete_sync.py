from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RoutesClient()
    request = compute_v1.DeleteRouteRequest(project='project_value', route='route_value')
    response = client.delete(request=request)
    print(response)