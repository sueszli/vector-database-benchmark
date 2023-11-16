from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RoutesClient()
    request = compute_v1.GetRouteRequest(project='project_value', route='route_value')
    response = client.get(request=request)
    print(response)