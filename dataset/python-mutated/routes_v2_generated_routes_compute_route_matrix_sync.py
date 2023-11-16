from google.maps import routing_v2

def sample_compute_route_matrix():
    if False:
        for i in range(10):
            print('nop')
    client = routing_v2.RoutesClient()
    request = routing_v2.ComputeRouteMatrixRequest()
    stream = client.compute_route_matrix(request=request)
    for response in stream:
        print(response)