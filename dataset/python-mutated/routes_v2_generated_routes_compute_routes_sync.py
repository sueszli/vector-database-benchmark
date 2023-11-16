from google.maps import routing_v2

def sample_compute_routes():
    if False:
        for i in range(10):
            print('nop')
    client = routing_v2.RoutesClient()
    request = routing_v2.ComputeRoutesRequest()
    response = client.compute_routes(request=request)
    print(response)