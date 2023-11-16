from google.cloud import optimization_v1

def sample_optimize_tours():
    if False:
        while True:
            i = 10
    client = optimization_v1.FleetRoutingClient()
    request = optimization_v1.OptimizeToursRequest(parent='parent_value')
    response = client.optimize_tours(request=request)
    print(response)