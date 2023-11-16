from google.cloud import compute_v1

def sample_get_router_status():
    if False:
        while True:
            i = 10
    client = compute_v1.RoutersClient()
    request = compute_v1.GetRouterStatusRouterRequest(project='project_value', region='region_value', router='router_value')
    response = client.get_router_status(request=request)
    print(response)