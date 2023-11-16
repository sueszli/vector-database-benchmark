from google.cloud import compute_v1

def sample_update():
    if False:
        while True:
            i = 10
    client = compute_v1.RoutersClient()
    request = compute_v1.UpdateRouterRequest(project='project_value', region='region_value', router='router_value')
    response = client.update(request=request)
    print(response)