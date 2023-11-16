from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.RoutersClient()
    request = compute_v1.DeleteRouterRequest(project='project_value', region='region_value', router='router_value')
    response = client.delete(request=request)
    print(response)