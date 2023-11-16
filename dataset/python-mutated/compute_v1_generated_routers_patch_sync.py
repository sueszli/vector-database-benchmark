from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RoutersClient()
    request = compute_v1.PatchRouterRequest(project='project_value', region='region_value', router='router_value')
    response = client.patch(request=request)
    print(response)