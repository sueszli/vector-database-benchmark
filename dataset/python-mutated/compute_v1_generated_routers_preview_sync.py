from google.cloud import compute_v1

def sample_preview():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RoutersClient()
    request = compute_v1.PreviewRouterRequest(project='project_value', region='region_value', router='router_value')
    response = client.preview(request=request)
    print(response)