from google.cloud import compute_v1

def sample_update():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionAutoscalersClient()
    request = compute_v1.UpdateRegionAutoscalerRequest(project='project_value', region='region_value')
    response = client.update(request=request)
    print(response)