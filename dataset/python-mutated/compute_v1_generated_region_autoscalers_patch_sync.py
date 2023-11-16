from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.RegionAutoscalersClient()
    request = compute_v1.PatchRegionAutoscalerRequest(project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)