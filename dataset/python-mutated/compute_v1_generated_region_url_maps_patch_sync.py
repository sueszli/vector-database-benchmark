from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.RegionUrlMapsClient()
    request = compute_v1.PatchRegionUrlMapRequest(project='project_value', region='region_value', url_map='url_map_value')
    response = client.patch(request=request)
    print(response)