from google.cloud import compute_v1

def sample_update():
    if False:
        return 10
    client = compute_v1.RegionUrlMapsClient()
    request = compute_v1.UpdateRegionUrlMapRequest(project='project_value', region='region_value', url_map='url_map_value')
    response = client.update(request=request)
    print(response)