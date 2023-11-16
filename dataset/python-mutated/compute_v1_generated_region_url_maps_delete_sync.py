from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionUrlMapsClient()
    request = compute_v1.DeleteRegionUrlMapRequest(project='project_value', region='region_value', url_map='url_map_value')
    response = client.delete(request=request)
    print(response)