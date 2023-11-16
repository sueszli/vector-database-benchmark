from google.cloud import compute_v1

def sample_validate():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionUrlMapsClient()
    request = compute_v1.ValidateRegionUrlMapRequest(project='project_value', region='region_value', url_map='url_map_value')
    response = client.validate(request=request)
    print(response)