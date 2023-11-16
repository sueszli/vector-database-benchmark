from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.ZonesClient()
    request = compute_v1.GetZoneRequest(project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)