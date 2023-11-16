from google.cloud import compute_v1

def sample_update():
    if False:
        print('Hello World!')
    client = compute_v1.RegionHealthChecksClient()
    request = compute_v1.UpdateRegionHealthCheckRequest(health_check='health_check_value', project='project_value', region='region_value')
    response = client.update(request=request)
    print(response)