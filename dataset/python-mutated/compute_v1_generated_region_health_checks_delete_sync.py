from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.RegionHealthChecksClient()
    request = compute_v1.DeleteRegionHealthCheckRequest(health_check='health_check_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)