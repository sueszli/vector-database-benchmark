from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionHealthChecksClient()
    request = compute_v1.GetRegionHealthCheckRequest(health_check='health_check_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)