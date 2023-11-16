from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.RegionHealthChecksClient()
    request = compute_v1.PatchRegionHealthCheckRequest(health_check='health_check_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)