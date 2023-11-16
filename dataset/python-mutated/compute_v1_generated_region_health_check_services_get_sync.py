from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.RegionHealthCheckServicesClient()
    request = compute_v1.GetRegionHealthCheckServiceRequest(health_check_service='health_check_service_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)