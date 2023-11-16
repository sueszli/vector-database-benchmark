from google.cloud import compute_v1

def sample_patch():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionHealthCheckServicesClient()
    request = compute_v1.PatchRegionHealthCheckServiceRequest(health_check_service='health_check_service_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)