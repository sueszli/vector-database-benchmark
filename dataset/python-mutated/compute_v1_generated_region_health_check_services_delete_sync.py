from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.RegionHealthCheckServicesClient()
    request = compute_v1.DeleteRegionHealthCheckServiceRequest(health_check_service='health_check_service_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)