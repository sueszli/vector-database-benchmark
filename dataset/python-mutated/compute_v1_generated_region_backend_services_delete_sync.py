from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.RegionBackendServicesClient()
    request = compute_v1.DeleteRegionBackendServiceRequest(backend_service='backend_service_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)