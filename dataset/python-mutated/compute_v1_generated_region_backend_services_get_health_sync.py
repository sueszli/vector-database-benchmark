from google.cloud import compute_v1

def sample_get_health():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionBackendServicesClient()
    request = compute_v1.GetHealthRegionBackendServiceRequest(backend_service='backend_service_value', project='project_value', region='region_value')
    response = client.get_health(request=request)
    print(response)