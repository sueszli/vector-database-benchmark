from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.RegionBackendServicesClient()
    request = compute_v1.PatchRegionBackendServiceRequest(backend_service='backend_service_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)