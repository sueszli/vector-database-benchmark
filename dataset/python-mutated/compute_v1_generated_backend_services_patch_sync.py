from google.cloud import compute_v1

def sample_patch():
    if False:
        while True:
            i = 10
    client = compute_v1.BackendServicesClient()
    request = compute_v1.PatchBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.patch(request=request)
    print(response)