from google.cloud import compute_v1

def sample_get_health():
    if False:
        while True:
            i = 10
    client = compute_v1.BackendServicesClient()
    request = compute_v1.GetHealthBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.get_health(request=request)
    print(response)