from google.cloud import compute_v1

def sample_update():
    if False:
        while True:
            i = 10
    client = compute_v1.BackendServicesClient()
    request = compute_v1.UpdateBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.update(request=request)
    print(response)