from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.BackendServicesClient()
    request = compute_v1.DeleteBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.delete(request=request)
    print(response)