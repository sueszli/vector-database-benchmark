from google.cloud import compute_v1

def sample_add_signed_url_key():
    if False:
        return 10
    client = compute_v1.BackendServicesClient()
    request = compute_v1.AddSignedUrlKeyBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.add_signed_url_key(request=request)
    print(response)