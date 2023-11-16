from google.cloud import compute_v1

def sample_delete_signed_url_key():
    if False:
        print('Hello World!')
    client = compute_v1.BackendServicesClient()
    request = compute_v1.DeleteSignedUrlKeyBackendServiceRequest(backend_service='backend_service_value', key_name='key_name_value', project='project_value')
    response = client.delete_signed_url_key(request=request)
    print(response)