from google.cloud import compute_v1

def sample_set_security_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.BackendServicesClient()
    request = compute_v1.SetSecurityPolicyBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.set_security_policy(request=request)
    print(response)