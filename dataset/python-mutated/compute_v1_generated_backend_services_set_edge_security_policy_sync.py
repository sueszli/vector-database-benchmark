from google.cloud import compute_v1

def sample_set_edge_security_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.BackendServicesClient()
    request = compute_v1.SetEdgeSecurityPolicyBackendServiceRequest(backend_service='backend_service_value', project='project_value')
    response = client.set_edge_security_policy(request=request)
    print(response)